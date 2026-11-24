"""Part 3: what the genes and factors mean.

What this file is for
    Parts 1 and 2 ask how well the models predict. Part 3 asks what is behind
    them: which genes go with a worse prognosis, whether the gene lists found
    in different ways share genes, and how the proteins of those genes interact.
    scripts/04_biology.py and scripts/04b_cytoscape.py call the functions here.

    Everything here is descriptive and uses the whole cohort. Nothing in this
    module feeds a prediction, so there is no fold to protect.

The functions, in the order the pipeline uses them
    cox_score_z          one number per gene: does a higher value go with a
                         higher or a lower hazard, and how strongly?
    overlap_test         do two gene lists share more genes than chance?
    string_network       the known interactions among a set of genes (STRING).
    build_graph          turns those interactions into a network with modules.
    export_graph         writes the network as files Cytoscape can open.
    between_edges_test   are two gene lists linked to each other in the network
                         more or less than chance?
    string_enrichment    which pathways a set of genes is known for (STRING).
    module_layout, module_titles, cross_module_links, network_statistics
                         positions, titles and tables for the network figure.

Words used here
    hazard               the rate at which events (progression or death) happen
                         among the patients still being followed.
    z                    an effect divided by its uncertainty. Far from zero
                         (beyond about 2 either way) is unlikely to be chance.
    network (graph)      genes are the points (nodes); an interaction between
                         two proteins is a line (edge) between their genes. The
                         degree of a gene is its number of interactions.
    Louvain module       a group of genes with many interactions inside the
                         group and few to the outside. The Louvain method finds
                         such groups by moving genes between groups for as long
                         as that makes the groups more closed.
    hypergeometric test  the chance of a given overlap if one list were drawn
                         at random: put all genes in a bag, mark those of list
                         a, draw as many as list b has, count the marked ones.
    permutation test     shuffle the thing being tested many times and see how
                         often chance alone gives a result as extreme as the
                         real one.
"""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import requests
from scipy.stats import hypergeom

STRING_API = "https://string-db.org/api"
CALLER = "luad-multiomics"
# Gene Ontology processes, KEGG, Reactome and the MSigDB hallmark sets.
ENRICHMENT_CATEGORIES = ["Process", "KEGG", "RCTM", "HALLMARK"]


# ---------------------------------------------------------------- gene statistics
def cox_score_z(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed score-test z of a univariate Cox model for every column.

    X  patients x features (for example genes), one row per patient.
    y  outcome of the same patients (structured array: event, time).

    Returns one z per column of X. Positive z: higher value, higher hazard.
    Negative z: higher value, lower hazard.

    A univariate Cox model relates the hazard to one feature at a time. Its
    score test needs no fitting, so all columns are done together in one pass.
    At each event, the patient who had it is compared with the risk set R:
    everyone still being followed at that time. For feature x,
    U = sum(x_i - mean_R(x)), V = sum(var_R(x)), z = U / sqrt(V), the sums
    running over the events. U is positive when the patients who have events
    tend to have higher values than those still at risk. Tied times share a
    risk set (Breslow).
    """
    # Sort the patients from the longest follow-up to the shortest. The risk
    # set at any time is then the rows from the top down to that time, and
    # running totals down the rows give its mean and variance.
    order = np.argsort(-y["time"], kind="stable")
    X, events, times = X[order].astype(np.float64), y["event"][order], y["time"][order]
    n_at_risk = np.arange(1, len(times) + 1, dtype=np.float64)[:, None]
    mean_R = np.cumsum(X, axis=0) / n_at_risk
    # Variance as mean of squares minus squared mean; rounding can make it a
    # hair below zero, so it is floored at zero.
    var_R = np.maximum(np.cumsum(X * X, axis=0) / n_at_risk - mean_R ** 2, 0.0)
    # Last row of each block of tied times: the risk set every tied event shares.
    last = np.flatnonzero(np.r_[times[1:] != times[:-1], True])
    risk = last[np.searchsorted(last, np.arange(len(times)))]
    # Add up, over the patients who had an event, the gap between their value
    # and the mean of their risk set (U) and the variance of that risk set (V).
    event_rows = np.flatnonzero(events)
    U = (X[event_rows] - mean_R[risk[event_rows]]).sum(axis=0)
    V = var_R[risk[event_rows]].sum(axis=0)
    # A feature that never varies has V = 0; the floor avoids dividing by zero.
    return U / np.sqrt(np.maximum(V, 1e-12))


def overlap_test(a: set, b: set, universe: int) -> dict:
    """Is the overlap of two gene lists more than chance? Hypergeometric test.

    a, b      the two gene lists, as sets of gene names.
    universe  how many genes either list could have been drawn from.

    Returns a dict: the two list sizes ("n_a", "n_b"), the number of shared
    genes ("overlap"), the number expected by chance ("expected"), the Jaccard
    index ("jaccard": shared genes divided by genes in either list, 0 to 1) and
    "p_value", the chance of an overlap at least this large if the lists were
    unrelated.
    """
    k = len(a & b)
    # sf(k - 1) is the chance of MORE than k - 1 shared genes, that is, of k or
    # more. An empty list cannot overlap with anything, so its p-value is 1.
    p = float(hypergeom.sf(k - 1, universe, len(a), len(b))) if a and b else 1.0
    union = len(a | b)
    return {"n_a": len(a), "n_b": len(b), "overlap": k,
            "expected": len(a) * len(b) / universe if universe else 0.0,
            "jaccard": k / union if union else 0.0, "p_value": p}


# ---------------------------------------------------------------- STRING
# STRING is an online database of known and predicted protein interactions.
# Its answers are saved in results/network/ and reused on every later run, so
# the analysis needs the internet only the first time, and the network cannot
# change under the reader when STRING publishes a new release.

def _ask_string(endpoint: str, data: dict, tries: int = 4) -> str:
    """One question to STRING, repeated after a short wait if the site does
    not answer. Returns the reply as text."""
    for attempt in range(1, tries + 1):
        try:
            r = requests.post(f"{STRING_API}/tsv/{endpoint}", timeout=120,
                              data={**data, "caller_identity": CALLER})
            r.raise_for_status()
            return r.text
        except requests.RequestException as err:
            if attempt == tries:
                raise SystemExit(
                    f"STRING did not answer after {tries} tries ({type(err).__name__}). "
                    "Check the internet connection and run this step again; "
                    "finished steps do not need to be repeated.") from err
            wait = 10 * attempt
            print(f"  STRING did not answer (try {attempt} of {tries}); waiting {wait} s")
            time.sleep(wait)
    return ""


def string_network(genes: list[str], species: int, required_score: int,
                   cache: Path, refresh: bool = False) -> pd.DataFrame:
    """Protein-protein interactions among `genes` from STRING.

    Only gene symbols are sent. The answer is saved in `cache` and read from
    there afterwards; pass refresh=True to ask STRING again.
    """
    if cache.exists() and not refresh:
        return pd.read_csv(cache, sep="\t")
    text = _ask_string("network", {"identifiers": "\r".join(genes), "species": species,
                                   "required_score": required_score})
    df = pd.read_csv(io.StringIO(text), sep="\t")
    df = (df[["preferredName_A", "preferredName_B", "score"]]
          .rename(columns={"preferredName_A": "a", "preferredName_B": "b"})
          .drop_duplicates())
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, sep="\t", index=False)
    return df


def string_enrichment(genes: list[str], species: int, cache: Path,
                      refresh: bool = False, offline: bool = False) -> pd.DataFrame:
    """Which pathways and processes are over-represented among `genes`.

    One saved file holds the answers for every gene set asked so far; a gene
    set is recognised by a short fingerprint of its sorted gene names. With
    offline=True nothing is sent: a gene set that was never asked gives an
    empty table.
    """
    fingerprint = hashlib.sha1(",".join(sorted(genes)).encode()).hexdigest()[:16]
    saved = pd.read_csv(cache, sep="\t") if cache.exists() else pd.DataFrame(columns=["gene_set"])
    if not refresh and (saved["gene_set"] == fingerprint).any():
        return saved[saved["gene_set"] == fingerprint].drop(columns="gene_set").reset_index(drop=True)
    if offline:
        return pd.DataFrame()
    text = _ask_string("enrichment", {"identifiers": "\r".join(genes), "species": species})
    answer = pd.read_csv(io.StringIO(text), sep="\t") if text.strip() else pd.DataFrame()
    if len(answer):
        # Keep the four kinds of term the report uses; the full answer also
        # lists publications and protein domains, which is most of its size.
        answer = answer[answer["category"].isin(ENRICHMENT_CATEGORIES)].reset_index(drop=True)
        saved = pd.concat([saved[saved["gene_set"] != fingerprint],
                           answer.assign(gene_set=fingerprint)], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        saved.to_csv(cache, sep="\t", index=False)
    return answer


# ---------------------------------------------------------------- network
def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame, seed: int) -> nx.Graph:
    """The network: genes as nodes, STRING interactions as edges, with the gene
    table as node attributes and a Louvain module per node.

    edges  one row per interaction, columns a, b (gene names) and score
           (STRING's confidence in the interaction); from `string_network`.
    nodes  one row per gene, indexed by gene name; every column becomes an
           attribute of that gene's node.
    seed   random seed of the Louvain method.

    Returns the graph. Each edge has a "weight" (the score). Each node has the
    columns of `nodes` and "module", a number from 1 (the largest module)
    upwards. A gene with no interaction is not in the graph.
    """
    G = nx.Graph()
    # Keep an interaction only if both genes are in the gene table, and drop
    # an interaction of a gene with itself.
    known = set(nodes.index)
    for row in edges.itertuples():
        if row.a in known and row.b in known and row.a != row.b:
            G.add_edge(row.a, row.b, weight=float(row.score))
    # Copy the gene table onto the nodes. numpy numbers are turned into plain
    # Python numbers, which the network file formats can store.
    for gene in G.nodes:
        for column, value in nodes.loc[gene].items():
            G.nodes[gene][column] = (value.item() if hasattr(value, "item") else value)
    # Find the modules, then number them from the largest to the smallest.
    modules = nx.community.louvain_communities(G, weight="weight", seed=seed)
    for number, members in enumerate(sorted(modules, key=len, reverse=True), start=1):
        for gene in members:
            G.nodes[gene]["module"] = number
    return G


def between_edges_test(G: nx.Graph, a: set, b: set, n_perm: int, seed: int) -> dict:
    """Are genes of list a and list b linked to each other more or less than
    chance? A permutation test.

    G       the network.
    a, b    the two gene lists, as sets of gene names; genes that are not in
            the network are ignored.
    n_perm  number of shuffles.
    seed    random seed of the shuffles.

    The observed number of a-b edges is compared with the same count after
    shuffling which nodes carry which label. The network itself is not
    changed. What a shuffle keeps: the number of network genes in list a, in
    list b and in both. What it does not keep: how many interactions the genes
    of each list have. After a shuffle a list is a random set of network genes,
    so if the real list is made of genes with many (or few) interactions, part
    of the difference from "expected" is due to that alone. The direction of a
    large difference can be trusted; its exact size cannot.

    Returns a dict: "a_in_network" and "b_in_network" (list sizes inside the
    network), "edges_between" (observed), "expected" (mean over the shuffles),
    and "p_fewer" / "p_more", the share of shuffles with at most / at least as
    many edges as observed.
    """
    nodes = list(G.nodes)
    a, b = a & set(nodes), b & set(nodes)

    def count(genes_a, genes_b):
        """Edges with one end in genes_a and the other end in genes_b."""
        return sum(1 for u, v in G.edges
                   if (u in genes_a and v in genes_b) or (u in genes_b and v in genes_a))

    observed = count(a, b)
    rng = np.random.default_rng(seed)
    # One label per node: "a", "b", "ab" for a gene in both lists, or "" for a
    # gene in neither. Shuffling these labels moves the lists around the network.
    labels = np.array([("a" if g in a else "") + ("b" if g in b else "") for g in nodes])
    shuffled_counts = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(labels)
        shuffled_counts[i] = count({g for g, label in zip(nodes, shuffled) if "a" in label},
                                   {g for g, label in zip(nodes, shuffled) if "b" in label})
    # The + 1 on top and bottom counts the real network as one of the shuffles,
    # so a p-value can never be exactly zero.
    return {"a_in_network": len(a), "b_in_network": len(b), "edges_between": observed,
            "expected": float(shuffled_counts.mean()),
            "p_fewer": float((np.sum(shuffled_counts <= observed) + 1) / (n_perm + 1)),
            "p_more": float((np.sum(shuffled_counts >= observed) + 1) / (n_perm + 1))}


def export_graph(G: nx.Graph, out_dir: Path, stem: str = "network") -> None:
    """Write the network as files Cytoscape opens directly (GraphML and
    Cytoscape JSON), plus plain node and edge tables.

    G        the network.
    out_dir  folder to write into; it is created if it does not exist.
    stem     first part of the file names: <stem>.graphml, <stem>.cyjs,
             <stem>_nodes.tsv (one row per gene, with its attributes) and
             <stem>_edges.tsv (one row per interaction, with its score).
    """
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, out_dir / f"{stem}.graphml")
    json.dump(nx.cytoscape_data(G), open(out_dir / f"{stem}.cyjs", "w"))
    pd.DataFrame([{"gene": g, **d} for g, d in G.nodes(data=True)]).to_csv(
        out_dir / f"{stem}_nodes.tsv", sep="\t", index=False)
    pd.DataFrame([{"a": u, "b": v, "score": d["weight"]} for u, v, d in G.edges(data=True)]).to_csv(
        out_dir / f"{stem}_edges.tsv", sep="\t", index=False)


# ---------------------------------------------------------------- figure layout
def module_layout(G: nx.Graph, min_size: int = 4, spacing: float = 95.0, seed: int = 1):
    """Positions that keep each module together as its own cluster.

    A whole-network force layout packs the dense modules into one unreadable
    knot. Here every module of at least `min_size` genes is laid out on its
    own, given room in proportion to its size, and the modules are placed in
    rows, largest first. Smaller fragments are left out of the figure (they
    stay in the tables and the network files).

    G         the network; every node has a "module" attribute (see `build_graph`).
    min_size  smallest module that is drawn.
    spacing   the distance scale of the drawing, in canvas units: about the
              room one gene gets.
    seed      random seed of the force layout.

    Returns (positions {gene: (x, y)}, boxes {module: (centre_x, top_y, radius)}).
    y grows downwards, as on a Cytoscape canvas.
    """
    # The genes of each module, then the modules large enough to draw,
    # largest first.
    members: dict[int, list[str]] = {}
    for g, d in G.nodes(data=True):
        members.setdefault(int(d["module"]), []).append(g)
    big_modules = sorted((m for m, v in members.items() if len(v) >= min_size),
                         key=lambda m: -len(members[m]))

    # Step 1. Lay out each module on its own, centred on (0, 0).
    # local: {module: {gene: position inside the module}}.
    # radius: {module: half the side of the square the module fills}.
    local, radius = {}, {}
    for m in big_modules:
        # Built by hand, in alphabetical order. A subgraph view walks its
        # nodes in the order of a Python set, which changes from one run to
        # the next, and the force layout depends on that order.
        inside = set(members[m])
        sub = nx.Graph()
        sub.add_nodes_from(sorted(inside))
        sub.add_edges_from(sorted((min(u, v), max(u, v)) for u, v in G.edges(inside)
                                  if u in inside and v in inside))
        n = sub.number_of_nodes()
        # A force layout pulls interacting genes together and pushes all genes
        # apart until the picture settles.
        layout = nx.spring_layout(sub, seed=seed, k=3.0 / np.sqrt(n), iterations=300, weight=None)
        xy = np.array([layout[g] for g in sub])
        # Centre the module, then scale it so its size grows with the square
        # root of its number of genes: the area per gene stays the same.
        xy -= xy.mean(axis=0)
        xy *= spacing * np.sqrt(n) / max(np.abs(xy).max(), 1e-9)
        # Open up genes that sit on top of each other, and centre again.
        xy = _spread(xy, min_dist=1.25 * spacing)
        xy -= (xy.max(axis=0) + xy.min(axis=0)) / 2
        radius[m] = float(np.abs(xy).max())
        local[m] = {g: xy[i] for i, g in enumerate(sub)}

    # Step 2. Place the modules side by side in rows, like words on a page.
    # gap is the space between two modules, title the room above a module for
    # its title, min_width the least width a module gets so its title fits.
    gap, title, min_width = 150.0, 110.0, 560.0
    widths = {m: max(2 * radius[m] + gap, min_width) for m in big_modules}
    # The row width is chosen from the total area so that the whole figure
    # comes out wider than it is tall.
    row_width = 1.35 * np.sqrt(sum(w * (2 * radius[m] + gap) for m, w in widths.items()))
    # x, y: top left corner of the next module. row_height: the tallest module
    # in the current row so far.
    pos, boxes, x, y, row_height = {}, {}, 0.0, 0.0, 0.0
    for m in big_modules:
        width = widths[m]
        # Start a new row when this module would not fit in the current one.
        if x > 0 and x + width > row_width:
            x, y, row_height = 0.0, y + row_height + title + gap, 0.0
        # Move the module's genes from around (0, 0) to its place on the page.
        centre_x, centre_y = x + width / 2, y + title + radius[m]
        for g, (dx, dy) in local[m].items():
            pos[g] = (centre_x + dx, centre_y + dy)
        boxes[m] = (centre_x, y, radius[m])
        x, row_height = x + width, max(row_height, 2 * radius[m])
    return pos, boxes


def _spread(xy: np.ndarray, min_dist: float, rounds: int = 400) -> np.ndarray:
    """Push apart any two points closer than min_dist.

    xy        (n_points, 2) positions.
    min_dist  the least distance wanted between two points.
    rounds    the most rounds of pushing; it stops early once no two points
              are too close.

    Returns the new positions, same shape; the input is not changed.

    A force layout puts a densely connected module in a tight knot; this opens
    it up so every gene and label can be read, while neighbours stay neighbours.
    """
    xy = xy.copy()
    # A fixed seed: the random numbers are only used to separate two points
    # that sit exactly on top of each other, and must be the same on every run.
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        # d[i, j] is the arrow from point j to point i; dist[i, j] its length.
        d = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt((d ** 2).sum(axis=2))
        # A point is not too close to itself.
        np.fill_diagonal(dist, np.inf)
        close = dist < min_dist
        if not close.any():
            break
        # Two points at the same place have no direction to be pushed in, so
        # they are given a random one.
        d = np.where(dist[..., None] < 1e-6, rng.normal(size=d.shape), d)
        # Each point is pushed straight away from every point that is too
        # close, by an amount equal to how much too close the two are.
        push = np.where(close[..., None], d / np.maximum(dist[..., None], 1e-6)
                        * (min_dist - np.minimum(dist, min_dist))[..., None], 0.0)
        # Move by a part of the push only, so the points settle step by step
        # and do not jump past each other.
        xy += 0.3 * push.sum(axis=1)
    return xy


def network_statistics(G: nx.Graph, shown: set) -> pd.DataFrame:
    """How central each drawn gene is in the network, one row per gene.

    Degree is the number of interactions a gene has. Betweenness is how often
    it lies on the shortest path between two other genes; closeness is how few
    steps it is from the rest (both inside its own connected piece). The
    clustering coefficient is the share of its neighbours that also interact
    with each other, and neighbourhood connectivity is the mean degree of its
    neighbours. The column names and definitions are those of Cytoscape's
    "Analyze Network", and the values agree with it to 15 decimal places.
    """
    H = G.subgraph(sorted(shown)).copy()
    between, close = {}, {}
    for piece in nx.connected_components(H):
        part = H.subgraph(piece)
        between.update(nx.betweenness_centrality(part, normalized=True))
        close.update(nx.closeness_centrality(part))
    cluster, neighbours = nx.clustering(H), nx.average_neighbor_degree(H)
    rows = [{"name": g, "group": H.nodes[g].get("group"), "module": int(H.nodes[g]["module"]),
             "cox_z": float(H.nodes[g].get("cox_z", 0.0)), "Degree": H.degree[g],
             "BetweennessCentrality": between[g], "ClosenessCentrality": close[g],
             "ClusteringCoefficient": cluster[g], "NeighborhoodConnectivity": neighbours[g]}
            for g in H.nodes]
    # Ties in degree are broken by name so the table is the same on every run.
    return (pd.DataFrame(rows).sort_values(["Degree", "name"], ascending=[False, True])
            .reset_index(drop=True))


def cross_module_links(G: nx.Graph, shown: set) -> dict[tuple[int, int], int]:
    """How many interactions run between each pair of modules.

    G      the network.
    shown  the genes drawn in the figure; an interaction counts only if both
           of its genes are drawn.

    Returns {(module, other module): number of interactions}, the smaller
    module number first. Pairs with no interaction between them are absent.
    """
    out: dict[tuple[int, int], int] = {}
    for u, v in G.edges:
        if u in shown and v in shown:
            # Smaller number first, so that (2, 5) and (5, 2) are one pair.
            a, b = sorted((int(G.nodes[u]["module"]), int(G.nodes[v]["module"])))
            # An interaction inside one module is not a link between modules.
            if a != b:
                out[(a, b)] = out.get((a, b), 0) + 1
    return out


# Enrichment terms that name a disease or a mechanism where a figure needs the
# biology in two words. Used by every figure that titles a module.
SHORT_TERMS = {"Resolution of Sister Chromatid Cohesion": "Sister chromatid cohesion",
               "Formation of the cornified envelope": "Cornified envelope",
               "Defective GALNT3 causes HFTC": "Mucins (O-glycosylation)"}


def module_titles(G: nx.Graph, modules: pd.DataFrame, boxes: dict) -> dict[int, str]:
    """The title drawn above each module in the figure.

    G        the network (not used; kept so every figure helper is called alike).
    modules  the module table: columns "module", "genes" (how many), "hub_genes"
             (names separated by ", ") and, if enrichment was run, "top_term".
    boxes    the modules that are drawn, as returned by `module_layout`.

    Returns {module: title}, for example
    "Module 4: Surfactant metabolism" with "(27 genes)" on a second line. A
    module with no enriched term is named after its first two hub genes.
    """
    out = {}
    info = modules.set_index("module")
    for m in boxes:
        term = info.loc[m, "top_term"] if "top_term" in info.columns else None
        # An empty cell of the table is read back as NaN, which is not text.
        if not isinstance(term, str):
            term = ", ".join(str(info.loc[m, "hub_genes"]).split(", ")[:2]) + ", ..."
        out[m] = f"Module {m}: {SHORT_TERMS.get(term, term)}\n({int(info.loc[m, 'genes'])} genes)"
    return out
