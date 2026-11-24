"""Part 3: what the genes and factors mean.

Everything here is descriptive and uses the whole cohort. Nothing in this
module feeds a prediction, so there is no fold to protect.
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

    Closed form, one pass: for covariate x with risk set R at each event,
    U = sum(x_i - mean_R(x)), V = sum(var_R(x)), z = U / sqrt(V). Positive z:
    higher value, higher hazard. Tied times share a risk set (Breslow).
    """
    order = np.argsort(-y["time"], kind="stable")
    X, ev, t = X[order].astype(np.float64), y["event"][order], y["time"][order]
    n = np.arange(1, len(t) + 1, dtype=np.float64)[:, None]
    mean_R = np.cumsum(X, axis=0) / n
    var_R = np.maximum(np.cumsum(X * X, axis=0) / n - mean_R ** 2, 0.0)
    # Last row of each block of tied times: the risk set every tied event shares.
    last = np.flatnonzero(np.r_[t[1:] != t[:-1], True])
    risk = last[np.searchsorted(last, np.arange(len(t)))]
    rows = np.flatnonzero(ev)
    U = (X[rows] - mean_R[risk[rows]]).sum(axis=0)
    V = var_R[risk[rows]].sum(axis=0)
    return U / np.sqrt(np.maximum(V, 1e-12))


def overlap_test(a: set, b: set, universe: int) -> dict:
    """Is the overlap of two gene lists more than chance? Hypergeometric."""
    k = len(a & b)
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
    """Graph over the genes STRING connected, with the gene table as node
    attributes and a Louvain module per node."""
    G = nx.Graph()
    known = set(nodes.index)
    for r in edges.itertuples():
        if r.a in known and r.b in known and r.a != r.b:
            G.add_edge(r.a, r.b, weight=float(r.score))
    for g in G.nodes:
        for col, val in nodes.loc[g].items():
            G.nodes[g][col] = (val.item() if hasattr(val, "item") else val)
    modules = nx.community.louvain_communities(G, weight="weight", seed=seed)
    for i, members in enumerate(sorted(modules, key=len, reverse=True), start=1):
        for g in members:
            G.nodes[g]["module"] = i
    return G


def between_edges_test(G: nx.Graph, a: set, b: set, n_perm: int, seed: int) -> dict:
    """Are genes of list a and list b linked to each other more or less than
    chance? Observed a-b edges against the same count after shuffling which
    nodes carry which label (degrees and list sizes unchanged)."""
    nodes = list(G.nodes)
    a, b = a & set(nodes), b & set(nodes)

    def count(sa, sb):
        return sum(1 for u, v in G.edges if (u in sa and v in sb) or (u in sb and v in sa))

    observed = count(a, b)
    rng = np.random.default_rng(seed)
    labels = np.array([("a" if g in a else "") + ("b" if g in b else "") for g in nodes])
    perm = np.empty(n_perm)
    for i in range(n_perm):
        lab = rng.permutation(labels)
        perm[i] = count({g for g, l in zip(nodes, lab) if "a" in l},
                        {g for g, l in zip(nodes, lab) if "b" in l})
    return {"a_in_network": len(a), "b_in_network": len(b), "edges_between": observed,
            "expected": float(perm.mean()),
            "p_fewer": float((np.sum(perm <= observed) + 1) / (n_perm + 1)),
            "p_more": float((np.sum(perm >= observed) + 1) / (n_perm + 1))}


def export_graph(G: nx.Graph, out_dir: Path, stem: str = "network") -> None:
    """Files Cytoscape opens directly: GraphML and Cytoscape JSON, plus plain
    node and edge tables."""
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

    Returns (positions {gene: (x, y)}, boxes {module: (centre_x, top_y, radius)}).
    y grows downwards, as on a Cytoscape canvas.
    """
    members: dict[int, list[str]] = {}
    for g, d in G.nodes(data=True):
        members.setdefault(int(d["module"]), []).append(g)
    mods = sorted((m for m, v in members.items() if len(v) >= min_size),
                  key=lambda m: -len(members[m]))

    local, radius = {}, {}
    for m in mods:
        # Built by hand, in alphabetical order. A subgraph view walks its
        # nodes in the order of a Python set, which changes from one run to
        # the next, and the force layout depends on that order.
        inside = set(members[m])
        sub = nx.Graph()
        sub.add_nodes_from(sorted(inside))
        sub.add_edges_from(sorted((min(u, v), max(u, v)) for u, v in G.edges(inside)
                                  if u in inside and v in inside))
        n = sub.number_of_nodes()
        p = nx.spring_layout(sub, seed=seed, k=3.0 / np.sqrt(n), iterations=300, weight=None)
        xy = np.array([p[g] for g in sub])
        xy -= xy.mean(axis=0)
        xy *= spacing * np.sqrt(n) / max(np.abs(xy).max(), 1e-9)
        xy = _spread(xy, min_dist=1.25 * spacing)
        xy -= (xy.max(axis=0) + xy.min(axis=0)) / 2
        radius[m] = float(np.abs(xy).max())
        local[m] = {g: xy[i] for i, g in enumerate(sub)}

    gap, title, min_width = 150.0, 110.0, 560.0
    widths = {m: max(2 * radius[m] + gap, min_width) for m in mods}
    row_width = 1.35 * np.sqrt(sum(w * (2 * radius[m] + gap) for m, w in widths.items()))
    pos, boxes, x, y, row_h = {}, {}, 0.0, 0.0, 0.0
    for m in mods:
        w = widths[m]
        if x > 0 and x + w > row_width:
            x, y, row_h = 0.0, y + row_h + title + gap, 0.0
        cx, cy = x + w / 2, y + title + radius[m]
        for g, (dx, dy) in local[m].items():
            pos[g] = (cx + dx, cy + dy)
        boxes[m] = (cx, y, radius[m])
        x, row_h = x + w, max(row_h, 2 * radius[m])
    return pos, boxes


def _spread(xy: np.ndarray, min_dist: float, rounds: int = 400) -> np.ndarray:
    """Push apart any two points closer than min_dist. A force layout puts a
    densely connected module in a tight knot; this opens it up so every gene
    and label can be read, while neighbours stay neighbours."""
    xy = xy.copy()
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        d = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt((d ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        close = dist < min_dist
        if not close.any():
            break
        d = np.where(dist[..., None] < 1e-6, rng.normal(size=d.shape), d)
        push = np.where(close[..., None], d / np.maximum(dist[..., None], 1e-6)
                        * (min_dist - np.minimum(dist, min_dist))[..., None], 0.0)
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
    """How many interactions run between each pair of modules."""
    out: dict[tuple[int, int], int] = {}
    for u, v in G.edges:
        if u in shown and v in shown:
            a, b = sorted((int(G.nodes[u]["module"]), int(G.nodes[v]["module"])))
            if a != b:
                out[(a, b)] = out.get((a, b), 0) + 1
    return out


# Enrichment terms that name a disease or a mechanism where a figure needs the
# biology in two words. Used by every figure that titles a module.
SHORT_TERMS = {"Resolution of Sister Chromatid Cohesion": "Sister chromatid cohesion",
               "Formation of the cornified envelope": "Cornified envelope",
               "Defective GALNT3 causes HFTC": "Mucins (O-glycosylation)"}


def module_titles(G: nx.Graph, modules: pd.DataFrame, boxes: dict) -> dict[int, str]:
    """"Module 4: Surfactant metabolism (27 genes)"; hub genes where no term was enriched."""
    out = {}
    info = modules.set_index("module")
    for m in boxes:
        term = info.loc[m, "top_term"] if "top_term" in info.columns else None
        if not isinstance(term, str):
            term = ", ".join(str(info.loc[m, "hub_genes"]).split(", ")[:2]) + ", ..."
        out[m] = f"Module {m}: {SHORT_TERMS.get(term, term)}\n({int(info.loc[m, 'genes'])} genes)"
    return out
