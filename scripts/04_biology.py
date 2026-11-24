#!/usr/bin/env python
"""Step 4. Biology: factors, gene lists, and a gene network for Cytoscape.

    python scripts/04_biology.py
    python scripts/04_biology.py --refresh     # fetch the STRING network again
    python scripts/04_biology.py --offline     # no STRING calls; use the cached network only

What it is for
    Steps 1 to 3 ask how well the models predict. This step asks what is
    behind them. It is descriptive and uses the whole cohort (no folds).
    These gene lists are built:

    prognostic_risk / prognostic_protective   genes whose expression alone tracks the endpoint
    model                                     genes the penalised Cox model kept, across folds
    factor<k>                                 top genes of each survival-associated MOFA factor
    proliferation                             a fixed reference metagene

    They are put on the STRING protein-interaction network, split into modules,
    and the modules are tested for enrichment. The question the network answers:
    are the genes the model uses the same biology as the genes that are
    prognostic, or a different part of the network?

What it reads
    config/config.yaml                     the `biology` section: cut-offs and list sizes
    data/processed/, data/interim/         the cohort (expression, mutation, clinical, endpoint)
    results/tables/cv_<endpoint>_info.csv  which features each fold of step 1 kept
    results/network/string_edges.tsv       the saved STRING network, if there is one
    results/network/string_enrichment.tsv  the saved STRING enrichment answers, if any

What it writes
    results/tables/biology_factors.csv             one row per MOFA factor
    results/tables/biology_genes.csv               one row per gene in any list
    results/tables/biology_overlap.csv             genes shared by each pair of lists
    results/tables/biology_network_links.csv       interactions between pairs of lists
    results/tables/biology_network_centrality.csv  how central each drawn gene is
    results/tables/biology_modules.csv             one row per network module
    results/tables/biology_enrichment.csv          the top enriched terms of each module
    results/figures/network.png                    the network, one cluster per module
    results/network/                               network.graphml and network.cyjs to open in
                                                   Cytoscape, network_nodes.tsv, network_edges.tsv

How long it takes
    Under 2 minutes; 15 seconds when the MOFA fit is cached. It needs the
    internet only the first time: the gene symbols are sent to STRING
    (string-db.org) and the answers are saved in results/network/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_config, out_dirs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import norm

from luad import biology as B
from luad.data import OMICS, load_cohort
from luad.integrate import cached_mofa
from luad.transforms import build_blocks, expressed_genes

# One colour per gene list in results/figures/network.png. "factor" is used
# for every MOFA factor list; "several" is a gene that is in more than one list.
COLOURS = {"prognostic_risk": "#c2541c", "prognostic_protective": "#2a78d6",
           "model": "#1baf7a", "factor": "#b04a9a", "proliferation": "#eda100",
           "several": "#0b0b0b"}


def factor_table(cohort, cfg, seed, cache):
    """Fit MOFA on the whole cohort and describe every factor.

    Arguments
        cohort  the analysis cohort, from load_cohort()
        cfg     the configuration, from load_config()
        seed    the project seed, so the fit is the same on every run
        cache   folder where a finished MOFA fit is saved and found again

    Returns two tables:
        factors  one row per active factor: its Cox z with the endpoint, the
                 share of variance it explains in each omics block (r2_*), and
                 its correlation with stage, age, sex, smoking and tumour
                 mutation burden (r_*)
        weights  the weight of every expression gene (rows) on every active
                 factor (columns)
    """
    # The same blocks a fold would build, but with every patient as "training":
    # nothing is predicted here, so there is no test set to protect.
    everyone = np.arange(cohort.n)
    wide = build_blocks(cohort, everyone, cfg, seed, int(cfg["mofa"]["top_k"]))
    views = {v: wide[v].X for v in OMICS}
    mofa = cached_mofa(views, cfg, seed, cache)

    # Each patient's value on each factor, and how strongly each factor tracks
    # the endpoint on its own.
    Z = mofa.project(views)
    z = B.cox_score_z(Z, cohort.y)

    # What a factor is: which block it explains, and which patient
    # characteristics it follows.
    clinical = pd.DataFrame(wide["clinical"].X, columns=cohort.clinical_names)
    rows = []
    for j, k in enumerate(mofa.active):
        row = {"factor": f"factor{k + 1}", "cox_z": z[j]}
        for i, v in enumerate(mofa.views):
            row[f"r2_{v}"] = mofa.r2[i, k]
        for name in ("stage_ordinal", "age", "sex_male", "ever_smoker"):
            if name in clinical:
                row[f"r_{name}"] = float(np.corrcoef(Z[:, j], clinical[name])[0, 1])
        tmb = cohort.mut_feat[:, cohort.mut_feat_names.index("tmb_log10")]
        row["r_tmb"] = float(np.corrcoef(Z[:, j], tmb)[0, 1])
        rows.append(row)

    # Expression features are named "rna__<gene>"; the weights table is indexed
    # by the gene symbol alone so it can be matched with the gene lists.
    genes = np.array([n.split("__", 1)[1] for n in wide["rna"].names])
    weights = pd.DataFrame(mofa.W["rna"][:, mofa.active], index=genes,
                           columns=[f"factor{k + 1}" for k in mofa.active])
    return pd.DataFrame(rows), weights


def model_genes(tables: Path, endpoint: str, min_folds: int) -> pd.Series:
    """How many folds the penalised Cox model kept each expression gene in,
    across the arms that contain expression.

    Arguments
        tables     folder of the result tables (results/tables)
        endpoint   "pfi" or "os": which cross-validation run to read
        min_folds  a gene is returned only if it was kept in at least this many folds

    Returns a Series, gene symbol -> number of folds, most often kept first.
    """
    info = pd.read_csv(tables / f"cv_{endpoint}_info.csv")
    info = info[(info["model"] == "coxnet") & info["arm"].isin(["rna", "clinical+rna", "early"])]
    # A gene counts once per fold, however many of the three arms kept it
    # there: the folds are collected in a set.
    counts: dict[str, set] = {}
    for row in info.itertuples():
        if isinstance(row.selected, str):          # an empty cell is read as NaN, not text
            for name in row.selected.split(";"):
                if name.startswith("rna__"):
                    counts.setdefault(name[5:], set()).add(row.fold)
    n_folds = pd.Series({g: len(f) for g, f in counts.items()}, dtype=float)
    n_folds = n_folds.sort_values(ascending=False)
    return n_folds[n_folds >= min_folds]


# The legend text for each colour in COLOURS.
GROUP_LABELS = {"prognostic_risk": "Prognostic, higher risk",
                "prognostic_protective": "Prognostic, protective",
                "model": "Kept by the Cox model", "factor": "MOFA factor gene",
                "proliferation": "Proliferation metagene", "several": "In more than one list"}


def draw(G: nx.Graph, modules: pd.DataFrame, path: Path) -> None:
    """Draw the network: one cluster per module, titled, every gene labelled,
    with a legend.

    Arguments
        G        the gene network; every node carries `module`, `group` and `cox_z`
        modules  the module table (biology_modules.csv), used for the titles
        path     where the PNG is written

    Returns nothing. Colour is the gene list, marker size is |Cox z|.
    """
    # Where every gene goes. Only modules of 4 or more genes get a place, so
    # H, the part of the network that is drawn, can be smaller than G.
    pos, boxes = B.module_layout(G)
    titles = B.module_titles(G, modules, boxes)
    H = G.subgraph(pos)
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    # The page grows with the layout, so the gene labels keep the same size
    # whatever the number of modules.
    width, height = xs.max() - xs.min() + 400, ys.max() - ys.min() + 500
    fig, ax = plt.subplots(figsize=(width / 190, height / 190))

    # Interactions inside a module are drawn. Interactions between modules
    # would cross the whole page, so each pair of modules gets one dashed
    # line with the number of interactions on it (pairs with at least 5).
    for u, v in H.edges:
        if H.nodes[u]["module"] == H.nodes[v]["module"]:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#8a8a8a",
                    lw=0.5, alpha=0.3, zorder=1)
    centre = {m: (cx, top + 110 + r) for m, (cx, top, r) in boxes.items()}
    for (module_a, module_b), n_links in B.cross_module_links(G, set(pos)).items():
        if n_links >= 5:
            (x1, y1), (x2, y2) = centre[module_a], centre[module_b]
            ax.plot([x1, x2], [y1, y2], color="#555555", lw=0.6 + np.log10(n_links), ls=(0, (4, 3)),
                    alpha=0.5, zorder=0)
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{n_links} links", fontsize=8, ha="center",
                    va="center", color="#333333", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", lw=0.5))

    # The genes: one marker each, with the gene symbol just below it. A larger
    # marker is a stronger association with the endpoint (capped at |z| = 6).
    for gene in H:
        attributes = H.nodes[gene]
        ax.scatter(*pos[gene], s=60 + 45 * min(abs(attributes["cox_z"]), 6),
                   color=COLOURS[attributes["group"]], edgecolor="white", linewidth=0.8, zorder=2)
        ax.annotate(gene, pos[gene], xytext=(0, -9), textcoords="offset points", ha="center",
                    va="top", fontsize=6.5, color="#222222", zorder=3)

    # The module titles, above each cluster.
    for module, (centre_x, top, _radius) in boxes.items():
        ax.text(centre_x, top + 40, titles[module], ha="center", va="center", fontsize=9,
                fontweight="bold", color="#111111", linespacing=1.3)

    # The legend. Empty scatters draw nothing on the page; they only give the
    # legend its entries: the colours first, then three marker sizes.
    for name, colour in COLOURS.items():
        ax.scatter([], [], color=colour, s=90, label=GROUP_LABELS[name])
    for z in (1, 3, 5):
        ax.scatter([], [], color="#bdbdbd", s=60 + 45 * z, label=f"|Cox z| = {z}")
    ax.legend(fontsize=8.5, loc="lower center", ncol=5, frameon=False,
              bbox_to_anchor=(0.5, -0.02), columnspacing=1.6, handletextpad=0.5)

    # The title says how many genes are not drawn, so nobody takes the figure
    # for the whole network.
    left_out = G.number_of_nodes() - H.number_of_nodes()
    ax.set_title(f"Protein-interaction network of the gene lists, by module "
                 f"({H.number_of_nodes()} genes shown; {left_out} in fragments of fewer than 4 not drawn)\n"
                 "Lines inside a module are interactions; "
                 "a dashed line counts the interactions between two modules",
                 fontsize=11, pad=12)
    ax.set_xlim(xs.min() - 200, xs.max() + 200)
    ax.set_ylim(ys.max() + 330, ys.min() - 130)       # y grows downwards
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    """Run the whole step: gene lists, tables, network, modules, figure.

    Takes its options from the command line. Returns 0, the exit code of a
    run that finished.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true",
                        help="ask STRING again instead of using the saved answers")
    parser.add_argument("--offline", action="store_true",
                        help="send nothing to STRING; use the saved answers only")
    args = parser.parse_args()

    cfg = load_config()
    bio, seed = cfg["biology"], int(cfg["project"]["seed"])
    endpoint = cfg["endpoint"]["primary"].lower()
    tables, figures = out_dirs(cfg)
    net_dir = Path(cfg["paths"]["network"])
    cohort = load_cohort(cfg, endpoint)

    # ---- gene-level association with the endpoint ----------------------
    # One Cox z per expressed gene, each gene on its own. This is the
    # "prognostic" view: what a gene says about the endpoint without a model.
    expressed = expressed_genes(cohort, np.arange(cohort.n), cfg)
    genes = cohort.rna_genes[expressed]
    z = pd.Series(B.cox_score_z(cohort.rna[:, expressed], cohort.y), index=genes)
    z = z[~z.index.duplicated()]        # if a symbol occurs twice, the first one is kept
    cut, top = float(bio["prognostic_z"]), int(bio["max_genes_per_list"])
    # How many genes pass the cut-off, beside how many would pass if no gene
    # were related to the endpoint: the share of a normal distribution that
    # lies beyond the cut-off on either side (0.0027 for a cut-off of 3).
    n_tail = int((z.abs() > cut).sum())
    expected = 2 * norm.sf(cut) * len(z)
    print(f"  {n_tail} of {len(z)} expressed genes at |z| > {cut} "
          f"(about {expected:.0f} expected by chance)")

    # The strongest genes on each side, and the reference metagene.
    lists = {
        "prognostic_risk": list(z[z > cut].sort_values(ascending=False).index[:top]),
        "prognostic_protective": list(z[z < -cut].sort_values().index[:top]),
        "proliferation": [g for g in bio["proliferation_metagene"] if g in z.index],
    }

    # ---- genes the model kept ------------------------------------------
    # The "model" view: the genes the penalised Cox model chose in step 1,
    # where every gene competes with all the others.
    kept = model_genes(tables, endpoint, int(bio["model_gene_min_folds"]))
    lists["model"] = list(kept.index[:top])

    # ---- MOFA factors ---------------------------------------------------
    # A factor gets a gene list only if it is associated with the endpoint
    # (|Cox z| > 2): the genes with the largest weights on it.
    factors, weights = factor_table(cohort, cfg, seed, Path(cfg["paths"]["cache"]) / "mofa")
    factors.to_csv(tables / "biology_factors.csv", index=False)
    print(factors.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    for factor_name in factors.loc[factors["cox_z"].abs() > 2, "factor"]:
        factor_weights = weights[factor_name]
        lists[factor_name] = list(factor_weights.abs().sort_values(ascending=False)
                                  .index[:int(bio["factor_top_genes"])])

    # ---- one row per gene ------------------------------------------------
    # The gene table: every gene that is in at least one list, with its Cox z,
    # the folds that kept it, one True/False column per list, and its factor.
    union = sorted({g for members in lists.values() for g in members})
    nodes = pd.DataFrame(index=union)
    nodes["cox_z"] = z.reindex(union).fillna(0.0).to_numpy()
    nodes["folds_selected"] = kept.reindex(union).fillna(0).astype(int).to_numpy()
    for name, members in lists.items():
        nodes[name] = nodes.index.isin(members)
    # Only genes in the MOFA expression view (the most variable genes) have a weight.
    weights = weights[~weights.index.duplicated()]
    in_view = [g for g in union if g in weights.index]
    top_factor = weights.loc[in_view].abs().idxmax(axis=1)
    nodes["top_factor"] = top_factor.reindex(union).fillna("none").to_numpy()
    nodes["top_factor_weight"] = [float(weights.loc[g, top_factor[g]]) if g in top_factor.index else 0.0
                                  for g in union]

    def group(row):
        """The one label that colours a gene in the figures: its list,
        "factor" for any factor list, or "several" if it is in more than one."""
        hit = [n for n in lists if row[n]]
        if len(hit) > 1:
            return "several"
        return "factor" if hit[0].startswith("factor") else hit[0]
    nodes["group"] = nodes.apply(group, axis=1)
    nodes.rename_axis("gene").to_csv(tables / "biology_genes.csv")
    print("  gene lists: " + ", ".join(f"{k} {len(v)}" for k, v in lists.items()))

    # ---- do the lists overlap? -------------------------------------------
    # The first way to compare two lists: do they share genes, and is that
    # more than chance? Risk and protective genes are taken together here.
    prognostic = set(lists["prognostic_risk"]) | set(lists["prognostic_protective"])
    named = {"prognostic": prognostic, **{k: set(v) for k, v in lists.items()
                                         if not k.startswith("prognostic_")}}
    names = list(named)
    overlap = pd.DataFrame([{"list_a": a, "list_b": b, **B.overlap_test(named[a], named[b], len(z))}
                            for i, a in enumerate(names) for b in names[i + 1:]])
    overlap.to_csv(tables / "biology_overlap.csv", index=False)

    # ---- network ----------------------------------------------------------
    # Two lists can share no gene and still be the same biology, if their
    # proteins interact. STRING gives the interactions; its answer is saved,
    # so only the first run needs the internet.
    cache = net_dir / "string_edges.tsv"
    if args.offline and not cache.exists():
        print("  offline and no cached STRING network: stopping after the gene tables")
        return 0
    string_settings = bio["string"]
    edges = B.string_network(union, int(string_settings["species"]),
                             int(string_settings["required_score"]), cache, args.refresh)
    # The graph holds the genes with at least one interaction; build_graph also
    # gives every gene its module. The files for Cytoscape are written here.
    G = B.build_graph(edges, nodes, seed)
    B.export_graph(G, net_dir)
    print(f"  network: {G.number_of_nodes()} of {len(union)} genes connected, {G.number_of_edges()} edges")

    # ---- are model genes and prognostic genes linked? ---------------------
    # The second way to compare two lists: the number of interactions between
    # them, against the number after shuffling the list labels 2,000 times.
    tests = []
    for a, b in [("model", "prognostic"), ("model", "proliferation"), ("prognostic", "proliferation")]:
        if named.get(a) and named.get(b):
            tests.append({"list_a": a, "list_b": b,
                          **B.between_edges_test(G, named[a], named[b], 2000, seed)})
    pd.DataFrame(tests).to_csv(tables / "biology_network_links.csv", index=False)

    # ---- how central is each gene? -----------------------------------------
    # On the genes the figures draw (modules of 4 or more). Step 4b checks
    # these numbers against Cytoscape's own analysis when Cytoscape is there.
    drawn, _ = B.module_layout(G)
    B.network_statistics(G, set(drawn)).to_csv(tables / "biology_network_centrality.csv", index=False)

    # ---- modules and what they are enriched for ---------------------------
    # One row per module: its size, how many genes of each list it holds, and
    # its best-connected genes. A module of 8 or more genes is also sent to
    # STRING, to ask which pathways are over-represented in it; the five best
    # terms are kept and the first one becomes the module's title.
    node_table = pd.DataFrame([{"gene": g, **d} for g, d in G.nodes(data=True)])
    rows, enrich = [], []
    for module, members_table in node_table.groupby("module"):
        row = {"module": module, "genes": len(members_table), "mean_cox_z": members_table["cox_z"].mean(),
               **{name: int(members_table[name].sum()) for name in lists},
               "hub_genes": ", ".join(sorted(members_table["gene"], key=lambda x: -G.degree[x])[:8])}
        if len(members_table) >= 8:
            terms = B.string_enrichment(list(members_table["gene"]), int(string_settings["species"]),
                                        net_dir / "string_enrichment.tsv", args.refresh, args.offline)
            if len(terms):
                terms = terms.sort_values("fdr").head(5)
                for term in terms.itertuples():
                    enrich.append({"module": module, "category": term.category, "term": term.description,
                                   "genes_in_term": int(term.number_of_genes), "fdr": float(term.fdr)})
                if len(terms):
                    row["top_term"] = terms.iloc[0]["description"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables / "biology_modules.csv", index=False)

    # ---- figure and last messages ------------------------------------------
    draw(G, pd.DataFrame(rows), figures / "network.png")
    # Always written, even when empty, so that a table left by an earlier run
    # can never be mistaken for this run's result.
    pd.DataFrame(enrich, columns=["module", "category", "term", "genes_in_term", "fdr"]).to_csv(
        tables / "biology_enrichment.csv", index=False)
    print(pd.DataFrame(rows).head(10).to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print(f"\n  open in Cytoscape: {net_dir}/network.graphml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
