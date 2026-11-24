#!/usr/bin/env python
"""Step 4. Biology: factors, gene lists, and a gene network for Cytoscape.

    python scripts/04_biology.py
    python scripts/04_biology.py --refresh     # fetch the STRING network again
    python scripts/04_biology.py --offline     # no STRING calls; use the cached network only

Descriptive, on the whole cohort. Four gene lists are built:

    prognostic_risk / prognostic_protective   genes whose expression alone tracks the endpoint
    model                                     genes the penalised Cox model kept, across folds
    factor<k>                                 top genes of each survival-associated MOFA factor
    proliferation                             a fixed reference metagene

They are put on the STRING protein-interaction network, split into modules,
and the modules are tested for enrichment. The question the network answers:
are the genes the model uses the same biology as the genes that are
prognostic, or a different part of the network?

Writes results/tables/biology_*.csv, results/figures/network.png and, in
results/network/, network.graphml and network.cyjs to open in Cytoscape.
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

from luad import biology as B
from luad.data import OMICS, load_cohort
from luad.integrate import cached_mofa
from luad.transforms import build_blocks, expressed_genes

COLOURS = {"prognostic_risk": "#c2541c", "prognostic_protective": "#2a78d6",
           "model": "#1baf7a", "factor": "#b04a9a", "proliferation": "#eda100",
           "several": "#0b0b0b"}


def factor_table(c, cfg, seed, cache):
    """MOFA on the whole cohort; one row per factor."""
    everyone = np.arange(c.n)
    wide = build_blocks(c, everyone, cfg, seed, int(cfg["mofa"]["top_k"]))
    views = {v: wide[v].X for v in OMICS}
    mofa = cached_mofa(views, cfg, seed, cache)
    Z = mofa.project(views)
    z = B.cox_score_z(Z, c.y)
    clin = pd.DataFrame(wide["clinical"].X, columns=c.clinical_names)
    rows = []
    for j, k in enumerate(mofa.active):
        row = {"factor": f"factor{k + 1}", "cox_z": z[j]}
        for i, v in enumerate(mofa.views):
            row[f"r2_{v}"] = mofa.r2[i, k]
        for name in ("stage_ordinal", "age", "sex_male", "ever_smoker"):
            if name in clin:
                row[f"r_{name}"] = float(np.corrcoef(Z[:, j], clin[name])[0, 1])
        tmb = c.mut_feat[:, c.mut_feat_names.index("tmb_log10")]
        row["r_tmb"] = float(np.corrcoef(Z[:, j], tmb)[0, 1])
        rows.append(row)
    genes = np.array([n.split("__", 1)[1] for n in wide["rna"].names])
    weights = pd.DataFrame(mofa.W["rna"][:, mofa.active], index=genes,
                           columns=[f"factor{k + 1}" for k in mofa.active])
    return pd.DataFrame(rows), weights


def model_genes(tables: Path, endpoint: str, min_folds: int) -> pd.Series:
    """How many folds the penalised Cox model kept each expression gene in,
    across the arms that contain expression."""
    info = pd.read_csv(tables / f"cv_{endpoint}_info.csv")
    info = info[(info["model"] == "coxnet") & info["arm"].isin(["rna", "clinical+rna", "early"])]
    counts: dict[str, set] = {}
    for r in info.itertuples():
        if isinstance(r.selected, str):
            for name in r.selected.split(";"):
                if name.startswith("rna__"):
                    counts.setdefault(name[5:], set()).add(r.fold)
    s = pd.Series({g: len(f) for g, f in counts.items()}, dtype=float).sort_values(ascending=False)
    return s[s >= min_folds]


GROUP_LABELS = {"prognostic_risk": "Prognostic, higher risk", "prognostic_protective": "Prognostic, protective",
                "model": "Kept by the Cox model", "factor": "MOFA factor gene",
                "proliferation": "Proliferation metagene", "several": "In more than one list"}


def draw(G: nx.Graph, modules: pd.DataFrame, path: Path) -> None:
    """One cluster per module, titled, every gene labelled, with a legend."""
    pos, boxes = B.module_layout(G)
    titles = B.module_titles(G, modules, boxes)
    H = G.subgraph(pos)
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    w, h = xs.max() - xs.min() + 400, ys.max() - ys.min() + 500
    fig, ax = plt.subplots(figsize=(w / 190, h / 190))
    # Interactions inside a module are drawn. Interactions between modules
    # would cross the whole page, so each pair of modules gets one dashed
    # line with the number of interactions on it (pairs with at least 5).
    for u, v in H.edges:
        if H.nodes[u]["module"] == H.nodes[v]["module"]:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#8a8a8a",
                    lw=0.5, alpha=0.3, zorder=1)
    centre = {m: (cx, top + 110 + r) for m, (cx, top, r) in boxes.items()}
    for (a, b), n in B.cross_module_links(G, set(pos)).items():
        if n >= 5:
            (x1, y1), (x2, y2) = centre[a], centre[b]
            ax.plot([x1, x2], [y1, y2], color="#555555", lw=0.6 + np.log10(n), ls=(0, (4, 3)),
                    alpha=0.5, zorder=0)
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{n} links", fontsize=8, ha="center",
                    va="center", color="#333333", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", lw=0.5))
    for g in H:
        d = H.nodes[g]
        ax.scatter(*pos[g], s=60 + 45 * min(abs(d["cox_z"]), 6), color=COLOURS[d["group"]],
                   edgecolor="white", linewidth=0.8, zorder=2)
        ax.annotate(g, pos[g], xytext=(0, -9), textcoords="offset points", ha="center",
                    va="top", fontsize=6.5, color="#222222", zorder=3)
    for m, (cx, top, r) in boxes.items():
        ax.text(cx, top + 40, titles[m], ha="center", va="center", fontsize=9,
                fontweight="bold", color="#111111", linespacing=1.3)
    for name, col in COLOURS.items():
        ax.scatter([], [], color=col, s=90, label=GROUP_LABELS[name])
    for z in (1, 3, 5):
        ax.scatter([], [], color="#bdbdbd", s=60 + 45 * z, label=f"|Cox z| = {z}")
    ax.legend(fontsize=8.5, loc="lower center", ncol=5, frameon=False,
              bbox_to_anchor=(0.5, -0.02), columnspacing=1.6, handletextpad=0.5)
    left_out = G.number_of_nodes() - H.number_of_nodes()
    ax.set_title(f"Protein-interaction network of the gene lists, by module "
                 f"({H.number_of_nodes()} genes shown; {left_out} in fragments of fewer than 4 not drawn)\n"
                 "Lines inside a module are interactions; a dashed line counts the interactions between two modules",
                 fontsize=11, pad=12)
    ax.set_xlim(xs.min() - 200, xs.max() + 200)
    ax.set_ylim(ys.max() + 330, ys.min() - 130)       # y grows downwards
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    bio, seed = cfg["biology"], int(cfg["project"]["seed"])
    endpoint = cfg["endpoint"]["primary"].lower()
    tables, figures = out_dirs(cfg)
    net_dir = Path(cfg["paths"]["network"])
    c = load_cohort(cfg, endpoint)

    # ---- gene-level association with the endpoint ----------------------
    expressed = expressed_genes(c, np.arange(c.n), cfg)
    genes = c.rna_genes[expressed]
    z = pd.Series(B.cox_score_z(c.rna[:, expressed], c.y), index=genes)
    z = z[~z.index.duplicated()]
    cut, top = float(bio["prognostic_z"]), int(bio["max_genes_per_list"])
    n_tail = int((z.abs() > cut).sum())
    expected = 0.0027 * len(z)
    print(f"  {n_tail} of {len(z)} expressed genes at |z| > {cut} (about {expected:.0f} expected by chance)")

    lists = {
        "prognostic_risk": list(z[z > cut].sort_values(ascending=False).index[:top]),
        "prognostic_protective": list(z[z < -cut].sort_values().index[:top]),
        "proliferation": [g for g in bio["proliferation_metagene"] if g in z.index],
    }

    # ---- genes the model kept ------------------------------------------
    kept = model_genes(tables, endpoint, int(bio["model_gene_min_folds"]))
    lists["model"] = list(kept.index[:top])

    # ---- MOFA factors ---------------------------------------------------
    factors, weights = factor_table(c, cfg, seed, Path(cfg["paths"]["cache"]) / "mofa")
    factors.to_csv(tables / "biology_factors.csv", index=False)
    print(factors.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    for f in factors.loc[factors["cox_z"].abs() > 2, "factor"]:
        w = weights[f]
        lists[f] = list(w.abs().sort_values(ascending=False).index[:int(bio["factor_top_genes"])])

    # ---- one row per gene ------------------------------------------------
    union = sorted({g for l in lists.values() for g in l})
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
        hit = [n for n in lists if row[n]]
        if len(hit) > 1:
            return "several"
        return "factor" if hit[0].startswith("factor") else hit[0]
    nodes["group"] = nodes.apply(group, axis=1)
    nodes.rename_axis("gene").to_csv(tables / "biology_genes.csv")
    print("  gene lists: " + ", ".join(f"{k} {len(v)}" for k, v in lists.items()))

    # ---- do the lists overlap? -------------------------------------------
    prognostic = set(lists["prognostic_risk"]) | set(lists["prognostic_protective"])
    named = {"prognostic": prognostic, **{k: set(v) for k, v in lists.items()
                                         if not k.startswith("prognostic_")}}
    names = list(named)
    overlap = pd.DataFrame([{"list_a": a, "list_b": b, **B.overlap_test(named[a], named[b], len(z))}
                            for i, a in enumerate(names) for b in names[i + 1:]])
    overlap.to_csv(tables / "biology_overlap.csv", index=False)

    # ---- network ----------------------------------------------------------
    cache = net_dir / "string_edges.tsv"
    if args.offline and not cache.exists():
        print("  offline and no cached STRING network: stopping after the gene tables")
        return 0
    s = bio["string"]
    edges = B.string_network(union, int(s["species"]), int(s["required_score"]), cache, args.refresh)
    G = B.build_graph(edges, nodes, seed)
    B.export_graph(G, net_dir)
    print(f"  network: {G.number_of_nodes()} of {len(union)} genes connected, {G.number_of_edges()} edges")

    # ---- are model genes and prognostic genes linked? ---------------------
    tests = []
    for a, b in [("model", "prognostic"), ("model", "proliferation"), ("prognostic", "proliferation")]:
        if named.get(a) and named.get(b):
            tests.append({"list_a": a, "list_b": b, **B.between_edges_test(G, named[a], named[b], 2000, seed)})
    pd.DataFrame(tests).to_csv(tables / "biology_network_links.csv", index=False)

    # ---- modules and what they are enriched for ---------------------------
    mod = pd.DataFrame([{"gene": g, **d} for g, d in G.nodes(data=True)])
    rows, enrich = [], []
    for m, g in mod.groupby("module"):
        row = {"module": m, "genes": len(g), "mean_cox_z": g["cox_z"].mean(),
               **{name: int(g[name].sum()) for name in lists},
               "hub_genes": ", ".join(sorted(g["gene"], key=lambda x: -G.degree[x])[:8])}
        if len(g) >= 8 and not args.offline:
            e = B.string_enrichment(list(g["gene"]), int(s["species"]))
            if len(e):
                e = e[e["category"].isin(["Process", "KEGG", "RCTM", "HALLMARK"])].sort_values("fdr").head(5)
                for r in e.itertuples():
                    enrich.append({"module": m, "category": r.category, "term": r.description,
                                   "genes_in_term": r.number_of_genes, "fdr": r.fdr})
                if len(e):
                    row["top_term"] = e.iloc[0]["description"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables / "biology_modules.csv", index=False)
    draw(G, pd.DataFrame(rows), figures / "network.png")
    if enrich:
        pd.DataFrame(enrich).to_csv(tables / "biology_enrichment.csv", index=False)
    print(pd.DataFrame(rows).head(10).to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print(f"\n  open in Cytoscape: {net_dir}/network.graphml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
