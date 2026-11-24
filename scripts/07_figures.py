#!/usr/bin/env python
"""Step 7. Publication figures for results/REPORT.md, drawn from the tables.

    python scripts/07_figures.py

Nothing is fitted here. Every mark is read from results/tables/, so a figure
cannot disagree with RESULTS.md. Each figure is written to results/figures/
as a 300 dpi PNG (for the report) and a vector PDF (for a manuscript). The
workflow diagram is written as an SVG, and as a PDF where Chrome or Chromium
is installed to convert it. The numbers behind every panel are written to
results/source_data/, one CSV per panel.

Colour is used for one thing per figure. The model family (penalised Cox,
random survival forest) is always a panel, never a colour. Integration
methods keep one colour and one marker shape everywhere: early fusion a blue
circle, late fusion an orange square, MOFA a green diamond. A filled marker means the 95% interval excludes zero.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from _common import load_config, out_dirs

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from luad.biology import SHORT_TERMS  # noqa: E402
from luad.data import cohort_flow, load_cohort  # noqa: E402
from luad.models import MODEL_LABELS  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
METHOD = {"early": BLUE, "late": ORANGE, "mofa": AQUA, "clinical+rna": INK2}
# Each method also keeps one marker shape, so a figure reads without colour
# (greyscale print, colour-vision deficiency). Sizes are scaled to look equal.
MARK = {"early": "o", "late": "s", "mofa": "D", "clinical+rna": "^"}
MSCALE = {"o": 1.0, "s": 0.88, "D": 0.84, "^": 1.05}
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#256abf", "#9ec5f4", "#f0efec", "#f0a3a2", "#d03b3b"])

ARM = {"clinical": "Clinical", "rna": "Expression", "mutation": "Mutation",
       "methylation": "Methylation", "clinical+rna": "Clinical + expression",
       "clinical+mutation": "Clinical + mutation",
       "clinical+methylation": "Clinical + methylation",
       "early": "Early fusion", "late": "Late fusion", "mofa": "MOFA"}
SCEN = {"full": "nothing missing", "no_rna": "no expression",
        "no_methylation": "no methylation", "no_mutation": "no mutation",
        "clinical_only": "no omics", "geo": "GEO"}
PART = {"prediction": "Part 1. Does a block add to clinical?",
        "integration": "Part 2. Integration methods, nothing missing",
        "missing": "Part 2. A block missing at test time"}
W = 7.2  # inches: a double-column figure


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig, figures, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {figures / stem}.png and .pdf")


SOURCE = Path("results/source_data")


def source(stem: str, **panels: pd.DataFrame) -> None:
    """The numbers a figure is drawn from, one CSV per panel."""
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name, df in panels.items():
        df.to_csv(SOURCE / f"{stem}_{name}.csv", index=False)


def svg_to_pdf(svg: Path, width: int, height: int) -> None:
    """A vector PDF of an SVG, through headless Chrome. Skipped where no
    Chrome or Chromium is installed; the SVG is the figure of record."""
    chrome = next((p for p in map(shutil.which, ("google-chrome", "chromium", "chromium-browser")) if p), None)
    if chrome is None:
        print(f"  no Chrome or Chromium found: {svg.with_suffix('.pdf').name} not written")
        return
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(f"<style>@page{{size:{width}px {height}px;margin:0}}body{{margin:0}}"
                        f"svg{{display:block}}</style>{svg.read_text()}")
        out = svg.with_suffix(".pdf")
        r = subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                            f"--user-data-dir={tmp}/profile", f"--print-to-pdf={out.resolve()}",
                            page.as_uri()], capture_output=True, timeout=120)
        print(f"wrote {out}" if r.returncode == 0 and out.exists() else f"  Chrome could not write {out.name}")


def panel(ax, letter: str, title: str) -> None:
    """Journal style: a bold lowercase 8 pt panel letter, then a short neutral label."""
    ax.set_title(f"      {title}" if letter else title, loc="left", fontweight="bold", color=INK, pad=6)
    if letter:
        ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=INK, va="baseline", ha="left")


def split(arm: str) -> tuple[str, str]:
    a, _, s = arm.partition("@")
    return a, s or "full"


def contrast_label(a: str, b: str) -> str:
    (aa, sa), (bb, _) = split(a), split(b)
    lab = f"{ARM.get(aa, aa)} − {ARM.get(bb, bb).lower()}"
    return lab if sa == "full" else f"{lab}  ({SCEN[sa]})"


def interval(ax, x, lo, hi, y, colour, resolved=True, ms=4.5, lw=1.4, marker="o"):
    ax.plot([lo, hi], [y, y], color=colour, lw=lw, solid_capstyle="round", zorder=3)
    ax.plot(x, y, marker, ms=ms * MSCALE[marker], mfc=colour if resolved else "white", mec=colour, mew=1.1, zorder=4)


# ---------------------------------------------------------------- figure 1: workflow (SVG)

def workflow_svg(figures, cfg, n: int, events: int) -> None:
    cv, boot = cfg["cv"], cfg["bootstrap"]
    n_con = sum(len(v) for k, v in cfg["contrasts"].items() if k != "control") * len(cfg["models"])
    o = []
    font = "Arial, Helvetica, sans-serif"

    def box(x, y, w, h, title, lines, fill="#ffffff", stroke=AXIS, tcol=INK, step=None):
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        tx = x + 14
        if step:
            o.append(f'<rect x="{x + w - 78}" y="{y + 10}" width="66" height="18" rx="9" fill="#f0efec"/>')
            o.append(f'<text x="{x + w - 45}" y="{y + 23}" font-size="10.5" text-anchor="middle" fill="{INK2}" '
                     f'font-family="DejaVu Sans Mono, Menlo, monospace">{step}</text>')
        o.append(f'<text x="{tx}" y="{y + 25}" font-size="14" font-weight="700" fill="{tcol}">{title}</text>')
        for i, t in enumerate(lines):
            o.append(f'<text x="{tx}" y="{y + 46 + 17 * i}" font-size="12" fill="{INK2}">{t}</text>')

    def arrow(x1, y1, x2, y2):
        o.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{MUTED}" stroke-width="1.6" fill="none" marker-end="url(#ah)"/>')

    def band(y, text):
        o.append(f'<text x="40" y="{y}" font-size="11" font-weight="700" letter-spacing="1.2" fill="{MUTED}">{text}</text>')

    Wd, Hd = 1120, 1010
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wd} {Hd}" width="{Wd}" height="{Hd}" '
             f'font-family="{font}" role="img" aria-label="Analysis workflow">')
    o.append(f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker></defs>')
    o.append(f'<rect width="{Wd}" height="{Hd}" fill="#ffffff"/>')
    o.append(f'<text x="40" y="44" font-size="20" font-weight="700" fill="{INK}">LUAD multi-omics integration: analysis workflow</text>')
    o.append(f'<text x="40" y="66" font-size="12.5" fill="{INK2}">Does omics add prognostic information to clinical covariates, '
             'and which integration method survives a missing block?</text>')

    X0, FULL = 40, 1040
    y = 86

    def section(text, arrows_from=None):
        """An arrow down from the previous box, then the section label."""
        nonlocal y
        if arrows_from:
            for x in arrows_from:
                arrow(x, y, x, y + 24)
            y += 34
        band(y + 12, text)
        y += 24

    section("1  DATA   TCGA-LUAD, FOUR BLOCKS PER PATIENT")
    bw, gap = 245, 20
    blocks = [("Clinical", ["10 covariates", "stage, age, sex, smoking"]),
              ("Expression", ["19,499 protein-coding genes", "within-sample ranks"]),
              ("Mutation", ["recurrent genes, burden,", "10 pathway indicators"]),
              ("Methylation", ["450k array", "beta values"])]
    for i, (t, ls) in enumerate(blocks):
        box(X0 + i * (bw + gap), y, bw, 80, t, ls)
        arrow(X0 + i * (bw + gap) + bw / 2, y + 80, X0 + i * (bw + gap) + bw / 2, y + 106)
    y += 110
    box(X0, y, FULL, 62, f"Analysis cohort: {n} patients with an endpoint and all four blocks",
        [f"Primary endpoint: progression-free interval, censored at {cfg['endpoint']['admin_censor_months']} months "
         f"({events} events).  Sensitivity endpoint: overall survival."], fill="#f5f9fe", stroke="#9ec5f4")
    y += 62

    section("2  ESTIMATION   EVERYTHING LEARNED FROM DATA IS FITTED ON TRAINING ROWS ONLY", [560])
    box(X0, y, FULL, 204, f"Nested cross-validation: {cv['outer_folds']} folds x {cv['outer_repeats']} repeats, "
        f"inner {cv['inner_folds']}-fold tuning", [], step="01_cv")
    cw = 326
    box(56, y + 38, cw, 150, "Arms: how the blocks are used",
        ["One block alone", "Clinical + one omics block",
         "Early fusion: concatenate all four", "Late fusion: one model per block, risks averaged",
         "MOFA: shared latent factors + clinical"], fill="#fafaf8")
    box(56 + cw + 15, y + 38, cw, 150, "Two model families",
        ["Penalised Cox (elastic net)", "Random survival forest", "",
         "Tuned inside the fold: penalty, leaf size,", "block width (50 or 500 features)"], fill="#fafaf8")
    box(56 + 2 * (cw + 15), y + 38, cw, 150, "Missing at test time",
        ["Nothing  ·  expression  ·  methylation  ·  mutation", "All omics (clinical only)", "",
         "Models always train on complete data;", "each method copes in its own way"], fill="#fafaf8")
    y += 204

    section("3  INFERENCE", [560])
    box(X0, y, FULL, 62, f"Paired patient-level bootstrap, {boot['n_resamples']:,} resamples",
        [f"Uno's C (primary) and Harrell's C.  {n_con} contrasts fixed in config.yaml before the run; "
         "a contrast is resolved when its 95% interval excludes zero."], step="02_boot")
    y += 62

    section("4  THREE QUESTIONS", [560])
    pw = 333
    box(X0, y, pw, 134, "Part 1. Prediction",
        ["Does expression, mutation or", "methylation add to clinical covariates?", "",
         "Clinical + block against clinical"], fill="#f5f9fe", stroke="#9ec5f4", step="01-02")
    box(X0 + pw + 20, y, pw, 134, "Part 2. Integration",
        ["Early, late or MOFA: which is best,", "and which survives a missing block?", "",
         "External: 3 GEO cohorts, where", "mutation and methylation are absent"],
        fill="#fef6f2", stroke="#f4b79f", step="01-03")
    box(X0 + 2 * (pw + 20), y, pw + 1, 134, "Part 3. Biology",
        ["Which genes and factors are behind", "the models?", "",
         "MOFA factors, gene lists, STRING", "network, modules, Cytoscape"],
        fill="#f1faf6", stroke="#9bd9c0", step="04-04b")
    y += 134

    section("5  CHECKS AND REPORT", [560])
    box(X0, y, 686, 66, "Four robustness checks",
        ["Positive controls  ·  leakage and shuffled outcome  ·  block width sweep  ·  metric agreement"],
        step="05_checks")
    box(746, y, 334, 66, "Report",
        ["RESULTS.md and figures, from the tables"], step="06-07")
    y += 66 + 28
    o[0] = o[0].replace(f'viewBox="0 0 {Wd} {Hd}"', f'viewBox="0 0 {Wd} {y}"').replace(f'height="{Hd}"', f'height="{y}"')
    o[2] = o[2].replace(f'height="{Hd}"', f'height="{y}"')
    o.append("</svg>")
    path = figures / "fig1_workflow.svg"
    path.write_text("\n".join(o))
    print(f"wrote {path}")
    svg_to_pdf(path, Wd, y)


# ---------------------------------------------------------------- cohort flow

def fig_flow(figures, cfg) -> None:
    flow = cohort_flow(cfg)
    source("figS_cohort_flow", a=flow)
    n = len(flow)
    fig, ax = plt.subplots(figsize=(4.6, 0.62 * n + 0.5))
    ax.axis("off")
    ax.set_xlim(0, 10.4); ax.set_ylim(-n + 0.35, 0.65)
    for i, r in enumerate(flow.itertuples()):
        last = i == n - 1
        ax.add_patch(plt.Rectangle((0.2, -i - 0.3), 6.0, 0.6, fc="#f5f9fe" if last else "white",
                                   ec="#9ec5f4" if last else AXIS, lw=0.8))
        label = "Analysis cohort: all four blocks" if last else (r.step[0].upper() + r.step[1:])
        ax.text(0.4, -i + 0.09, label, va="center", fontsize=6.5, color=INK2)
        ax.text(0.4, -i - 0.13, f"n = {r.patients}", va="center", fontsize=7, fontweight="bold", color=INK)
        if i:
            ax.annotate("", (3.0, -i + 0.3), (3.0, -i + 0.7), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.8})
            ax.plot([3.0, 6.6], [-i + 0.5, -i + 0.5], color=MUTED, lw=0.8)
            what = r.step.replace("with ", "")
            ax.text(6.75, -i + 0.5, f"{r.excluded} without {what}", va="center", fontsize=6.5,
                    color=INK2 if r.excluded else MUTED)
    save(fig, figures, "figS_cohort_flow")


# ---------------------------------------------------------------- figure 2: prediction

def fig_prediction(tables, figures, endpoint: str) -> None:
    arms = pd.read_csv(tables / f"arms_{endpoint}.csv")
    d = arms[arms["metric"] == "uno_c"]
    groups = [["clinical"], ["rna", "mutation", "methylation"],
              ["clinical+rna", "clinical+mutation", "clinical+methylation", "early"]]
    names = {"early": "Early fusion (all four)"}
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True, sharex=True)
    for ax, (model, label) in zip(axes, MODEL_LABELS.items()):
        dm = d[d["model"] == model].set_index("arm")
        ref = dm.loc["clinical", "c_index"]
        ax.axvline(0.5, color=AXIS, lw=0.8)
        ax.axvline(ref, color=INK, lw=0.8, ls=(0, (1, 2)))
        y, ticks, labs = 0.0, [], []
        for g in groups:
            for arm in g:
                r = dm.loc[arm]
                interval(ax, r.c_index, r.ci_low, r.ci_high, y, INK if arm == "clinical" else BLUE)
                ax.text(0.70, y, f"{r.c_index:.3f}", va="center", ha="right", fontsize=6.5, color=INK2)
                ticks.append(y); labs.append(names.get(arm, ARM[arm]))
                y -= 1
            y -= 0.6
        ax.set_yticks(ticks, labs)
        ax.set_xlim(0.43, 0.70)
        ax.set_xlabel("Uno's C (95% interval)")
        panel(ax, "ab"[list(MODEL_LABELS).index(model)], label)
        ax.text(0.5, ticks[-1] - 0.95, "chance", ha="center", va="top", fontsize=6.5, color=MUTED)
        ax.text(ref, ticks[-1] - 0.95, "clinical", ha="center", va="top", fontsize=6.5, color=INK2,
                bbox={"fc": "white", "ec": "none", "pad": 1})
        ax.set_ylim(ticks[-1] - 1.7, 0.7)
    fig.tight_layout(w_pad=2)
    source(f"fig2_prediction_{endpoint}", ab=d[d["arm"].isin([a for g in groups for a in g])])
    save(fig, figures, f"fig2_prediction_{endpoint}")


# ---------------------------------------------------------------- figure 3: integration

def fig_integration(tables, figures, cfg, endpoint: str) -> None:
    arms = pd.read_csv(tables / f"arms_{endpoint}.csv")
    con = pd.read_csv(tables / f"contrasts_{endpoint}.csv")
    d, c = arms[arms["metric"] == "uno_c"], con[con["metric"] == "uno_c"]
    scen = list(cfg["scenarios"])
    fig, axes = plt.subplots(2, 2, figsize=(W, 5.6), sharey="row", gridspec_kw={"height_ratios": [1.15, 1]})
    off = {"early": -0.24, "late": 0.0, "mofa": 0.24}
    for j, (model, label) in enumerate(MODEL_LABELS.items()):
        # a, b: every method under every scenario
        ax, dm = axes[0, j], d[d["model"] == model].set_index("arm")
        ref = dm.loc["clinical", "c_index"]
        ax.grid(axis="y"); ax.grid(axis="x", visible=False)
        ax.axhline(ref, color=INK, lw=0.8, ls=(0, (1, 2)))
        ax.text(len(scen) - 0.45, ref, " clinical\n alone", va="center", ha="left", fontsize=6.5, color=INK2)
        for m, dx in off.items():
            for i, s in enumerate(scen):
                r = dm.loc[m if s == "full" else f"{m}@{s}"]
                ax.plot([i + dx, i + dx], [r.ci_low, r.ci_high], color=METHOD[m], lw=1.2, solid_capstyle="round")
                ax.plot(i + dx, r.c_index, MARK[m], ms=4.5 * MSCALE[MARK[m]], color=METHOD[m], mec="white", mew=0.6, zorder=4)
        ax.set_xticks(range(len(scen)), [SCEN[s].replace(" ", "\n") for s in scen], fontsize=6.5)
        ax.set_xlim(-0.55, len(scen) - 0.45)
        ax.tick_params(axis="x", length=0)
        if j == 0:
            ax.set_ylabel("Uno's C (95% interval)")
        panel(ax, "ab"[j], f"{label}: by missing block")
        # c, d: the paired differences against early fusion
        ax = axes[1, j]
        cm = c[(c["model"] == model) & (c["b"].map(lambda b: split(b)[0]) == "early")]
        ax.axvline(0, color=INK, lw=0.8)
        ticks = []
        for i, s in enumerate(scen[:-1]):
            for m, dy in (("mofa", 0.17), ("late", -0.17)):
                r = cm[cm["a"] == (m if s == "full" else f"{m}@{s}")].iloc[0]
                interval(ax, r.difference, r.ci_low, r.ci_high, -i + dy, METHOD[m], r.verdict != "not resolved",
                         marker=MARK[m])
            ticks.append(-i)
        ax.set_yticks(ticks, [SCEN[s] for s in scen[:-1]])
        ax.set_ylim(-len(scen) + 1.45, 0.55)
        ax.set_xlim(-0.03, 0.12)
        ax.set_xlabel("Difference in Uno's C against early fusion")
        panel(ax, "cd"[j], f"{label}: gain over early fusion")
    hs = [Line2D([], [], marker=MARK[m], ls="-", color=METHOD[m], ms=4.5 * MSCALE[MARK[m]], lw=1.2, label=ARM[m])
          for m in off]
    hs += [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=4.5, label="interval excludes zero"),
           Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=4.5, label="not resolved")]
    fig.legend(handles=hs, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.035), columnspacing=1.6)
    fig.tight_layout(w_pad=2, h_pad=1.6)
    shown = ["clinical"] + [m if s == "full" else f"{m}@{s}" for m in off for s in scen]
    source(f"fig3_integration_{endpoint}", ab=d[d["arm"].isin(shown)],
           cd=c[c["b"].map(lambda b: split(b)[0]) == "early"])
    save(fig, figures, f"fig3_integration_{endpoint}")


# ---------------------------------------------------------------- figure 4: every contrast

def fig_contrasts(tables, figures, endpoint: str, stem: str) -> None:
    con = pd.read_csv(tables / f"contrasts_{endpoint}.csv")
    c = con[(con["metric"] == "uno_c") & (con["part"] != "control")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 5.4), sharey=True, sharex=True)
    for ax, (model, label) in zip(axes, MODEL_LABELS.items()):
        cm = c[c["model"] == model]
        ax.axvline(0, color=INK, lw=0.8)
        y, ticks, labs = 0.0, [], []
        for part, title in PART.items():
            if ax is axes[0]:
                ax.text(-0.02, y, title, transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=6.5, fontweight="bold", color=INK)
            y -= 1
            for r in cm[cm["part"] == part].itertuples():
                res = r.verdict != "not resolved"
                interval(ax, r.difference, r.ci_low, r.ci_high, y, INK if res else MUTED, res, ms=4)
                ticks.append(y); labs.append(contrast_label(r.a, r.b))
                y -= 1
            y -= 0.5
        ax.set_yticks(ticks, labs)
        ax.set_ylim(y + 0.9, 0.7)
        ax.set_xticks([-0.05, 0, 0.05], ["−0.05", "0", "0.05"])
        ax.set_xlabel("Difference in Uno's C (95% interval)")
        panel(ax, "ab"[list(MODEL_LABELS).index(model)], label)
        n_res = int((cm["verdict"] != "not resolved").sum())
        ax.text(1, 1.012, f"{n_res} of {len(cm)} resolved", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color=INK2)
    fig.tight_layout(w_pad=1.5)
    source(stem, ab=c)
    save(fig, figures, stem)


# ---------------------------------------------------------------- figure 5: external

def fig_external(tables, figures) -> None:
    con = pd.read_csv(tables / "external_contrasts.csv")
    c = con[(con["metric"] == "uno_c") & (con["b"] == "clinical")]
    cells = (c[["endpoint", "cohort", "n", "n_events"]].drop_duplicates()
             .sort_values(["endpoint", "cohort"]).reset_index(drop=True))
    order = ["clinical+rna", "early@geo", "late@geo", "mofa@geo"]
    labs = ["Clinical + expression", "Early fusion", "Late fusion", "MOFA"]
    fig, axes = plt.subplots(2, len(cells), figsize=(W, 3.5), sharex=True, sharey=True)
    for i, (model, label) in enumerate(MODEL_LABELS.items()):
        for j, cell in cells.iterrows():
            ax = axes[i, j]
            ax.axvline(0, color=INK, lw=0.8)
            cc = c[(c["model"] == model) & (c["cohort"] == cell.cohort) & (c["endpoint"] == cell.endpoint)].set_index("a")
            for k, a in enumerate(order):
                r = cc.loc[a]
                interval(ax, r.difference, r.ci_low, r.ci_high, -k, METHOD[split(a)[0]],
                         r.verdict != "not resolved", ms=4, lw=1.2, marker=MARK[split(a)[0]])
            ax.set_yticks(range(0, -len(order), -1), labs)
            ax.set_ylim(-len(order) + 0.4, 0.6)
            ax.set_xlim(-0.14, 0.30)
            ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3], ["−0.1", "0", "0.1", "0.2", "0.3"])
            if i == 0:
                kind = "relapse-free" if "RFS" in cell.endpoint else "overall"
                ax.set_title(f"{cell.cohort}\n{kind} survival\nn = {cell.n}, {cell.n_events} events",
                             fontsize=6.5, color=INK, pad=5)
            if j == len(cells) - 1:
                ax.text(1.06, 0.5, label, transform=ax.transAxes, rotation=270, va="center", ha="left",
                        fontsize=6.5, fontweight="bold", color=INK)
    fig.supxlabel("Difference in Uno's C against the clinical model (95% interval)", fontsize=7, color=INK2, y=0.075)
    hs = [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=4.5, label="interval excludes zero"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=4.5, label="not resolved")]
    fig.legend(handles=hs, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0, 0.06, 0.985, 1))
    source("fig5_external", a=c[c["a"].isin(order)])
    save(fig, figures, "fig5_external")


# ---------------------------------------------------------------- figure 6: MOFA factors

def heat(ax, M, cmap, vmin, vmax, xt, yt, fmt, flip=0.6):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(xt)), xt)
    ax.set_yticks(range(len(yt)), yt)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(xt)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(yt)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6.5,
                color="white" if abs(v) / max(abs(vmin), abs(vmax)) > flip else INK)


def fig_factors(tables, figures) -> None:
    f = pd.read_csv(tables / "biology_factors.csv")
    names = [x.replace("factor", "Factor ") for x in f["factor"]]
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.4), gridspec_kw={"width_ratios": [1, 1.05, 1.6]})
    heat(axes[0], f[["r2_rna", "r2_mutation", "r2_methylation"]].to_numpy() * 100, SEQ, 0, 20,
         ["Expression", "Mutation", "Methylation"], names, lambda v: f"{v:.0f}")
    panel(axes[0], "a", "Variance explained (%)")
    ax = axes[1]
    y = np.arange(len(f))
    hit = f["cox_z"].abs() > 2
    ax.barh(y, f["cox_z"], height=0.62, color=[BLUE if h else AXIS for h in hit])
    for v in (-2, 2):
        ax.axvline(v, color=MUTED, lw=0.7, ls=(0, (1, 2)))
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [])
    ax.set_ylim(len(f) - 0.5, -0.5)
    ax.set_xlim(-4, 4)
    ax.set_xlabel("Cox z with progression")
    ax.text(2.12, len(f) - 0.55, "|z| = 2", ha="left", va="bottom", fontsize=6, color=MUTED)
    panel(ax, "b", "Association with outcome")
    cols = ["r_stage_ordinal", "r_age", "r_sex_male", "r_ever_smoker", "r_tmb"]
    heat(axes[2], f[cols].to_numpy(), DIV, -0.6, 0.6,
         ["Stage", "Age", "Male sex", "Ever\nsmoker", "Mutation\nburden"], [""] * len(f),
         lambda v: f"{v:+.2f}".replace("-", "−"), flip=0.7)
    panel(axes[2], "c", "Correlation with patient covariates")
    fig.tight_layout(w_pad=1.2)
    source("fig6_mofa_factors", abc=f)
    save(fig, figures, "fig6_mofa_factors")


# ---------------------------------------------------------------- figure 7: biology

def fig_biology(tables, figures) -> None:
    mod = pd.read_csv(tables / "biology_modules.csv")
    ov = pd.read_csv(tables / "biology_overlap.csv")
    ln = pd.read_csv(tables / "biology_network_links.csv")
    cen = pd.read_csv(tables / "biology_network_centrality.csv")
    fig = plt.figure(figsize=(W, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.25, 1])

    # a: who lives in which module
    ax = fig.add_subplot(gs[0, 0])
    m = mod[mod["genes"] >= 12].copy()
    short = SHORT_TERMS
    cols = ["prognostic_risk", "prognostic_protective", "model", "factor2", "proliferation"]
    counts = m[cols].to_numpy()
    share = counts / m["genes"].to_numpy()[:, None] * 100
    ax.imshow(share, cmap=SEQ, vmin=0, vmax=90, aspect="auto")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(len(cols)), ["Higher\nrisk", "Protective", "Kept by\nCox model",
                                     "MOFA\nfactor 2", "Prolif.\nmetagene"], fontsize=6)
    ax.set_yticks(range(len(m)), [f"M{r.module}  {short.get(r.top_term, r.top_term)}  ({r.genes})" for r in m.itertuples()])
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(m)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for (i, j), v in np.ndenumerate(counts):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5,
                color="white" if share[i, j] > 50 else (INK if v else MUTED))
    panel(ax, "a", "Genes of each list in each network module")

    # b: degree by list
    ax = fig.add_subplot(gs[0, 1])
    order = [("prognostic_risk", "Prognostic, higher risk"), ("proliferation", "Proliferation metagene"),
             ("several", "In more than one list"), ("factor", "MOFA factor 2"),
             ("prognostic_protective", "Prognostic, protective"), ("model", "Kept by Cox model")]
    rng = np.random.default_rng(0)
    for i, (g, _) in enumerate(order):
        v = cen.loc[cen["group"] == g, "Degree"].to_numpy()
        ax.plot(v, -i + rng.uniform(-0.2, 0.2, len(v)), "o", ms=3, color=BLUE, alpha=0.55, mec="none")
        ax.plot([np.median(v)] * 2, [-i - 0.33, -i + 0.33], color=INK, lw=1.6, solid_capstyle="round")
    ax.set_yticks(range(0, -len(order), -1),
                  [f"{lab}  ({int((cen['group'] == g).sum())})" for g, lab in order])
    ax.set_ylim(-len(order) + 0.4, 0.6)
    ax.set_xlim(-2, 68)
    ax.set_xlabel("Interactions per gene (degree); bar is the median")
    panel(ax, "b", "Connectivity by gene list")

    # c, d: observed against expected
    def dumbbell(ax, rows, xlabel, title, letter):
        for i, (lab, obs, exp) in enumerate(rows):
            ax.plot([exp, obs], [-i, -i], color=AXIS, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(exp, -i, "o", ms=5.5, mfc="white", mec=INK2, mew=1.1, zorder=3)
            ax.plot(obs, -i, "o", ms=5.5, color=BLUE, zorder=4)
            ax.annotate(f"{obs:.0f}", (obs, -i), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=6.5, fontweight="bold", color=INK)
            ax.annotate(f"{exp:.0f}" if exp >= 10 else f"{exp:.1f}", (exp, -i), xytext=(0, -12),
                        textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
        ax.set_yticks(range(0, -len(rows), -1), [r[0] for r in rows])
        ax.set_ylim(-len(rows) + 0.35, 0.65)
        ax.set_xlabel(xlabel)
        panel(ax, letter, title)

    nm = {"prognostic": "prognostic", "model": "Cox model", "factor2": "factor 2", "proliferation": "proliferation"}
    keep = [("prognostic", "model"), ("prognostic", "factor2"), ("model", "factor2")]
    rows = [(f"{nm[a].capitalize()} and {nm[b]}", *ov[(ov.list_a == a) & (ov.list_b == b)][["overlap", "expected"]].iloc[0])
            for a, b in keep]
    ax = fig.add_subplot(gs[1, 0])
    dumbbell(ax, rows, "Number of genes", "Genes shared by two lists", "c")
    ax.set_xlim(-0.8, 15)
    rows = [(f"{nm[r.list_a].capitalize()} and {nm[r.list_b]}", r.edges_between, r.expected) for r in ln.itertuples()]
    ax = fig.add_subplot(gs[1, 1])
    dumbbell(ax, rows, "Number of protein interactions", "Interactions between two lists", "d")
    ax.set_xlim(-25, 470)
    hs = [Line2D([], [], marker="o", ls="", color=BLUE, ms=5.5, label="observed"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5.5, label="expected by chance")]
    fig.legend(handles=hs, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=1.5, h_pad=1.8)
    source("fig7_biology", a=m, b=cen[["name", "group", "Degree"]],
           c=ov[[(a, b) in keep for a, b in zip(ov.list_a, ov.list_b)]], d=ln)
    save(fig, figures, "fig7_biology")


# ---------------------------------------------------------------- figure 8: checks

def fig_checks(tables, figures, endpoint: str) -> None:
    real = pd.read_csv(tables / f"cv_{endpoint}_summary.csv")
    perm = pd.read_csv(tables / f"cv_{endpoint}_permuted_summary.csv")
    sweep = pd.read_csv(tables / "width_sweep.csv")
    con = pd.read_csv(tables / f"contrasts_{endpoint}.csv")
    con = con[con["part"] != "control"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.1, 1, 1]})

    ax = axes[0]
    both = perm.merge(real, on=["model", "arm"], suffixes=("_perm", "_real"))
    rng = np.random.default_rng(0)
    ax.axvline(0.5, color=INK, lw=0.8)
    for i, (col, colour) in enumerate((("uno_c_real", BLUE), ("uno_c_perm", MUTED))):
        v = both[col].to_numpy()
        ax.plot(v, -i + rng.uniform(-0.18, 0.18, len(v)), "o", ms=4, color=colour, alpha=0.75, mec="white", mew=0.4)
    ax.set_yticks([0, -1], ["Real\noutcome", "Shuffled\noutcome"])
    ax.set_ylim(-1.6, 0.6)
    ax.set_xlim(0.46, 0.63)
    ax.set_xlabel(f"Uno's C, {len(both)} arm-model pairs")
    panel(ax, "a", "Shuffled outcome")

    ax = axes[1]
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    ax.axhline(0, color=INK, lw=0.8)
    ks = sorted(sweep["k"].unique())
    styles = {"coxnet": "-", "rsf": (0, (4, 2))}
    for (model, arm), g in sweep.groupby(["model", "arm"], sort=False):
        g = g.sort_values("k")
        ax.plot(range(len(ks)), g["minus_clinical"], ls=styles[model], lw=1.5, marker=MARK[arm],
                    ms=4 * MSCALE[MARK[arm]],
                color=METHOD[arm], mec="white", mew=0.5)
    ax.set_xticks(range(len(ks)), [f"{k:,}" for k in ks])
    ax.set_xlim(-0.25, len(ks) - 0.75)
    ax.set_ylim(-0.05, 0.03)
    ax.set_xlabel("Features per omics block (fixed)")
    ax.set_ylabel("Uno's C minus clinical")
    hs = [Line2D([], [], color=METHOD["clinical+rna"], lw=1.5, marker=MARK["clinical+rna"], ms=4,
                 mec="white", mew=0.5, label="Clinical + expression"),
          Line2D([], [], color=METHOD["early"], lw=1.5, marker=MARK["early"], ms=4, mec="white", mew=0.5,
                 label="Early fusion"),
          Line2D([], [], color=INK2, lw=1.2, ls="-", label="Cox"),
          Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 2)), label="Forest")]
    ax.legend(handles=hs, loc="upper right", ncol=2, fontsize=6, handlelength=2.2, columnspacing=1,
              borderaxespad=0.1)
    panel(ax, "b", "Fixed block width")

    ax = axes[2]
    w = con.pivot_table(index=["model", "a", "b"], columns="metric", values="difference").reset_index()
    v = con[con["metric"] == "uno_c"].set_index(["model", "a", "b"])["verdict"]
    res = np.array([v[(r.model, r.a, r.b)] != "not resolved" for r in w.itertuples()])
    lim = 0.06
    ax.grid(axis="y")
    ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=0.8)
    ax.axhline(0, color=INK, lw=0.6); ax.axvline(0, color=INK, lw=0.6)
    ax.plot(w["uno_c"][~res], w["harrell_c"][~res], "o", ms=4, mfc="white", mec=MUTED, mew=1)
    ax.plot(w["uno_c"][res], w["harrell_c"][res], "o", ms=4, color=INK)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([-0.04, 0, 0.04], ["−0.04", "0", "0.04"])
    ax.set_yticks([-0.04, 0, 0.04], ["−0.04", "0", "0.04"])
    ax.tick_params(axis="y", length=2.5)
    ax.set_xlabel("Difference in Uno's C")
    ax.set_ylabel("Difference in Harrell's C")
    r = np.corrcoef(w["uno_c"], w["harrell_c"])[0, 1]
    ax.text(0.04, 0.96, f"r = {r:.2f}\n{len(w)} contrasts", transform=ax.transAxes, va="top", fontsize=6.5, color=INK2)
    panel(ax, "c", "Metric agreement")
    fig.tight_layout(w_pad=1.3)
    source("fig8_checks", a=both, b=sweep, c=w.assign(resolved_uno_c=res))
    save(fig, figures, "fig8_checks")


def main() -> int:
    cfg = load_config()
    tables, figures = out_dirs(cfg)
    primary, sens = cfg["endpoint"]["primary"].lower(), cfg["endpoint"]["sensitivity"].lower()
    style()
    c = load_cohort(cfg, primary, verbose=False)
    workflow_svg(figures, cfg, c.n, int(c.y["event"].sum()))
    fig_flow(figures, cfg)
    fig_prediction(tables, figures, primary)
    fig_integration(tables, figures, cfg, primary)
    fig_contrasts(tables, figures, primary, f"fig4_contrasts_{primary}")
    fig_external(tables, figures)
    fig_factors(tables, figures)
    fig_biology(tables, figures)
    fig_checks(tables, figures, primary)
    fig_prediction(tables, figures, sens)
    fig_integration(tables, figures, cfg, sens)
    fig_contrasts(tables, figures, sens, f"figS_contrasts_{sens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
