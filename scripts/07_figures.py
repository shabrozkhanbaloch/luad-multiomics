#!/usr/bin/env python
"""Step 7. Publication figures for results/REPORT.md, drawn from the tables.

    python scripts/07_figures.py

What it is for
    It draws every figure of results/REPORT.md except the Cytoscape network
    (that one is step 4b). Nothing is fitted here. Every mark is read from
    results/tables/, so a figure cannot disagree with RESULTS.md.

    Colour is used for one thing per figure. The model family (penalised Cox,
    random survival forest) is always a panel, never a colour. Integration
    methods keep one colour and one marker shape everywhere: early fusion a
    blue circle, late fusion an orange square, MOFA a green diamond. A filled
    marker means the 95% interval excludes zero.

What it reads
    results/tables/        the tables of the earlier steps (arms, contrasts, external,
                           biology, checks); it must run after them
    config/config.yaml     the scenarios, and the numbers quoted in the workflow diagram
    data/processed/, data/interim/
                           only to count patients and events for the workflow
                           diagram and the cohort flow

What it writes
    results/figures/fig*.png, fig*.pdf   each figure as a 300 dpi PNG (for the report)
                                         and a vector PDF (for a manuscript)
    results/figures/fig1_workflow.svg    the workflow diagram, and a PDF of it where
                                         Chrome or Chromium is installed to convert it
    results/figures/figS_*               the cohort flow and the overall-survival contrasts
    results/source_data/                 the numbers behind every panel, one CSV per panel

How long it takes
    About 20 seconds.
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

# Text colours, from darkest to lightest, then the grid and axis greys.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# One colour per integration method, the same in every figure.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
METHOD = {"early": BLUE, "late": ORANGE, "mofa": AQUA, "clinical+rna": INK2}
# Each method also keeps one marker shape, so a figure reads without colour
# (greyscale print, colour-vision deficiency). Sizes are scaled to look equal.
MARK = {"early": "o", "late": "s", "mofa": "D", "clinical+rna": "^"}
MSCALE = {"o": 1.0, "s": 0.88, "D": 0.84, "^": 1.05}
# Colour scales for the heatmaps: SEQ for a share (white to dark blue), DIV for
# a signed number (blue below zero, red above).
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#256abf", "#9ec5f4", "#f0efec", "#f0a3a2", "#d03b3b"])

# The names in the tables (arm, scenario, part of the plan) and the words a
# figure shows for them.
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
    """Set the look shared by every figure: fonts, sizes, colours, grid.

    Takes no arguments and returns nothing; it changes matplotlib's settings
    for the rest of the run. Text is kept as text in the PDF and SVG files
    (fonttype 42, "none"), so it can still be edited in a drawing program.
    """
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
    """Write a finished figure as PNG and as PDF, then close it.

    Arguments
        fig      the matplotlib figure
        figures  the folder to write into (results/figures)
        stem     the file name without its ending, for example "fig5_external"
    """
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {figures / stem}.png and .pdf")


# Where the numbers behind the figures are written.
SOURCE = Path("results/source_data")


def source(stem: str, **panels: pd.DataFrame) -> None:
    """The numbers a figure is drawn from, one CSV per panel.

    Arguments
        stem    the figure's file name without its ending
        panels  one table per panel, given by the panel letters:
                `source("fig7_biology", a=..., b=...)` writes fig7_biology_a.csv
                and fig7_biology_b.csv
    """
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name, table in panels.items():
        table.to_csv(SOURCE / f"{stem}_{name}.csv", index=False)


def svg_to_pdf(svg: Path, width: int, height: int) -> None:
    """A vector PDF of an SVG, through headless Chrome. Skipped where no
    Chrome or Chromium is installed; the SVG is the figure of record.

    Arguments
        svg            the SVG file; the PDF is written beside it, same name
        width, height  the size of the drawing in pixels, used as the page size
    """
    chrome = next((found for found in map(shutil.which, ("google-chrome", "chromium", "chromium-browser"))
                   if found), None)
    if chrome is None:
        print(f"  no Chrome or Chromium found: {svg.with_suffix('.pdf').name} not written")
        return
    # Chrome prints web pages, not SVG files. The SVG is put inside a small
    # page whose paper size is exactly the drawing, with no margins.
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(f"<style>@page{{size:{width}px {height}px;margin:0}}body{{margin:0}}"
                        f"svg{{display:block}}</style>{svg.read_text()}")
        pdf_path = svg.with_suffix(".pdf")
        result = subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                                 f"--user-data-dir={tmp}/profile", f"--print-to-pdf={pdf_path.resolve()}",
                                 page.as_uri()], capture_output=True, timeout=120)
        print(f"wrote {pdf_path}" if result.returncode == 0 and pdf_path.exists()
              else f"  Chrome could not write {pdf_path.name}")


def panel(ax, letter: str, title: str) -> None:
    """Journal style: a bold lowercase 8 pt panel letter, then a short neutral label.

    Arguments
        ax      the panel
        letter  "a", "b", ...; an empty string gives a title without a letter
        title   the label after the letter
    """
    # The title starts with spaces to leave room for the letter, which is
    # drawn separately because it is a larger font.
    ax.set_title(f"      {title}" if letter else title, loc="left", fontweight="bold", color=INK, pad=6)
    if letter:
        ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=INK, va="baseline", ha="left")


def split(arm: str) -> tuple[str, str]:
    """Split an arm name from the tables into its method and its scenario.

    "mofa@no_rna" gives ("mofa", "no_rna"). A name without "@" is the
    complete-data case: "mofa" gives ("mofa", "full").
    """
    name, _, scenario = arm.partition("@")
    return name, scenario or "full"


def contrast_label(a: str, b: str) -> str:
    """The row label of a contrast, as "First arm − second arm".

    Arguments
        a, b  the two arm names of the contrast (columns `a` and `b` of the
              contrasts table); the contrast is a minus b

    Returns the label, with the scenario in brackets when a block is missing.
    """
    (arm_a, scenario_a), (arm_b, _) = split(a), split(b)
    label = f"{ARM.get(arm_a, arm_a)} − {ARM.get(arm_b, arm_b).lower()}"
    return label if scenario_a == "full" else f"{label}  ({SCEN[scenario_a]})"


def interval(ax, x, lo, hi, y, colour, resolved=True, ms=4.5, lw=1.4, marker="o"):
    """Draw one estimate with its interval: a horizontal bar and a marker on it.

    Arguments
        ax        the panel
        x         the estimate (where the marker goes)
        lo, hi    the two ends of the interval
        y         the height of the row
        colour    the colour of bar and marker
        resolved  True draws a filled marker, False an open (white) one
        ms, lw    marker size and line width
        marker    the marker shape; its size is scaled by MSCALE to look equal
    """
    ax.plot([lo, hi], [y, y], color=colour, lw=lw, solid_capstyle="round", zorder=3)
    ax.plot(x, y, marker, ms=ms * MSCALE[marker], mfc=colour if resolved else "white", mec=colour,
            mew=1.1, zorder=4)


# ---------------------------------------------------------------- figure 1: workflow (SVG)

def workflow_svg(figures, cfg, n: int, events: int) -> None:
    """Figure 1 of REPORT.md: the analysis workflow, from data to report.

    This figure is boxes and arrows, not a plot, so it is written directly as
    SVG text: five bands from top to bottom (data, estimation, inference, the
    three questions, checks and report). The numbers in the boxes come from
    the configuration and the cohort, not from typing.

    Arguments
        figures  the folder to write into
        cfg      the configuration
        n        patients in the analysis cohort
        events   events of the primary endpoint

    Writes fig1_workflow.svg and, where Chrome is found, fig1_workflow.pdf.
    """
    cv, boot = cfg["cv"], cfg["bootstrap"]
    # The contrasts tested: every one in the plan except the control, once
    # per model family.
    n_contrasts = sum(len(v) for k, v in cfg["contrasts"].items() if k != "control") * len(cfg["models"])
    # The SVG is collected as a list of text pieces, joined at the end. The
    # helpers below each add the pieces of one shape.
    svg = []
    font = "Arial, Helvetica, sans-serif"

    def box(x, y, w, h, title, lines, fill="#ffffff", stroke=AXIS, tcol=INK, step=None):
        """A rounded box with a bold title and lines of text under it.
        `step`, if given, is the script name shown in a small pill at the top right."""
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
                   f'stroke="{stroke}" stroke-width="1.2"/>')
        text_x = x + 14
        if step:
            svg.append(f'<rect x="{x + w - 78}" y="{y + 10}" width="66" height="18" rx="9" fill="#f0efec"/>')
            svg.append(f'<text x="{x + w - 45}" y="{y + 23}" font-size="10.5" text-anchor="middle" '
                       f'fill="{INK2}" '
                       f'font-family="DejaVu Sans Mono, Menlo, monospace">{step}</text>')
        svg.append(f'<text x="{text_x}" y="{y + 25}" font-size="14" font-weight="700" '
                   f'fill="{tcol}">{title}</text>')
        for i, line in enumerate(lines):
            svg.append(f'<text x="{text_x}" y="{y + 46 + 17 * i}" font-size="12" fill="{INK2}">{line}</text>')

    def arrow(x1, y1, x2, y2):
        """A straight arrow from (x1, y1) to (x2, y2)."""
        svg.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{MUTED}" stroke-width="1.6" fill="none" '
                   f'marker-end="url(#ah)"/>')

    def band(y, text):
        """A small grey heading in capitals, at height y."""
        svg.append(f'<text x="40" y="{y}" font-size="11" font-weight="700" letter-spacing="1.2" '
                   f'fill="{MUTED}">{text}</text>')

    # The page: the opening tag, the arrow head, a white background, the title.
    # The height is a first guess here and is corrected at the end.
    page_width, page_height = 1120, 1010
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {page_width} {page_height}" '
               f'width="{page_width}" height="{page_height}" '
               f'font-family="{font}" role="img" aria-label="Analysis workflow">')
    svg.append(f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
               f'markerHeight="7" '
               f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker></defs>')
    svg.append(f'<rect width="{page_width}" height="{page_height}" fill="#ffffff"/>')
    svg.append(f'<text x="40" y="44" font-size="20" font-weight="700" fill="{INK}">'
               f'LUAD multi-omics integration: analysis workflow</text>')
    svg.append(f'<text x="40" y="66" font-size="12.5" fill="{INK2}">'
               'Does omics add prognostic information to clinical covariates, '
               'and which integration method survives a missing block?</text>')

    # X0 is the left edge and FULL the width of a box that spans the page.
    # y is the height reached so far; every box moves it down.
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

    # ---- band 1: the four data blocks, joined into the analysis cohort
    section("1  DATA   TCGA-LUAD, FOUR BLOCKS PER PATIENT")
    block_width, gap = 245, 20
    blocks = [("Clinical", ["10 covariates", "stage, age, sex, smoking"]),
              ("Expression", ["19,499 protein-coding genes", "within-sample ranks"]),
              ("Mutation", ["recurrent genes, burden,", "10 pathway indicators"]),
              ("Methylation", ["450k array", "beta values"])]
    for i, (title, lines) in enumerate(blocks):
        box(X0 + i * (block_width + gap), y, block_width, 80, title, lines)
        arrow(X0 + i * (block_width + gap) + block_width / 2, y + 80,
              X0 + i * (block_width + gap) + block_width / 2, y + 106)
    y += 110
    box(X0, y, FULL, 62, f"Analysis cohort: {n} patients with an endpoint and all four blocks",
        [f"Primary endpoint: progression-free interval, censored at "
         f"{cfg['endpoint']['admin_censor_months']} months "
         f"({events} events).  Sensitivity endpoint: overall survival."], fill="#f5f9fe", stroke="#9ec5f4")
    y += 62

    # ---- band 2: cross-validation, with three cards inside one wide box
    section("2  ESTIMATION   EVERYTHING LEARNED FROM DATA IS FITTED ON TRAINING ROWS ONLY", [560])
    box(X0, y, FULL, 204,
        f"Nested cross-validation: {cv['outer_folds']} folds x {cv['outer_repeats']} repeats, "
        f"inner {cv['inner_folds']}-fold tuning", [], step="01_cv")
    card_width = 326
    box(56, y + 38, card_width, 150, "Arms: how the blocks are used",
        ["One block alone", "Clinical + one omics block",
         "Early fusion: concatenate all four", "Late fusion: one model per block, risks averaged",
         "MOFA: shared latent factors + clinical"], fill="#fafaf8")
    box(56 + card_width + 15, y + 38, card_width, 150, "Two model families",
        ["Penalised Cox (elastic net)", "Random survival forest", "",
         "Tuned inside the fold: penalty, leaf size,", "block width (50 or 500 features)"], fill="#fafaf8")
    box(56 + 2 * (card_width + 15), y + 38, card_width, 150, "Missing at test time",
        ["Nothing  ·  expression  ·  methylation  ·  mutation", "All omics (clinical only)", "",
         "Models always train on complete data;", "each method copes in its own way"], fill="#fafaf8")
    y += 204

    # ---- band 3: the bootstrap
    section("3  INFERENCE", [560])
    box(X0, y, FULL, 62, f"Paired patient-level bootstrap, {boot['n_resamples']:,} resamples",
        [f"Uno's C (primary) and Harrell's C.  {n_contrasts} contrasts fixed in config.yaml before the run; "
         "a contrast is resolved when its 95% interval excludes zero."], step="02_boot")
    y += 62

    # ---- band 4: the three parts of the plan, side by side
    section("4  THREE QUESTIONS", [560])
    part_width = 333
    box(X0, y, part_width, 134, "Part 1. Prediction",
        ["Does expression, mutation or", "methylation add to clinical covariates?", "",
         "Clinical + block against clinical"], fill="#f5f9fe", stroke="#9ec5f4", step="01-02")
    box(X0 + part_width + 20, y, part_width, 134, "Part 2. Integration",
        ["Early, late or MOFA: which is best,", "and which survives a missing block?", "",
         "External: 3 GEO cohorts, where", "mutation and methylation are absent"],
        fill="#fef6f2", stroke="#f4b79f", step="01-03")
    box(X0 + 2 * (part_width + 20), y, part_width + 1, 134, "Part 3. Biology",
        ["Which genes and factors are behind", "the models?", "",
         "MOFA factors, gene lists, STRING", "network, modules, Cytoscape"],
        fill="#f1faf6", stroke="#9bd9c0", step="04-04b")
    y += 134

    # ---- band 5: checks and report
    section("5  CHECKS AND REPORT", [560])
    box(X0, y, 686, 66, "Four robustness checks",
        ["Positive controls  ·  leakage and shuffled outcome  ·  block width sweep  ·  metric agreement"],
        step="05_checks")
    box(746, y, 334, 66, "Report",
        ["RESULTS.md and figures, from the tables"], step="06-07")
    y += 66 + 28

    # Now the true height is known: put it into the opening tag (piece 0) and
    # the white background (piece 2), in place of the first guess.
    svg[0] = (svg[0].replace(f'viewBox="0 0 {page_width} {page_height}"', f'viewBox="0 0 {page_width} {y}"')
              .replace(f'height="{page_height}"', f'height="{y}"'))
    svg[2] = svg[2].replace(f'height="{page_height}"', f'height="{y}"')
    svg.append("</svg>")
    path = figures / "fig1_workflow.svg"
    path.write_text("\n".join(svg))
    print(f"wrote {path}")
    svg_to_pdf(path, page_width, y)


# ---------------------------------------------------------------- cohort flow

def fig_flow(figures, cfg) -> None:
    """The supplementary cohort-flow figure of REPORT.md (figS_cohort_flow).

    One panel: a column of boxes, one per inclusion rule in the order the
    rules are applied, each with the patients left. Beside each arrow is the
    number of patients that rule removed. The last box is the analysis cohort.

    Arguments
        figures  the folder to write into
        cfg      the configuration (where the data are)
    """
    flow = cohort_flow(cfg)
    # Source data: the flow table as it is.
    source("figS_cohort_flow", a=flow)
    n_rows = len(flow)
    # A blank page with its own coordinates: one unit down per box.
    fig, ax = plt.subplots(figsize=(4.6, 0.62 * n_rows + 0.5))
    ax.axis("off")
    ax.set_xlim(0, 10.4)
    ax.set_ylim(-n_rows + 0.35, 0.65)
    for i, row in enumerate(flow.itertuples()):
        last = i == n_rows - 1
        # The box, tinted blue for the final cohort, with its two lines of text.
        ax.add_patch(plt.Rectangle((0.2, -i - 0.3), 6.0, 0.6, fc="#f5f9fe" if last else "white",
                                   ec="#9ec5f4" if last else AXIS, lw=0.8))
        label = "Analysis cohort: all four blocks" if last else (row.step[0].upper() + row.step[1:])
        ax.text(0.4, -i + 0.09, label, va="center", fontsize=6.5, color=INK2)
        ax.text(0.4, -i - 0.13, f"n = {row.patients}", va="center", fontsize=7, fontweight="bold", color=INK)
        if i:
            # The arrow down from the box above, and a side branch that says
            # how many patients were removed (grey when it is none).
            ax.annotate("", (3.0, -i + 0.3), (3.0, -i + 0.7),
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.8})
            ax.plot([3.0, 6.6], [-i + 0.5, -i + 0.5], color=MUTED, lw=0.8)
            what = row.step.replace("with ", "")
            ax.text(6.75, -i + 0.5, f"{row.excluded} without {what}", va="center", fontsize=6.5,
                    color=INK2 if row.excluded else MUTED)
    save(fig, figures, "figS_cohort_flow")


# ---------------------------------------------------------------- figure 2: prediction

def fig_prediction(tables, figures, endpoint: str) -> None:
    """Figure 2 of REPORT.md: Uno's C of each arm, with its interval.

    Panel a is the penalised Cox model, panel b the random survival forest.
    Rows are the arms in three groups: clinical alone, each omics block
    alone, and clinical plus omics. The dotted line is the clinical model and
    the solid grey line is chance (0.5).

    Arguments
        tables    folder of the result tables
        figures   the folder to write into
        endpoint  "pfi" draws Figure 2; "os" draws the same figure for overall
                  survival (fig2_prediction_os)
    """
    arms = pd.read_csv(tables / f"arms_{endpoint}.csv")
    uno_rows = arms[arms["metric"] == "uno_c"]
    # The arms shown, top to bottom; a gap is left between the groups.
    groups = [["clinical"], ["rna", "mutation", "methylation"],
              ["clinical+rna", "clinical+mutation", "clinical+methylation", "early"]]
    names = {"early": "Early fusion (all four)"}
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True, sharex=True)
    for ax, (model, label) in zip(axes, MODEL_LABELS.items()):
        rows_of_model = uno_rows[uno_rows["model"] == model].set_index("arm")
        clinical_c = rows_of_model.loc["clinical", "c_index"]
        # The two reference lines: chance (solid grey) and the dotted clinical line.
        ax.axvline(0.5, color=AXIS, lw=0.8)
        ax.axvline(clinical_c, color=INK, lw=0.8, ls=(0, (1, 2)))
        # One row per arm: the interval, and the value as a number at the right edge.
        y, ticks, tick_labels = 0.0, [], []
        for group in groups:
            for arm in group:
                row = rows_of_model.loc[arm]
                interval(ax, row.c_index, row.ci_low, row.ci_high, y, INK if arm == "clinical" else BLUE)
                ax.text(0.70, y, f"{row.c_index:.3f}", va="center", ha="right", fontsize=6.5, color=INK2)
                ticks.append(y)
                tick_labels.append(names.get(arm, ARM[arm]))
                y -= 1
            y -= 0.6
        ax.set_yticks(ticks, tick_labels)
        ax.set_xlim(0.43, 0.70)
        ax.set_xlabel("Uno's C (95% interval)")
        panel(ax, "ab"[list(MODEL_LABELS).index(model)], label)
        # The words "chance" and "clinical" under their lines, below the last row.
        ax.text(0.5, ticks[-1] - 0.95, "chance", ha="center", va="top", fontsize=6.5, color=MUTED)
        ax.text(clinical_c, ticks[-1] - 0.95, "clinical", ha="center", va="top", fontsize=6.5, color=INK2,
                bbox={"fc": "white", "ec": "none", "pad": 1})
        ax.set_ylim(ticks[-1] - 1.7, 0.7)
    fig.tight_layout(w_pad=2)
    # Source data: the rows of the arms shown, both model families in one file.
    source(f"fig2_prediction_{endpoint}", ab=uno_rows[uno_rows["arm"].isin([a for g in groups for a in g])])
    save(fig, figures, f"fig2_prediction_{endpoint}")


# ---------------------------------------------------------------- figure 3: integration

def fig_integration(tables, figures, cfg, endpoint: str) -> None:
    """Figure 3 of REPORT.md: the integration methods, with and without a
    missing block.

    Panels a, b (top row; Cox, forest): Uno's C of early fusion, late fusion
    and MOFA under every scenario, with the clinical model as a dotted line.
    Panels c, d (bottom row; Cox, forest): the paired difference of MOFA and
    of late fusion against early fusion in the same scenario; a filled marker
    is a resolved difference.

    Arguments
        tables    folder of the result tables
        figures   the folder to write into
        cfg       the configuration, for the list of scenarios
        endpoint  "pfi" draws Figure 3; "os" the same for overall survival
    """
    arms = pd.read_csv(tables / f"arms_{endpoint}.csv")
    contrasts = pd.read_csv(tables / f"contrasts_{endpoint}.csv")
    arm_rows, contrast_rows = arms[arms["metric"] == "uno_c"], contrasts[contrasts["metric"] == "uno_c"]
    scenarios = list(cfg["scenarios"])
    fig, axes = plt.subplots(2, 2, figsize=(W, 5.6), sharey="row", gridspec_kw={"height_ratios": [1.15, 1]})
    # The three methods sit side by side within a scenario, not on top of
    # each other: each is shifted sideways by this much.
    offsets = {"early": -0.24, "late": 0.0, "mofa": 0.24}
    for j, (model, label) in enumerate(MODEL_LABELS.items()):
        # a, b: every method under every scenario
        ax = axes[0, j]
        rows_of_model = arm_rows[arm_rows["model"] == model].set_index("arm")
        clinical_c = rows_of_model.loc["clinical", "c_index"]
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        # The dotted clinical line, with its label at the right end.
        ax.axhline(clinical_c, color=INK, lw=0.8, ls=(0, (1, 2)))
        ax.text(len(scenarios) - 0.45, clinical_c, " clinical\n alone", va="center", ha="left",
                fontsize=6.5, color=INK2)
        # In the tables an arm under a scenario is named "method@scenario";
        # with nothing missing it is the method name alone.
        for method, shift in offsets.items():
            for i, scenario in enumerate(scenarios):
                row = rows_of_model.loc[method if scenario == "full" else f"{method}@{scenario}"]
                ax.plot([i + shift, i + shift], [row.ci_low, row.ci_high], color=METHOD[method], lw=1.2,
                        solid_capstyle="round")
                ax.plot(i + shift, row.c_index, MARK[method], ms=4.5 * MSCALE[MARK[method]],
                        color=METHOD[method], mec="white", mew=0.6, zorder=4)
        ax.set_xticks(range(len(scenarios)), [SCEN[s].replace(" ", "\n") for s in scenarios], fontsize=6.5)
        ax.set_xlim(-0.55, len(scenarios) - 0.45)
        ax.tick_params(axis="x", length=0)
        if j == 0:
            ax.set_ylabel("Uno's C (95% interval)")
        panel(ax, "ab"[j], f"{label}: by missing block")

        # c, d: the paired differences against early fusion
        ax = axes[1, j]
        contrasts_of_model = contrast_rows[(contrast_rows["model"] == model)
                                           & (contrast_rows["b"].map(lambda b: split(b)[0]) == "early")]
        ax.axvline(0, color=INK, lw=0.8)
        ticks = []
        # The last scenario in the configuration (no omics at all) is left out of these panels.
        # MOFA is drawn just above the row's centre, late fusion just below.
        for i, scenario in enumerate(scenarios[:-1]):
            for method, shift in (("mofa", 0.17), ("late", -0.17)):
                row = contrasts_of_model[contrasts_of_model["a"] == (method if scenario == "full"
                                                                     else f"{method}@{scenario}")].iloc[0]
                interval(ax, row.difference, row.ci_low, row.ci_high, -i + shift, METHOD[method],
                         row.verdict != "not resolved", marker=MARK[method])
            ticks.append(-i)
        ax.set_yticks(ticks, [SCEN[s] for s in scenarios[:-1]])
        ax.set_ylim(-len(scenarios) + 1.45, 0.55)
        ax.set_xlim(-0.03, 0.12)
        ax.set_xlabel("Difference in Uno's C against early fusion")
        panel(ax, "cd"[j], f"{label}: gain over early fusion")

    # One legend for the whole figure, under the panels: the three methods,
    # then what a filled and an open marker mean.
    handles = [Line2D([], [], marker=MARK[m], ls="-", color=METHOD[m], ms=4.5 * MSCALE[MARK[m]], lw=1.2,
                      label=ARM[m])
               for m in offsets]
    handles += [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=4.5, label="interval excludes zero"),
                Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=4.5, label="not resolved")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.035), columnspacing=1.6)
    fig.tight_layout(w_pad=2, h_pad=1.6)
    # Source data: the arms drawn in a, b and the contrasts against early fusion.
    shown = ["clinical"] + [m if s == "full" else f"{m}@{s}" for m in offsets for s in scenarios]
    source(f"fig3_integration_{endpoint}", ab=arm_rows[arm_rows["arm"].isin(shown)],
           cd=contrast_rows[contrast_rows["b"].map(lambda b: split(b)[0]) == "early"])
    save(fig, figures, f"fig3_integration_{endpoint}")


# ---------------------------------------------------------------- figure 4: every contrast

def fig_contrasts(tables, figures, endpoint: str, stem: str) -> None:
    """Figure 4 of REPORT.md: every pre-declared contrast, as a forest plot.

    Panel a is the penalised Cox model, panel b the random survival forest.
    Each row is "first arm minus second arm" in Uno's C with its interval,
    grouped by the part of the plan. Black and filled is resolved; grey and
    open is not. The control contrasts are not shown.

    Arguments
        tables    folder of the result tables
        figures   the folder to write into
        endpoint  "pfi" or "os"
        stem      the output file name: fig4_contrasts_pfi is Figure 4,
                  figS_contrasts_os the overall-survival version
    """
    all_contrasts = pd.read_csv(tables / f"contrasts_{endpoint}.csv")
    contrasts = all_contrasts[(all_contrasts["metric"] == "uno_c") & (all_contrasts["part"] != "control")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 5.4), sharey=True, sharex=True)
    for ax, (model, label) in zip(axes, MODEL_LABELS.items()):
        contrasts_of_model = contrasts[contrasts["model"] == model]
        ax.axvline(0, color=INK, lw=0.8)        # the line of no difference
        y, ticks, tick_labels = 0.0, [], []
        for part, title in PART.items():
            # A bold heading for the part, to the left of the first panel only.
            if ax is axes[0]:
                ax.text(-0.02, y, title, transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=6.5, fontweight="bold", color=INK)
            y -= 1
            for row in contrasts_of_model[contrasts_of_model["part"] == part].itertuples():
                resolved = row.verdict != "not resolved"
                interval(ax, row.difference, row.ci_low, row.ci_high, y, INK if resolved else MUTED,
                         resolved, ms=4)
                ticks.append(y)
                tick_labels.append(contrast_label(row.a, row.b))
                y -= 1
            y -= 0.5
        ax.set_yticks(ticks, tick_labels)
        ax.set_ylim(y + 0.9, 0.7)
        ax.set_xticks([-0.05, 0, 0.05], ["−0.05", "0", "0.05"])
        ax.set_xlabel("Difference in Uno's C (95% interval)")
        panel(ax, "ab"[list(MODEL_LABELS).index(model)], label)
        # The count of resolved contrasts, above the top right corner.
        n_resolved = int((contrasts_of_model["verdict"] != "not resolved").sum())
        ax.text(1, 1.012, f"{n_resolved} of {len(contrasts_of_model)} resolved", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, color=INK2)
    fig.tight_layout(w_pad=1.5)
    source(stem, ab=contrasts)
    save(fig, figures, stem)


# ---------------------------------------------------------------- figure 5: external

def fig_external(tables, figures) -> None:
    """Figure 5 of REPORT.md: gain over the clinical model in the GEO cohorts.

    A grid of small panels with no letters: one column per cohort and
    endpoint, the top row for the penalised Cox model and the bottom row for
    the random survival forest. In each panel the four rows are clinical +
    expression, early fusion, late fusion and MOFA: the difference in Uno's C
    against the clinical model, with its interval. A filled marker is a
    resolved difference.

    Arguments
        tables   folder of the result tables
        figures  the folder to write into
    """
    all_contrasts = pd.read_csv(tables / "external_contrasts.csv")
    contrasts = all_contrasts[(all_contrasts["metric"] == "uno_c") & (all_contrasts["b"] == "clinical")]
    # One column of panels per cohort and endpoint, sorted by endpoint, then cohort.
    cells = (contrasts[["endpoint", "cohort", "n", "n_events"]].drop_duplicates()
             .sort_values(["endpoint", "cohort"]).reset_index(drop=True))
    order = ["clinical+rna", "early@geo", "late@geo", "mofa@geo"]
    tick_labels = ["Clinical + expression", "Early fusion", "Late fusion", "MOFA"]
    fig, axes = plt.subplots(2, len(cells), figsize=(W, 3.5), sharex=True, sharey=True)
    for i, (model, label) in enumerate(MODEL_LABELS.items()):
        for j, cell in cells.iterrows():
            ax = axes[i, j]
            ax.axvline(0, color=INK, lw=0.8)        # the line of no difference
            contrasts_of_cell = contrasts[(contrasts["model"] == model) & (contrasts["cohort"] == cell.cohort)
                                          & (contrasts["endpoint"] == cell.endpoint)].set_index("a")
            for k, arm in enumerate(order):
                row = contrasts_of_cell.loc[arm]
                interval(ax, row.difference, row.ci_low, row.ci_high, -k, METHOD[split(arm)[0]],
                         row.verdict != "not resolved", ms=4, lw=1.2, marker=MARK[split(arm)[0]])
            ax.set_yticks(range(0, -len(order), -1), tick_labels)
            ax.set_ylim(-len(order) + 0.4, 0.6)
            ax.set_xlim(-0.14, 0.30)
            ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3], ["−0.1", "0", "0.1", "0.2", "0.3"])
            # Above the top row: the cohort, its endpoint, patients and events.
            if i == 0:
                kind = "relapse-free" if "RFS" in cell.endpoint else "overall"
                ax.set_title(f"{cell.cohort}\n{kind} survival\nn = {cell.n}, {cell.n_events} events",
                             fontsize=6.5, color=INK, pad=5)
            # Right of the last column: the model family, written sideways.
            if j == len(cells) - 1:
                ax.text(1.06, 0.5, label, transform=ax.transAxes, rotation=270, va="center", ha="left",
                        fontsize=6.5, fontweight="bold", color=INK)
    fig.supxlabel("Difference in Uno's C against the clinical model (95% interval)", fontsize=7, color=INK2,
                  y=0.075)
    # The legend: what a filled and an open marker mean.
    handles = [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=4.5, label="interval excludes zero"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=4.5, label="not resolved")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0, 0.06, 0.985, 1))
    source("fig5_external", a=contrasts[contrasts["a"].isin(order)])
    save(fig, figures, "fig5_external")


# ---------------------------------------------------------------- figure 6: MOFA factors

def heat(ax, values, cmap, vmin, vmax, column_labels, row_labels, fmt, flip=0.6):
    """Draw a table of numbers as a heatmap, with the number written in every cell.

    Arguments
        ax             the panel
        values         the numbers, a 2-D array (rows by columns)
        cmap           the colour scale
        vmin, vmax     the values at the two ends of the colour scale
        column_labels  the text under each column
        row_labels     the text beside each row
        fmt            a function that turns a number into the text of its cell
        flip           a cell this far along the colour scale (0 to 1) is dark,
                       so its number is written in white instead of black
    """
    ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(column_labels)), column_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # White lines between the cells: minor ticks are put on the cell edges
    # and only the minor grid is drawn.
    ax.set_xticks(np.arange(-0.5, len(column_labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    # The number in every cell.
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, fmt(value), ha="center", va="center", fontsize=6.5,
                color="white" if abs(value) / max(abs(vmin), abs(vmax)) > flip else INK)


def fig_factors(tables, figures) -> None:
    """Figure 6 of REPORT.md: the MOFA factors, one row per factor.

    Panel a: the share of each omics block's variance a factor explains, in
    per cent. Panel b: the Cox z of each factor with progression; dotted
    lines mark |z| = 2 and a factor beyond them is blue. Panel c: the
    correlation of each factor with stage, age, sex, smoking and tumour
    mutation burden.

    Arguments
        tables   folder of the result tables
        figures  the folder to write into
    """
    factors = pd.read_csv(tables / "biology_factors.csv")
    names = [x.replace("factor", "Factor ") for x in factors["factor"]]
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.4), gridspec_kw={"width_ratios": [1, 1.05, 1.6]})

    # a: variance explained. This panel carries the factor names for all three.
    heat(axes[0], factors[["r2_rna", "r2_mutation", "r2_methylation"]].to_numpy() * 100, SEQ, 0, 20,
         ["Expression", "Mutation", "Methylation"], names, lambda v: f"{v:.0f}")
    panel(axes[0], "a", "Variance explained (%)")

    # b: association with the outcome, as bars in the same row order as a.
    ax = axes[1]
    y = np.arange(len(factors))
    associated = factors["cox_z"].abs() > 2
    ax.barh(y, factors["cox_z"], height=0.62, color=[BLUE if h else AXIS for h in associated])
    for v in (-2, 2):
        ax.axvline(v, color=MUTED, lw=0.7, ls=(0, (1, 2)))
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [])
    ax.set_ylim(len(factors) - 0.5, -0.5)       # factor 1 at the top, as in the heatmaps
    ax.set_xlim(-4, 4)
    ax.set_xlabel("Cox z with progression")
    ax.text(2.12, len(factors) - 0.55, "|z| = 2", ha="left", va="bottom", fontsize=6, color=MUTED)
    panel(ax, "b", "Association with outcome")

    # c: correlation with patient covariates. A true minus sign replaces the hyphen.
    columns = ["r_stage_ordinal", "r_age", "r_sex_male", "r_ever_smoker", "r_tmb"]
    heat(axes[2], factors[columns].to_numpy(), DIV, -0.6, 0.6,
         ["Stage", "Age", "Male sex", "Ever\nsmoker", "Mutation\nburden"], [""] * len(factors),
         lambda v: f"{v:+.2f}".replace("-", "−"), flip=0.7)
    panel(axes[2], "c", "Correlation with patient covariates")
    fig.tight_layout(w_pad=1.2)
    # Source data: the whole factor table serves the three panels.
    source("fig6_mofa_factors", abc=factors)
    save(fig, figures, "fig6_mofa_factors")


# ---------------------------------------------------------------- figure 7: biology

def fig_biology(tables, figures) -> None:
    """Figure 7 of REPORT.md: the gene lists on the protein network.

    Panel a: a heatmap of the larger network modules (rows) against the gene
    lists (columns); the number is the genes of that list in the module and
    the shading is their share of the module. Panel b: interactions per gene
    (degree) for each list, one dot per gene, with the median as a bar.
    Panel c: genes shared by two lists, observed against expected by chance.
    Panel d: protein interactions between two lists, observed against
    expected.

    Arguments
        tables   folder of the result tables
        figures  the folder to write into
    """
    modules = pd.read_csv(tables / "biology_modules.csv")
    overlap = pd.read_csv(tables / "biology_overlap.csv")
    links = pd.read_csv(tables / "biology_network_links.csv")
    centrality = pd.read_csv(tables / "biology_network_centrality.csv")
    fig = plt.figure(figsize=(W, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.25, 1])

    # a: who lives in which module
    ax = fig.add_subplot(gs[0, 0])
    big_modules = modules[modules["genes"] >= 12].copy()       # small modules are left out
    short = SHORT_TERMS
    columns = ["prognostic_risk", "prognostic_protective", "model", "factor2", "proliferation"]
    counts = big_modules[columns].to_numpy()
    # The colour is the share of the module (per cent); the number written is the count.
    share = counts / big_modules["genes"].to_numpy()[:, None] * 100
    ax.imshow(share, cmap=SEQ, vmin=0, vmax=90, aspect="auto")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(range(len(columns)), ["Higher\nrisk", "Protective", "Kept by\nCox model",
                                        "MOFA\nfactor 2", "Prolif.\nmetagene"], fontsize=6)
    # Row labels: module number, its top enriched term (shortened), and its size.
    ax.set_yticks(range(len(big_modules)),
                  [f"M{row.module}  {short.get(row.top_term, row.top_term)}  ({row.genes})"
                   for row in big_modules.itertuples()])
    ax.tick_params(length=0)
    # White lines between the cells, as in heat().
    ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(big_modules)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    # The count in every cell: white on a dark cell, grey for a zero.
    for (i, j), count in np.ndenumerate(counts):
        ax.text(j, i, str(count), ha="center", va="center", fontsize=6.5,
                color="white" if share[i, j] > 50 else (INK if count else MUTED))
    panel(ax, "a", "Genes of each list in each network module")

    # b: degree by list
    ax = fig.add_subplot(gs[0, 1])
    order = [("prognostic_risk", "Prognostic, higher risk"), ("proliferation", "Proliferation metagene"),
             ("several", "In more than one list"), ("factor", "MOFA factor 2"),
             ("prognostic_protective", "Prognostic, protective"), ("model", "Kept by Cox model")]
    # The dots are moved up and down at random so they do not hide each other.
    # The generator is seeded, so the figure is the same on every run.
    rng = np.random.default_rng(0)
    for i, (group, _) in enumerate(order):
        degrees = centrality.loc[centrality["group"] == group, "Degree"].to_numpy()
        ax.plot(degrees, -i + rng.uniform(-0.2, 0.2, len(degrees)), "o", ms=3, color=BLUE, alpha=0.55,
                mec="none")
        ax.plot([np.median(degrees)] * 2, [-i - 0.33, -i + 0.33], color=INK, lw=1.6, solid_capstyle="round")
    # Row labels: the list and, in brackets, its number of genes in the network.
    ax.set_yticks(range(0, -len(order), -1),
                  [f"{label}  ({int((centrality['group'] == group).sum())})" for group, label in order])
    ax.set_ylim(-len(order) + 0.4, 0.6)
    ax.set_xlim(-2, 68)
    ax.set_xlabel("Interactions per gene (degree); bar is the median")
    panel(ax, "b", "Connectivity by gene list")

    # c, d: observed against expected
    def dumbbell(ax, rows, xlabel, title, letter):
        """One row per pair of lists: an open dot at the expected number, a
        filled dot at the observed number, and a grey bar joining them.

        `rows` is a list of (label, observed, expected). The observed number
        is written above its dot and the expected number below its dot.
        """
        for i, (label, observed, expected) in enumerate(rows):
            ax.plot([expected, observed], [-i, -i], color=AXIS, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(expected, -i, "o", ms=5.5, mfc="white", mec=INK2, mew=1.1, zorder=3)
            ax.plot(observed, -i, "o", ms=5.5, color=BLUE, zorder=4)
            ax.annotate(f"{observed:.0f}", (observed, -i), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=6.5, fontweight="bold", color=INK)
            ax.annotate(f"{expected:.0f}" if expected >= 10 else f"{expected:.1f}", (expected, -i),
                        xytext=(0, -12), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
        ax.set_yticks(range(0, -len(rows), -1), [r[0] for r in rows])
        ax.set_ylim(-len(rows) + 0.35, 0.65)
        ax.set_xlabel(xlabel)
        panel(ax, letter, title)

    # c: the three pairs of lists the report discusses, from the overlap table.
    list_names = {"prognostic": "prognostic", "model": "Cox model", "factor2": "factor 2",
                  "proliferation": "proliferation"}
    keep = [("prognostic", "model"), ("prognostic", "factor2"), ("model", "factor2")]
    rows = [(f"{list_names[a].capitalize()} and {list_names[b]}",
             *overlap[(overlap.list_a == a) & (overlap.list_b == b)][["overlap", "expected"]].iloc[0])
            for a, b in keep]
    ax = fig.add_subplot(gs[1, 0])
    dumbbell(ax, rows, "Number of genes", "Genes shared by two lists", "c")
    ax.set_xlim(-0.8, 15)
    # d: every pair in the network-links table.
    rows = [(f"{list_names[row.list_a].capitalize()} and {list_names[row.list_b]}", row.edges_between,
             row.expected)
            for row in links.itertuples()]
    ax = fig.add_subplot(gs[1, 1])
    dumbbell(ax, rows, "Number of protein interactions", "Interactions between two lists", "d")
    ax.set_xlim(-25, 470)

    # The legend for c and d, under the figure.
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, ms=5.5, label="observed"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5.5, label="expected by chance")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=1.5, h_pad=1.8)
    # Source data: one file per panel, holding only what the panel shows.
    source("fig7_biology", a=big_modules, b=centrality[["name", "group", "Degree"]],
           c=overlap[[(a, b) in keep for a, b in zip(overlap.list_a, overlap.list_b)]], d=links)
    save(fig, figures, "fig7_biology")


# ---------------------------------------------------------------- figure 9: checks (file fig8_checks)

def fig_checks(tables, figures, endpoint: str) -> None:
    """Figure 9 of REPORT.md: three of the robustness checks. The file is
    named fig8_checks; Figure 8 of the report is the Cytoscape network.

    Panel a: Uno's C of every arm and model on the real outcome and on a
    shuffled outcome, which should sit at chance (0.5). Panel b: the gain
    over the clinical model when the width of the omics blocks is fixed
    instead of tuned. Panel c: each contrast under the two concordance
    metrics, Uno's C across and Harrell's C up; filled points are resolved
    under Uno's C.

    Arguments
        tables    folder of the result tables
        figures   the folder to write into
        endpoint  "pfi" or "os": which run the checks are drawn for
    """
    real = pd.read_csv(tables / f"cv_{endpoint}_summary.csv")
    shuffled = pd.read_csv(tables / f"cv_{endpoint}_permuted_summary.csv")
    sweep = pd.read_csv(tables / "width_sweep.csv")
    contrasts = pd.read_csv(tables / f"contrasts_{endpoint}.csv")
    contrasts = contrasts[contrasts["part"] != "control"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.1, 1, 1]})

    # a: real against shuffled outcome. Only the arms scored in both runs are kept.
    ax = axes[0]
    both = shuffled.merge(real, on=["model", "arm"], suffixes=("_perm", "_real"))
    # Random up-and-down positions so the dots do not hide each other; seeded,
    # so the figure is the same on every run.
    rng = np.random.default_rng(0)
    ax.axvline(0.5, color=INK, lw=0.8)      # chance
    for i, (column, colour) in enumerate((("uno_c_real", BLUE), ("uno_c_perm", MUTED))):
        scores = both[column].to_numpy()
        ax.plot(scores, -i + rng.uniform(-0.18, 0.18, len(scores)), "o", ms=4, color=colour, alpha=0.75,
                mec="white", mew=0.4)
    ax.set_yticks([0, -1], ["Real\noutcome", "Shuffled\noutcome"])
    ax.set_ylim(-1.6, 0.6)
    ax.set_xlim(0.46, 0.63)
    ax.set_xlabel(f"Uno's C, {len(both)} arm-model pairs")
    panel(ax, "a", "Shuffled outcome")

    # b: one line per model and arm across the fixed block widths. The colour
    # and marker are the arm; a solid line is Cox, a dashed line the forest.
    ax = axes[1]
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color=INK, lw=0.8)        # equal to the clinical model
    widths = sorted(sweep["k"].unique())
    styles = {"coxnet": "-", "rsf": (0, (4, 2))}
    for (model, arm), line_rows in sweep.groupby(["model", "arm"], sort=False):
        line_rows = line_rows.sort_values("k")
        ax.plot(range(len(widths)), line_rows["minus_clinical"], ls=styles[model], lw=1.5, marker=MARK[arm],
                ms=4 * MSCALE[MARK[arm]],
                color=METHOD[arm], mec="white", mew=0.5)
    ax.set_xticks(range(len(widths)), [f"{k:,}" for k in widths])
    ax.set_xlim(-0.25, len(widths) - 0.75)
    ax.set_ylim(-0.05, 0.03)
    ax.set_xlabel("Features per omics block (fixed)")
    ax.set_ylabel("Uno's C minus clinical")
    # The legend of panel b: the two arms, then the two line styles.
    handles = [Line2D([], [], color=METHOD["clinical+rna"], lw=1.5, marker=MARK["clinical+rna"], ms=4,
                      mec="white", mew=0.5, label="Clinical + expression"),
               Line2D([], [], color=METHOD["early"], lw=1.5, marker=MARK["early"], ms=4, mec="white", mew=0.5,
                      label="Early fusion"),
               Line2D([], [], color=INK2, lw=1.2, ls="-", label="Cox"),
               Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 2)), label="Forest")]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=6, handlelength=2.2, columnspacing=1,
              borderaxespad=0.1)
    panel(ax, "b", "Fixed block width")

    # c: the two metrics against each other. The table is turned so that each
    # contrast is one row with a column per metric.
    ax = axes[2]
    wide = contrasts.pivot_table(index=["model", "a", "b"], columns="metric",
                                 values="difference").reset_index()
    verdicts = contrasts[contrasts["metric"] == "uno_c"].set_index(["model", "a", "b"])["verdict"]
    resolved = np.array([verdicts[(row.model, row.a, row.b)] != "not resolved" for row in wide.itertuples()])
    lim = 0.06
    ax.grid(axis="y")
    # The diagonal is where the two metrics agree exactly.
    ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=0.8)
    ax.axhline(0, color=INK, lw=0.6)
    ax.axvline(0, color=INK, lw=0.6)
    # Open points first, filled points after, so a resolved contrast is on top.
    ax.plot(wide["uno_c"][~resolved], wide["harrell_c"][~resolved], "o", ms=4, mfc="white", mec=MUTED, mew=1)
    ax.plot(wide["uno_c"][resolved], wide["harrell_c"][resolved], "o", ms=4, color=INK)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([-0.04, 0, 0.04], ["−0.04", "0", "0.04"])
    ax.set_yticks([-0.04, 0, 0.04], ["−0.04", "0", "0.04"])
    ax.tick_params(axis="y", length=2.5)
    ax.set_xlabel("Difference in Uno's C")
    ax.set_ylabel("Difference in Harrell's C")
    # The correlation of the two metrics, written in the top left corner.
    correlation = np.corrcoef(wide["uno_c"], wide["harrell_c"])[0, 1]
    ax.text(0.04, 0.96, f"r = {correlation:.2f}\n{len(wide)} contrasts", transform=ax.transAxes, va="top",
            fontsize=6.5, color=INK2)
    panel(ax, "c", "Metric agreement")
    fig.tight_layout(w_pad=1.3)
    source("fig8_checks", a=both, b=sweep, c=wide.assign(resolved_uno_c=resolved))
    save(fig, figures, "fig8_checks")


def main() -> int:
    """Draw every figure, in the order of the report.

    Takes no arguments. Returns 0, the exit code of a run that finished.
    """
    cfg = load_config()
    tables, figures = out_dirs(cfg)
    primary, sensitivity = cfg["endpoint"]["primary"].lower(), cfg["endpoint"]["sensitivity"].lower()
    style()

    # ---- the two diagrams: workflow and cohort flow -------------------------
    # The cohort is loaded only for its two counts, patients and events.
    cohort = load_cohort(cfg, primary, verbose=False)
    workflow_svg(figures, cfg, cohort.n, int(cohort.y["event"].sum()))
    fig_flow(figures, cfg)

    # ---- the primary endpoint: Figures 2 to 7 and 9 --------------------------
    fig_prediction(tables, figures, primary)
    fig_integration(tables, figures, cfg, primary)
    fig_contrasts(tables, figures, primary, f"fig4_contrasts_{primary}")
    fig_external(tables, figures)
    fig_factors(tables, figures)
    fig_biology(tables, figures)
    fig_checks(tables, figures, primary)

    # ---- the sensitivity endpoint: Figures 2 to 4 again ----------------------
    fig_prediction(tables, figures, sensitivity)
    fig_integration(tables, figures, cfg, sensitivity)
    fig_contrasts(tables, figures, sensitivity, f"figS_contrasts_{sensitivity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
