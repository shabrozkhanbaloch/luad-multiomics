#!/usr/bin/env python
"""Step 4b. Send the gene network to Cytoscape: style, layout, analyse, export.

    python scripts/04b_cytoscape.py              # uses Cytoscape if open, else starts it
    python scripts/04b_cytoscape.py --no-launch  # do not start Cytoscape if it is closed
    python scripts/04b_cytoscape.py --require    # fail, instead of skipping, without Cytoscape

What it is for
    Step 4 builds the gene network and draws it with matplotlib. This step
    loads the same network into the Cytoscape desktop program, where it can
    be explored by hand, and redraws the figures there. It drives Cytoscape
    through its automation port (CyREST, localhost:1234), so what you see in
    the Cytoscape window is exactly what the pipeline produced:

    - the network, one cluster per module, each titled with its top enriched term
    - style "LUAD lists":  node colour = which gene list, size = |Cox z|
    - style "LUAD Cox z":  node colour = Cox z, blue (protective) to red (risk)
    - Cytoscape's own network analysis (degree, betweenness, ...) on every node

    It also checks that Cytoscape's network statistics agree with the table
    step 4 wrote. Cytoscape is left open afterwards so the network can be
    explored.

    This step is optional. On a machine without Cytoscape it prints a message
    and stops without an error, so `make all` still finishes; nothing later
    depends on it, and the network figures it would redraw are already in the
    repository.

What it reads
    results/network/network.graphml                the network of step 4
    results/tables/biology_modules.csv             the module table, for the titles
    results/tables/biology_network_centrality.csv  step 4's network statistics, for the check
    config/config.yaml                             `cytoscape: launcher`, used to start Cytoscape

What it writes
    results/figures/network_cytoscape_lists.png and .pdf   coloured by gene list
    results/figures/network_cytoscape_coxz.png and .pdf    coloured by Cox z
    results/network/network.cys                            the session file, which reopens
                                                           everything in Cytoscape

How long it takes
    About 30 seconds with Cytoscape open. Starting Cytoscape first adds about
    a minute. Where Cytoscape is not installed the step is skipped with a
    message, at once.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from _common import load_config, out_dirs

import networkx as nx
import pandas as pd
import py4cytoscape as p4c

from luad import biology as B

# Checked for colour-vision deficiency over every pair (worst pair 9.1 in OKLab
# x100 under protanopia; 8 is the target). Each list also has its own node
# shape, so the figure does not depend on colour.
GROUP_COLOURS = {"prognostic_risk": "#C2541C", "prognostic_protective": "#2A78D6",
                 "model": "#1BAF7A", "factor": "#B04A9A", "proliferation": "#EDA100",
                 "several": "#0B0B0B"}
GROUP_SHAPES = {"prognostic_risk": "ELLIPSE", "prognostic_protective": "DIAMOND",
                "model": "ROUND_RECTANGLE", "factor": "TRIANGLE", "proliferation": "HEXAGON",
                "several": "OCTAGON"}
# The legend text for each gene list, and the three legend entries of the
# "LUAD Cox z" style (colour, text).
GROUP_LABELS = {"prognostic_risk": "Prognostic, higher risk",
                "prognostic_protective": "Prognostic, protective",
                "model": "Kept by the Cox model", "factor": "MOFA factor gene",
                "proliferation": "Proliferation metagene", "several": "In more than one list"}
COXZ_LEGEND = [("#2A78D6", "Cox z -5, protective"), ("#F2F2F2", "Cox z 0"),
               ("#C2541C", "Cox z +5, higher risk")]


def ready() -> bool:
    """True if Cytoscape is running and answers on its automation port.

    Takes no arguments. Any failure to connect counts as "not running".
    """
    try:
        p4c.cytoscape_version_info()
        return True
    except Exception:
        return False


def ensure_cytoscape(launcher: str | None, launch: bool, wait_s: int = 300) -> bool:
    """True once Cytoscape answers. False if it is not running and cannot be
    started from here.

    Arguments
        launcher  path of the program that starts Cytoscape, or None
        launch    False means: never start Cytoscape, only use it if it is open
        wait_s    how many seconds to wait for a started Cytoscape to answer
    """
    if ready():
        return True
    if not launch or not launcher or not Path(launcher).expanduser().exists():
        return False
    print(f"  starting Cytoscape ({launcher}); this takes about a minute")
    # Started in its own session so it stays open after this script finishes.
    subprocess.Popen([str(Path(launcher).expanduser())], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    # Cytoscape needs a while before its automation port answers: ask every
    # 5 seconds until it does or the waiting time is over.
    started = time.time()
    while time.time() - started < wait_s:
        if ready():
            return True
        time.sleep(5)
    return False


def load_graph(path: Path) -> nx.Graph:
    """Read the network step 4 wrote and prepare it for Cytoscape.

    Arguments
        path  the GraphML file

    Returns the graph. Every node gains `abs_cox_z`, the size of its Cox z
    without the sign, which the node size is mapped to, and its `module` is
    made a whole number.
    """
    G = nx.read_graphml(path)
    for gene, attributes in G.nodes(data=True):
        attributes["abs_cox_z"] = abs(float(attributes.get("cox_z", 0.0)))
        attributes["module"] = int(attributes.get("module", 0))
    return G


def make_styles() -> None:
    """Create the two visual styles in Cytoscape, "LUAD lists" and "LUAD Cox z".

    Takes no arguments and returns nothing. The two styles share the node
    size, the labels and the hidden edges; they differ only in what the node
    colour (and shape) means. A style of the same name left by an earlier run
    is deleted first.
    """
    # What every node and edge looks like unless a mapping below says otherwise.
    defaults = {"NODE_SHAPE": "ELLIPSE", "NODE_BORDER_WIDTH": 1.5, "NODE_BORDER_PAINT": "#FFFFFF",
                "EDGE_WIDTH": 1, "EDGE_STROKE_UNSELECTED_PAINT": "#8A8A8A",
                "NODE_LABEL_FONT_SIZE": 17, "NODE_LABEL_COLOR": "#222222",
                "NODE_LABEL_POSITION": "S,N,c,0.00,3.00",      # label under the node
                "NETWORK_BACKGROUND_PAINT": "#FFFFFF"}
    # A mapping ties a visual property to a column of the node or edge table:
    # "c" is continuous (a number range), "d" is discrete (one value per
    # category), "p" is passthrough (the column's value is used as it is).
    size = p4c.map_visual_property("node size", "abs_cox_z", "c", [0.0, 6.0], [26, 74])
    label = p4c.map_visual_property("node label", "name", "p")
    # Interactions inside a module are drawn; those between modules would
    # cross the whole canvas, so they are hidden and counted in a note.
    edges = p4c.map_visual_property("edge transparency", "within_module", "d",
                                    ["true", "false"], [90, 0])
    by_list = p4c.map_visual_property("node fill color", "group", "d",
                                      list(GROUP_COLOURS), list(GROUP_COLOURS.values()))
    by_z = p4c.map_visual_property("node fill color", "cox_z", "c", [-5.0, 0.0, 5.0],
                                   ["#2A78D6", "#F2F2F2", "#C2541C"])
    shape = p4c.map_visual_property("node shape", "group", "d",
                                    list(GROUP_SHAPES), list(GROUP_SHAPES.values()))
    for name, extra in (("LUAD lists", [by_list, shape]), ("LUAD Cox z", [by_z])):
        if name in p4c.get_visual_style_names():
            p4c.delete_visual_style(name)
        p4c.create_visual_style(name, defaults=defaults, mappings=[size, label, edges, *extra])


def clear_annotations(prefix: str = "") -> None:
    """Delete annotations (titles, notes, legend marks) from the Cytoscape canvas.

    Arguments
        prefix  only annotations whose name starts with this are deleted;
                the default, an empty prefix, deletes all of them
    """
    for annotation in p4c.get_annotation_list() or []:
        if annotation.get("name", "").startswith(prefix):
            p4c.delete_annotation(annotation["uuid"])


def text(content: str, cx: float, y: float, size: int, name: str, bold: bool = False) -> None:
    """Text centred on cx. Cytoscape anchors text at its left edge, so the
    width is estimated from the character count.

    Arguments
        content  the text; each line of it becomes one annotation
        cx, y    the centre of the text across, and the top of its first line
        size     the font size
        name     the annotation name; the line number is added to it
        bold     True for bold text
    """
    for line_number, line in enumerate(content.split("\n")):
        p4c.add_annotation_text(text=line, x_pos=cx - 0.27 * size * len(line),
                                y_pos=y + line_number * size * 1.25,
                                font_size=size, font_style="bold" if bold else "plain",
                                color="#111111", name=f"{name}_{line_number}", canvas="foreground")


def legend(items, x0: float, y: float, name: str) -> float:
    """A row of coloured marks with labels; returns the x where the row ends.

    Arguments
        items  one (colour, label) or (colour, label, shape) per entry; an
               entry without a shape is drawn as a circle
        x0, y  where the row starts
        name   the start of the annotation names, so the row can be deleted again
    """
    x = x0
    for index, (colour, label, *shape) in enumerate(items):
        p4c.add_annotation_shape(type=shape[0] if shape else "ELLIPSE", x_pos=x, y_pos=y,
                                 fill_color=colour, opacity=100,
                                 border_thickness=1, border_color="#999999", height=40, width=40,
                                 name=f"{name}_dot{index}", canvas="foreground")
        p4c.add_annotation_text(text=label, x_pos=x + 55, y_pos=y + 4, font_size=26, color="#111111",
                                name=f"{name}_txt{index}", canvas="foreground")
        # Move on by the mark, the estimated width of the label, and a gap.
        x += 55 + 15.5 * len(label) + 70
    return x


def trim(path: Path, margin: int = 40) -> None:
    """Cut the empty border Cytoscape leaves around the network.

    Arguments
        path    the PNG file; it is replaced by the trimmed image
        margin  how many pixels of white to keep on every side
    """
    from PIL import Image, ImageChops
    image = Image.open(path).convert("RGB")
    # The smallest rectangle that holds every pixel that is not white.
    box = ImageChops.difference(image, Image.new("RGB", image.size, (255, 255, 255))).getbbox()
    if box:
        left, top, right, bottom = box
        image.crop((max(left - margin, 0), max(top - margin, 0),
                    min(right + margin, image.width), min(bottom + margin, image.height))).save(path)


def main() -> int:
    """Run the whole step: load, style, check, annotate, export, save.

    Takes its options from the command line. Returns 0 when the step
    finished or was skipped because Cytoscape is not there.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-launch", action="store_true",
                        help="do not start Cytoscape if it is closed")
    parser.add_argument("--require", action="store_true",
                        help="end with an error, instead of skipping, where Cytoscape is not available")
    args = parser.parse_args()
    cfg = load_config()
    tables, figures = out_dirs(cfg)
    net_dir = Path(cfg["paths"]["network"])
    graphml = net_dir / "network.graphml"
    if not graphml.exists():
        raise SystemExit(f"{graphml} not found: run scripts/04_biology.py first")

    # ---- is Cytoscape there? ----------------------------------------------
    # Without Cytoscape the step is skipped, not failed, unless --require
    # was given.
    if not ensure_cytoscape(cfg.get("cytoscape", {}).get("launcher"), not args.no_launch):
        message = ("Cytoscape is not running and could not be started here. "
                   "Step 4b is optional: open Cytoscape and run `make cytoscape` to redraw "
                   "the network figures.")
        if args.require:
            raise SystemExit(message)
        print(f"  SKIPPED. {message}")
        return 0
    version = p4c.cytoscape_version_info()
    print(f"  Cytoscape {version['cytoscapeVersion']} (automation API {version['apiVersion']})")

    # ---- the network and where each gene goes ------------------------------
    G = load_graph(graphml)
    modules = pd.read_csv(tables / "biology_modules.csv")
    # The same module-by-module layout as the matplotlib figure of step 4.
    pos, boxes = B.module_layout(G)
    titles = B.module_titles(G, modules, boxes)
    # H is the part that is drawn: the genes in modules of 4 or more. Each
    # interaction is marked as inside one module or between two, because the
    # styles hide the interactions between modules.
    H = G.subgraph(pos).copy()
    for u, v, attributes in H.edges(data=True):
        attributes["within_module"] = "true" if H.nodes[u]["module"] == H.nodes[v]["module"] else "false"

    # ---- send it to Cytoscape ----------------------------------------------
    # A network left by an earlier run is replaced, so Cytoscape never shows
    # two copies.
    for old in p4c.get_network_list():
        if old == "LUAD gene network":
            p4c.delete_network(old)
    p4c.create_network_from_networkx(H, title="LUAD gene network", collection="LUAD multi-omics")
    make_styles()
    p4c.set_visual_style("LUAD lists")
    # Every gene is put at the position computed above, in place of
    # Cytoscape's own layout. The short pause lets Cytoscape finish drawing.
    gene_names = list(pos)
    p4c.set_node_position_bypass(gene_names, [pos[g][0] for g in gene_names],
                                 [pos[g][1] for g in gene_names])
    time.sleep(2)

    # ---- check the network statistics --------------------------------------
    # Cytoscape's own network statistics, compared with the table step 4
    # computed. Two independent programs giving the same numbers is the check.
    p4c.analyze_network(directed=False)
    theirs = p4c.get_table_columns("node").set_index("name")
    ours = pd.read_csv(tables / "biology_network_centrality.csv").set_index("name")
    measures = ["Degree", "BetweennessCentrality", "ClosenessCentrality",
                "ClusteringCoefficient", "NeighborhoodConnectivity"]
    gap = float((ours[measures] - theirs.loc[ours.index, measures].astype(float)).abs().max().max())
    print(f"  network statistics: largest difference from Cytoscape's own analysis {gap:.1e}")
    if gap > 1e-9:
        raise SystemExit("Cytoscape's network statistics disagree with step 4's table")

    # ---- titles and notes ---------------------------------------------------
    # Titles, notes and legend are annotations on the Cytoscape canvas.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    left, mid, bottom = min(xs), (min(xs) + max(xs)) / 2, max(ys)
    # The hidden interactions are counted per pair of modules and written out
    # in a note, largest count first (pairs with at least 5).
    links = B.cross_module_links(G, set(pos))
    note = ("Lines are protein interactions inside a module. Interactions between modules are hidden: "
            + "; ".join(f"modules {a} and {b}: {n}" for (a, b), n in
                        sorted(links.items(), key=lambda pair: -pair[1]) if n >= 5) + ".")
    clear_annotations()
    for module, (centre_x, top, _radius) in boxes.items():
        text(titles[module], centre_x, top + 10, 30, f"title{module}", bold=True)
    text(f"{H.number_of_nodes()} genes in modules of 4 or more "
         f"({G.number_of_nodes() - H.number_of_nodes()} in smaller fragments not drawn). "
         "Node size: |Cox z|.", mid, bottom + 150, 26, "note_a")
    text(note, mid, bottom + 195, 26, "note_b")

    # ---- one figure per style -----------------------------------------------
    # Each style needs its own legend, so the legend is deleted and drawn
    # again before each export. The pauses let Cytoscape finish redrawing
    # before the picture is taken.
    for style, stem, items in (
            ("LUAD lists", "network_cytoscape_lists",
             [(GROUP_COLOURS[k], GROUP_LABELS[k], GROUP_SHAPES[k]) for k in GROUP_COLOURS]),
            ("LUAD Cox z", "network_cytoscape_coxz", COXZ_LEGEND)):
        clear_annotations("legend")
        # The width of the whole legend row, estimated as legend() spaces it,
        # so the row can be centred under the network.
        width = sum(55 + 15.5 * len(item[1]) + 70 for item in items)
        legend(items, mid - width / 2, bottom + 260, "legend")
        p4c.set_visual_style(style)
        time.sleep(2)
        p4c.fit_content()
        time.sleep(2)
        png_path = figures / f"{stem}.png"
        png_path.unlink(missing_ok=True)
        # A fixed width in pixels: with a zoom factor the image size follows the
        # size of the Cytoscape window, so two machines give different figures.
        p4c.export_image(str(png_path), type="PNG", units="pixels", width=3000, overwrite_file=True)
        trim(png_path)
        # A vector copy for a manuscript: gene labels stay sharp at any size.
        pdf_path = figures / f"{stem}.pdf"
        pdf_path.unlink(missing_ok=True)
        p4c.export_image(str(pdf_path), type="PDF", overwrite_file=True)
        print(f"  wrote {png_path} and {pdf_path}")

    # ---- leave Cytoscape ready to explore, and save the session --------------
    # Leave Cytoscape on the "lists" view with its legend.
    clear_annotations("legend")
    items = [(GROUP_COLOURS[k], GROUP_LABELS[k], GROUP_SHAPES[k]) for k in GROUP_COLOURS]
    legend(items, mid - sum(55 + 15.5 * len(item[1]) + 70 for item in items) / 2, bottom + 260, "legend")
    p4c.set_visual_style("LUAD lists")

    session = net_dir / "network.cys"
    p4c.save_session(str(session), overwrite_file=True)
    print(f"  wrote {session}")
    print(f"  {H.number_of_nodes()} genes drawn, {G.number_of_edges()} interactions in the full network; "
          "Cytoscape is open with the network loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
