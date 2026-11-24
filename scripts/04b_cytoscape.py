#!/usr/bin/env python
"""Step 4b. Send the gene network to Cytoscape: style, layout, analyse, export.

    python scripts/04b_cytoscape.py              # uses Cytoscape if open, else starts it
    python scripts/04b_cytoscape.py --no-launch  # fail instead of starting Cytoscape

Reads results/network/network.graphml (step 4). Drives the Cytoscape desktop
program through its automation port (CyREST, localhost:1234), so what you see
in the Cytoscape window is exactly what the pipeline produced:

    - the network, one cluster per module, each titled with its top enriched term
    - style "LUAD lists":  node colour = which gene list, size = |Cox z|
    - style "LUAD Cox z":  node colour = Cox z, blue (protective) to red (risk)
    - Cytoscape's own network analysis (degree, betweenness, ...) on every node

Writes results/figures/network_cytoscape_lists.png and _coxz.png,
results/tables/biology_network_centrality.csv and the session file
results/network/network.cys, which reopens everything in Cytoscape.
Cytoscape is left open afterwards so the network can be explored.
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
GROUP_LABELS = {"prognostic_risk": "Prognostic, higher risk",
                "prognostic_protective": "Prognostic, protective",
                "model": "Kept by the Cox model", "factor": "MOFA factor gene",
                "proliferation": "Proliferation metagene", "several": "In more than one list"}
COXZ_LEGEND = [("#2A78D6", "Cox z -5, protective"), ("#F2F2F2", "Cox z 0"),
               ("#C2541C", "Cox z +5, higher risk")]


def ready() -> bool:
    try:
        p4c.cytoscape_version_info()
        return True
    except Exception:
        return False


def ensure_cytoscape(launcher: str | None, launch: bool, wait_s: int = 300) -> None:
    if ready():
        return
    if not launch or not launcher or not Path(launcher).expanduser().exists():
        raise SystemExit("Cytoscape is not running. Open it and run this again"
                         + (f" (looked for {launcher})" if launcher else "") + ".")
    print(f"  starting Cytoscape ({launcher}); this takes about a minute")
    # Started in its own session so it stays open after this script finishes.
    subprocess.Popen([str(Path(launcher).expanduser())], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if ready():
            return
        time.sleep(5)
    raise SystemExit(f"Cytoscape did not answer on port 1234 within {wait_s} s")


def load_graph(path: Path) -> nx.Graph:
    G = nx.read_graphml(path)
    for g, d in G.nodes(data=True):
        d["abs_cox_z"] = abs(float(d.get("cox_z", 0.0)))
        d["module"] = int(d.get("module", 0))
    return G


def make_styles() -> None:
    defaults = {"NODE_SHAPE": "ELLIPSE", "NODE_BORDER_WIDTH": 1.5, "NODE_BORDER_PAINT": "#FFFFFF",
                "EDGE_WIDTH": 1, "EDGE_STROKE_UNSELECTED_PAINT": "#8A8A8A",
                "NODE_LABEL_FONT_SIZE": 17, "NODE_LABEL_COLOR": "#222222",
                "NODE_LABEL_POSITION": "S,N,c,0.00,3.00",      # label under the node
                "NETWORK_BACKGROUND_PAINT": "#FFFFFF"}
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
    for a in p4c.get_annotation_list() or []:
        if a.get("name", "").startswith(prefix):
            p4c.delete_annotation(a["uuid"])


def text(s: str, cx: float, y: float, size: int, name: str, bold: bool = False) -> None:
    """Text centred on cx. Cytoscape anchors text at its left edge, so the
    width is estimated from the character count."""
    for k, line in enumerate(s.split("\n")):
        p4c.add_annotation_text(text=line, x_pos=cx - 0.27 * size * len(line), y_pos=y + k * size * 1.25,
                                font_size=size, font_style="bold" if bold else "plain",
                                color="#111111", name=f"{name}_{k}", canvas="foreground")


def legend(items, x0: float, y: float, name: str) -> float:
    """A row of coloured marks with labels; returns the x where the row ends."""
    x = x0
    for k, (colour, lab, *shape) in enumerate(items):
        p4c.add_annotation_shape(type=shape[0] if shape else "ELLIPSE", x_pos=x, y_pos=y, fill_color=colour, opacity=100,
                                 border_thickness=1, border_color="#999999", height=40, width=40,
                                 name=f"{name}_dot{k}", canvas="foreground")
        p4c.add_annotation_text(text=lab, x_pos=x + 55, y_pos=y + 4, font_size=26, color="#111111",
                                name=f"{name}_txt{k}", canvas="foreground")
        x += 55 + 15.5 * len(lab) + 70
    return x


def trim(path: Path, margin: int = 40) -> None:
    """Cut the empty border Cytoscape leaves around the network."""
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if box:
        l, t, r, b = box
        im.crop((max(l - margin, 0), max(t - margin, 0),
                 min(r + margin, im.width), min(b + margin, im.height))).save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-launch", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    tables, figures = out_dirs(cfg)
    net_dir = Path(cfg["paths"]["network"])
    graphml = net_dir / "network.graphml"
    if not graphml.exists():
        raise SystemExit(f"{graphml} not found: run scripts/04_biology.py first")

    ensure_cytoscape(cfg.get("cytoscape", {}).get("launcher"), not args.no_launch)
    v = p4c.cytoscape_version_info()
    print(f"  Cytoscape {v['cytoscapeVersion']} (automation API {v['apiVersion']})")

    G = load_graph(graphml)
    modules = pd.read_csv(tables / "biology_modules.csv")
    # The same module-by-module layout as the matplotlib figure of step 4.
    pos, boxes = B.module_layout(G)
    titles = B.module_titles(G, modules, boxes)
    H = G.subgraph(pos).copy()
    for u, v, d in H.edges(data=True):
        d["within_module"] = "true" if H.nodes[u]["module"] == H.nodes[v]["module"] else "false"

    for old in p4c.get_network_list():
        if old == "LUAD gene network":
            p4c.delete_network(old)
    p4c.create_network_from_networkx(H, title="LUAD gene network", collection="LUAD multi-omics")
    make_styles()
    p4c.set_visual_style("LUAD lists")
    names = list(pos)
    p4c.set_node_position_bypass(names, [pos[g][0] for g in names], [pos[g][1] for g in names])
    time.sleep(2)

    # Cytoscape's own topology statistics, written back as a table. They are
    # computed on the drawn network, which leaves out fragments under 4 genes.
    p4c.analyze_network(directed=False)
    node_table = p4c.get_table_columns("node")
    keep = [c for c in ["name", "group", "module", "cox_z", "Degree", "BetweennessCentrality",
                        "ClosenessCentrality", "ClusteringCoefficient", "NeighborhoodConnectivity"]
            if c in node_table.columns]
    (node_table[keep].sort_values(["Degree", "name"], ascending=[False, True])
     .to_csv(tables / "biology_network_centrality.csv", index=False))

    # Titles, notes and legend are annotations on the Cytoscape canvas.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    left, mid, bottom = min(xs), (min(xs) + max(xs)) / 2, max(ys)
    links = B.cross_module_links(G, set(pos))
    note = ("Lines are protein interactions inside a module. Interactions between modules are hidden: "
            + "; ".join(f"modules {a} and {b}: {n}" for (a, b), n in
                        sorted(links.items(), key=lambda kv: -kv[1]) if n >= 5) + ".")
    clear_annotations()
    for m, (cx, top, r) in boxes.items():
        text(titles[m], cx, top + 10, 30, f"title{m}", bold=True)
    text(f"{H.number_of_nodes()} genes in modules of 4 or more "
         f"({G.number_of_nodes() - H.number_of_nodes()} in smaller fragments not drawn). "
         "Node size: |Cox z|.", mid, bottom + 150, 26, "note_a")
    text(note, mid, bottom + 195, 26, "note_b")

    for style, stem, items in (
            ("LUAD lists", "network_cytoscape_lists", [(GROUP_COLOURS[k], GROUP_LABELS[k], GROUP_SHAPES[k]) for k in GROUP_COLOURS]),
            ("LUAD Cox z", "network_cytoscape_coxz", COXZ_LEGEND)):
        clear_annotations("legend")
        width = sum(55 + 15.5 * len(it[1]) + 70 for it in items)
        legend(items, mid - width / 2, bottom + 260, "legend")
        p4c.set_visual_style(style)
        time.sleep(2)
        p4c.fit_content()
        time.sleep(2)
        out = figures / f"{stem}.png"
        out.unlink(missing_ok=True)
        # A fixed width in pixels: with a zoom factor the image size follows the
        # size of the Cytoscape window, so two machines give different figures.
        p4c.export_image(str(out), type="PNG", units="pixels", width=3000, overwrite_file=True)
        trim(out)
        # A vector copy for a manuscript: gene labels stay sharp at any size.
        vec = figures / f"{stem}.pdf"
        vec.unlink(missing_ok=True)
        p4c.export_image(str(vec), type="PDF", overwrite_file=True)
        print(f"  wrote {out} and {vec}")
    # Leave Cytoscape on the "lists" view with its legend.
    clear_annotations("legend")
    items = [(GROUP_COLOURS[k], GROUP_LABELS[k], GROUP_SHAPES[k]) for k in GROUP_COLOURS]
    legend(items, mid - sum(55 + 15.5 * len(it[1]) + 70 for it in items) / 2, bottom + 260, "legend")
    p4c.set_visual_style("LUAD lists")

    session = net_dir / "network.cys"
    p4c.save_session(str(session), overwrite_file=True)
    print(f"  wrote {session}")
    print(f"  {H.number_of_nodes()} genes drawn, {G.number_of_edges()} interactions in the full network; "
          "Cytoscape is open with the network loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
