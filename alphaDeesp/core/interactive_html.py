# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of ExpertOp4Grid, an expert system approach to solve flow congestions in power grids

"""Build an interactive HTML viewer around a Graphviz-rendered SVG.

The viewer keeps the exact dot-computed positions and bezier-curved edges
(it reuses the SVG produced by Graphviz verbatim) and layers JS-driven
interactions on top: pan/zoom, hover tooltip, click-to-highlight
neighborhood, search, layer toggles by edge color/style.

The companion JSON (``dot -Tjson`` flavour) is parsed in Python so the
embedded data structure exposed to the browser is a flat node/edge model
with a pre-computed adjacency map — interactions stay O(1) at runtime.
"""

from __future__ import annotations

import html as html_mod
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pydot

logger = logging.getLogger(__name__)

# Section names rendered as <h3> headers in the sidebar layer list.
# Layers carry their section in the model so the JS just groups by it.
_SECTION_STRUCTURAL = "Structural Paths"
_SECTION_PROPERTIES = "Individual entities properties"
_SECTION_FLOWS = "Flow redispatch values"

# Edge color → human-readable layer label. Restricted to the three flow
# polarities (positive / negative / null). The historical "black" /
# "gray" / "darkred" buckets are dropped because they are redundant
# with the explicit semantic flags (is_overload / is_monitored) or
# carry no operational meaning on their own.
_LAYER_LABELS: Dict[str, str] = {
    "coral": "Positive",
    "blue": "Negative",
    "dimgray": "Null",
}

# Edge style → layer label (orthogonal to color).
_STYLE_LAYERS: Dict[str, str] = {
    "dotted": "Non-reconnectable",
    "dashed": "Reconnectable",
    "tapered": "Swapped flow",
}

# Source-of-truth attribute layers — values produced upstream by
# alphaDeesp / expert_op4grid_recommender as explicit boolean flags on
# nodes and/or edges. The viewer scans for them and exposes a layer
# toggle for each. Defining them here (rather than guessing from edge
# colours / shapes) keeps the layer list semantically stable when the
# visual palette evolves.
#
# Each entry:
#   key            — `data-attr-*` flag scanned on node and edge groups
#   label          — human-readable sidebar label
#   swatch         — special-case identifier consumed by the JS
#                    template to render an inline SVG glyph (no colour
#                    chip — these layers cut across the colour palette)
#   scope          — "node", "edge", or "both"
_SEMANTIC_LAYERS: List[Dict[str, str]] = [
    {"key": "on_constrained_path", "label": "Constrained path", "swatch": "constrained-path", "scope": "both"},
    {"key": "in_red_loop", "label": "Red-loop paths", "swatch": "red-loop", "scope": "both"},
    {"key": "is_overload", "label": "Overloads", "swatch": "overload", "scope": "edge"},
    {"key": "is_monitored", "label": "Low margin lines", "swatch": "monitored", "scope": "edge"},
    # Operator-supplied extras (ExpertAgent's `additionalLinesToCut`):
    # cut in the analysis like overloads but rendered with their
    # natural flow colour and excluded from the Overloads /
    # Low margin lines layers.  Surfaced as a dedicated layer so the
    # operator can still see how their choice materialised.
    {"key": "is_extra_cut", "label": "Extra lines to prevent flow increase", "swatch": "extra-cut", "scope": "edge"},
    {"key": "is_hub", "label": "Hubs", "swatch": "diamond", "scope": "node"},
]

# Threshold below which a node's ``value`` (prod − load, in MW) is
# treated as "no real prod/load here". Build-time conventions in the
# upstream simulators tag every node with ``prod_or_load="load"`` and
# ``value="0.0"`` even when no consumption exists, so a strict
# ``prod_or_load == "load"`` test would flood the layer with empty
# nodes. The 1 MW floor matches operator practice.
_PROD_LOAD_VALUE_FLOOR_MW = 1.0

# Per-kind config for the value-based node layers. Matched against the
# ``prod_or_load`` attribute set by ``build_nodes`` upstream
# (alphaDeesp/core/graphs/power_flow_graph.py and the simulator-specific
# build_nodes_v2 helpers). Each entry produces a single layer in the
# "Individual entities properties" section, populated only with the
# nodes whose absolute ``value`` clears ``_PROD_LOAD_VALUE_FLOOR_MW``.
_VALUE_NODE_LAYERS: List[Dict[str, str]] = [
    {"key": "prod", "label": "Production nodes", "swatch": "prod-node"},
    {"key": "load", "label": "Consumption nodes", "swatch": "load-node"},
]

# Per-layer-key section assignment. The JS renders one ``<h3>`` per
# section in the order the sections are first encountered.
_LAYER_SECTIONS: Dict[str, str] = {
    # Structural paths — multi-edge structures.
    "semantic:on_constrained_path": _SECTION_STRUCTURAL,
    "semantic:in_red_loop": _SECTION_STRUCTURAL,
    # Individual entities properties — per-edge / per-node flags.
    "semantic:is_overload": _SECTION_PROPERTIES,
    "semantic:is_monitored": _SECTION_PROPERTIES,
    "semantic:is_extra_cut": _SECTION_PROPERTIES,
    "semantic:is_hub": _SECTION_PROPERTIES,
    "style:dashed": _SECTION_PROPERTIES,
    "style:dotted": _SECTION_PROPERTIES,
    "style:tapered": _SECTION_PROPERTIES,
    # Value-based node layers — see _VALUE_NODE_LAYERS / build_nodes.
    "node:prod": _SECTION_PROPERTIES,
    "node:load": _SECTION_PROPERTIES,
    # Flow polarity buckets.
    "color:coral": _SECTION_FLOWS,
    "color:blue": _SECTION_FLOWS,
    "color:dimgray": _SECTION_FLOWS,
}

# Render order: sections appear top-to-bottom in this order; layers
# within a section appear in the order the model emits them.
_SECTION_ORDER: List[str] = [
    _SECTION_STRUCTURAL,
    _SECTION_PROPERTIES,
    _SECTION_FLOWS,
]


def _decode_title(text: str) -> str:
    """Graphviz HTML-escapes node/edge titles in SVG (``A&#45;&gt;B``)."""
    return html_mod.unescape(text or "")


def _split_edge_title(title: str) -> Tuple[str, str]:
    """Edge titles are ``"<src>->[<dst>"`` (digraph) or ``"<src>--<dst>"``."""
    title = _decode_title(title)
    for sep in ("->", "--"):
        if sep in title:
            src, dst = title.split(sep, 1)
            return src.strip(), dst.strip()
    return title, ""


def _color_to_layer_key(color: str) -> Optional[str]:
    """Map a (possibly compound or hex) color to a known layer key."""
    if not color:
        return None
    base = color.split(":", 1)[0].strip().strip('"').lower()
    if base in _LAYER_LABELS:
        return base
    return None


def _normalize_attrs(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Strip Graphviz-internal _draw_/_ldraw_ keys and quoted strings."""
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if k.startswith("_") or k in ("nodes", "edges", "objects", "subgraphs"):
            continue
        if isinstance(v, str):
            v = v.strip().strip('"')
        out[k] = v
    return out


def _model_from_dot_json(dot_json: bytes) -> Dict[str, Any]:
    """Parse ``dot -Tjson`` into a flat node/edge model + adjacency map."""
    data = json.loads(dot_json.decode("utf-8"))
    objects: List[Dict[str, Any]] = data.get("objects", [])
    raw_edges: List[Dict[str, Any]] = data.get("edges", [])

    nodes: List[Dict[str, Any]] = []
    name_by_gvid: Dict[int, str] = {}
    for i, obj in enumerate(objects):
        if "nodes" in obj or "subgraphs" in obj:
            # cluster/subgraph entry — skip in v1
            continue
        name = obj.get("name", f"node{i}")
        name_by_gvid[obj.get("_gvid", i)] = name
        nodes.append({
            "name": name,
            "attrs": _normalize_attrs(obj),
        })

    edges: List[Dict[str, Any]] = []
    adjacency: Dict[str, List[Dict[str, str]]] = {n["name"]: [] for n in nodes}
    for j, edge in enumerate(raw_edges):
        src = name_by_gvid.get(edge.get("tail"))
        dst = name_by_gvid.get(edge.get("head"))
        if src is None or dst is None:
            continue
        attrs = _normalize_attrs(edge)
        edges.append({
            "id": f"edge{j + 1}",  # matches Graphviz SVG id naming
            "source": src,
            "target": dst,
            "attrs": attrs,
        })
        adjacency.setdefault(src, []).append({"node": dst, "edge": f"edge{j + 1}"})
        adjacency.setdefault(dst, []).append({"node": src, "edge": f"edge{j + 1}"})

    layers = _build_layer_index(edges, nodes)
    return {
        "nodes": nodes,
        "edges": edges,
        "adjacency": adjacency,
        "layers": layers,
    }


def _is_truthy_flag(value: Any) -> bool:
    """Check whether a graph attribute represents a True boolean flag.

    Boolean attributes round-trip through pydot/graphviz/dot-json as
    string ``"True"``. We accept the native Python ``True`` for
    in-process callers and the string form for the JSON path.
    """
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _build_layer_index(
    edges: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Group edges & nodes by colour / style / semantic flag so the UI
    can offer toggles. Each layer carries both ``nodes`` and ``edges``
    id lists (either may be empty).
    """
    by_color: Dict[str, List[str]] = {}
    by_style: Dict[str, List[str]] = {}
    edge_id_lookup = {e["id"]: e for e in edges}
    for e in edges:
        color_key = _color_to_layer_key(e["attrs"].get("color", ""))
        if color_key:
            by_color.setdefault(color_key, []).append(e["id"])
        style = (e["attrs"].get("style") or "").lower()
        if style in _STYLE_LAYERS:
            by_style.setdefault(style, []).append(e["id"])

    # Semantic flags scanned on both nodes and edges. Only emit a layer
    # entry if at least one element carries the flag — otherwise the
    # checkbox would be useless and noise.
    semantic_buckets: Dict[str, Dict[str, List[str]]] = {
        cfg["key"]: {"nodes": [], "edges": []} for cfg in _SEMANTIC_LAYERS
    }
    if nodes:
        for n in nodes:
            for cfg in _SEMANTIC_LAYERS:
                if cfg["scope"] in ("node", "both") and _is_truthy_flag(
                    n["attrs"].get(cfg["key"])
                ):
                    semantic_buckets[cfg["key"]]["nodes"].append(n["name"])
    for e in edges:
        for cfg in _SEMANTIC_LAYERS:
            if cfg["scope"] in ("edge", "both") and _is_truthy_flag(
                e["attrs"].get(cfg["key"])
            ):
                semantic_buckets[cfg["key"]]["edges"].append(e["id"])

    # For each colour / style layer, the endpoint nodes of every
    # claimed edge are also added to the layer so toggling, e.g.,
    # "Positive overflow" alone keeps the substations the coral edges
    # connect visible (the operator can still read the topology around
    # the highlighted edges instead of seeing them float in dimmed
    # space). We dedupe while preserving first-seen order.
    edge_id_to_endpoints: Dict[str, Tuple[str, str]] = {
        e["id"]: (e["source"], e["target"]) for e in edges
    }

    def _endpoint_nodes(edge_ids: List[str]) -> List[str]:
        seen: Dict[str, None] = {}
        for eid in edge_ids:
            ends = edge_id_to_endpoints.get(eid)
            if not ends:
                continue
            for n in ends:
                if n not in seen:
                    seen[n] = None
        return list(seen.keys())

    # Edge-only semantic layers (Overloads, Low margin lines) carry
    # their edges' endpoints too — same UX rationale as colour/style
    # layers: when the operator ticks "Overloads" alone the affected
    # substations stay visible.
    _EDGE_ONLY_SEMANTIC_KEYS = {
        cfg["key"] for cfg in _SEMANTIC_LAYERS if cfg["scope"] == "edge"
    }

    def _merge_dedup(base: List[str], extra: List[str]) -> List[str]:
        seen: Dict[str, None] = {n: None for n in base}
        for n in extra:
            if n not in seen:
                seen[n] = None
        return list(seen.keys())

    raw_layers: List[Dict[str, Any]] = []
    for key, ids in by_color.items():
        raw_layers.append({
            "key": f"color:{key}",
            "label": _LAYER_LABELS[key],
            "swatch": key,
            "nodes": _endpoint_nodes(ids),
            "edges": ids,
        })
    for key, ids in by_style.items():
        raw_layers.append({
            "key": f"style:{key}",
            "label": _STYLE_LAYERS[key],
            "swatch": "",
            "nodes": _endpoint_nodes(ids),
            "edges": ids,
        })
    for cfg in _SEMANTIC_LAYERS:
        bucket = semantic_buckets[cfg["key"]]
        if not bucket["nodes"] and not bucket["edges"]:
            continue
        nodes_for_layer = bucket["nodes"]
        if cfg["key"] in _EDGE_ONLY_SEMANTIC_KEYS:
            nodes_for_layer = _merge_dedup(
                nodes_for_layer, _endpoint_nodes(bucket["edges"])
            )
        raw_layers.append({
            "key": f"semantic:{cfg['key']}",
            "label": cfg["label"],
            "swatch": cfg["swatch"],
            "nodes": nodes_for_layer,
            "edges": bucket["edges"],
        })

    # Value-based node layers (Production / Consumption). Built from
    # the ``prod_or_load`` attribute upstream tagged on every node by
    # ``build_nodes`` — see _VALUE_NODE_LAYERS. The white-coloured
    # zero-balance nodes carry ``prod_or_load="load"`` AND
    # ``value="0.0"`` upstream by convention; the 1 MW floor filters
    # them out so the operator's "Consumption nodes" toggle doesn't
    # also tag every passive substation.
    if nodes:
        value_buckets: Dict[str, List[str]] = {
            cfg["key"]: [] for cfg in _VALUE_NODE_LAYERS
        }
        for n in nodes:
            kind = n["attrs"].get("prod_or_load")
            if kind not in value_buckets:
                continue
            try:
                magnitude = abs(float(n["attrs"].get("value", "0")))
            except (TypeError, ValueError):
                continue
            if magnitude < _PROD_LOAD_VALUE_FLOOR_MW:
                continue
            value_buckets[kind].append(n["name"])
        for cfg in _VALUE_NODE_LAYERS:
            ids = value_buckets[cfg["key"]]
            if not ids:
                continue
            raw_layers.append({
                "key": f"node:{cfg['key']}",
                "label": cfg["label"],
                "swatch": cfg["swatch"],
                "nodes": ids,
                "edges": [],
            })

    # Drop layers without a section assignment (e.g. ``color:black``,
    # ``color:gray``, ``color:darkred`` — historically redundant
    # buckets). Then group by section in the canonical order so the
    # JS can render them with section headers.
    sectioned: Dict[str, List[Dict[str, Any]]] = {s: [] for s in _SECTION_ORDER}
    for layer in raw_layers:
        section = _LAYER_SECTIONS.get(layer["key"])
        if section is None:
            continue
        layer["section"] = section
        sectioned.setdefault(section, []).append(layer)

    layers: List[Dict[str, Any]] = []
    for section in _SECTION_ORDER:
        layers.extend(sectioned.get(section, []))
    # Silence unused-var warning; lookup retained for future hover xref.
    del edge_id_lookup
    return layers


def _inject_svg_data_attrs(svg_bytes: bytes, model: Dict[str, Any]) -> str:
    """Annotate Graphviz SVG nodes/edges with stable data-* attributes.

    Graphviz emits ``<g id="nodeN" class="node"><title>NAME</title>``; we
    rely on the title to look up our model entries and append data-*
    attributes (which the JS uses for selectors and tooltips).
    """
    svg = svg_bytes.decode("utf-8")
    edge_by_id = {e["id"]: e for e in model["edges"]}
    node_by_name = {n["name"]: n for n in model["nodes"]}

    def _attrs_to_data(prefix: str, attrs: Dict[str, Any]) -> str:
        out: List[str] = []
        for k, v in attrs.items():
            safe_v = html_mod.escape(str(v), quote=True)
            out.append(f' data-{prefix}-{k}="{safe_v}"')
        return "".join(out)

    def _node_repl(match: re.Match) -> str:
        gid = match.group(1)
        title = match.group(2)
        name = _decode_title(title)
        node = node_by_name.get(name)
        if not node:
            return match.group(0)
        data = _attrs_to_data("attr", node["attrs"])
        return (
            f'<g id="{gid}" class="node" data-name="{html_mod.escape(name, quote=True)}"'
            f'{data}><title>{title}</title>'
        )

    def _edge_repl(match: re.Match) -> str:
        gid = match.group(1)
        title = match.group(2)
        edge = edge_by_id.get(gid)
        if not edge:
            return match.group(0)
        src, dst = edge["source"], edge["target"]
        layers: List[str] = []
        color_key = _color_to_layer_key(edge["attrs"].get("color", ""))
        if color_key:
            layers.append(f"color:{color_key}")
        style = (edge["attrs"].get("style") or "").lower()
        if style in _STYLE_LAYERS:
            layers.append(f"style:{style}")
        data = _attrs_to_data("attr", edge["attrs"])
        layer_attr = f' data-layers="{html_mod.escape(" ".join(layers), quote=True)}"' if layers else ""
        return (
            f'<g id="{gid}" class="edge"'
            f' data-source="{html_mod.escape(src, quote=True)}"'
            f' data-target="{html_mod.escape(dst, quote=True)}"'
            f'{layer_attr}{data}><title>{title}</title>'
        )

    svg = re.sub(
        r'<g id="(node\d+)" class="node">\s*<title>([^<]*)</title>',
        _node_repl,
        svg,
    )
    svg = re.sub(
        r'<g id="(edge\d+)" class="edge">\s*<title>([^<]*)</title>',
        _edge_repl,
        svg,
    )
    return svg


# Self-contained HTML template — keeps the SVG inline so the file can be
# opened directly from disk (no web server, no CDN). The JS is a tiny
# ~150-line bundle: pan/zoom, hover tooltip, click-highlight, search,
# layer toggles. All state is driven by CSS classes for cheap toggling.
_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #fafafa; --panel: #ffffff; --border: #d0d7de;
    --muted: #6b7280; --accent: #0969da; --dim-opacity: 0.12;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; background: var(--bg); color: #111; }
  #app { display: flex; height: 100vh; }
  #sidebar {
    width: 280px; flex-shrink: 0; padding: 12px 14px; overflow-y: auto;
    background: var(--panel); border-right: 1px solid var(--border); font-size: 13px;
  }
  #sidebar h1 { font-size: 14px; margin: 0 0 6px; }
  #sidebar h2 { font-size: 12px; text-transform: uppercase; color: var(--muted); margin: 14px 0 6px; letter-spacing: 0.04em; }
  #sidebar .layer-section-header { font-size: 11px; text-transform: uppercase; color: var(--muted); margin: 10px 0 4px; letter-spacing: 0.03em; font-weight: 600; border-top: 1px solid var(--border); padding-top: 6px; }
  #sidebar .layer-section-header:first-of-type { border-top: none; padding-top: 0; margin-top: 4px; }
  #sidebar input[type=text] { width: 100%; padding: 6px 8px; border: 1px solid var(--border); border-radius: 4px; font-size: 12px; }
  #sidebar label { display: flex; align-items: center; gap: 6px; padding: 3px 0; cursor: pointer; }
  #sidebar label .swatch { width: 12px; height: 12px; border-radius: 2px; border: 1px solid #ccc; flex-shrink: 0; display: inline-flex; align-items: center; justify-content: center; }
  #sidebar label .swatch svg { width: 10px; height: 10px; display: block; }
  #sidebar .layer-controls { display: flex; gap: 8px; font-size: 11px; margin-bottom: 4px; }
  #sidebar .layer-controls button { background: transparent; border: none; padding: 2px 4px; cursor: pointer; color: var(--accent); font-size: 11px; text-decoration: underline; }
  #sidebar .layer-controls button:hover { background: #eef; }
  #sidebar .hint { color: var(--muted); font-size: 11px; line-height: 1.5; margin-top: 6px; }
  #info { font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-size: 11px; white-space: pre-wrap; word-break: break-word; background: #f6f8fa; border-radius: 4px; padding: 8px; min-height: 60px; }
  #stage { flex: 1; position: relative; overflow: hidden; background: var(--bg); }
  #stage svg { display: block; width: 100%; height: 100%; cursor: grab; }
  #stage svg.dragging { cursor: grabbing; }
  #stage .toolbar { position: absolute; top: 10px; right: 10px; display: flex; gap: 4px; background: var(--panel); border: 1px solid var(--border); border-radius: 4px; padding: 2px; }
  #stage .toolbar button { border: none; background: transparent; padding: 4px 8px; cursor: pointer; font-size: 14px; font-weight: bold; color: #333; }
  #stage .toolbar button:hover { background: #eef; }
  #tooltip { position: absolute; pointer-events: none; background: rgba(20,20,20,0.92); color: #fff; padding: 6px 8px; border-radius: 4px; font-size: 11px; line-height: 1.4; max-width: 260px; z-index: 10; display: none; }
  #tooltip .key { color: #9ad; }
  /* Interaction states applied via classes on the root <g class="graph"> */
  .graph.has-selection .node:not(.hl):not(.selected),
  .graph.has-selection .edge:not(.hl) { opacity: var(--dim-opacity); }
  .graph .node.selected ellipse,
  .graph .node.selected polygon,
  .graph .node.selected circle { stroke-width: 5px; }
  /* Unchecked layers stay rendered but recede to a near-transparent
     light-grey so the user retains spatial context. Applied to both
     nodes and edges. Pointer-events kept enabled so hover/click still
     works on dimmed elements. */
  .graph .layer-off { opacity: 0.12; }
  .graph.has-search .node:not(.match) { opacity: var(--dim-opacity); }
  .graph .node.match ellipse,
  .graph .node.match polygon,
  .graph .node.match circle { stroke: #0969da !important; stroke-width: 4px; }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <h1>__TITLE__</h1>
    <div class="hint">Click a node to highlight its neighbourhood. Wheel to zoom, drag to pan. Esc clears selection.</div>

    <h2>Search</h2>
    <input type="text" id="search" placeholder="filter nodes by name or id…" autocomplete="off">

    <h2>Layers</h2>
    <div class="layer-controls">
      <button id="layers-select-all" type="button" title="Show every layer">Select all</button>
      <span style="color: var(--muted)">·</span>
      <button id="layers-select-none" type="button" title="Dim every layer">Unselect all</button>
    </div>
    <div id="layers"></div>

    <h2>Selection</h2>
    <div id="info">Nothing selected.</div>

    <h2>Stats</h2>
    <div class="hint" id="stats"></div>
  </aside>
  <section id="stage">
    <div class="toolbar">
      <button id="zoom-in" title="Zoom in">+</button>
      <button id="zoom-out" title="Zoom out">−</button>
      <button id="zoom-reset" title="Reset view">⟲</button>
    </div>
    __SVG__
    <div id="tooltip"></div>
  </section>
</div>
<script>
const MODEL = __MODEL_JSON__;
(function(){
  const svg = document.querySelector('#stage svg');
  const root = svg.querySelector('g.graph') || svg.querySelector('g');
  if (root && !root.classList.contains('graph')) root.classList.add('graph');

  // ---- Pan & zoom: native viewBox manipulation (~30 lines, no deps) ----
  let vb = (svg.getAttribute('viewBox') || '').split(/\\s+/).map(Number);
  if (vb.length !== 4) { const r = svg.getBBox(); vb = [r.x, r.y, r.width, r.height]; }
  const initial = vb.slice();
  function setVB() { svg.setAttribute('viewBox', vb.join(' ')); }
  function clientToSvg(x, y) {
    const rect = svg.getBoundingClientRect();
    return { x: vb[0] + (x - rect.left) / rect.width * vb[2], y: vb[1] + (y - rect.top) / rect.height * vb[3] };
  }
  svg.addEventListener('wheel', (e) => {
    e.preventDefault();
    const k = e.deltaY < 0 ? 0.85 : 1.18;
    const p = clientToSvg(e.clientX, e.clientY);
    vb[0] = p.x - (p.x - vb[0]) * k;
    vb[1] = p.y - (p.y - vb[1]) * k;
    vb[2] *= k; vb[3] *= k;
    setVB();
  }, { passive: false });
  let dragStart = null;
  svg.addEventListener('mousedown', (e) => { if (e.button !== 0) return; dragStart = { x: e.clientX, y: e.clientY, vb: vb.slice() }; svg.classList.add('dragging'); });
  window.addEventListener('mouseup', () => { dragStart = null; svg.classList.remove('dragging'); });
  window.addEventListener('mousemove', (e) => {
    if (!dragStart) return;
    const rect = svg.getBoundingClientRect();
    vb[0] = dragStart.vb[0] - (e.clientX - dragStart.x) / rect.width * vb[2];
    vb[1] = dragStart.vb[1] - (e.clientY - dragStart.y) / rect.height * vb[3];
    setVB();
  });
  document.getElementById('zoom-in').onclick = () => zoomBy(0.8);
  document.getElementById('zoom-out').onclick = () => zoomBy(1.25);
  document.getElementById('zoom-reset').onclick = () => { vb = initial.slice(); setVB(); };
  function zoomBy(k) {
    const cx = vb[0] + vb[2]/2, cy = vb[1] + vb[3]/2;
    vb[0] = cx - (cx - vb[0]) * k; vb[1] = cy - (cy - vb[1]) * k;
    vb[2] *= k; vb[3] *= k; setVB();
  }

  // ---- Adjacency lookup, hover, click-highlight ----
  const tooltip = document.getElementById('tooltip');
  const info = document.getElementById('info');
  function fmtAttrs(prefix, el, skip) {
    const out = [];
    const skipSet = skip ? new Set(skip) : null;
    for (const a of el.attributes) {
      if (a.name.startsWith('data-' + prefix + '-')) {
        const k = a.name.slice(('data-' + prefix + '-').length);
        if (skipSet && skipSet.has(k)) continue;
        out.push('<span class="key">' + k + '</span>: ' + a.value);
      }
    }
    return out.join('<br>');
  }
  // Display name for a node: the readable ``label`` (e.g. a voltage-level
  // name such as "Saucats 400kV") when present, else the stable node id
  // (``data-name``). The id is preserved as the node identity for
  // selection / adjacency / double-click resolution; ``label`` only
  // changes what the operator reads.
  function nodeDisplayName(el) {
    if (!el) return '';
    const label = el.getAttribute('data-attr-label');
    // Graphviz stores an *unset* label as the placeholder "\\N" (meaning
    // "node name"); any backslash escape (\\N, \\G, …) is not a readable
    // name, so fall back to the stable id (data-name) in that case.
    if (label && label.indexOf('\\\\') === -1) return label;
    return el.getAttribute('data-name') || '';
  }
  // Tooltip / selection header for a node: readable name in bold, with the
  // raw id shown underneath only when it differs from the readable name.
  function nodeHeaderHtml(el) {
    const disp = nodeDisplayName(el);
    const id = el.getAttribute('data-name') || '';
    let head = '<b>' + escapeHtml(disp) + '</b>';
    if (id && id !== disp) head += '<br><span class="key">id</span>: ' + escapeHtml(id);
    return head;
  }
  function showTooltip(e, html) {
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    const r = svg.getBoundingClientRect();
    tooltip.style.left = (e.clientX - r.left + 12) + 'px';
    tooltip.style.top = (e.clientY - r.top + 12) + 'px';
  }
  function hideTooltip() { tooltip.style.display = 'none'; }

  function clearSelection() {
    root.classList.remove('has-selection');
    root.querySelectorAll('.hl, .selected').forEach(el => el.classList.remove('hl', 'selected'));
    info.textContent = 'Nothing selected.';
  }
  function selectNode(name) {
    clearSelection();
    const node = root.querySelector('.node[data-name="' + cssEscape(name) + '"]');
    if (!node) return;
    root.classList.add('has-selection');
    node.classList.add('selected', 'hl');
    const neighbours = MODEL.adjacency[name] || [];
    for (const n of neighbours) {
      const nb = root.querySelector('.node[data-name="' + cssEscape(n.node) + '"]');
      if (nb) nb.classList.add('hl');
      const ed = document.getElementById(n.edge);
      if (ed) ed.classList.add('hl');
    }
    info.innerHTML = nodeHeaderHtml(node) + '<br>' + fmtAttrs('attr', node, ['label'])
      + '<br><br>degree: ' + neighbours.length;
  }
  function cssEscape(s) { return (window.CSS && CSS.escape) ? CSS.escape(s) : s.replace(/(["\\\\])/g, '\\\\$1'); }
  function escapeHtml(s) { return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

  root.addEventListener('mouseover', (e) => {
    const g = e.target.closest('.node, .edge'); if (!g) return;
    if (g.classList.contains('node')) {
      showTooltip(e, nodeHeaderHtml(g) + '<br>' + fmtAttrs('attr', g, ['label']));
    } else {
      const lbl = g.querySelector('text'); const name = g.getAttribute('data-attr-name') || '';
      showTooltip(e, '<b>' + escapeHtml(g.getAttribute('data-source')) + ' → ' + escapeHtml(g.getAttribute('data-target')) + '</b>'
        + (name ? '<br>' + escapeHtml(name) : '')
        + '<br>' + fmtAttrs('attr', g));
    }
  });
  root.addEventListener('mousemove', (e) => {
    if (tooltip.style.display === 'block') {
      const r = svg.getBoundingClientRect();
      tooltip.style.left = (e.clientX - r.left + 12) + 'px';
      tooltip.style.top = (e.clientY - r.top + 12) + 'px';
    }
  });
  root.addEventListener('mouseout', hideTooltip);
  root.addEventListener('click', (e) => {
    const g = e.target.closest('.node'); if (!g) return;
    e.stopPropagation();
    selectNode(g.getAttribute('data-name'));
  });
  svg.addEventListener('click', (e) => { if (e.target === svg || e.target === root) clearSelection(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { clearSelection(); document.getElementById('search').value = ''; applySearch(); }});

  // ---- Search ----
  function applySearch() {
    const q = document.getElementById('search').value.trim().toLowerCase();
    root.querySelectorAll('.node.match').forEach(n => n.classList.remove('match'));
    if (!q) { root.classList.remove('has-search'); return; }
    root.classList.add('has-search');
    let count = 0;
    root.querySelectorAll('.node').forEach(n => {
      // Match against both the stable id (data-name) and the resolved
      // readable display name (e.g. a voltage-level name), so operators can
      // find a node by either spelling. nodeDisplayName ignores the
      // graphviz "\\N" placeholder, so label-less nodes match on their id.
      const id = (n.getAttribute('data-name') || '').toLowerCase();
      const disp = nodeDisplayName(n).toLowerCase();
      if (id.indexOf(q) !== -1 || (disp && disp.indexOf(q) !== -1)) {
        n.classList.add('match'); count++;
      }
    });
  }
  document.getElementById('search').addEventListener('input', applySearch);

  // ---- Layer toggles ----
  // Membership-based dim model: every node and edge knows which
  // layers claim it. An element is **visible** iff at least one of
  // its claiming layers is currently checked. Elements with no
  // memberships at all are dimmed whenever any layer toggle differs
  // from the default "all checked" state — that matches the
  // user-facing intent that ticking a single layer focuses the view
  // on that layer only and recedes everything else.
  const layersEl = document.getElementById('layers');
  function swatchInner(swatch) {
    if (!swatch) return '<span style="display:block;width:100%;height:100%;border:1px dashed #999"></span>';
    const COLORED = {coral:1, blue:1, black:1, gray:1, dimgray:1, darkred:1, red:1, green:1};
    if (COLORED[swatch]) return '';
    if (swatch === 'diamond') return '<svg viewBox="0 0 10 10"><polygon points="5,0 10,5 5,10 0,5" fill="#444"/></svg>';
    if (swatch === 'red-loop') return '<svg viewBox="0 0 10 10"><circle cx="5" cy="5" r="2" fill="coral"/><circle cx="5" cy="5" r="4" fill="none" stroke="coral" stroke-width="1"/></svg>';
    if (swatch === 'constrained-path') return '<svg viewBox="0 0 14 6"><line x1="0" y1="3" x2="14" y2="3" stroke="black" stroke-width="2"/></svg>';
    if (swatch === 'overload') return '<svg viewBox="0 0 14 6"><line x1="0" y1="3" x2="14" y2="3" stroke="black" stroke-width="2.5"/><line x1="0" y1="3" x2="14" y2="3" stroke="yellow" stroke-width="0.8"/></svg>';
    if (swatch === 'monitored') return '<svg viewBox="0 0 14 6"><line x1="0" y1="3" x2="14" y2="3" stroke="coral" stroke-width="2.5"/><line x1="0" y1="3" x2="14" y2="3" stroke="yellow" stroke-width="0.8"/></svg>';
    if (swatch === 'extra-cut') return '<svg viewBox="0 0 14 6"><line x1="0" y1="3" x2="14" y2="3" stroke="blue" stroke-width="2" stroke-dasharray="3 2"/></svg>';
    // Match the upstream node fillcolors set in build_nodes:
    //   prod (prod_minus_load > 0)  → coral
    //   load (prod_minus_load < 0)  → lightblue
    if (swatch === 'prod-node') return '<svg viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="coral" stroke="#444" stroke-width="0.6"/></svg>';
    if (swatch === 'load-node') return '<svg viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="lightblue" stroke="#444" stroke-width="0.6"/></svg>';
    return '';
  }
  function swatchStyle(swatch) {
    const COLORED = {coral:1, blue:1, black:1, gray:1, dimgray:1, darkred:1, red:1, green:1};
    if (COLORED[swatch]) return 'background:' + swatch;
    return 'background:#fff';
  }

  // Build per-element layer membership maps once. These are consulted
  // by `applyAllLayers()` on every checkbox change so an element
  // claimed by multiple layers never gets stuck in `layer-off`
  // because an unrelated checkbox flipped the wrong way.
  const nodeMemberships = new Map();  // data-name -> Array<layerIndex>
  const edgeMemberships = new Map();  // edge id  -> Array<layerIndex>
  for (let i = 0; i < MODEL.layers.length; i++) {
    const layer = MODEL.layers[i];
    for (const n of (layer.nodes || [])) {
      const arr = nodeMemberships.get(n);
      if (arr) arr.push(i); else nodeMemberships.set(n, [i]);
    }
    for (const e of (layer.edges || [])) {
      const arr = edgeMemberships.get(e);
      if (arr) arr.push(i); else edgeMemberships.set(e, [i]);
    }
  }

  const layerCheckboxes = [];
  let lastSection = null;
  for (const layer of MODEL.layers) {
    if (layer.section && layer.section !== lastSection) {
      const header = document.createElement('h3');
      header.className = 'layer-section-header';
      header.textContent = layer.section;
      layersEl.appendChild(header);
      lastSection = layer.section;
    }
    const id = 'layer-' + layer.key.replace(/[^a-z0-9]/gi, '-');
    const wrap = document.createElement('label');
    const total = (layer.nodes ? layer.nodes.length : 0) + (layer.edges ? layer.edges.length : 0);
    wrap.innerHTML = '<input type="checkbox" id="' + id + '" checked>'
      + '<span class="swatch" style="' + swatchStyle(layer.swatch) + '">' + swatchInner(layer.swatch) + '</span>'
      + '<span>' + escapeHtml(layer.label) + ' <span style="color:var(--muted)">(' + total + ')</span></span>';
    layersEl.appendChild(wrap);
    const cb = wrap.querySelector('input');
    layerCheckboxes.push(cb);
    cb.addEventListener('change', (e) => {
      applyAllLayers();
      window.parent.postMessage({
        type: 'cs4g:overflow-layer-toggled',
        key: layer.key,
        label: layer.label,
        visible: e.target.checked
      }, '*');
    });
  }

  function applyAllLayers() {
    const checkedSet = new Set();
    let allChecked = true;
    for (let i = 0; i < layerCheckboxes.length; i++) {
      if (layerCheckboxes[i].checked) checkedSet.add(i);
      else allChecked = false;
    }
    function shouldDim(memberships) {
      if (allChecked) return false;
      if (!memberships || memberships.length === 0) return true;
      for (const idx of memberships) {
        if (checkedSet.has(idx)) return false;
      }
      return true;
    }
    root.querySelectorAll('.node').forEach((el) => {
      const name = el.getAttribute('data-name');
      el.classList.toggle('layer-off', shouldDim(nodeMemberships.get(name)));
    });
    root.querySelectorAll('.edge').forEach((el) => {
      const id = el.getAttribute('id');
      el.classList.toggle('layer-off', shouldDim(edgeMemberships.get(id)));
    });
  }
  // Expose for testability and external triggers (e.g. parent app
  // requesting a full repaint after dynamic content updates).
  window.__cs4gApplyAllLayers = applyAllLayers;

  function setAllLayers(visible) {
    let changed = false;
    for (let i = 0; i < layerCheckboxes.length; i++) {
      if (layerCheckboxes[i].checked !== visible) {
        layerCheckboxes[i].checked = visible;
        changed = true;
      }
    }
    if (changed) applyAllLayers();
    window.parent.postMessage({
      type: 'cs4g:overflow-select-all-layers',
      visible: visible
    }, '*');
  }
  document.getElementById('layers-select-all').addEventListener('click', () => setAllLayers(true));
  document.getElementById('layers-select-none').addEventListener('click', () => setAllLayers(false));

  // Double-click on a graph node bubbles up to the parent window,
  // which is responsible for opening the corresponding Single-Line
  // Diagram view. The node `data-name` is the voltage-level (or
  // substation, depending on backend) identifier the parent will
  // resolve to a SLD endpoint.
  root.addEventListener('dblclick', (ev) => {
    const g = ev.target.closest('.node');
    if (!g) return;
    ev.preventDefault();
    ev.stopPropagation();
    const name = g.getAttribute('data-name') || '';
    if (!name) return;
    window.parent.postMessage({
      type: 'cs4g:overflow-node-double-clicked',
      name: name
    }, '*');
  });

  document.getElementById('stats').textContent =
    MODEL.nodes.length + ' nodes, ' + MODEL.edges.length + ' edges';
})();
</script>
</body>
</html>
"""


def _align_edge_ids_with_svg(svg_bytes: bytes, model: Dict[str, Any]) -> Dict[str, Any]:
    """Re-key edges in ``model`` so their ``id`` field matches the SVG's
    ``<g id="edgeN" class="edge">`` for the SAME (src, dst) endpoints.

    Background
    ----------
    Graphviz emits edge IDs ``edgeN`` in **two independent orderings** for
    the SVG and the JSON outputs of the same graph. ``_model_from_dot_json``
    assigns IDs by JSON-edge index but the SVG element with the same
    ``edgeN`` id often refers to a different edge (different (src, dst)
    pair). The downstream JS dim layer queries SVG elements **by id** —
    so a mismatch makes the wrong edges dim/highlight when a layer
    toggle is flipped (this is exactly the user-reported confusion
    SSV.OP7→GROSNP7 ↔ SSV.OP7→CREYSP7 / CHALOP6→CPVANP6 ↔
    CHALOP6→CHALOP3 in the small-grid scenario).

    Fix
    ---
    Walk the SVG, parse each edge's ``<title>`` to extract its true
    (src, dst), and greedily pair it with a JSON-side edge of matching
    endpoints. Each JSON edge is consumed at most once (parallel edges
    are paired in their relative order, which is stable across SVG and
    JSON). The model's edge IDs are updated in place; adjacency and
    layer membership lists are remapped through the same dict.

    Returns the updated model.
    """
    svg = svg_bytes.decode("utf-8")
    # Walk SVG edges in document order: ``<g id="edgeN" class="edge">
    # <title>SRC->DST</title>``. Graphviz HTML-escapes the title.
    pattern = re.compile(
        r'<g id="(edge\d+)" class="edge">\s*<title>([^<]*)</title>'
    )
    svg_edges_in_order: List[Tuple[str, str, str]] = []
    for m in pattern.finditer(svg):
        gid = m.group(1)
        src, dst = _split_edge_title(m.group(2))
        svg_edges_in_order.append((gid, src, dst))

    # Build a per-(src, dst) FIFO of JSON edges keeping their original order.
    json_edges = model["edges"]
    by_pair: Dict[Tuple[str, str], List[int]] = {}
    for i, e in enumerate(json_edges):
        by_pair.setdefault((e["source"], e["target"]), []).append(i)

    # Greedily match each SVG edge to the next un-consumed JSON edge of the
    # same endpoints. The remap dict translates "old (JSON-order) id" → "new
    # (SVG-order) id".
    remap: Dict[str, str] = {}
    for svg_id, s, t in svg_edges_in_order:
        candidates = by_pair.get((s, t)) or by_pair.get((t, s))
        if not candidates:
            continue
        json_idx = candidates.pop(0)
        old_id = json_edges[json_idx]["id"]
        if old_id == svg_id:
            continue
        remap[old_id] = svg_id

    if not remap:
        return model

    # Apply the remap. ``remap`` may contain swaps (a→b and b→a). To avoid
    # collisions we materialise the new IDs through a fresh dict in two
    # passes: first relabel each JSON edge to its SVG-aligned id, then walk
    # adjacency / layers to substitute the references.
    edge_id_lookup = {e["id"]: e for e in json_edges}
    for old_id, new_id in remap.items():
        edge = edge_id_lookup[old_id]
        edge["id"] = new_id

    # Adjacency entries reference edge ids by string — apply the same
    # substitution there.
    for entries in model.get("adjacency", {}).values():
        for entry in entries:
            if entry.get("edge") in remap:
                entry["edge"] = remap[entry["edge"]]

    # Layer membership lists use the same string ids.
    for layer in model.get("layers", []):
        layer["edges"] = [remap.get(eid, eid) for eid in layer.get("edges", [])]

    return model


def build_interactive_html(
    pydot_graph: pydot.Graph,
    prog: Any = "dot",
    title: str = "ExpertOp4Grid — interactive overflow graph",
) -> str:
    """Render ``pydot_graph`` to interactive HTML.

    Returns the HTML string. Caller decides where to write it.
    """
    svg_bytes = pydot_graph.create(prog=prog, format="svg")
    json_bytes = pydot_graph.create(prog=prog, format="json")
    model = _model_from_dot_json(json_bytes)
    # Align JSON edge ids with the SVG element ids — graphviz emits the
    # two orderings independently and the downstream JS toggles SVG
    # elements by id, so a mismatch silently dims the wrong edges.
    model = _align_edge_ids_with_svg(svg_bytes, model)
    annotated_svg = _inject_svg_data_attrs(svg_bytes, model)
    return (
        _HTML_TEMPLATE
        .replace("__TITLE__", html_mod.escape(title))
        .replace("__SVG__", annotated_svg)
        .replace("__MODEL_JSON__", json.dumps(model))
    )
