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

# Edge color → human-readable layer label. The set is small and maps the
# semantic palette used across the codebase (overflow_graph.py,
# null_flow_graph.py): black=constrained, coral=positive overflow,
# blue=negative flow, gray=low/inactive, dimgray=null-flow recoloured.
_LAYER_LABELS: Dict[str, str] = {
    "black": "Constrained line",
    "coral": "Positive overflow",
    "blue": "Negative flow",
    "gray": "Low / inactive",
    "dimgray": "Null-flow",
    "darkred": "Highlighted loading",
}

# Edge style → layer label (orthogonal to color).
_STYLE_LAYERS: Dict[str, str] = {
    "dotted": "Non-reconnectable",
    "dashed": "Reconnectable",
    "tapered": "Swapped flow",
}


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

    layers = _build_layer_index(edges)
    return {
        "nodes": nodes,
        "edges": edges,
        "adjacency": adjacency,
        "layers": layers,
    }


def _build_layer_index(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group edges by color/style so the UI can offer toggles."""
    by_color: Dict[str, List[str]] = {}
    by_style: Dict[str, List[str]] = {}
    for e in edges:
        color_key = _color_to_layer_key(e["attrs"].get("color", ""))
        if color_key:
            by_color.setdefault(color_key, []).append(e["id"])
        style = (e["attrs"].get("style") or "").lower()
        if style in _STYLE_LAYERS:
            by_style.setdefault(style, []).append(e["id"])

    layers: List[Dict[str, Any]] = []
    for key, ids in by_color.items():
        layers.append({
            "key": f"color:{key}",
            "label": _LAYER_LABELS[key],
            "swatch": key,
            "edges": ids,
        })
    for key, ids in by_style.items():
        layers.append({
            "key": f"style:{key}",
            "label": _STYLE_LAYERS[key],
            "swatch": "",
            "edges": ids,
        })
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
  #sidebar input[type=text] { width: 100%; padding: 6px 8px; border: 1px solid var(--border); border-radius: 4px; font-size: 12px; }
  #sidebar label { display: flex; align-items: center; gap: 6px; padding: 3px 0; cursor: pointer; }
  #sidebar label .swatch { width: 12px; height: 12px; border-radius: 2px; border: 1px solid #ccc; flex-shrink: 0; }
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
  .graph .edge.layer-off { display: none; }
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
    <input type="text" id="search" placeholder="filter nodes by name…" autocomplete="off">

    <h2>Layers</h2>
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
  function fmtAttrs(prefix, el) {
    const out = [];
    for (const a of el.attributes) {
      if (a.name.startsWith('data-' + prefix + '-')) {
        const k = a.name.slice(('data-' + prefix + '-').length);
        out.push('<span class="key">' + k + '</span>: ' + a.value);
      }
    }
    return out.join('<br>');
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
    info.innerHTML = '<b>' + escapeHtml(name) + '</b><br>' + fmtAttrs('attr', node)
      + '<br><br>degree: ' + neighbours.length;
  }
  function cssEscape(s) { return (window.CSS && CSS.escape) ? CSS.escape(s) : s.replace(/(["\\\\])/g, '\\\\$1'); }
  function escapeHtml(s) { return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

  root.addEventListener('mouseover', (e) => {
    const g = e.target.closest('.node, .edge'); if (!g) return;
    if (g.classList.contains('node')) {
      showTooltip(e, '<b>' + escapeHtml(g.getAttribute('data-name') || '') + '</b><br>' + fmtAttrs('attr', g));
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
      const name = (n.getAttribute('data-name') || '').toLowerCase();
      if (name.indexOf(q) !== -1) { n.classList.add('match'); count++; }
    });
  }
  document.getElementById('search').addEventListener('input', applySearch);

  // ---- Layer toggles ----
  const layersEl = document.getElementById('layers');
  for (const layer of MODEL.layers) {
    const id = 'layer-' + layer.key.replace(/[^a-z0-9]/gi, '-');
    const wrap = document.createElement('label');
    wrap.innerHTML = '<input type="checkbox" id="' + id + '" checked>'
      + (layer.swatch ? '<span class="swatch" style="background:' + layer.swatch + '"></span>' : '<span class="swatch" style="border-style:dashed"></span>')
      + '<span>' + escapeHtml(layer.label) + ' <span style="color:var(--muted)">(' + layer.edges.length + ')</span></span>';
    layersEl.appendChild(wrap);
    wrap.querySelector('input').addEventListener('change', (e) => {
      for (const eid of layer.edges) {
        const el = document.getElementById(eid);
        if (el) el.classList.toggle('layer-off', !e.target.checked);
      }
    });
  }

  document.getElementById('stats').textContent =
    MODEL.nodes.length + ' nodes, ' + MODEL.edges.length + ' edges';
})();
</script>
</body>
</html>
"""


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
    annotated_svg = _inject_svg_data_attrs(svg_bytes, model)
    return (
        _HTML_TEMPLATE
        .replace("__TITLE__", html_mod.escape(title))
        .replace("__SVG__", annotated_svg)
        .replace("__MODEL_JSON__", json.dumps(model))
    )
