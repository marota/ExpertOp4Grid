# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of ExpertOp4Grid, an expert system approach to solve flow congestions in power grids

"""Unit tests for the interactive HTML viewer builder.

These tests exercise ``build_interactive_html`` against a small handcrafted
graph so they do not need a Grid2op backend or chronics on disk. They are
the smoke contract: the HTML must contain the SVG, the model JSON, the
expected data-* attributes, and the layer index derived from edge colors.
"""

import json
import re
from typing import List

import networkx as nx
import pytest

pydot = pytest.importorskip("pydot")

from alphaDeesp.core.interactive_html import (
    _build_layer_index,
    _color_to_layer_key,
    _model_from_dot_json,
    _split_edge_title,
    build_interactive_html,
)


def _toy_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("VALDI", color="red", shape="oval", penwidth=3)
    g.add_node("CHEVI", color="green", shape="oval", penwidth=3)
    g.add_node("MARSI", color="blue", shape="circle", penwidth=3)
    g.add_edge("VALDI", "CHEVI", color="coral", label="42",
               name="line_1", penwidth=4, capacity=42.0)
    g.add_edge("CHEVI", "MARSI", color="black", label="-100",
               name="line_2", penwidth=8, constrained=True)
    g.add_edge("VALDI", "MARSI", color="gray", label="0",
               name="line_3", penwidth=0.5, style="dotted")
    return g


def test_split_edge_title_handles_directed_and_html_escaped():
    assert _split_edge_title("A->B") == ("A", "B")
    assert _split_edge_title("VALDI&#45;&gt;CHEVI") == ("VALDI", "CHEVI")
    assert _split_edge_title("X--Y") == ("X", "Y")


def test_color_to_layer_key_handles_compound_colors():
    assert _color_to_layer_key("coral") == "coral"
    assert _color_to_layer_key('"coral:yellow:coral"') == "coral"
    assert _color_to_layer_key("BLUE") == "blue"
    # Black / gray / darkred are no longer recognised colour layers
    # (their semantic role moved onto the explicit ``is_overload`` /
    # ``is_monitored`` flags).
    assert _color_to_layer_key("black") is None
    assert _color_to_layer_key("#abcdef") is None


def test_model_from_dot_json_extracts_nodes_edges_adjacency():
    pg = nx.drawing.nx_pydot.to_pydot(_toy_graph())
    model = _model_from_dot_json(pg.create(prog="dot", format="json"))

    assert {n["name"] for n in model["nodes"]} == {"VALDI", "CHEVI", "MARSI"}
    assert len(model["edges"]) == 3
    # adjacency is undirected for highlight-neighbourhood UX
    assert sorted(n["node"] for n in model["adjacency"]["VALDI"]) == ["CHEVI", "MARSI"]
    assert sorted(n["node"] for n in model["adjacency"]["MARSI"]) == ["CHEVI", "VALDI"]


def test_layer_index_groups_by_color_and_style():
    pg = nx.drawing.nx_pydot.to_pydot(_toy_graph())
    model = _model_from_dot_json(pg.create(prog="dot", format="json"))
    layers = {l["key"]: l for l in model["layers"]}

    # After the section restructure only the three flow polarities
    # remain as colour layers (positive/negative/null). The historical
    # ``black`` / ``gray`` / ``darkred`` buckets are dropped — their
    # semantic role is carried by the ``is_overload`` / ``is_monitored``
    # flags instead.
    assert "color:coral" in layers
    assert "color:black" not in layers
    assert "color:gray" not in layers
    assert "style:dotted" in layers
    assert len(layers["color:coral"]["edges"]) == 1
    assert len(layers["style:dotted"]["edges"]) == 1


def test_build_interactive_html_is_self_contained_and_has_data_attrs():
    pg = nx.drawing.nx_pydot.to_pydot(_toy_graph())
    html = build_interactive_html(pg, title="toy")

    # Self-contained: no external script/style references.
    assert "<script src=" not in html
    assert "<link " not in html
    # Inlined SVG with the expected nodes and edges.
    assert "<svg " in html
    assert "id=\"node1\"" in html
    assert "id=\"edge1\"" in html
    # Data-* attributes injected by the post-processor.
    assert 'data-name="VALDI"' in html
    assert re.search(r'data-source="(VALDI|CHEVI)"', html)
    assert 'data-attr-name="line_1"' in html
    # Layer attribute composed from color + style. ``gray`` is no
    # longer recognised as a colour layer so the dotted edge advertises
    # only its style layer.
    assert 'data-layers="style:dotted"' in html
    # Embedded JS model is valid JSON and contains adjacency + layers.
    m = re.search(r"const MODEL = (\{.*?\});\n\(function", html, re.S)
    assert m, "expected MODEL constant to be embedded as JSON"
    model = json.loads(m.group(1))
    assert "adjacency" in model and "layers" in model
    assert "VALDI" in model["adjacency"]


def test_layer_index_ignores_unknown_colors():
    layers = _build_layer_index([
        {"id": "edge1", "source": "A", "target": "B",
         "attrs": {"color": "#123456"}},
        {"id": "edge2", "source": "A", "target": "B",
         "attrs": {"color": "coral"}},
    ])
    keys = {l["key"] for l in layers}
    assert "color:coral" in keys
    assert not any(k.startswith("color:#") for k in keys)


def test_layer_index_emits_semantic_layers_from_source_flags():
    """Source-of-truth attributes drive new semantic layer toggles."""
    edges = [
        {"id": "edge1", "source": "A", "target": "B",
         "attrs": {"color": "coral", "is_monitored": True, "in_red_loop": True}},
        {"id": "edge2", "source": "B", "target": "C",
         "attrs": {"color": "black", "is_overload": "True",
                   "on_constrained_path": True}},
    ]
    nodes = [
        {"name": "A", "attrs": {"is_hub": True}},
        {"name": "B", "attrs": {"in_red_loop": True,
                                "on_constrained_path": "True"}},
        {"name": "C", "attrs": {}},
    ]
    layers = _build_layer_index(edges, nodes)
    by_key = {l["key"]: l for l in layers}

    # Hubs is node-only.
    assert by_key["semantic:is_hub"]["nodes"] == ["A"]
    assert by_key["semantic:is_hub"]["edges"] == []
    # Red-loop spans both.
    assert set(by_key["semantic:in_red_loop"]["nodes"]) == {"B"}
    assert set(by_key["semantic:in_red_loop"]["edges"]) == {"edge1"}
    # Constrained path spans both.
    assert set(by_key["semantic:on_constrained_path"]["nodes"]) == {"B"}
    assert set(by_key["semantic:on_constrained_path"]["edges"]) == {"edge2"}
    # Overloads / monitored are edge-driven but include the edge
    # endpoint nodes so the substations remain visible when those
    # layers are toggled on alone.
    assert by_key["semantic:is_overload"]["edges"] == ["edge2"]
    assert set(by_key["semantic:is_overload"]["nodes"]) == {"B", "C"}
    assert by_key["semantic:is_monitored"]["edges"] == ["edge1"]
    assert set(by_key["semantic:is_monitored"]["nodes"]) == {"A", "B"}


def test_layer_index_emits_extra_cut_layer_with_endpoints():
    """``is_extra_cut`` is an edge-only semantic flag; like the other
    edge-only layers it must include the endpoint nodes so the
    substations stay visible when the operator ticks it on alone, and
    the layer must be assigned to the "Properties" section."""
    edges = [
        {"id": "edge1", "source": "A", "target": "B",
         "attrs": {"color": "blue", "is_extra_cut": "True"}},
        {"id": "edge2", "source": "B", "target": "C",
         "attrs": {"color": "coral"}},
    ]
    nodes = [
        {"name": "A", "attrs": {}},
        {"name": "B", "attrs": {}},
        {"name": "C", "attrs": {}},
    ]
    layers = _build_layer_index(edges, nodes)
    by_key = {l["key"]: l for l in layers}

    assert "semantic:is_extra_cut" in by_key
    layer = by_key["semantic:is_extra_cut"]
    assert layer["edges"] == ["edge1"]
    assert set(layer["nodes"]) == {"A", "B"}
    assert layer["swatch"] == "extra-cut"
    assert layer["label"] == "Extra lines to prevent flow increase"
    # Section assignment matches the other per-entity property layers.
    assert layer["section"] == "Individual entities properties"


def test_layer_index_skips_semantic_layer_when_no_match():
    """No noise: empty semantic buckets do NOT produce a layer entry."""
    edges = [
        {"id": "edge1", "source": "A", "target": "B",
         "attrs": {"color": "coral"}},
    ]
    nodes = [{"name": "A", "attrs": {}}, {"name": "B", "attrs": {}}]
    keys = {l["key"] for l in _build_layer_index(edges, nodes)}
    assert "semantic:is_hub" not in keys
    assert "semantic:in_red_loop" not in keys
    assert "semantic:is_extra_cut" not in keys
    assert "color:coral" in keys


def test_layer_index_node_arg_is_optional_for_legacy_callers():
    layers = _build_layer_index([
        {"id": "edge1", "source": "A", "target": "B",
         "attrs": {"color": "coral"}},
    ])
    # All emitted layers gain the new ``nodes`` field for shape uniformity.
    for layer in layers:
        assert "nodes" in layer and "edges" in layer


def test_readable_node_label_surfaces_without_losing_id():
    """A node ``label`` (e.g. a readable voltage-level name) is rendered and
    exposed as ``data-attr-label`` while the node identity (``<title>`` /
    ``data-name``) stays the original id — so selection, adjacency and
    double-click resolution keep using the stable id."""
    g = nx.MultiDiGraph()
    g.add_node("VL_way_1", color="red", shape="oval", label="Saucats 400kV")
    g.add_node("VL_way_2", color="blue", shape="oval")
    g.add_edge("VL_way_1", "VL_way_2", color="coral", label="42", name="line_1")
    pg = nx.drawing.nx_pydot.to_pydot(g)
    html = build_interactive_html(pg, title="toy")

    # Readable name is exposed for the JS (search + tooltip header) …
    assert 'data-attr-label="Saucats 400kV"' in html
    # … and rendered as the visible node text.
    assert "Saucats 400kV" in html
    # Node identity is preserved as the id.
    assert "<title>VL_way_1</title>" in html
    assert 'data-name="VL_way_1"' in html


def test_search_matches_readable_label_and_id():
    """The viewer's search filters on both the readable label and the id."""
    pg = nx.drawing.nx_pydot.to_pydot(_toy_graph())
    html = build_interactive_html(pg, title="toy")
    # Search reads the readable label attribute in addition to data-name.
    assert "data-attr-label" in html
    assert "function nodeDisplayName" in html
    assert "function nodeHeaderHtml" in html
    # The search routine consults BOTH the stable id (data-name) and the
    # resolved readable display name (via nodeDisplayName).
    search_fn = html.split("function applySearch()", 1)[1].split(
        "document.getElementById('search').addEventListener", 1
    )[0]
    assert "data-name" in search_fn
    assert "nodeDisplayName(n)" in search_fn


def test_label_less_nodes_fall_back_to_id_for_display():
    """Backward compatibility: when no readable ``label`` is set, Graphviz
    stores the placeholder ``\\N`` in the label attribute. The viewer must
    NOT surface that placeholder — ``nodeDisplayName`` ignores any
    backslash escape and falls back to the stable id (data-name)."""
    pg = nx.drawing.nx_pydot.to_pydot(_toy_graph())  # _toy_graph sets no labels
    html = build_interactive_html(pg, title="toy")
    assert 'data-name="VALDI"' in html
    # Graphviz emits the "\N" placeholder for label-less nodes …
    assert 'data-attr-label="\\N"' in html
    # … and the viewer guards against leaking any backslash placeholder.
    assert "label.indexOf('\\\\') === -1" in html


def test_node_tooltip_skips_duplicated_label_attr():
    """Node tooltips/selection use the readable name as the header and pass
    a skip-list so the ``label`` attribute is not also repeated in the
    attribute dump."""
    pg = nx.drawing.nx_pydot.to_pydot(_toy_graph())
    html = build_interactive_html(pg, title="toy")
    # fmtAttrs accepts a skip list and node renders pass ['label'].
    assert "function fmtAttrs(prefix, el, skip)" in html
    assert "fmtAttrs('attr', g, ['label'])" in html
    assert "fmtAttrs('attr', node, ['label'])" in html


def test_template_uses_dim_class_not_display_none():
    """Unchecked layers must DIM elements (not hide them) so spatial
    context is preserved."""
    pg = nx.drawing.nx_pydot.to_pydot(_toy_graph())
    html = build_interactive_html(pg, title="toy")
    # The historical hide rule is gone…
    assert ".graph .edge.layer-off { display: none; }" not in html
    # …replaced by a dim rule that applies to both nodes & edges.
    assert ".graph .layer-off { opacity: 0.12; }" in html


def test_template_has_select_all_unselect_all_buttons():
    pg = nx.drawing.nx_pydot.to_pydot(_toy_graph())
    html = build_interactive_html(pg, title="toy")
    assert 'id="layers-select-all"' in html
    assert 'id="layers-select-none"' in html


def test_build_interactive_html_propagates_semantic_attrs_to_svg():
    """Boolean source-of-truth flags must round-trip into ``data-attr-*``
    on the resulting SVG so the HTML viewer's JS can scan them."""
    g = nx.MultiDiGraph()
    g.add_node("HUB1", color="red", shape="diamond", penwidth=3, is_hub=True)
    g.add_node("REGN", color="green", shape="oval", penwidth=3)
    g.add_edge("HUB1", "REGN", color="coral", label="42",
               name="line_1", penwidth=4, capacity=42.0,
               is_monitored=True, on_constrained_path=True)
    pg = nx.drawing.nx_pydot.to_pydot(g)
    html = build_interactive_html(pg, title="semantic")

    assert 'data-attr-is_hub="True"' in html
    assert 'data-attr-is_monitored="True"' in html
    assert 'data-attr-on_constrained_path="True"' in html

    m = re.search(r"const MODEL = (\{.*?\});\n\(function", html, re.S)
    assert m
    model = json.loads(m.group(1))
    layer_keys = {l["key"] for l in model["layers"]}
    assert "semantic:is_hub" in layer_keys
    assert "semantic:is_monitored" in layer_keys
    assert "semantic:on_constrained_path" in layer_keys


def test_edge_ids_align_with_svg_titles_after_alignment_pass():
    """Regression for the user-reported visual bug: graphviz emits SVG
    and JSON edge IDs in independent orders, so JSON-derived IDs may
    point at the wrong SVG element. After ``_align_edge_ids_with_svg``,
    every SVG ``<g id="edgeN" class="edge">`` must carry data-* that
    agree with its own ``<title>``."""
    g = nx.MultiDiGraph()
    # Nodes
    for n in ["A", "B", "C", "D"]:
        g.add_node(n, color="red", shape="oval", penwidth=3)
    # Mix of single + parallel edges to exercise greedy matching.
    g.add_edge("A", "B", color="coral", name="L_AB1", capacity=1.0)
    g.add_edge("A", "B", color="coral", name="L_AB2", capacity=2.0)
    g.add_edge("A", "C", color="blue",  name="L_AC",  capacity=-3.0)
    g.add_edge("B", "C", color="black", name="L_BC",  capacity=-4.0)
    g.add_edge("C", "D", color="coral", name="L_CD",  capacity=5.0)
    g.add_edge("D", "A", color="blue",  name="L_DA",  capacity=-6.0)
    pg = nx.drawing.nx_pydot.to_pydot(g)
    html = build_interactive_html(pg, title="alignment")

    # For every SVG edge group, the parsed title src/dst must match
    # the data-source / data-target attrs.
    edge_blocks = re.findall(
        r'<g id="(edge\d+)" class="edge"[^>]*data-source="([^"]*)"[^>]*'
        r'data-target="([^"]*)"[^>]*>\s*<title>([^<]*)</title>',
        html,
    )
    assert edge_blocks, "no edge blocks found in rendered HTML"
    for gid, src, tgt, title in edge_blocks:
        title_src, title_dst = _split_edge_title(title)
        assert (src, tgt) == (title_src, title_dst), (
            f"{gid}: data ({src!r}, {tgt!r}) ≠ title ({title_src!r}, {title_dst!r})"
        )


def test_alignment_preserves_layer_membership_through_remap():
    """An edge that is in (say) ``color:blue`` before the alignment
    pass must still be in ``color:blue`` after, just under its new
    SVG-aligned id."""
    g = nx.MultiDiGraph()
    g.add_node("X", color="red", shape="oval")
    g.add_node("Y", color="green", shape="oval")
    g.add_edge("X", "Y", color="blue", name="negative", capacity=10.0)
    g.add_edge("Y", "X", color="coral", name="redispatch", capacity=10.0)
    pg = nx.drawing.nx_pydot.to_pydot(g)
    html = build_interactive_html(pg, title="layer-remap")

    m = re.search(r"const MODEL = (\{.*?\});\n\(function", html, re.S)
    model = json.loads(m.group(1))
    layers = {l["key"]: l for l in model["layers"]}
    edges_by_id = {e["id"]: e for e in model["edges"]}

    # Each layer's edge ids must correspond to edges whose attrs match
    # the layer's semantic.
    if "color:blue" in layers:
        for eid in layers["color:blue"]["edges"]:
            assert edges_by_id[eid]["attrs"].get("color") == "blue"
    if "color:coral" in layers:
        for eid in layers["color:coral"]["edges"]:
            assert edges_by_id[eid]["attrs"].get("color") == "coral"

    # AND the SVG element with id == layer's edge id must have the
    # title that references the correct endpoints.
    for layer_key in ("color:blue", "color:coral"):
        layer = layers.get(layer_key)
        if not layer:
            continue
        for eid in layer["edges"]:
            patt = re.compile(
                r'<g id="' + re.escape(eid) + r'" class="edge"[^>]*>\s*<title>([^<]*)</title>'
            )
            mm = patt.search(html)
            assert mm, f"layer {layer_key!r} references missing SVG element {eid}"
            t_src, t_dst = _split_edge_title(mm.group(1))
            edge = edges_by_id[eid]
            assert (edge["source"], edge["target"]) == (t_src, t_dst)


def test_color_layers_include_edge_endpoint_nodes():
    """When the user toggles a colour layer (e.g. "Positive overflow"
    coral) alone, the endpoint substations of those edges should also
    stay visible — not just the edges themselves. The layer index
    therefore embeds those endpoints in its `nodes` list."""
    pg = nx.drawing.nx_pydot.to_pydot(_toy_graph())
    model = _model_from_dot_json(pg.create(prog="dot", format="json"))
    layers = {l["key"]: l for l in model["layers"]}

    coral_layer = layers["color:coral"]
    assert set(coral_layer["nodes"]) == {"VALDI", "CHEVI"}, (
        f"coral layer endpoints: {coral_layer['nodes']}"
    )
    # ``color:black`` is no longer a colour layer (its semantic role
    # moved to the ``is_overload`` flag) so it must be absent.
    assert "color:black" not in layers


def test_style_layers_include_edge_endpoint_nodes():
    """Same contract for style layers (dashed / dotted / tapered)."""
    pg = nx.drawing.nx_pydot.to_pydot(_toy_graph())
    model = _model_from_dot_json(pg.create(prog="dot", format="json"))
    layers = {l["key"]: l for l in model["layers"]}

    dotted_layer = layers["style:dotted"]
    # The toy graph's only dotted edge goes VALDI->MARSI.
    assert set(dotted_layer["nodes"]) == {"VALDI", "MARSI"}


def test_layers_carry_section_field_in_canonical_order():
    """Layers must declare a ``section`` field and be ordered so that
    Structural Paths appear before Individual entities properties,
    which appear before Flow redispatch values."""
    edges = [
        {"id": "edge1", "source": "A", "target": "B",
         "attrs": {"color": "coral", "is_monitored": True,
                   "in_red_loop": True}},
        {"id": "edge2", "source": "B", "target": "C",
         "attrs": {"color": "blue", "is_overload": True,
                   "on_constrained_path": True}},
    ]
    nodes = [
        {"name": "A", "attrs": {"is_hub": True}},
        {"name": "B", "attrs": {}},
        {"name": "C", "attrs": {}},
    ]
    layers = _build_layer_index(edges, nodes)
    # Every emitted layer must carry a section.
    for layer in layers:
        assert "section" in layer, f"layer {layer['key']} missing section"

    section_seq = [l["section"] for l in layers]
    canonical = ["Structural Paths",
                 "Individual entities properties",
                 "Flow redispatch values"]
    # The first occurrence of each section must respect canonical order.
    seen_sections: List[str] = []
    for s in section_seq:
        if s not in seen_sections:
            seen_sections.append(s)
    assert seen_sections == [s for s in canonical if s in seen_sections]


def test_html_embeds_section_field_and_inserts_section_headers():
    """The embedded MODEL must carry a ``section`` on every layer,
    AND the bundled JS must include the header-insertion logic that
    will emit one ``<h3 class="layer-section-header">`` per section."""
    g = nx.MultiDiGraph()
    g.add_node("HUB", color="red", shape="oval", is_hub=True)
    g.add_node("LO", color="green", shape="oval", on_constrained_path=True)
    g.add_node("HI", color="blue", shape="oval", in_red_loop=True)
    g.add_edge("HUB", "LO", color="coral", name="lA",
               on_constrained_path=True)
    g.add_edge("LO", "HI", color="blue", name="lB", in_red_loop=True,
               is_overload=True)
    pg = nx.drawing.nx_pydot.to_pydot(g)
    html = build_interactive_html(pg, title="sections")

    # JS template owns the header-insertion logic.
    assert "layer-section-header" in html
    assert "createElement('h3')" in html

    # Embedded model must surface the three section names.
    m = re.search(r"const MODEL = (\{.*?\});\n\(function", html, re.S)
    assert m
    model = json.loads(m.group(1))
    sections = {layer.get("section") for layer in model["layers"]}
    assert "Structural Paths" in sections
    assert "Individual entities properties" in sections
    assert "Flow redispatch values" in sections
