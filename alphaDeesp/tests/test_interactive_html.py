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
    assert _color_to_layer_key("BLACK") == "black"
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

    assert "color:coral" in layers
    assert "color:black" in layers
    assert "color:gray" in layers
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
    # Layer attribute composed from color + style.
    assert 'data-layers="color:gray style:dotted"' in html or \
           'data-layers="style:dotted color:gray"' in html
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
