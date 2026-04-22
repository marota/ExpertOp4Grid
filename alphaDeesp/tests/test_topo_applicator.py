"""Unit tests for :mod:`alphaDeesp.core.topo_applicator`.

Tests the static helpers exposed by TopoApplicatorMixin which do not depend
on self.g or the full AlphaDeesp pipeline.
"""

import pytest
from alphaDeesp.core.topo_applicator import TopoApplicatorMixin
from alphaDeesp.core.elements import (
    Consumption,
    ExtremityLine,
    OriginLine,
    Production,
)


# ──────────────────────────────────────────────────────────────────────
# _compute_prod_load_per_bus
# ──────────────────────────────────────────────────────────────────────

class TestComputeProdLoadPerBus:

    def test_single_production(self):
        elements = [Production(busbar_id=0, value=10.0)]
        prod, load = TopoApplicatorMixin._compute_prod_load_per_bus(elements)
        assert prod == {0: 10.0}
        assert load == {}

    def test_single_consumption(self):
        elements = [Consumption(busbar_id=1, value=5.0)]
        prod, load = TopoApplicatorMixin._compute_prod_load_per_bus(elements)
        assert prod == {}
        assert load == {1: 5.0}

    def test_mixed_on_same_bus(self):
        elements = [
            Production(busbar_id=0, value=10.0),
            Consumption(busbar_id=0, value=3.0),
        ]
        prod, load = TopoApplicatorMixin._compute_prod_load_per_bus(elements)
        assert prod == {0: 10.0}
        assert load == {0: 3.0}

    def test_multiple_productions_summed(self):
        elements = [
            Production(busbar_id=0, value=4.0),
            Production(busbar_id=0, value=6.0),
        ]
        prod, load = TopoApplicatorMixin._compute_prod_load_per_bus(elements)
        assert prod == {0: pytest.approx(10.0)}

    def test_production_on_two_buses(self):
        elements = [
            Production(busbar_id=0, value=8.0),
            Production(busbar_id=1, value=2.0),
        ]
        prod, load = TopoApplicatorMixin._compute_prod_load_per_bus(elements)
        assert prod == {0: 8.0, 1: 2.0}

    def test_origin_line_ignored(self):
        elements = [OriginLine(busbar_id=0, end_substation_id=2)]
        prod, load = TopoApplicatorMixin._compute_prod_load_per_bus(elements)
        assert prod == {}
        assert load == {}

    def test_empty_elements(self):
        prod, load = TopoApplicatorMixin._compute_prod_load_per_bus([])
        assert prod == {}
        assert load == {}


# ──────────────────────────────────────────────────────────────────────
# _classify_bus
# ──────────────────────────────────────────────────────────────────────

class TestClassifyBus:

    def test_prod_only(self):
        kind, value = TopoApplicatorMixin._classify_bus(0, {0: 10.0}, {})
        assert kind == "prod"
        assert value == pytest.approx(10.0)

    def test_load_only(self):
        kind, value = TopoApplicatorMixin._classify_bus(1, {}, {1: 5.0})
        assert kind == "load"
        assert value == pytest.approx(5.0)

    def test_both_prod_dominant(self):
        kind, value = TopoApplicatorMixin._classify_bus(0, {0: 10.0}, {0: 3.0})
        assert kind == "prod"
        assert value == pytest.approx(7.0)

    def test_both_load_dominant(self):
        kind, value = TopoApplicatorMixin._classify_bus(0, {0: 2.0}, {0: 8.0})
        assert kind == "load"
        assert value == pytest.approx(-6.0)

    def test_neither_returns_none(self):
        kind, value = TopoApplicatorMixin._classify_bus(0, {}, {})
        assert kind is None
        assert value == 0

    def test_bus_not_present_returns_none(self):
        kind, value = TopoApplicatorMixin._classify_bus(2, {0: 5.0}, {1: 3.0})
        assert kind is None
        assert value == 0
