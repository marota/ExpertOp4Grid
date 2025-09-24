# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of ExpertOp4Grid, an expert system approach to solve flow congestions in power grids
"""
This file contains substation elements, ie, Objects: Production, Consumption, Line.
"""


class Production:
    ID = 0

    def __init__(self, busbar_id, value=None):
        # print("Production created...")
        self.ID = Production.ID
        self.busbar_id = busbar_id
        self.value = value
        Production.ID += 1

    def __repr__(self):
        return "<< PRODUCTION Object ID: {}, busbar_id: {}, value: {} >>".format(self.ID, self.busbar_id, self.value)

    @property
    def busbar(self):
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar):
        self.busbar_id = new_busbar


class Consumption:
    ID = 0

    def __init__(self, busbar_id, value=None):
        # print("Consumption created...")
        self.ID = Consumption.ID
        self.busbar_id = busbar_id
        self.value = value
        Consumption.ID += 1

    def __repr__(self):
        return "<< CONSUMPTION Object ID: {}, busbar_id: {}, value: {} >>".format(self.ID, self.busbar_id, self.value)

    @property
    def busbar(self):
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar):
        print("debug inside busbar setter, new busbar =", new_busbar)
        self.busbar_id = new_busbar


class OriginLine:
    ID = 0

    def __init__(self, busbar_id, end_substation_id=None, flow_value=None):
        # print("OriginLine created...")
        self.ID = OriginLine.ID
        self.busbar_id = busbar_id
        self.end_substation_id = end_substation_id
        self.flow_value = flow_value
        OriginLine.ID += 1

    def __repr__(self):
        return "<< ORIGINLINE Object ID: {}, busbar_id: {}," \
               " connected to substation: {}, flow_value: {} >>".format(self.ID, self.busbar_id, self.end_substation_id,
                                                                        self.flow_value)

    @property
    def busbar(self):
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar):
        self.busbar_id = new_busbar


class ExtremityLine:
    ID = 0

    def __init__(self, busbar_id, start_substation_id=None, flow_value=None):
        # print("ExtremityLine created...")
        self.ID = ExtremityLine.ID
        self.busbar_id = busbar_id
        self.start_substation_id = start_substation_id
        self.flow_value = flow_value
        ExtremityLine.ID += 1

    def __repr__(self):
        return "<< EXTREMITYLINE Object ID: {}, busbar_id: {}," \
               " connected to substation: {}, flow_value: {} >>".format(self.ID, self.busbar_id,
                                                                        self.start_substation_id, self.flow_value)

    @property
    def busbar(self):
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar):
        self.busbar_id = new_busbar
