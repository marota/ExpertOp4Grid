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

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class Production:
    ID: int = 0

    def __init__(self, busbar_id: int, value: Optional[float] = None) -> None:
        self.ID: int = Production.ID
        self.busbar_id: int = busbar_id
        self.value: Optional[float] = value
        Production.ID += 1

    def __repr__(self) -> str:
        return "<< PRODUCTION Object ID: {}, busbar_id: {}, value: {} >>".format(self.ID, self.busbar_id, self.value)

    @property
    def busbar(self) -> int:
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar: int) -> None:
        self.busbar_id = new_busbar


class Consumption:
    ID: int = 0

    def __init__(self, busbar_id: int, value: Optional[float] = None) -> None:
        self.ID: int = Consumption.ID
        self.busbar_id: int = busbar_id
        self.value: Optional[float] = value
        Consumption.ID += 1

    def __repr__(self) -> str:
        return "<< CONSUMPTION Object ID: {}, busbar_id: {}, value: {} >>".format(self.ID, self.busbar_id, self.value)

    @property
    def busbar(self) -> int:
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar: int) -> None:
        logger.debug("busbar setter, new busbar = %s", new_busbar)
        self.busbar_id = new_busbar


class OriginLine:
    ID: int = 0

    def __init__(
        self,
        busbar_id: int,
        end_substation_id: Optional[int] = None,
        flow_value: Optional[List[float]] = None,
    ) -> None:
        self.ID: int = OriginLine.ID
        self.busbar_id: int = busbar_id
        self.end_substation_id: Optional[int] = end_substation_id
        self.flow_value: Optional[List[float]] = flow_value
        OriginLine.ID += 1

    def __repr__(self) -> str:
        return "<< ORIGINLINE Object ID: {}, busbar_id: {}," \
               " connected to substation: {}, flow_value: {} >>".format(self.ID, self.busbar_id, self.end_substation_id,
                                                                        self.flow_value)

    @property
    def busbar(self) -> int:
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar: int) -> None:
        self.busbar_id = new_busbar


class ExtremityLine:
    ID: int = 0

    def __init__(
        self,
        busbar_id: int,
        start_substation_id: Optional[int] = None,
        flow_value: Optional[List[float]] = None,
    ) -> None:
        self.ID: int = ExtremityLine.ID
        self.busbar_id: int = busbar_id
        self.start_substation_id: Optional[int] = start_substation_id
        self.flow_value: Optional[List[float]] = flow_value
        ExtremityLine.ID += 1

    def __repr__(self) -> str:
        return "<< EXTREMITYLINE Object ID: {}, busbar_id: {}," \
               " connected to substation: {}, flow_value: {} >>".format(self.ID, self.busbar_id,
                                                                        self.start_substation_id, self.flow_value)

    @property
    def busbar(self) -> int:
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar: int) -> None:
        self.busbar_id = new_busbar
