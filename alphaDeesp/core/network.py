# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of ExpertOp4Grid, an expert system approach to solve flow congestions in power grids

import logging
import pprint
from typing import Any, Dict, List, Tuple

import numpy as np

from alphaDeesp.core.elements import (
    Consumption,
    ExtremityLine,
    OriginLine,
    Production,
)
from alphaDeesp.core.simulation import SubstationElement
from alphaDeesp.core.twin_nodes import (
    is_twin_node_id,
    original_substation_id,
    twin_node_id,
)

logger = logging.getLogger(__name__)


class Network:
    """
    A Network represents an electrical network with Nodes that are composed of Elements

    For the moment used to created structures:

        self.nodes_prod_values = final_array_for_drawing_nodes
    and
        self.substation_id_busbar_id_node_id_mapping = substation_id_busbar_id_to_node_id_mapping

    """
    def __init__(self, substations_elements: Dict[int, List[SubstationElement]]) -> None:
        logger.debug("A Network got created...")

        nodes = sorted(list(substations_elements.keys()))
        logger.debug("Nodes = %s", nodes)

        #####################################################################################################
        # ##################################### NODE PART ###################################################
        #####################################################################################################

        mapping_node_id_to_prod_minus_load = {}
        for substation_id in nodes:
            first_time_bool = True
            prods_minus_loads = {
                0: None,
                1: None
            }

            mapping_node_id_to_prod_minus_load[substation_id] = prods_minus_loads
            logger.debug("Node ID = %s", substation_id)

            # LOOP THROUGH SUBSTATIONS
            for element in substations_elements[substation_id]:
                logger.debug("%s", element)
                for busbar_id in [0, 1]:  # TODO IMPORTANT, DO A PREPROCESSING OR CONFIG INI TO GET NB TOTAL BUSBARS
                    if element.busbar_id == busbar_id:
                        if isinstance(element, Production) or isinstance(element, Consumption):
                            if first_time_bool:
                                if isinstance(element, Production):
                                    prods_minus_loads[busbar_id] = np.round(element.value, 2)
                                elif isinstance(element, Consumption):
                                    prods_minus_loads[busbar_id] = np.round(-element.value)
                                first_time_bool = False
                            else:
                                if isinstance(element, Production):
                                    prods_minus_loads[busbar_id] = np.round(
                                        prods_minus_loads[busbar_id] + element.value[0], 2)
                                elif isinstance(element, Consumption):
                                    if prods_minus_loads[busbar_id] is not None:
                                        prods_minus_loads[busbar_id] = np.round(
                                            prods_minus_loads[busbar_id] - element.value)
                                    else:
                                        prods_minus_loads[busbar_id] = np.round(-element.value)

                        # here we continue filling dictionary mapping_node_id_to_prod_minus_load with information
                        # from elements OriginLine and ExtremityLine
                        # Important to detect new "twin nodes" of format 666XX..
                        # Important to get everything right
                        elif isinstance(element, OriginLine) or isinstance(element, ExtremityLine):
                            if prods_minus_loads[busbar_id] is None:
                                prods_minus_loads[busbar_id] = "XXX"
                            # if element.flow_value is None:
                            #     prods_minus_loads[busbar_id] = "XXX"

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("PROD MINUS LOAD\n%s", pprint.pformat(prods_minus_loads))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("mapping_node_id_to_prod_minus_load\n%s",
                         pprint.pformat(mapping_node_id_to_prod_minus_load))

        ################################
        ################################
        final_array_for_drawing_nodes: List[Tuple[Any, Any]] = []
        save_for_complementary_nodes: List[Tuple[Any, Any]] = []
        # LOOP THROUGH SUBSTATIONS
        for substation_id in nodes:
            for busbar in mapping_node_id_to_prod_minus_load[substation_id].keys():
                if busbar == 0:
                    value = mapping_node_id_to_prod_minus_load[substation_id][busbar]
                    final_array_for_drawing_nodes.append((substation_id, value))

                elif busbar == 1 and mapping_node_id_to_prod_minus_load[substation_id][busbar] is not None:
                    twin_node_name = twin_node_id(substation_id)
                    value = mapping_node_id_to_prod_minus_load[substation_id][busbar]
                    # twin nodes for elements moved to busbar 1
                    save_for_complementary_nodes.append((twin_node_name, value))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("final_array_for_drawing_nodes\n%s",
                         pprint.pformat(final_array_for_drawing_nodes))
            logger.debug("save_for_complementary_nodes\n%s",
                         pprint.pformat(save_for_complementary_nodes))

        for elem in save_for_complementary_nodes:
            final_array_for_drawing_nodes.append(elem)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("final_array_for_drawing_nodes (with complementaries)\n%s",
                         pprint.pformat(final_array_for_drawing_nodes))
        self.nodes_prod_values: List[Tuple[Any, Any]] = final_array_for_drawing_nodes

        #####################################################################################################
        # ##################################### EDGE PART PREPROCESSING #####################################
        #####################################################################################################

        # here we create a mapping between: substation_id + busbar_id = node_id

        # from self.nodes_prod_values create a mapping
        # from a list create a dic mapping[substation_id][busbar_id] = node_id
        substation_id_busbar_id_to_node_id_mapping = {}
        # node_id is a tuple(node_id, value) value is a prod or cons value, or None, if there is none on this node
        for node_id in reversed(self.nodes_prod_values):
            logger.debug("node_id = %s", node_id)
            substation_id = node_id[0]

            # first create dict
            if substation_id not in list(substation_id_busbar_id_to_node_id_mapping.keys()):
                substation_id_busbar_id_to_node_id_mapping[substation_id] = {0: None, 1: None}

            # meaning there is a busbar 1 used on this substation, referenced by its twin-node id
            if is_twin_node_id(substation_id):
                initial_node_id = original_substation_id(substation_id)
                logger.debug("twin node original substation id = %s", initial_node_id)
                logger.debug("substation_id = %s", substation_id)

                # create dict for initial_node_id if not exists
                substation_id_busbar_id_to_node_id_mapping[initial_node_id] = {0: None, 1: None}

                # then update accordingly
                substation_id_busbar_id_to_node_id_mapping[initial_node_id][1] = substation_id
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("substation_id_busbar_id_to_node_id_mapping\n%s",
                                 pprint.pformat(substation_id_busbar_id_to_node_id_mapping))

            # meaning there is no busbar1 used on this node, we just do a standard mapping
            else:
                # change only if neither bus has been set
                logger.debug("substation_id_busbar_id_to_node_id_mapping[substation_id][0] = %s",
                             substation_id_busbar_id_to_node_id_mapping[substation_id][0])
                logger.debug("substation_id_busbar_id_to_node_id_mapping[substation_id][1] = %s",
                             substation_id_busbar_id_to_node_id_mapping[substation_id][1])
                if substation_id_busbar_id_to_node_id_mapping[substation_id][0] is None or \
                        substation_id_busbar_id_to_node_id_mapping[substation_id][1] is None:
                    logger.debug("we were in the IFFFFFFF")
                    substation_id_busbar_id_to_node_id_mapping[substation_id][0] = substation_id

        logger.debug("End of Network's init")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("substation_id_busbar_id_to_node_id_mapping\n%s",
                         pprint.pformat(substation_id_busbar_id_to_node_id_mapping))

        self.substation_id_busbar_id_node_id_mapping: Dict[int, Dict[int, Any]] = (
            substation_id_busbar_id_to_node_id_mapping
        )
        self.nb_graphical_nodes: int = len(list(substation_id_busbar_id_to_node_id_mapping.keys()))
        logger.debug("There are %s graphical nodes in this graph.", self.nb_graphical_nodes)


    def get_number_total_number_of_nodes(self) -> None:
        """
        Nodes, meaning counting splits if diff busbars
        :return:
        """
        pass

    def get_graphical_number_of_nodes(self) -> int:
        return self.nb_graphical_nodes
