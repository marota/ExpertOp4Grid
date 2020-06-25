from alphaDeesp.core.elements import *
import pprint
import numpy as np


class Network:
    """
    A Network represents an electrical network with Nodes that are composed of Elements

    For the moment used to created structures:

        self.nodes_prod_values = final_array_for_drawing_nodes
    and
        self.substation_id_busbar_id_node_id_mapping = substation_id_busbar_id_to_node_id_mapping

    """
    def __init__(self, substations_elements: dict):
        print("A Network got created...")

        nodes = sorted(list(substations_elements.keys()))
        print("Nodes = ", nodes)

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
            print("Node ID = ", substation_id)

            # LOOP THROUGH SUBSTATIONS
            for element in substations_elements[substation_id]:
                print(element)
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

            print("PROD MINUS LOAD")
            pprint.pprint(prods_minus_loads)

        print("mapping_node_id_to_prod_minus_load")
        pprint.pprint(mapping_node_id_to_prod_minus_load)

        ################################
        ################################
        final_array_for_drawing_nodes = []
        save_for_complementary_nodes = []
        # LOOP THROUGH SUBSTATIONS
        for substation_id in nodes:
            for busbar in mapping_node_id_to_prod_minus_load[substation_id].keys():
                if busbar == 0:
                    value = mapping_node_id_to_prod_minus_load[substation_id][busbar]
                    final_array_for_drawing_nodes.append((substation_id, value))

                elif busbar == 1 and mapping_node_id_to_prod_minus_load[substation_id][busbar] is not None:
                    twin_node_name = "666" + str(substation_id)
                    value = mapping_node_id_to_prod_minus_load[substation_id][busbar]
                    # nodes 666+ that are on busbar1 +
                    save_for_complementary_nodes.append((twin_node_name, value))

        pprint.pprint(final_array_for_drawing_nodes)
        pprint.pprint(save_for_complementary_nodes)

        for elem in save_for_complementary_nodes:
            final_array_for_drawing_nodes.append(elem)

        pprint.pprint(final_array_for_drawing_nodes)
        self.nodes_prod_values = final_array_for_drawing_nodes

        #####################################################################################################
        # ##################################### EDGE PART PREPROCESSING #####################################
        #####################################################################################################

        # here we create a mapping between: substation_id + busbar_id = node_id

        # from self.nodes_prod_values create a mapping
        # from a list create a dic mapping[substation_id][busbar_id] = node_id
        substation_id_busbar_id_to_node_id_mapping = {}
        # node_id is a tuple(node_id, value) value is a prod or cons value, or None, if there is none on this node
        for node_id in reversed(self.nodes_prod_values):
            print("\nnode_id = ", node_id)
            substation_id = node_id[0]

            # first create dict
            if substation_id not in list(substation_id_busbar_id_to_node_id_mapping.keys()):
                substation_id_busbar_id_to_node_id_mapping[substation_id] = {0: None, 1: None}

            # meaning there is a busbar 1 used on node X, from 666X
            if "666" in str(substation_id):
                initial_node_id = int(str(substation_id)[3:])  # we remove 666 and take the rest, from 6662, we get 2
                print("REST = ", initial_node_id)
                print("substation_id = ", substation_id)

                # create dict for initial_node_id if not exists
                substation_id_busbar_id_to_node_id_mapping[initial_node_id] = {0: None, 1: None}

                # then update accordingly
                substation_id_busbar_id_to_node_id_mapping[initial_node_id][1] = substation_id
                print("###############################===================================###############################")
                print("###############################===================================###############################")
                pprint.pprint(substation_id_busbar_id_to_node_id_mapping)

            # meaning there is no busbar1 used on this node, we just do a standard mapping
            else:
                # change only if neither bus has been set
                print("substation_id_busbar_id_to_node_id_mapping[substation_id][0] = ", substation_id_busbar_id_to_node_id_mapping[substation_id][0])
                print("substation_id_busbar_id_to_node_id_mapping[substation_id][1] = ", substation_id_busbar_id_to_node_id_mapping[substation_id][1])
                if substation_id_busbar_id_to_node_id_mapping[substation_id][0] is None or \
                        substation_id_busbar_id_to_node_id_mapping[substation_id][1] is None:
                    print("we where in the IFFFFFFF")
                    substation_id_busbar_id_to_node_id_mapping[substation_id][0] = substation_id

        print("###############################===================================###############################")
        print("###############################===================================###############################")
        print("End of Network's init")
        pprint.pprint(substation_id_busbar_id_to_node_id_mapping)

        self.substation_id_busbar_id_node_id_mapping = substation_id_busbar_id_to_node_id_mapping
        self.nb_graphical_nodes = len(list(substation_id_busbar_id_to_node_id_mapping.keys()))
        print("There are {} graphical nodes in this graph.".format(self.nb_graphical_nodes))


    def get_number_total_number_of_nodes(self):
        """
        Nodes, meaning counting splits if diff busbars
        :return:
        """
        pass

    def get_graphical_number_of_nodes(self):
        return self.nb_graphical_nodes
