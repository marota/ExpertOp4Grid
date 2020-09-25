****************************
AlphaDeesp algorithm details
****************************

Call
====

Calling the alphaDeesp engine is done like so :

``alphadeesp = AlphaDeesp(g_over, df_of_g, custom_layout, printer, simulator_data,sim.substation_in_cooldown, debug = debug)``
``ranked_combinations = alphadeesp.get_ranked_combinations()``

Inputs
======
The following inputs will be required to be computed by the Simulation override.

* ``g_over``
    A newtorkx graph representation of the grid with flow values

* ``df_of_g``
    A dataframe representing a detailed view of the graph
.. image:: ../alphaDeesp/ressources/df_of_g_explained.jpg

* ``custom_layout``
    The layout of the graph (list of (X,Y) coordinate for edges. Used for plotting.

* ``printer``
    A printer service for logs and graphs

* ``simulator_data``
    A dict composed of :

    * ``substations_elements``
        A local representation of the network using AlphaDeesp model objects from ``network.py``
    .. image:: ../alphaDeesp/ressources/internal_structure_explained.jpg

    * ``substation_to_node_mapping``

    * ``internal_to_external_mapping``
        A dict linking the substation ids from substations_elements (internal) to the observation substations (external)
    .. image:: ../alphaDeesp/ressources/internal_to_external_mapping_explanation_console.png


* ``substation_in_cooldown``
    List of substation that are in cooldown

* ``debug``
    Boolean flag for debugging purposes

Outputs
=======
The alphaDeesp object then provides a list : ``ranked_combinations``

This is a list of dataframes with the following columns :

* ``score``
    the score of the topology from 0(worst) to 4(best)
* ``topology``
    An array of integers (bus_ids) showing the topology of a node
* ``node``
    The node on which the topology was applied

This list is then used to simulate all topologies with the Simulation override :
``expert_system_results, actions = sim.compute_new_network_changes(ranked_combinations)``