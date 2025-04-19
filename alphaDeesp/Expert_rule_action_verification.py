#!/usr/bin/python3
# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of ExpertOp4Grid, an expert system approach to solve flow congestions in power grids

from make_evaluation_env import make_grid2op_evaluation_env
from make_training_env import make_grid2op_training_env
from load_evaluation_data import list_all_chronics, get_first_obs_on_chronic
from datetime import datetime
from data_utils import StateInfo
import sys

import configparser
import numpy as np
import os
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.graphsAndPaths import OverFlowGraph,Structured_Overload_Distribution_Graph
import networkx as nx
import pypowsybl as pp
import pandas as pd
from load_training_data import aux_prevent_asset_reconnection,load_interesting_lines,DELETED_LINE_NAME


import glob
import shutil
import json
from packaging.version import Version as version_packaging

from packaging.version import Version as version_packaging
from importlib.metadata import version
EXOP_MIN_VERSION = version_packaging("0.2.6")
if version_packaging(version("expertop4grid")) < EXOP_MIN_VERSION:
    raise RuntimeError(f"Incompatible version found for expertOp4Grid, make sureit is >= {EXOP_MIN_VERSION}")

param_options_expertOp={
    # 0.2 is 20 percent of the max overload flow
    "ThresholdReportOfLine": 0.05,#0.2,#0.05,#
    # 10 percent de la surcharge max
    "ThersholdMinPowerOfLoop": 0.1,
    # If at least a loop is detected, only keep the ones with a flow  of at least 25 percent the biggest one
    "ratioToKeepLoop": 0.25,
    # Ratio percentage for reconsidering the flow direction
    "ratioToReconsiderFlowDirection": 0.75,
    # max unused lines
    "maxUnusedLines": 3,
    # number of simulated topologies node at the final simulation step
    "totalnumberofsimulatedtopos": 30,
    # number of simulated topologies per node at the final simulation step
    "numberofsimulatedtopospernode": 10
    }


def identify_overload_lines_to_keep_overflow_graph_connected(obs_simu, lines_overloaded_ids):
    """
    This function identifies which overloaded lines to keep in order to maintain a connected overflow graph.

    Parameters:
    obs_simu (Grid2opSimulation): The simulation object containing the current state of the grid.
    lines_overloaded_ids (list): List of IDs of the overloaded lines.

    Returns:
    list: List of overloaded line IDs to keep in the overflow graph.
    """
    # Get the energy graph from the simulation object
    obs_graph = obs_simu.get_energy_graph()

    # Find the maximum rho value among all lines
    max_rho = max(obs_simu.rho)

    # Identify edges that are disconnected (rho value is 0)
    edges_disconnected = [edge for edge, rho in nx.get_edge_attributes(obs_graph, 'a_or').items() if np.round(rho, 3) == 0]

    # Identify the edge with the maximum rho value
    recover_max_overload_edge = [edge for edge, rho in nx.get_edge_attributes(obs_graph, 'rho').items() if rho == max(obs_simu.rho)]

    # Identify other overloaded edges
    recover_other_overload_edge = [edge for edge, rho in nx.get_edge_attributes(obs_graph, 'rho').items() for line_id in lines_overloaded_ids if rho == obs_simu.rho[line_id] and rho != max_rho]

    # Create a graph from the energy graph
    obs_graph = nx.Graph(obs_graph)

    # Remove disconnected edges from the graph
    obs_graph.remove_edges_from(edges_disconnected)

    # Calculate the number of connected components initially
    n_connected_comp_init = len([c for c in nx.connected_components(obs_graph)])

    # Remove the edge with the maximum overload from the graph
    obs_graph.remove_edges_from(recover_max_overload_edge)

    # Calculate the number of connected components after removing the max overload edge
    n_connected_comp_max_overload = len([c for c in nx.connected_components(obs_graph)])

    # Remove other overloaded edges from the graph
    obs_graph.remove_edges_from(recover_other_overload_edge)

    # Calculate the number of connected components after removing all overload edges
    n_connected_comp_all_overload = len([c for c in nx.connected_components(obs_graph)])

    # Determine which overloaded lines to keep based on the number of connected components
    if n_connected_comp_init == n_connected_comp_all_overload:
        # If removing all overloads does not change the number of connected components, keep all overloaded lines
        lines_overloaded_ids_to_keep = lines_overloaded_ids
    elif n_connected_comp_max_overload == n_connected_comp_init:
        # If removing the max overload edge does not change the number of connected components, keep only the max overload line
        lines_overloaded_ids_to_keep = [line_id for line_id in lines_overloaded_ids if obs_simu.rho[line_id] == max_rho]
        print(f"we reduce the problem by focusing on the deepest overload {obs_simu.name_line[lines_overloaded_ids_to_keep[0]]} as considering all overloads cut break the network appart")
    else:
        # If removing any overload edge changes the number of connected components, it is not solvable without load shedding
        lines_overloaded_ids_to_keep = None

    return lines_overloaded_ids_to_keep

def inhibit_swapped_flows(df_of_g):
    """
    This function readjusts the flow direction for swapped flows in the overflow graph, inhibiting this legacy behavior from expertOp4grid.

    Parameters:
    df_of_g (pandas.DataFrame): DataFrame containing the overflow graph data.

    Returns:
    pandas.DataFrame: Updated DataFrame with adjusted flow directions for swapped flows.
    """
    # Reverse delta_flows value for swapped flows
    df_of_g.loc[df_of_g.new_flows_swapped, "delta_flows"] = -df_of_g[df_of_g.new_flows_swapped]["delta_flows"]

    # Swap idx_or and idx_ex for swapped flows
    idx_or = df_of_g[df_of_g.new_flows_swapped]["idx_or"]
    df_of_g.loc[df_of_g.new_flows_swapped, "idx_or"] = df_of_g[df_of_g.new_flows_swapped]["idx_ex"]
    df_of_g.loc[df_of_g.new_flows_swapped, "idx_ex"] = idx_or

    return df_of_g

def build_overflow_graph(obs_overloaded,action_space,observation_space,overloaded_line_ids,non_connected_reconnectable_lines,lines_non_reconnectable,param_options,timestep,do_consolidate_graph=True,inhibit_swapped_flow_reversion=True):
    """
    This function builds an overflow graph based on the given overloaded observation and parameters.
    sim = Grid2opSimulation(obs_overloaded, action_space, observation_space, param_options=param_options, debug=False,
    Parameters:
    obs_overloaded (Grid2opObservation): The observation containing overloaded lines.
    action_space (Grid2opActionSpace): The action space for the environment.
    observation_space (Grid2opObservationSpace): The observation space for the environment.
    overloaded_line_ids (list): List of IDs of the overloaded lines.
    non_connected_reconnectable_lines (list): List of reconnectable lines.
    lines_non_reconnectable (list): List of non reconnectable lines.
    param_options (dict): Dictionary of parameters for the simulation.
    timestep (int): The timestep for the simulation.
    inhibit_swapped_flow_reversion (bool): Cancel the swapped edge direction for swapped flows (possibly not needed anymore given the new consolidate graph functions)

    Returns:
    tuple: A tuple containing five objects:
           - df_of_g: pandas dataframe with changes in flows for each lines after disconnecting overflows
           - overflow_sim (Grid2opSimulation): The simulation object.
           - g_overflow (OverFlowGraph): The overflow graph.
           - real_hubs: list of hubs in g_distribution_graph
           - g_distribution_graph (Structured_Overload_Distribution_Graph): The structured overload distribution graph.
    """
    overflow_sim = Grid2opSimulation(obs_overloaded, action_space, observation_space, param_options=param_options, debug=False,
                            ltc=overloaded_line_ids, plot=True, simu_step=timestep)

    df_of_g = overflow_sim.get_dataframe()

    #add line names
    df_of_g["line_name"] = obs_overloaded.name_line

    #inhibit swap flows, reverse delta_flows value for swapped flows
    if inhibit_swapped_flow_reversion:
        df_of_g=inhibit_swapped_flows(df_of_g)

    g_overflow = OverFlowGraph(overflow_sim.topo, overloaded_line_ids, df_of_g, float_precision="%.0f")

    mapping = {i: name for i, name in enumerate(obs_overloaded.name_sub)}
    g_overflow.g = nx.relabel_nodes(g_overflow.g, mapping, copy=True)

    g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)

    if len(g_distribution_graph.g_only_red_components.nodes)!=0 and do_consolidate_graph:
        g_overflow.consolidate_graph(g_distribution_graph)
    #
        g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)

    #get real hubs before we add disconected but reconnectable lines
    real_hubs = g_distribution_graph.get_hubs()

    #add disconected but reconnectable lines
    # TO DO - reminder: uncomment!!
    for i in range(2):#need two iterations to identify CHALOL31LOUHA reconnectable path under contingency "BEON L31CPVAN" at timestep 1 on chronic 28th august
        g_overflow.add_relevant_null_flow_lines_all_paths(g_distribution_graph, non_connected_lines=non_connected_reconnectable_lines,non_reconnectable_lines=lines_non_reconnectable)
        g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)

    return df_of_g,overflow_sim,g_overflow,real_hubs,g_distribution_graph

def is_nodale_grid2op_action(act):

    #check if bus set happens for multiple objects at a substation
    subs, counts = np.unique(act._topo_vect_to_sub[(act._set_topo_vect >= 1)], return_counts=True)
    is_nodale_action = np.any(counts>=2)#if at least two objects at a given substation appear with a set bus
    concerned_subs=[]
    is_splitting_subs=[]
    if is_nodale_action:
        concerned_subs=[sub for i,sub in enumerate(subs) if counts[i]>=2]
        for sub in concerned_subs:
            bus_for_set=np.unique(act._set_topo_vect[act._topo_vect_to_sub==sub])
            is_splitting=len(bus_for_set[bus_for_set>=1])>=2
            is_splitting_subs.append(is_splitting)

    return is_nodale_action, concerned_subs, is_splitting_subs

def is_line_disconnection(grid2op_action):
    line_or_change = grid2op_action.line_or_change_bus
    line_ex_change = grid2op_action.line_ex_change_bus
    line_change_status = grid2op_action.line_change_status

    line_or_set = grid2op_action.line_or_set_bus
    line_ex_set = grid2op_action.line_ex_set_bus
    line_set_status = grid2op_action.line_set_status

    is_line_deconnection=False
    if np.any(line_change_status!=0) or np.any(line_or_change!=0) or np.any(line_ex_change!=0):
        print("WARNING: line_change_status is not supported in this is_line_deconnection function ")

    if np.any(line_or_set==-1) or np.any(line_ex_set==-1) or np.any(line_set_status==-1):
        is_line_deconnection=True

    return is_line_deconnection


def is_line_reconnection(grid2op_action):
    line_or_change = grid2op_action.line_or_change_bus
    line_ex_change = grid2op_action.line_ex_change_bus
    line_change_status = grid2op_action.line_change_status

    line_or_set = grid2op_action.line_or_set_bus
    line_ex_set = grid2op_action.line_ex_set_bus
    line_set_status = grid2op_action.line_set_status

    is_line_reconnection=False
    if np.any(line_change_status!=0) or np.any(line_or_change!=0) or np.any(line_ex_change!=0):
        print("WARNING: line_change_status is not supported in this is_line_reconnection function ")

    if np.any(line_or_set*line_ex_set == 1) or np.any(line_set_status == 1):
        is_line_reconnection = True

    return is_line_reconnection

def is_load_disconnection(grid2op_action):
    load_change_bus = grid2op_action.load_change_bus
    load_set_bus = grid2op_action.load_set_bus

    is_load_disconnection=False
    if np.any(load_change_bus!=0):
        print("WARNING: load_change_bus is not supported in this is_load_disconnection function ")

    if np.any(load_set_bus == -1):
        is_load_disconnection = True

    return is_load_disconnection

def identify_grid2op_action_type(grid2op_action):
    """
    This function identifies the type of action based on the provided grid2op action object.

    Parameters:
    grid2op_action (grid2op.Action): The action object from the Grid2Op library.

    Returns:
    str: A string representing the type of action. Possible values are:
         - "open_coupling"
         - "close_coupling"
         - "open_line"
         - "close_line"
         - "open_load"
         - "close_load"
         - "open_line_load"
         - "close_line_load"
    """
    is_nodale_action, concerned_subs, is_splitting_subs=is_nodale_grid2op_action(grid2op_action)
    if is_nodale_action:
        if any(is_splitting_subs):
            return "open_coupling"
        else:
            return "close_coupling"
    else:
        is_line_disco=is_line_disconnection(grid2op_action)
        is_line_reco=is_line_reconnection(grid2op_action)
        is_load_disco=is_load_disconnection(grid2op_action)

        if is_line_disco:
            if is_load_disco:
                return "open_line_load"
            else:
                return "open_line"
        if is_line_reco:
            return "close_line"
        if is_load_disco:
            return "open_load"

    return "unknown"


def identify_action_type(actions_desc, by_description=True,grid2op_action_space=None):
    """
    This function identifies the type of action based on the provided action description.

    Parameters:
    actions_desc (dict): A dictionary containing the action description.
                         Expected keys include "description_unitaire" and "content".
    by_description (bool): A flag indicating whether to identify the action type based on the description.
                          If False, the function will identify the action type based on the content directly.

    Returns:
    str: A string representing the type of action. Possible values are:
         - "open_coupling"
         - "close_coupling"
         - "open_line"
         - "close_line"
         - "open_load"
         - "close_load"
         - "open_line_load"
         - "close_line_load"
    """
    type = None
    if by_description:
        # Extract the set_bus dictionary from the action description
        dict_action = actions_desc["content"]["set_bus"]

        # Check if there are any loads or lines involved in the action
        has_load = len(dict_action["loads_id"]) != 0
        has_line = (len(dict_action["lines_or_id"]) != 0) or (len(dict_action["lines_ex_id"]) != 0)

        # Get the description of the action
        description = actions_desc["description_unitaire"]

        # Determine the action type based on the description
        if ("COUPL" in description or "TRO." in description) and "Ouverture" in description:
            type = "open_coupling"
        elif ("COUPL" in description or "TRO." in description) and "Fermeture" in description:
            type = "close_coupling"
        elif "Ouverture" in description:
            if has_load and has_line:
                type = "open_line_load"
            elif has_line:
                type = "open_line"
            else:
                type = "open_load"
        elif "Fermeture" in description:
            if has_load and has_line:
                type = "close_line_load"
            elif has_line:
                type = "close_line"
            else:
                type = "close_load"
    else:
        # If by_description is False, the function does not process the action type based on the content directly
        grid2op_action_str=actions_desc["content"]
        grid2op_action=grid2op_action_space(grid2op_action_str)
        type=identify_grid2op_action_type(grid2op_action)

    return type

def localize_line_action(lines, lines_constrained_path, lines_dispatch):
    """
    This function determines the localization of a set of lines within the constrained path or dispatch path.

    Parameters:
    lines (list): List of line IDs to be localized.
    lines_constrained_path (list): List of line IDs that are part of the constrained path.
    lines_dispatch (list): List of line IDs that are part of the dispatch path.

    Returns:
    str: A string indicating the localization of the lines. Possible values are:
         - "out_of_graph": The lines are not part of the constrained path or dispatch path.
         - "constrained_path": The lines are part of the constrained path.
         - "dispatch_path": The lines are part of the dispatch path.
    """
    localization = "out_of_graph"

    lines_intersect_constrained_path = set(lines).intersection(set(lines_constrained_path))
    lines_intersect_dispatch_lines = set(lines).intersection(set(lines_dispatch))

    if len(lines_intersect_constrained_path) != 0:
        localization = "constrained_path"
    elif len(lines_intersect_dispatch_lines) != 0:
        localization = "dispatch_path"

    return localization

def localize_coupling_action(action_subs, hubs, nodes_constrained_path, nodes_dispatch_path):
    """
    This function determines the localization of a coupling action within the constrained path, dispatch path, or hubs.

    Parameters:
    action_subs (list): List of action sub-topologies to be localized.
    hubs (list): List of hub nodes in the graph.
    nodes_constrained_path (list): List of nodes that are part of the constrained path.
    nodes_dispatch_path (list): List of nodes that are part of the dispatch path.

    Returns:
    str: A string indicating the localization of the coupling action. Possible values are:
         - "out_of_graph": The action is not part of the constrained path, dispatch path, or hubs.
         - "hubs": The action is part of the hubs.
         - "constrained_path": The action is part of the constrained path.
         - "dispatch_path": The action is part of the dispatch path.
    """
    # Initialize the localization to "out_of_graph"
    localization = "out_of_graph"

    # Check if the action involves any hubs
    action_in_hubs = len(set(action_subs).intersection(set(hubs))) != 0

    # Check if the action involves any nodes in the constrained path
    action_in_constrained_path = len(set(action_subs).intersection(set(nodes_constrained_path))) != 0

    # Check if the action involves any nodes in the dispatch path
    action_in_dispatch_path = len(set(action_subs).intersection(set(nodes_dispatch_path))) != 0

    # Determine the localization based on the checks
    if action_in_hubs:
        localization = "hubs"
    elif action_in_constrained_path:
        localization = "constrained_path"
    elif action_in_dispatch_path:
        localization = "dispatch_path"

    return localization

def check_rules(action_type, localization, subs_topology):
    """
    This function checks if the given action violates any expert-defined rules based on its type, localization, and sub-topology.

    Parameters:
    action_type (str): The type of the action (e.g., "open_line", "close_coupling").
    localization (str): The localization of the action within the graph (e.g., "dispatch_path", "constrained_path").
    subs_topology (list): A list of sub-topologies involved in the action.

    Returns:
    tuple: A tuple containing two elements:
           - do_filter_action (bool): A flag indicating whether the action should be filtered out.
           - broken_rule (str): A string describing the rule that was broken, if any.
    """
    do_filter_action = False
    broken_rule = None

    if "load" not in action_type:  # We don't filter actions that disconnect loads for now
        # Check if the topology is in a single node
        is_topo_subs_one_node = np.all([len(set(sub_topo) - set([-1])) == 1 for sub_topo in subs_topology])

        # Expert rules to prevent
        out_of_graph = (localization == "out_of_graph")
        line_disconnection_dispatch_path = ("line" in action_type) and ("open" in action_type) and (localization == "dispatch_path")
        line_reconnection_constrained_path = ("line" in action_type) and ("close" in action_type) and (localization == "constrained_path")
        # Filter node splitting only if the sub-topology was initially one node
        node_splitting_dispatch_path = ("coupling" in action_type) and ("open" in action_type) and (localization == "dispatch_path") and is_topo_subs_one_node
        node_merging_constrained_path = ("coupling" in action_type) and ("close" in action_type) and (localization == "constrained_path")
        # Check if any rules are broken
        do_filter_action = out_of_graph or line_disconnection_dispatch_path or line_reconnection_constrained_path or node_splitting_dispatch_path or node_merging_constrained_path

        if out_of_graph:
            broken_rule = "No action out of the overflow graph"
        elif line_reconnection_constrained_path:
            broken_rule = "No line reconnection on constrained path"
        elif line_disconnection_dispatch_path:
            broken_rule = "No line disconnection on dispatch path"
        elif node_merging_constrained_path:
            broken_rule = "No node merging on constrained path"
        elif node_splitting_dispatch_path:
            broken_rule = "No node splitting on dispatch path"

    return do_filter_action, broken_rule

def verify_action(action_desc, hubs, lines_constrained_path, nodes_constrained_path, lines_dispatch, nodes_dispatch_path, subs_topology=[],by_description=True,grid2op_action_space=None):
    """
    This function verifies whether an action should be filtered based on expert-defined rules.

    Parameters:
    action_desc (dict): A dictionary containing the action description.
                         Expected keys include "description_unitaire", "content", and "VoltageLevelId".
    hubs (list): List of hub nodes in the graph.
    lines_constrained_path (list): List of line IDs that are part of the constrained path.
    nodes_constrained_path (list): List of nodes that are part of the constrained path.
    lines_dispatch (list): List of line IDs that are part of the dispatch path.
    nodes_dispatch_path (list): List of nodes that are part of the dispatch path.
    subs_topology (list): A list of sub-topologies involved in the action. Default is an empty list.

    Returns:
    tuple: A tuple containing two elements:
           - do_filter_action (bool): A flag indicating whether the action should be filtered out.
           - broken_rule (str): A string describing the rule that was broken, if any.
    """
    # in current implementation, we need "description_unitaire","content (with "set_bus") and "VoltageLevelId" fields in actions_desc
    # Identify the type of action based on the description
    action_type = identify_action_type(action_desc, by_description,grid2op_action_space)

    # Determine the localization of the action within the graph
    if "line" in action_type:
        grid2op_actions_set_bus = action_desc["content"]["set_bus"]
        lines = list(grid2op_actions_set_bus["lines_or_id"].keys()) + list(
            grid2op_actions_set_bus["lines_ex_id"].keys())
        localization = localize_line_action(lines, lines_constrained_path, lines_dispatch)
    else:
        action_subs = [action_desc["VoltageLevelId"]]
        localization = localize_coupling_action(action_subs, hubs, nodes_constrained_path, nodes_dispatch_path)

    # Check if the action violates any expert-defined rules
    do_filter_action, broken_rule = check_rules(action_type, localization, subs_topology)

    # If the action is filtered out and no specific rule is broken, print a check message
    if do_filter_action and broken_rule is None:
        print("check")

    return do_filter_action, broken_rule

def check_rho_reduction(obs, timestep, act_defaut, action, overload_ids,act_reco_maintenance, rho_tolerance=0.01):
    """
    This function checks if the action reduces the rho values of overloaded lines.
    is_rho_reduction=None
    Parameters:
    obs (Grid2opObservation): The initial observation of the grid.
    timestep (int): The timestep for the simulation.
    act_defaut (Grid2opAction): The default action to be applied.
    action (Grid2opAction): The action to be checked.
    overload_ids (list): List of IDs of the overloaded lines.
    rho_tolerance (float): Tolerance value to determine if a reduction in rho is significant. Default is 0.01.

    Returns:
    bool: True if the action results in a reduction of rho values for the overloaded lines, False otherwise.
    """
    # Initialize the flag to check if rho reduction is observed
    is_rho_reduction = None

    # Simulate the default action to get the initial rho values
    obs_defaut, reward, done, info = obs.simulate(act_defaut+act_reco_maintenance, time_step=timestep)
    rho_init = obs_defaut.rho[overload_ids]

    # Simulate the combined action (default action + additional action) to get the final rho values
    obs_simu_action, reward, done, info = obs.simulate(action + act_defaut+act_reco_maintenance, time_step=timestep)
    rho_final = obs_simu_action.rho[overload_ids]

    # Check if there are no exceptions in the simulation
    if len(info["exception"]) == 0:
        # Check if all final rho values are less than the initial rho values within the tolerance
        if np.all(rho_final + rho_tolerance < rho_init):
            is_rho_reduction = True
            print("We saw a reduction in rho from " + str(rho_init) + " to " + str(rho_final))
        else:
            is_rho_reduction = False

    return is_rho_reduction

def categorize_action_space(dict_action, hubs, lines_constrained_path, nodes_constrained_path, lines_dispatch,
                            nodes_dispatch_path, obs, timestep, defaut, action_space, overload_ids,lines_reco_maintenance, action_rule_preprocessing=True,by_description=True):
    """
    This function categorizes the action space based on expert-defined rules and verifies whether each action should be filtered out.
    actions_to_filter={}
    Parameters:
    dict_action (dict): A dictionary containing the action descriptions.
    hubs (list): List of hub nodes in the graph.
    lines_constrained_path (list): List of line IDs that are part of the constrained path.
    nodes_constrained_path (list): List of nodes that are part of the constrained path.
    lines_dispatch (list): List of line IDs that are part of the dispatch path.
    nodes_dispatch_path (list): List of nodes that are part of the dispatch path.
    obs (Grid2opObservation): The initial observation of the grid.
    timestep (int): The timestep for the simulation.
    defaut (str): The default line to be disconnected.
    action_space (Grid2opActionSpace): The action space for the environment.
    overload_ids (list): List of IDs of the overloaded lines.
    action_rule_preprocessing (bool): Flag to determine if action rule preprocessing should be applied. Default is True.

    Returns:
    tuple: A tuple containing two dictionaries:
           - actions_to_filter: Dictionary of actions that should be filtered out.
           - actions_unfiltered: Dictionary of actions that should not be filtered out.
    """
    actions_to_filter = {}
    actions_unfiltered = {}

    for action_id, action_desc in dict_action.items():
        # Determine the sub-topologies involved in the action
        action_subs = [action_desc["VoltageLevelId"]]
        subs_topology = [obs.sub_topology(np.where(obs.name_sub == sub_name)[0][0]) for sub_name in action_subs]

        # Verify if the action should be filtered based on expert-defined rules
        do_filter_action, broken_rule = verify_action(action_desc, hubs, lines_constrained_path, nodes_constrained_path,
                                                      lines_dispatch, nodes_dispatch_path, subs_topology,by_description,action_space)

        action_grid2op = action_desc["content"]

        if do_filter_action:
            # Create default and additional actions to simulate
            act_defaut = action_space({"set_bus": {"lines_ex_id": {defaut: -1}, "lines_or_id": {defaut: -1}}})
            action = action_space(action_grid2op)

            if action_rule_preprocessing:
                state = StateInfo()
                action = aux_prevent_asset_reconnection(obs, state, action)

            # Check if the action reduces the rho values of overloaded lines
            act_reco_maintenance = action_space(
                {"set_line_status": [(line_reco, 1) for line_reco in lines_reco_maintenance]})
            is_rho_reduction = check_rho_reduction(obs, timestep, act_defaut, action, overload_ids,act_reco_maintenance, rho_tolerance=0.01)

            if is_rho_reduction:
                print(action_desc["description_unitaire"])
                print(broken_rule)

            # Add the filtered action to the dictionary
            actions_to_filter[action_id] = {
                "description_unitaire": action_desc["description_unitaire"],
                "broken_rule": broken_rule,
                "is_rho_reduction": is_rho_reduction
            }
        else:
            # Add the unfiltered action to the dictionary
            actions_unfiltered[action_id] = {
                "description_unitaire": action_desc["description_unitaire"]
            }

    return actions_to_filter, actions_unfiltered

def make_overflow_graph_visualization(env_path, overflow_sim, g_overflow, obs_simu, save_folder, graph_file_name, lines_swapped,custom_layout=None, draw_only_significant_edges=True):
    """
    This function creates and saves a visualization of the overflow graph based on the given parameters.

    Parameters:
    env_path (str): The path to the environment folder.
    overflow_sim (Grid2opSimulation): The simulation object containing the overflow state.
    g_overflow (OverFlowGraph): The overflow graph object.
    obs_simu (Grid2opObservation): The observation containing the simulation state.
    save_folder (str): The folder where the visualization will be saved.
    graph_file_name (str): The name of the file to save the visualization.
    lines_swapped (list): List of line names that have swapped flows.
    draw_only_significant_edges (bool): Flag to determine if only significant edges should be drawn. Default is True.

    Returns:
    svg: The generated svg.
    """

    # Rescale factor for better layout visualization
    rescale_factor = 3#5  # Adjust this value to change the zoom level
    fontsize = 10
    node_thickness = 2
    shape_hub = "diamond"

    #####
    # Add voltage levels to the graph
    file_iidm = "grid.xiidm"
    network_file_path = os.path.join(env_path, file_iidm)
    n_zone = pp.network.load(network_file_path)
    df_volt = n_zone.get_voltage_levels()

    # Create a dictionary mapping sub-stations to their nominal voltages
    df_volt_dict = {sub: volt for sub, volt in zip(df_volt.index, df_volt.nominal_v)}

    # Define colors for different voltage levels
    voltage_colors = {
        400: "red", 225: "darkgreen", 90: "gold", 63: "purple", 20: "pink",
        24: "pink", 10: "pink", 15: "pink", 33: "pink"
    }

    # Set voltage level colors in the overflow graph
    g_overflow.set_voltage_level_color(df_volt_dict, voltage_colors)

    #####
    # Add node numbers to the graph
    number_nodal_dict = {
        sub_name: len(set(obs_simu.sub_topology(i)) - set([-1]))
        for i, sub_name in enumerate(obs_simu.name_sub)
    }
    g_overflow.set_electrical_node_number(number_nodal_dict)

    #######
    # Add hubs to the graph
    g_overflow.set_hubs_shape(g_distribution_graph.hubs, shape_hub=shape_hub)

    #######
    # Highlight limiting lines in the graph
    dict_significant_change_in_line_loading = {}

    ######
    # Highlight swapped flows in the graph
    g_overflow.highlight_swapped_flows(lines_swapped)

    # Identify assets to monitor based on rho values
    # But first check if load flow run in DC, because in that case we cannot really estimate the rho
    is_DC = obs_simu._obs_env._parameters.ENV_DC
    if is_DC:
        print("we ill not highlight the lines with high rho because the load flow is in DC and rho is not properly estimated in that case")
    else:
        ind_assets_to_monitor = np.where(overflow_sim.obs_linecut.rho >= 0.9)[0]

        lines_overloaded_ids = overflow_sim.ltc
        ind_assets_to_monitor = np.append(ind_assets_to_monitor, lines_overloaded_ids)
        line_loaded_of_interest = env.name_line[ind_assets_to_monitor]

        # Record significant changes in line loading before and after the simulation
        for ind_line, line_name in zip(ind_assets_to_monitor, line_loaded_of_interest):
            dict_significant_change_in_line_loading[line_name] = {
                "before": int(obs_simu.rho[ind_line] * 100),
                "after": int(overflow_sim.obs_linecut.rho[ind_line] * 100)
            }

        # Highlight significant line loading changes in the graph
        g_overflow.highlight_significant_line_loading(dict_significant_change_in_line_loading)

    #####
    # Generate SVG and save the visualization
    tmp_save_folder = os.path.join(save_folder, graph_file_name)
    svg = g_overflow.plot(
        custom_layout,
        save_folder=tmp_save_folder,
        fontsize=fontsize,
        without_gray_edges=draw_only_significant_edges,
        node_thickness=node_thickness,
        rescale_factor=rescale_factor
    )

    # Move and rename the saved PDF file
    pdf_files = glob.glob(f"{tmp_save_folder}/Base graph/*.pdf")
    file_path = os.path.join(save_folder, graph_file_name + ".pdf")
    shutil.move(pdf_files[0], file_path)
    shutil.rmtree(tmp_save_folder)

    print("Overflow graph visualization has been saved in: " + file_path)

    return svg

def save_to_json(data, output_file):
    """Saves extracted data to a json file."""
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

def check_simu_overloads(obs,line_defaut,lines_overloaded_ids,lines_reco_maintenance):
    has_converged=False
    has_lost_load=False
    act_deco_overloads=env.action_space({"set_line_status": [(env.name_line[line_id], -1) for line_id in lines_overloaded_ids]})
    act_deco_defaut=env.action_space({"set_line_status": [(line_defaut, -1)]})
    act_reco_maintenance = env.action_space(
        {"set_line_status": [(line_reco, 1) for line_reco in lines_reco_maintenance]})
    obs_simu_overloads, reward, done, info = obs.simulate(act_deco_overloads+act_deco_defaut+act_reco_maintenance, time_step=timestep)
    if len(info["exception"])!=0:
        print(f"error in simulation of all overloads : {[env.name_line[line_id] for line_id in lines_overloaded_ids]} : {info['exception']}")
    elif obs_simu_overloads.load_p.sum()+1<obs_simu.load_p.sum():
        print(f"lost loads in simulation of all overloads : {[env.name_line[line_id] for line_id in lines_overloaded_ids]} : {obs_simu_overloads.load_p.sum()} < {obs_simu.load_p.sum()}")
        has_converged=True
        has_lost_load=True
    else:
        has_converged=True
        has_lost_load=False
    return has_converged,has_lost_load



if __name__ == "__main__":

    date = datetime(2024, 11, 25)#datetime(2024, 11, 25)#datetime(2024, 12, 9)#datetime(2024, 12, 2)#datetime(2024, 8, 28)  # we choose a date for the chronic
    timestep = 14#14#13#22 #1 # 36
    line_defaut = "BEON L31CPVAN"#"MAGNYY633"#"BEON L31CPVAN"##AISERL31RONCI, P.SAOL31RONCI, AISERL31MAGNY, BEON L31CPVAN, "FRON5L31LOUHA"
    env_folder="./"
    env_name = "env_dijon_v2_assistant"
    env_path=os.path.join(env_folder,env_name)

    #User parameters
    draw_only_significant_edges=True#True#False
    use_grid_layout=True #True use geo layout, otherwise hierarchical automatic layout
    check_with_action_description=True#if False it will use the grid2op action object directly for checking the rules, if True it only uses the human description field of the action
    use_evaluation_config = False  # if false, use training config
    use_dc_load_flow=False #if problem of convergence of AC load flow because of too many iterations when disconnecting overloads, use DC load flow for building the overflow graph. We will still run the first load flow with N-1 disconnection with AC load flow to detect the overloads
    do_consolidate_graph=True #use False in case too many actions have been unfortunately filtered out which might indicate that the graph consolidation heuristic might not have been so relevant in that case

    #additional inputs
    action_space_folder="action_space"
    file_action_space_desc="actions_repas_most_frequent_topologies_revised.json"
    file_path = os.path.join(action_space_folder, file_action_space_desc)

    #load actions
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        dict_action = json.load(file)

    # make the environment
    # env = grid2op.make(env_name, backend=backend, n_busbar=6, param=p)
    if use_evaluation_config:
        env = make_grid2op_evaluation_env(env_folder, env_name)
    else:
        env = make_grid2op_training_env(env_folder, env_name)


    custom_layout=None
    if use_grid_layout:
        custom_layout = [env.grid_layout[sub] for sub in env.name_sub]

    chronics_name = list_all_chronics(env)
    print("chronics names are:")
    print(chronics_name)

    # we get the first observation for the chronic at the desired date
    path_chronic = [path for path in env.chronics_handler.real_data.subpaths if date.strftime('%Y%m%d') in path][0]
    obs = get_first_obs_on_chronic(date, env,path_thermal_limits=path_chronic)
    chronic_name = env.chronics_handler.get_name()



    # read non reconnectable lines
    lines_non_reconnectable = list(load_interesting_lines(path=path_chronic,file_name="non_reconnectable_lines.csv"))
    lines_should_not_reco_2024_and_beyond =DELETED_LINE_NAME

    #detect initially disconnected lines that might need to be reconnected
    maintenance_df = pd.DataFrame(env.chronics_handler.real_data.data.maintenance_handler.array, columns=obs.name_line)
    lines_in_maintenance_obs_start = list(maintenance_df.iloc[0][(maintenance_df.iloc[0])].index)
    reconnectabble_line_in_maintenance_at_start = list(
        set(lines_in_maintenance_obs_start) - set(lines_non_reconnectable))
    print(f"lines in maintenance at start: {lines_in_maintenance_obs_start}")
    print(f"lines in maintenance at start and reconnectable in scenario: {reconnectabble_line_in_maintenance_at_start}")

    #act to reconnect some initially disconnected lines ?
    do_reco_maintenance_at_t = ~maintenance_df[reconnectabble_line_in_maintenance_at_start].iloc[timestep]
    maintenance_to_reco_at_t = list(do_reco_maintenance_at_t[do_reco_maintenance_at_t].index)
    if len(maintenance_to_reco_at_t) != 0:
        print(f"reconnecting lines not in maintenance anymore {maintenance_to_reco_at_t} at timestep {timestep}")
    act_reco_maintenance = env.action_space(
        {"set_line_status": [(line_reco, 1) for line_reco in maintenance_to_reco_at_t]})

    # simulate contingency tp detect overloads
    act_deco_defaut = env.action_space({"set_line_status": [(line_defaut, -1)]})
    obs_simu, reward, done, info = obs.simulate(act_deco_defaut+act_reco_maintenance, time_step=timestep)
    if len(info["exception"])!=0:
        print(f"error in simulation of contingency : {line_defaut} : {info['exception']}")
        print("we cannot analyze the problem then and stop the computation here")
        sys.exit(0)#we stop here the execution of the script

    lines_overloaded_ids = [i for i, rho in enumerate(obs_simu.rho) if rho >= 1]

    non_connected_reconnectable_lines = [l_name for i,l_name in enumerate(env.name_line) if l_name not in lines_non_reconnectable+lines_should_not_reco_2024_and_beyond and not obs_simu.line_status[i]]

    ##########
    has_converged,has_lost_load=check_simu_overloads(obs,line_defaut, lines_overloaded_ids,maintenance_to_reco_at_t)

    #check if disconnecting overloads, either the largest or all, leads to disconnected components. Only keep the toughest one if that's the case
    lines_overloaded_ids_kept=identify_overload_lines_to_keep_overflow_graph_connected(obs_simu, lines_overloaded_ids)

    if lines_overloaded_ids_kept:
        #inspect other simu
        #act=act_deco_defaut
        #for ov_id in lines_overloaded_ids_kept[:1]:
        #    act+=env.action_space({"set_line_status": [(env.name_line[ov_id], -1)]})
#
        ###########
        if use_dc_load_flow:
            env_params = env.parameters
            env_params.ENV_DC = True
            if use_evaluation_config:
                env = make_grid2op_evaluation_env(env_folder, env_name,params=env_params)
            else:
                env = make_grid2op_training_env(env_folder, env_name,params=env_params)

            obs = get_first_obs_on_chronic(date, env)#reset the env and get first obs with DC load flow
            obs_simu, reward, done, info = obs.simulate(act_deco_defaut+act_reco_maintenance, time_step=timestep)
        ########"

        #check if diconnecting kept overloads still breaks the graph with loast loads
        if lines_overloaded_ids_kept!=lines_overloaded_ids:
            has_converged, has_lost_load = check_simu_overloads(obs,line_defaut, lines_overloaded_ids_kept,maintenance_to_reco_at_t)
            if not has_converged:
                print("this prevents us from building the overflow graph and the issue might not be solvable with topology except maybe with some line reconnections")
                sys.exit(0)  # we stop here the execution of the script

        inhibit_swapped_flow_reversion=True#Cancel the swapped edge direction for swapped flows (possibly not needed anymore given the new consolidate graph functions)
        df_of_g,overflow_sim,g_overflow,hubs, g_distribution_graph = build_overflow_graph(obs_simu, env.action_space, env.observation_space,
                                                                lines_overloaded_ids_kept, non_connected_reconnectable_lines,lines_non_reconnectable,
                                                                param_options_expertOp, timestep,do_consolidate_graph,inhibit_swapped_flow_reversion)

        ###########
        # make graph visualization and save
        graph_file_name = "Overflow_Graph_" + line_defaut +"_chronic_"+chronic_name+"_timestep_"+str(timestep)
        if use_grid_layout:
            graph_file_name += "_geo"
        else:
            graph_file_name += "_hierarchi"
        if draw_only_significant_edges:
            graph_file_name += "_only_signif_edges"
        else:
            graph_file_name += "_all_edges"
        if do_consolidate_graph:
            graph_file_name += "_consoli"
        else:
            graph_file_name += "_no_consoli"
        if use_dc_load_flow:
            graph_file_name += "_in_DC"
        save_folder="./Overflow_Graph"
        lines_swapped=list(df_of_g[df_of_g.new_flows_swapped].line_name)
        make_overflow_graph_visualization(env_path, overflow_sim, g_overflow, obs_simu,save_folder, graph_file_name,lines_swapped,custom_layout, draw_only_significant_edges)
        print("ok")

        ##########
        # get useful paths for action verification
        lines_constrained_path, nodes_constrained_path = g_distribution_graph.get_constrained_edges_nodes()
        #if lines_overloaded_ids_kept!=lines_overloaded_ids, check that all overloads are on constrained path
        if len(lines_overloaded_ids_kept)!=len(lines_overloaded_ids):
            missing_overload_constrained_path=set(env.name_line[lines_overloaded_ids])-set(lines_constrained_path)
            if len(missing_overload_constrained_path)!=0:
                print("WARNING: our overload graphs should be considered with caution as lines "+str(missing_overload_constrained_path)+" are not seen on the constrained path which would be more consistent")


        lines_dispatch, nodes_dispatch_path = g_distribution_graph.get_dispatch_edges_nodes()

        #########
        # check rules for each action
        actions_to_filter,actions_unfiltered=categorize_action_space(dict_action, hubs, lines_constrained_path, nodes_constrained_path, lines_dispatch,
                                nodes_dispatch_path, obs, timestep, line_defaut, env.action_space, lines_overloaded_ids,maintenance_to_reco_at_t,by_description=check_with_action_description)#here we check for all overloaded lines

        print("#################     NOW printing the actions that have been filtered out     #################")
        for action_id, action_content in actions_to_filter.items():
            print(action_content['description_unitaire'])
            print("    " + action_content['broken_rule'])
            print("\n")

        n_actions=len(dict_action.keys())
        n_actions_filtered=len(actions_to_filter.keys())
        n_actions_badly_filtered=len([id for id,act_filter_content in actions_to_filter.items() if act_filter_content["is_rho_reduction"]])

        print("#################     NOW printing the actions that have been UNFORTUNATELY filtered out     #################")
        for action_id, action_content in actions_to_filter.items():
            if action_content["is_rho_reduction"]:
                print(action_content['description_unitaire'])
                print("    " + action_content['broken_rule'])
                print("\n")

        print(str(n_actions_filtered)+" actions have been filtered out of "+str(n_actions))
        print(str(n_actions_badly_filtered) + " actions have been unfortunately filtered out of " + str(n_actions_filtered)+" since they showed a tendency to reduce a bit the overflow")


        save_to_json(actions_to_filter, "actions_to_filter_expert_rules.json")
        save_to_json(actions_unfiltered, "actions_unfiltered_expert_rules.json")

        if use_dc_load_flow:
            print("Warning: you have used the DC load flow, so results are more approximate")


    else:
        print("Overload breaks the grid apart, only load shedding actions are an option")