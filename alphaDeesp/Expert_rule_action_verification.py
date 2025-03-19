#!/usr/bin/python3
# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of ExpertOp4Grid, an expert system approach to solve flow congestions in power grids

#from make_evaluation_env import make_grid2op_evaluation_env
from make_training_env import make_grid2op_training_env
from load_evaluation_data import list_all_chronics, get_first_obs_on_chronic
from datetime import datetime
from data_utils import StateInfo

import configparser
import numpy as np
import os
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.graphsAndPaths import OverFlowGraph,Structured_Overload_Distribution_Graph
import networkx as nx
import pypowsybl as pp
from load_training_data import aux_prevent_asset_reconnection

import glob
import shutil
import json

param_options_expertOp={
    # 2 percent of the max overload flow
    "ThresholdReportOfLine": 0.2,#0.05,#
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

def inhibit_swapped_flows(df_of_g):
    df_of_g.loc[df_of_g.new_flows_swapped,"delta_flows"]=-df_of_g[df_of_g.new_flows_swapped]["delta_flows"]
    idx_or=df_of_g[df_of_g.new_flows_swapped]["idx_or"]
    df_of_g.loc[df_of_g.new_flows_swapped,"idx_or"] = df_of_g[df_of_g.new_flows_swapped]["idx_ex"]
    df_of_g.loc[df_of_g.new_flows_swapped, "idx_ex"]=idx_or
    # df_of_g[["new_flows_swapped"]]=False

    return df_of_g

def build_overflow_graph(obs_overloaded,action_space,observation_space,overloaded_line_ids,reconnectable_lines,param_options,timestep,inhibit_swapped_flow_reversion=True):
    """
    This function builds an overflow graph based on the given overloaded observation and parameters.
    sim = Grid2opSimulation(obs_overloaded, action_space, observation_space, param_options=param_options, debug=False,
    Parameters:
    obs_overloaded (Grid2opObservation): The observation containing overloaded lines.
    action_space (Grid2opActionSpace): The action space for the environment.
    observation_space (Grid2opObservationSpace): The observation space for the environment.
    overloaded_line_ids (list): List of IDs of the overloaded lines.
    reconnectable_lines (list): List of reconnectable lines.
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

    g_overflow.consolidate_graph(g_distribution_graph)

    g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)

    #get real hubs before we add disconected but reconnectable lines
    real_hubs = g_distribution_graph.get_hubs()

    #add disconected but reconnectable lines
    g_overflow.add_relevant_null_flow_lines_all_paths(g_distribution_graph, reconnectable_lines)
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_overflow.g)

    return df_of_g,overflow_sim,g_overflow,real_hubs,g_distribution_graph

def get_constrained_path(g_distribution_graph):
    """
    This function identifies the constrained path within the distribution graph.

    Parameters:
    g_distribution_graph (Structured_Overload_Distribution_Graph): The structured overload distribution graph.

    Returns:
    tuple: A tuple containing two lists:
           - edges_constrained_path: List of edges that are part of the constrained path.
           - nodes_constrained_path: List of nodes that are part of the constrained path.
    """
    constrained_path_object = g_distribution_graph.find_constrained_path()
    nodes_constrained_path = constrained_path_object.full_n_constrained_path()
    edges_constrained_path = []

    edge_names = nx.get_edge_attributes(g_distribution_graph.g_init, 'name')
    edges_constrained_path +=[edge_name for edge, edge_name in edge_names.items() if edge in constrained_path_object.amont_edges]
    edges_constrained_path +=[edge_name for edge, edge_name in edge_names.items() if edge in constrained_path_object.aval_edges]

    if type(constrained_path_object.constrained_edge) is list:
        edges_constrained_path +=[edge_name for edge, edge_name in edge_names.items() if edge in constrained_path_object.constrained_edge]
    else:
        edges_constrained_path.append([edge_name for edge, edge_name in edge_names.items() if edge==constrained_path_object.constrained_edge][0])

    return list(set(edges_constrained_path)),nodes_constrained_path

def get_dispatch_path(g_distribution_graph):
    """
    This function identifies the dispatch path within the distribution graph.

    Parameters:
    g_distribution_graph (Structured_Overload_Distribution_Graph): The structured overload distribution graph.

    Returns:
    tuple: A tuple containing two lists:
           - lines_redispatch: List of lines that are part of the dispatch path.
           - list_nodes_dispatch_path: List of nodes that are part of the dispatch path.
    """
    list_nodes_dispatch_path = list(set(g_distribution_graph.find_loops()["Path"].sum()))

    g_red = g_distribution_graph.g_only_red_components
    edge_names_red = nx.get_edge_attributes(g_red, 'name')

    lines_redispatch = [edge_name for edge, edge_name in edge_names_red.items() if (edge[0] in list_nodes_dispatch_path) and (edge[1] in list_nodes_dispatch_path)]

    return lines_redispatch, list_nodes_dispatch_path

def identify_action_type(actions_desc,by_description=True):
    type=None
    if by_description:
        dict_action = actions_desc["content"]["set_bus"]
        has_load = len(dict_action["loads_id"]) != 0
        has_line = (len(dict_action["lines_or_id"]) != 0) or (len(dict_action["lines_ex_id"]) != 0)

        description=actions_desc["description_unitaire"]
        if ("COUPL" in description or "TRO." in description) and "Ouverture" in description:
            type="open_coupling"
        elif ("COUPL" in description or "TRO." in description) and "Fermeture" in description:
            type="close_coupling"
        elif "Ouverture" in description:
            if has_load and has_line:
                type="open_line_load"
            elif has_line:
                type="open_line"
            else:
                type = "open_load"
        elif "Fermeture" in description:
            if has_load and has_line:
                type = "close_line_load"
            elif has_line:
                type = "close_line"
            else:
                type = "close_load"

    else:#by content directly
        pass

    return type

def localize_line_action(lines, lines_constrained_path,lines_dispatch):
    localization="out_of_graph"

    lines_intersect_constrained_path=set(lines).intersection(set(lines_constrained_path))
    lines_intersect_dispatch_lines=set(lines).intersection(set(lines_dispatch))

    if len(lines_intersect_constrained_path)!=0:
        localization="constrained_path"
    elif len(lines_intersect_dispatch_lines)!=0:
        localization = "dispatch_path"

    return localization

def localize_coupling_action(action_subs, hubs, nodes_constrained_path, nodes_dispatch_path):
    localization="out_of_graph"

    action_in_hubs=len(set(action_subs).intersection(set(hubs)))!=0
    action_in_constrained_path=len(set(action_subs).intersection(set(nodes_constrained_path)))!=0
    action_in_dispatch_path=len(set(action_subs).intersection(set(nodes_dispatch_path)))!=0

    if action_in_hubs:
        localization="hubs"
    elif action_in_constrained_path:
        localization="constrained_path"
    elif action_in_dispatch_path:
        localization = "dispatch_path"

    return localization

def check_rules(action_type,localization,subs_topology):
    do_filter_action=False
    broken_rule=None

    if ("load" not in action_type):#We don't filter actions that disconnect loads for now
        #check if topology in multi nodes
        is_topo_subs_one_node=np.all([len(set(sub_topo)-set([-1]))==1 for sub_topo in subs_topology])

        ###
        #expert rules to prevent
        out_of_graph=(localization=="out_of_graph")
        line_disconnection_dispatch_path=("line"in action_type) and ("open" in action_type) and (localization=="dispatch_path")
        line_reconnection_constrained_path=("line"in action_type) and ("close" in action_type) and (localization=="constrained_path")
        #filter node splitting only if the sub topology was initially one node
        node_splitting_dispatch_path=("coupling"in action_type) and ("open" in action_type) and (localization=="dispatch_path") and is_topo_subs_one_node
        node_merging_constrained_path = ("coupling" in action_type) and ("close" in action_type) and (
                    localization == "constrained_path")

        #check if rules are broken
        do_filter_action = out_of_graph or line_disconnection_dispatch_path or line_reconnection_constrained_path or node_splitting_dispatch_path or node_merging_constrained_path

        if out_of_graph:
            broken_rule = "No action out of the overflow graph"
        elif line_reconnection_constrained_path:
            broken_rule="No line reconnection on constrained path"
        elif line_disconnection_dispatch_path:
            broken_rule="No line disconnection on dispatch path"
        elif node_merging_constrained_path:
            broken_rule = "No node merging on constrained path"
        elif node_splitting_dispatch_path:
            broken_rule = "No node splitting on dispatch path"

        if do_filter_action and broken_rule is None:
            print("check")
    else:
        print("check")
    return do_filter_action,broken_rule

def verify_action(action_desc,hubs,lines_constrained_path, nodes_constrained_path,lines_dispatch, nodes_dispatch_path,subs_topology=[]):
    #in current implementation, we need "description_unitaire","content (with "set_bus") and "VoltageLevelId" fields in actions_desc
    action_type=identify_action_type(action_desc, by_description=True)

    if "line" in action_type:
        grid2op_actions_set_bus = action_desc["content"]["set_bus"]
        lines = list(grid2op_actions_set_bus["lines_or_id"].keys()) + list(
            grid2op_actions_set_bus["lines_ex_id"].keys())
        localization=localize_line_action(lines, lines_constrained_path,lines_dispatch)
    else:
        action_subs=[action_desc["VoltageLevelId"]]
        localization = localize_coupling_action(action_subs, hubs, nodes_constrained_path, nodes_dispatch_path)

    do_filter_action, broken_rule=check_rules(action_type,localization,subs_topology)
    if do_filter_action and broken_rule is None:
        print("check")
    return do_filter_action, broken_rule

def check_rho_reduction(obs,timestep,act_defaut,action,overload_ids,rho_tolerance=0.01):
    is_rho_reduction=None
    
    obs_defaut, reward, done, info = obs.simulate(act_defaut, time_step=timestep)
    rho_init = obs_defaut.rho[overload_ids]

    obs_simu_action, reward, done, info = obs.simulate(action + act_defaut, time_step=timestep)
    rho_final = obs_simu_action.rho[overload_ids]
    
    #1
    if len(info["exception"]) == 0:
        if np.all(rho_final+rho_tolerance<rho_init):
            is_rho_reduction = True
            print("we saw a reduction a rho from "+str(rho_init)+" to "+str(rho_final))
        else:
            is_rho_reduction=False
    return is_rho_reduction

def categorize_action_space(dict_action,hubs, lines_constrained_path, nodes_constrained_path, lines_dispatch,
                  nodes_dispatch_path,obs,timestep,defaut,action_space,overload_ids,action_rule_preprocessing=True):
    actions_to_filter={}
    actions_unfiltered={}
    for action_id,action_desc in dict_action.items():

        #to cjheck later for coupling actions if subs already at multiple node or not. Case of PYMONP3 for instance
        action_subs = [action_desc["VoltageLevelId"]]
        subs_topology = [obs.sub_topology(np.where(obs.name_sub==sub_name)[0][0]) for sub_name in action_subs]

        do_filter_action, broken_rule=verify_action(action_desc, hubs, lines_constrained_path, nodes_constrained_path, lines_dispatch,
                  nodes_dispatch_path,subs_topology)

        action_grid2op=action_desc["content"]

        if do_filter_action:
            print(action_desc["description_unitaire"])
            print(broken_rule)


            act_defaut=action_space({"set_bus": {"lines_ex_id":{defaut:-1}, "lines_or_id":{defaut:-1}}})
            action=action_space(action_grid2op)
            if action_rule_preprocessing:
                state = StateInfo()
                action=aux_prevent_asset_reconnection(obs, state,action)

            is_rho_reduction=check_rho_reduction(obs, timestep, act_defaut, action, overload_ids, rho_tolerance=0.01)
            actions_to_filter[action_id] = {"description_unitaire": action_desc["description_unitaire"],
                                           "broken_rule": broken_rule,"is_rho_reduction":is_rho_reduction}
        else:
            actions_unfiltered[action_id]={"description_unitaire": action_desc["description_unitaire"]}

    return actions_to_filter,actions_unfiltered

def make_overflow_graph_visualization(env_path,overflow_sim,g_overflow,obs_simu,save_folder,graph_file_name,lines_swapped,draw_only_significant_edges=True):

    rescale_factor = 5  # for better layout, you can play with it to change the zoom level
    fontsize = 10
    node_thickness = 2
    shape_hub = "diamond"

    #####
    # add voltage levels
    file_iidm = "grid.xiidm"
    network_file_path = os.path.join(env_path, file_iidm)  # pf_20240711T1450Z_20240711T1450Z
    n_zone = pp.network.load(network_file_path)
    df_volt = n_zone.get_voltage_levels()

    df_volt_dict = {sub: volt for sub, volt in zip(df_volt.index, df_volt.nominal_v)}

    voltage_colors = {400: "red", 225: "darkgreen", 90: "gold", 63: "purple", 20: "pink", 24: "pink", 10: "pink",
                      15: "pink", 33: "pink" }  # [400., 225.,  63.,  24.,  20.,  33.,  10.]

    g_overflow.set_voltage_level_color(df_volt_dict,voltage_colors)

    #####
    # add node number
    number_nodal_dict = {sub_name: len(set(obs_simu.sub_topology(i)) - set([-1])) for i, sub_name in
                         enumerate(obs_simu.name_sub)}
    g_overflow.set_electrical_node_number(number_nodal_dict)

    #######
    # add hubs
    g_overflow.set_hubs_shape(g_distribution_graph.hubs, shape_hub=shape_hub)

    #######
    # highlight limiting lines
    dict_significant_change_in_line_loading = {}

    ######
    # show swapped
    g_overflow.highlight_swapped_flows(lines_swapped)

    ind_assets_to_monitor = np.where(overflow_sim.obs_linecut.rho >= 0.9)[0]

    lines_overloaded_ids=overflow_sim.ltc
    ind_assets_to_monitor = np.append(ind_assets_to_monitor, lines_overloaded_ids)
    line_loaded_of_interest = env.name_line[ind_assets_to_monitor]

    for ind_line, line_name in zip(ind_assets_to_monitor, line_loaded_of_interest):
        dict_significant_change_in_line_loading[line_name] = {"before": int(obs_simu.rho[ind_line] * 100),
                                                              "after": int(overflow_sim.obs_linecut.rho[ind_line] * 100)}

    g_overflow.highlight_significant_line_loading(dict_significant_change_in_line_loading)

    #####
    #make svg and save visualization
    tmp_save_folder=os.path.join(save_folder,graph_file_name)
    svg = g_overflow.plot(None, save_folder=tmp_save_folder, fontsize=fontsize, without_gray_edges=draw_only_significant_edges, node_thickness=node_thickness)

    #file saved as pdf in save_folder/Base graph/*.pdf
    pdf_files = glob.glob(f"{tmp_save_folder}/Base graph/*.pdf")
    # Move and rename the file
    file_path=os.path.join(save_folder,graph_file_name+".pdf")
    shutil.move(pdf_files[0], file_path)
    shutil.rmtree(tmp_save_folder)

    print("overFlow graph visualization has been saved in: "+file_path)

    return svg

def save_to_json(data, output_file):
    """Saves extracted data to a json file."""
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

#################################
### TEST Section
def test_overflow_graph_construction():
    """
    This function tests the construction of the overflow graph and the identification of constrained and dispatch paths.
    """
    date = datetime(2024, 8, 28)  # we choose a date for the chronic
    timestep=1#36
    line_defaut="P.SAOL31RONCI"#"FRON5L31LOUHA"
    env_name = "env_dijon_v2_assistant"
    non_connected_reconnectable_lines = ['BOISSL61GEN.P', 'CHALOL31LOUHA', 'CRENEL71VIELM', 'CURTIL61ZCUR5',
                                         'GEN.PL73VIELM',
                                         'P.SAOL31RONCI',
                                         'PYMONL61VOUGL', 'BUGEYY715', 'CPVANY632', 'GEN.PY762', 'PYMONY632']

    #env = grid2op.make(env_name, backend=backend, n_busbar=6, param=p)
    env = make_grid2op_training_env(".", env_name)#make_grid2op_evaluation_env(".", env_name)

    # make the environment
    chronics_name = list_all_chronics(env)
    print("chronics names are:")
    print(chronics_name)

    # we get the first observation for the chronic at the desired date
    obs = get_first_obs_on_chronic(date, env)

    act_deco_defaut=env.action_space({"set_line_status": [(line_defaut, -1)]})



    obs_simu, reward, done, info=obs.simulate(act_deco_defaut,time_step=timestep)

    lines_overloaded_ids=[i for i,rho in enumerate(obs_simu.rho) if rho>=1]

    param_options_test={
        # 2 percent of the max overload flow
        "ThresholdReportOfLine": 0.2,  # 0.05,#
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
    df_of_g, overflow_sim,g_overflow,hubs,g_distribution_graph=build_overflow_graph(obs_simu, env.action_space, env.observation_space, lines_overloaded_ids,non_connected_reconnectable_lines, param_options_test, timestep)

    ##########
    # get useful paths for action verification

    lines_constrained_path, nodes_constrained_path=get_constrained_path(g_distribution_graph)

    lines_redispatch,list_nodes_dispatch_path=get_dispatch_path(g_distribution_graph)

    ############
    # Pour tests
    list_nodes_constrained_path_test=['NAVILP3','CPVANP6','CPVANP3','CHALOP6','GROSNP6', '1GROSP7',
                                      'GROSNP7', 'VIELMP7', 'H.PAUP7', 'SSV.OP7', 'ZCUR5P6', 'H.PAUP6', '2H.PAP7',
                                      'COUCHP6', 'VIELMP6', '1VIELP7', 'COMMUP6', 'ZMAGNP6', 'C.REGP6', 'BEON P3', 'P.SAOP3']

    list_lines_contrained_path_test=['GROSNY761','COMMUL61VIELM', 'GROSNY771', 'COUCHL61CPVAN', 'VIELMY771', 'VIELMY763', 'GROSNY762',
                                     'H.PAUL61ZCUR5', 'VIELMY762', 'CPVANY632', 'GROSNL61ZCUR5', 'C.REGL61VIELM', 'H.PAUL71VIELM',
                                     'H.PAUY762', 'CPVANY633', 'C.REGL62VIELM', 'CHALOL62GROSN', 'CHALOL61CPVAN', 'C.REGL61ZMAGN',
                                     'COMMUL61H.PAU', 'CHALOL61GROSN', 'GROSNL71VIELM', 'GROSNL71SSV.O', 'CPVANL61ZMAGN', 'COUCHL61VIELM',
                                     'VIELMY761', 'BEON L31P.SAO','BEON L31CPVAN', 'H.PAUY772', 'CPVANY631','NAVILL31P.SAO']

    list_nodes_dispatch_path_test=['1GEN.P7', 'BOISSP6', 'C.FOUP3', 'CHALOP3', 'CHALOP6', 'CIZE P6', 'CPVANP6', 'CREYSP7', 'CUISEP3',
     'FLEYRP6', 'FRON5P3', 'G.CHEP3', 'GEN.PP6', 'GEN.PP7', 'GROSNP6', 'H.PAUP7', 'IZERNP6', 'LOUHAP3', 'MACONP6','MERVAP3',
     'NAVILP3', 'PYMONP3', 'PYMONP6', 'SAISSP3', 'SSUSUP3', 'SSV.OP7', 'VIELMP7', 'VOUGLP3', 'VOUGLP6','ZJOUXP6']

    list_lines_redispatch_path_test=['GEN.PY761', 'GEN.PL61IZERN', 'GEN.PL61VOUGL', 'GEN.PL73VIELM', 'GEN.PY771', 'GEN.PY762', 'GROSNL61MACON',
     'GEN.PL71VIELM', 'GEN.PL72VIELM','H.PAUL71SSV.O', 'BOISSL61GEN.P', 'CHALOL31LOUHA', 'CHALOY633', 'CHALOY631', 'CHALOY632',
     'CIZE L61FLEYR', 'CPVANL61PYMON', 'CREYSL71GEN.P', 'CREYSL72GEN.P','CUISEL31G.CHE', 'FLEYRL61VOUGL', 'FRON5L31LOUHA', 'FRON5L31G.CHE', 'CIZE L61IZERN', 'LOUHAL31SSUSU',
     'MACONL61ZJOUX', 'C.FOUL31MERVA', 'LOUHAL31PYMON', 'PYMONL61VOUGL','PYMONY632', 'PYMONL31SAISS', 'MERVAL31SSUSU',
     'CREYSL71SSV.O', 'CREYSL72SSV.O', 'CUISEL31VOUGL', 'SAISSL31VOUGL','VOUGLY632', 'VOUGLY631', 'BOISSL61ZJOUX','C.FOUL31NAVIL']

    list_hubs_test=[ 'VIELMP7', 'H.PAUP7', 'SSV.OP7','NAVILP3']#[ 'CPVANP6', 'CHALOP6', 'GROSNP6', 'VIELMP7', 'H.PAUP7', 'SSV.OP7','NAVILP3']#'P.SAOP3',

    assert(set(list_nodes_constrained_path_test)==set(nodes_constrained_path))
    assert (set(list_lines_contrained_path_test) == set(lines_constrained_path))
    assert (set(list_nodes_dispatch_path_test) == set(list_nodes_dispatch_path))
    assert (set(list_lines_redispatch_path_test) == set(lines_redispatch))
    assert (set(list_hubs_test) == set(hubs))

def test_overflow_graph_actions_filtered():

    date = datetime(2024, 8, 28)  # we choose a date for the chronic
    timestep = 1  # 36
    line_defaut = "P.SAOL31RONCI"  # "FRON5L31LOUHA"
    env_folder="./"
    env_name = "env_dijon_v2_assistant"

    non_connected_reconnectable_lines = ['BOISSL61GEN.P', 'CHALOL31LOUHA', 'CRENEL71VIELM', 'CURTIL61ZCUR5',
                                         'GEN.PL73VIELM',
                                         'P.SAOL31RONCI',
                                         'PYMONL61VOUGL', 'BUGEYY715', 'CPVANY632', 'GEN.PY762', 'PYMONY632']

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
    env = make_grid2op_training_env(env_folder, env_name)  # make_grid2op_evaluation_env(".", env_name)

    chronics_name = list_all_chronics(env)
    print("chronics names are:")
    print(chronics_name)

    # we get the first observation for the chronic at the desired date
    obs = get_first_obs_on_chronic(date, env)

    act_deco_defaut = env.action_space({"set_line_status": [(line_defaut, -1)]})

    obs_simu, reward, done, info = obs.simulate(act_deco_defaut, time_step=timestep)

    inhibit_swapped_flow_reversion=True#Cancel the swapped edge direction for swapped flows (possibly not needed anymore given the new consolidate graph functions)
    lines_overloaded_ids = [i for i, rho in enumerate(obs_simu.rho) if rho >= 1]
    df_of_g,overflow_sim,g_overflow,hubs, g_distribution_graph = build_overflow_graph(obs_simu, env.action_space, env.observation_space,
                                                            lines_overloaded_ids, non_connected_reconnectable_lines,
                                                            param_options_expertOp, timestep,inhibit_swapped_flow_reversion)

    ##########
    # get useful paths for action verification
    lines_constrained_path, nodes_constrained_path = get_constrained_path(g_distribution_graph)

    lines_dispatch, nodes_dispatch_path = get_dispatch_path(g_distribution_graph)

    #########
    # check rules for each action
    actions_to_filter,actions_unfiltered=categorize_action_space(dict_action, hubs, lines_constrained_path, nodes_constrained_path, lines_dispatch,
                            nodes_dispatch_path, obs, timestep, line_defaut, env.action_space, lines_overloaded_ids)

    n_actions=len(dict_action.keys())
    n_actions_filtered=len(actions_to_filter.keys())
    n_actions_unfiltered = len(actions_unfiltered.keys())
    n_actions_badly_filtered=len([id for id,act_filter_content in actions_to_filter.items() if act_filter_content["is_rho_reduction"]])

    #could also directly compare to saved dictionnaries "actions_to_filter_expert_rules.json" and "actions_unfiltered_expert_rules.json"
    assert(n_actions == 102)
    assert(n_actions_filtered == 56)
    assert (n_actions_unfiltered == 46)
    assert(n_actions_badly_filtered == 1) # Ouverture OC 'MAGNY3TR633 DJ_OC' dans le poste 'MAGNYP3'

def test_action_type_open_coupling():
    actions_desc={
        "description_unitaire": " Ouverture OC 'VOUGL6COUPL DJ_OC' dans le poste 'VOUGLP6'",
        "content": {
            "set_bus": {
                "lines_or_id": {
                    "VOUGLY612": 1,
                    "VOUGLY631": 1,
                    "VOUGLY632": 2
                },
                "lines_ex_id": {
                    "FLEYRL61VOUGL": 2,
                    "GEN.PL61VOUGL": 1,
                    "PYMONL61VOUGL": 2
                },
                "loads_id": {},
                "generators_id": {},
                "shunts_id": {}
            }
        },
        "VoltageLevelId": "VOUGLP6"
    }

    action_type=identify_action_type(actions_desc, by_description=True)
    assert(action_type=="open_coupling")

def test_action_type_open_line():
    actions_desc={
        "description_unitaire": "Ouverture OC 'PYMON3TR632 DJ_OC' dans le poste 'PYMONP3'",
        "content": {
            "set_bus": {
                "lines_or_id": {},
                "lines_ex_id": {
                    "PYMONY632": -1
                },
                "loads_id": {},
                "generators_id": {},
                "shunts_id": {}
            }
        },
        "VoltageLevelId": "PYMONP3"
    }

    action_type=identify_action_type(actions_desc, by_description=True)
    assert(action_type=="open_line")

def test_action_type_open_line_load():
    actions_desc={
        "description_unitaire": " Ouverture OC 'GEN.P6CHAV6.1 DJ_OC' dans le poste 'GEN.PP6'\n- Ouverture OC 'GEN.P6AT762 DJ_OC' dans le poste 'GEN.PP6'",
        "content": {
            "set_bus": {
                "lines_or_id": {},
                "lines_ex_id": {
                    "GEN.PY762": -1
                },
                "loads_id": {
                    "CHAV6L61GEN.P":-1
                },
                "generators_id": {},
                "shunts_id": {}
            }
        },
        "VoltageLevelId": "GEN.PP6"
    }

    action_type=identify_action_type(actions_desc, by_description=True)
    assert(action_type=="open_line_load")

def test_action_type_close_line():
    actions_desc={
        "description_unitaire": "Fermeture OC 'PYMON6CPVAN.1 DJ_OC' dans le poste 'PYMONP6'(reconnection sur noeuds 1 aux 2 postes extremites)",
        "content": {
            "set_bus": {
                "lines_or_id": {
                    "CPVANL61PYMON": 1
                },
                "lines_ex_id": {
                    "CPVANL61PYMON": 1
                },
                "loads_id": {},
                "generators_id": {},
                "shunts_id": {}
            }
        },
        "VoltageLevelId": "PYMONP6"
    }

    action_type=identify_action_type(actions_desc, by_description=True)
    assert(action_type=="close_line")

def test_action_type_close_coupling():
    actions_desc = {
        "description_unitaire": "Fermeture OC 'CPVAN3COUPL DJ_OC' dans le poste 'CPVANP3'",
        "content": {
            "set_bus": {
                "lines_or_id": {
                    "CPVANL31RIBAU": 1
                },
                "lines_ex_id": {
                    "BEON L31CPVAN": 1,
                    "CPVANY631": 1,
                    "CPVANY632": 1,
                    "CPVANY633": 1
                },
                "loads_id": {
                    "ARBOIL31CPVAN": 1,
                    "BREVAL31CPVAN": 1,
                    "CPDIVL32CPVAN": 1,
                    "CPVANL31MESNA": 1,
                    "CPVANL31ZBRE6": 1,
                    "CPVAN3TR312": 1,
                    "CPVAN3TR311": 1
                },
                "shunts_id": {},
                "generators_id": {}
            }
        },
        "VoltageLevelId": "CPVANP3"
    }

    action_type = identify_action_type(actions_desc, by_description=True)
    assert (action_type == "close_coupling")

def test_action_types():
    test_action_type_close_line()
    test_action_type_open_line()
    test_action_type_open_coupling()
    test_action_type_close_coupling()
    test_action_type_open_line_load()

def test_no_broken_rule_multi_node_dispatch_path():
    action_type="open_coupling"
    localization="dispatch_path"
    subs_topology=[[1,1,2,2,2,1]]#topology already in multi nodes
    do_filter_action, broken_rule=check_rules(action_type, localization, subs_topology)

    assert(do_filter_action==False)
    assert (broken_rule is None)

def test_broken_rule_open_line_dispatch_path():
    action_type="open_line"
    localization="dispatch_path"
    subs_topology=[]#topology already in multi nodes
    do_filter_action, broken_rule=check_rules(action_type, localization, subs_topology)

    assert(do_filter_action)
    assert (broken_rule=="No line disconnection on dispatch path")

def test_broken_rule_close_line_constrained_path():
    action_type="close_line"
    localization="constrained_path"
    subs_topology=[]#topology already in multi nodes
    do_filter_action, broken_rule=check_rules(action_type, localization, subs_topology)

    assert(do_filter_action)
    assert (broken_rule=="No line reconnection on constrained path")

def test_broken_rule_close_coupling_constrained_path():
    action_type="close_coupling"
    localization="constrained_path"
    subs_topology=[]#topology already in multi nodes
    do_filter_action, broken_rule=check_rules(action_type, localization, subs_topology)

    assert(do_filter_action)
    assert (broken_rule=="No node merging on constrained path")

def test_broken_rule_open_coupling_dispatch_path():
    action_type="open_coupling"
    localization="dispatch_path"
    subs_topology=[[1,1,1,1]]#topology in one node
    do_filter_action, broken_rule=check_rules(action_type, localization, subs_topology)

    assert(do_filter_action)
    assert (broken_rule=="No node splitting on dispatch path")

def test_load_action_no_filter():
    action_type = "open_line_load"
    localization = ""
    subs_topology = []

    do_filter_action, broken_rule = check_rules(action_type, localization, subs_topology)

    assert(do_filter_action==False)
    assert (broken_rule is None)

def test_rules():
    test_no_broken_rule_multi_node_dispatch_path()
    test_broken_rule_open_line_dispatch_path()
    test_broken_rule_close_line_constrained_path()
    test_broken_rule_close_coupling_constrained_path()
    test_broken_rule_open_coupling_dispatch_path()
    test_load_action_no_filter()

if __name__ == "__main__":
    #test_overflow_graph_construction()
    test_overflow_graph_actions_filtered()
    test_action_types()
    test_rules()

    date = datetime(2024, 8, 28)  # we choose a date for the chronic
    timestep = 1  # 36
    line_defaut = "P.SAOL31RONCI"  # "FRON5L31LOUHA"
    env_folder="./"
    env_name = "env_dijon_v2_assistant"
    env_path=os.path.join(env_folder,env_name)
    draw_only_significant_edges=True#True#False

    non_connected_reconnectable_lines = ['BOISSL61GEN.P', 'CHALOL31LOUHA', 'CRENEL71VIELM', 'CURTIL61ZCUR5',
                                         'GEN.PL73VIELM',
                                         'P.SAOL31RONCI',
                                         'PYMONL61VOUGL', 'BUGEYY715', 'CPVANY632', 'GEN.PY762', 'PYMONY632']

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
    env = make_grid2op_training_env(env_folder, env_name)  # make_grid2op_evaluation_env(".", env_name)

    chronics_name = list_all_chronics(env)
    print("chronics names are:")
    print(chronics_name)

    # we get the first observation for the chronic at the desired date
    obs = get_first_obs_on_chronic(date, env)
    chronic_name = env.chronics_handler.get_name()

    act_deco_defaut = env.action_space({"set_line_status": [(line_defaut, -1)]})

    obs_simu, reward, done, info = obs.simulate(act_deco_defaut, time_step=timestep)

    inhibit_swapped_flow_reversion=True#Cancel the swapped edge direction for swapped flows (possibly not needed anymore given the new consolidate graph functions)
    lines_overloaded_ids = [i for i, rho in enumerate(obs_simu.rho) if rho >= 1]
    df_of_g,overflow_sim,g_overflow,hubs, g_distribution_graph = build_overflow_graph(obs_simu, env.action_space, env.observation_space,
                                                            lines_overloaded_ids, non_connected_reconnectable_lines,
                                                            param_options_expertOp, timestep,inhibit_swapped_flow_reversion)

    ##########
    # get useful paths for action verification
    lines_constrained_path, nodes_constrained_path = get_constrained_path(g_distribution_graph)

    lines_dispatch, nodes_dispatch_path = get_dispatch_path(g_distribution_graph)

    #########
    # check rules for each action
    actions_to_filter,actions_unfiltered=categorize_action_space(dict_action, hubs, lines_constrained_path, nodes_constrained_path, lines_dispatch,
                            nodes_dispatch_path, obs, timestep, line_defaut, env.action_space, lines_overloaded_ids)

    n_actions=len(dict_action.keys())
    n_actions_filtered=len(actions_to_filter.keys())
    n_actions_badly_filtered=len([id for id,act_filter_content in actions_to_filter.items() if act_filter_content["is_rho_reduction"]])

    print(str(n_actions_filtered)+" actions have been filtered out of "+str(n_actions))
    print(str(n_actions_badly_filtered) + " actions have been unfortunately filtered out of " + str(n_actions_filtered)+" since they showed a tendency to reduce a bit the overflow")

    save_to_json(actions_to_filter, "actions_to_filter_expert_rules.json")
    save_to_json(actions_unfiltered, "actions_unfiltered_expert_rules.json")

    ###########
    # make graph visualization and save
    graph_file_name = "Overflow_Graph_" + line_defaut +"_chronic_"+chronic_name+"_timestep_"+str(timestep)
    save_folder="./Overflow_Graph"
    lines_swapped=list(df_of_g[df_of_g.new_flows_swapped].line_name)
    make_overflow_graph_visualization(env_path, overflow_sim, g_overflow, obs_simu,save_folder, graph_file_name,lines_swapped,draw_only_significant_edges)
    print("ok")