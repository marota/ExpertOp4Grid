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

import os
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.graphsAndPaths import OverFlowGraph,Structured_Overload_Distribution_Graph

import json

from Expert_rule_action_verification import build_overflow_graph, categorize_action_space, identify_action_type, check_rules, identify_grid2op_action_type, load_interesting_lines
from packaging.version import Version as version_packaging
from importlib.metadata import version

#: name of the powerline that are now removed (non existant)
#: but are in the environment because we need them when we will use
#: historical dataset
#Note: reimporté directement ici pour éviter les problèmes de dépendances, en effet DELETED_LINE_NAME changeait de valeur après l'éxécution de categorize_action_space dans test_overflow_graph_actions_filtered
DELETED_LINE_NAME = ['BXNE L32BXNE5', 'BXNE5L32MTAGN', 'BXNE5L32CORGO', 'BXNE5L31MTAGN']

EXOP_MIN_VERSION = version_packaging("0.2.6")
if version_packaging(version("expertop4grid")) < EXOP_MIN_VERSION:
    raise RuntimeError(f"Incompatible version found for expertOp4Grid, make sureit is >= {EXOP_MIN_VERSION}")
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
    inhibit_swapped_flow_reversion=False
    do_consolidate_graph=True
    lines_non_reconnectable = []  # TO DELETE, just for test
    df_of_g, overflow_sim,g_overflow,hubs,g_distribution_graph=build_overflow_graph(obs_simu, env.action_space, env.observation_space, lines_overloaded_ids,
                                                                                    non_connected_reconnectable_lines,lines_non_reconnectable, param_options_test, timestep,do_consolidate_graph,inhibit_swapped_flow_reversion)

    ##########
    # get useful paths for action verification

    lines_constrained_path, nodes_constrained_path = g_distribution_graph.get_constrained_edges_nodes()

    lines_redispatch,list_nodes_dispatch_path = g_distribution_graph.get_dispatch_edges_nodes()

    ############
    # Pour tests
    list_nodes_constrained_path_test=['NAVILP3','CPVANP6','CPVANP3','CHALOP6','GROSNP6', '1GROSP7',
                                      'GROSNP7', 'VIELMP7', 'H.PAUP7', 'SSV.OP7', 'ZCUR5P6', 'H.PAUP6', '2H.PAP7',
                                      'COUCHP6', 'VIELMP6', '1VIELP7', 'COMMUP6', 'ZMAGNP6', 'C.REGP6', 'BEON P3', 'P.SAOP3']

    list_lines_contrained_path_test=['GROSNY761','COMMUL61VIELM', 'GROSNY771', 'COUCHL61CPVAN', 'VIELMY771', 'VIELMY763', 'GROSNY762',
                                     'H.PAUL61ZCUR5', 'VIELMY762', 'CPVANY632', 'GROSNL61ZCUR5', 'C.REGL61VIELM', 'H.PAUL71VIELM',
                                     'H.PAUY762', 'CPVANY633', 'C.REGL62VIELM', 'CHALOL62GROSN', 'CHALOL61CPVAN', 'C.REGL61ZMAGN',
                                     'COMMUL61H.PAU', 'CHALOL61GROSN', 'GROSNL71SSV.O', 'CPVANL61ZMAGN', 'COUCHL61VIELM',
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

def test_overflow_graph_actions_filtered(check_with_action_description=True):
    """
    This function tests the filtering of actions based on the overflow graph and the identification of constrained and dispatch paths.
    It verifies that the actions are correctly categorized into filtered and unfiltered actions based on the expert rules.

    The test involves the following steps:
    1. Setting up the environment and loading the necessary data.
    2. Simulating the environment to get the initial observation.
    3. Identifying overloaded lines and building the overflow graph.
    4. Extracting constrained and dispatch paths from the overflow graph.
    5. Categorizing the actions based on the expert rules.
    6. Asserting that the number of actions and their categorization match the expected values.

    The test ensures that the expert rules are correctly applied to filter out inappropriate actions.
    """
    date = datetime(2024, 8, 28)  # we choose a date for the chronic
    timestep = 1  # 36
    line_defaut = "P.SAOL31RONCI"  # "FRON5L31LOUHA"
    env_folder = "./"
    env_name = "env_dijon_v2_assistant"

    action_space_folder = "action_space"
    file_action_space_desc = "actions_repas_most_frequent_topologies_revised.json"
    file_path = os.path.join(action_space_folder, file_action_space_desc)

    # Load actions
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        dict_action = json.load(file)

    # Make the environment
    env = make_grid2op_training_env(env_folder, env_name)

    chronics_name = list_all_chronics(env)
    print("chronics names are:")
    print(chronics_name)

    # Get the first observation for the chronic at the desired date
    obs = get_first_obs_on_chronic(date, env)

    # read non reconnectable lines
    path_chronic = [path for path in env.chronics_handler.real_data.subpaths if date.strftime('%Y%m%d') in path][0]
    lines_non_reconnectable = list(load_interesting_lines(path=path_chronic,file_name="non_reconnectable_lines.csv"))
    lines_should_not_reco_2024_and_beyond =DELETED_LINE_NAME


    # simulate contingency tp detect overloads
    act_deco_defaut = env.action_space({"set_line_status": [(line_defaut, -1)]})

    obs_simu, reward, done, info = obs.simulate(act_deco_defaut, time_step=timestep)

    non_connected_reconnectable_lines = [l_name for i,l_name in enumerate(env.name_line)
                                         if l_name not in lines_non_reconnectable+lines_should_not_reco_2024_and_beyond and not obs_simu.line_status[i]]
    param_options_test = {
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

    inhibit_swapped_flow_reversion = True  # Cancel the swapped edge direction for swapped flows (possibly not needed anymore given the new consolidate graph functions)
    lines_overloaded_ids = [i for i, rho in enumerate(obs_simu.rho) if rho >= 1]
    do_consolidate_graph = True
    lines_non_reconnectable=[]
    df_of_g, overflow_sim, g_overflow, hubs, g_distribution_graph = build_overflow_graph(obs_simu, env.action_space, env.observation_space,
                                                            lines_overloaded_ids, non_connected_reconnectable_lines,lines_non_reconnectable,
                                                            param_options_test, timestep,do_consolidate_graph, inhibit_swapped_flow_reversion)

    ##########
    # Get useful paths for action verification
    lines_constrained_path, nodes_constrained_path = g_distribution_graph.get_constrained_edges_nodes()

    lines_dispatch, nodes_dispatch_path = g_distribution_graph.get_dispatch_edges_nodes()

    #########
    # Check rules for each action
    lines_reco_maintenance=[]
    actions_to_filter, actions_unfiltered = categorize_action_space(dict_action, hubs, lines_constrained_path, nodes_constrained_path, lines_dispatch,
                            nodes_dispatch_path, obs, timestep, line_defaut, env.action_space, lines_overloaded_ids,lines_reco_maintenance,by_description=check_with_action_description)

    n_actions = len(dict_action.keys())
    n_actions_filtered = len(actions_to_filter.keys())
    n_actions_unfiltered = len(actions_unfiltered.keys())
    n_actions_badly_filtered = len([id for id, act_filter_content in actions_to_filter.items() if act_filter_content["is_rho_reduction"]])

    # Could also directly compare to saved dictionaries "actions_to_filter_expert_rules.json" and "actions_unfiltered_expert_rules.json"
    assert(n_actions == 102)
    assert(n_actions_filtered == 58)
    assert(n_actions_unfiltered == n_actions - n_actions_filtered)
    assert(n_actions_badly_filtered == 1)  # Opening OC 'MAGNY3TR633 DJ_OC' in the substation 'MAGNYP3'. This action is filtered because of the significant delta flow threshold.
    # If "ThresholdReportOfLine" is reduced from 0.2 to 0.05, then this action is not filtered anymore, and everything works as expected

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

def test_grid2op_action_type_open_line_load(action_space):
    actions_desc={
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
    }

    grid2op_action = action_space(actions_desc)
    action_type = identify_grid2op_action_type(grid2op_action)
    assert(action_type=="open_line_load")

def test_grid2op_action_type_close_line(action_space):
    actions_desc={
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
    }

    grid2op_action = action_space(actions_desc)
    action_type = identify_grid2op_action_type(grid2op_action)
    assert(action_type=="close_line")

def test_grid2op_action_type_open_line(action_space):
    actions_desc={
            "set_bus": {
                "lines_or_id": {},
                "lines_ex_id": {
                    "PYMONY632": -1
                },
                "loads_id": {},
                "generators_id": {},
                "shunts_id": {}
            }
    }

    grid2op_action = action_space(actions_desc)
    action_type = identify_grid2op_action_type(grid2op_action)
    assert(action_type=="open_line")


def test_grid2op_action_type_open_coupling(action_space):
    actions_desc={
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
    }

    grid2op_action = action_space(actions_desc)
    action_type = identify_grid2op_action_type(grid2op_action)
    assert(action_type=="open_coupling")


def test_grid2op_action_type_close_coupling(action_space):
    actions_desc={'set_bus': {'lines_or_id': {'CPVANL31RIBAU': 1},
          'lines_ex_id': {'BEON L31CPVAN': 1,
           'CPVANY631': 1,
           'CPVANY632': 1,
           'CPVANY633': 1},
          'loads_id': {'ARBOIL31CPVAN': 1,
           'BREVAL31CPVAN': 1,
           'CPDIVL32CPVAN': 1,
           'CPVANL31MESNA': 1,
           'CPVANL31ZBRE6': 1,
           'CPVAN3TR312': 1,
           'CPVAN3TR311': 1},
          'shunts_id': {},
          'generators_id': {}}}

    grid2op_action = action_space(actions_desc)
    action_type = identify_grid2op_action_type(grid2op_action)

    assert (action_type == "close_coupling")

def test_grid2op_action_types():
    #load action space
    env_name = "env_dijon_v2_assistant"
    env = make_grid2op_training_env(".", env_name)  # make_grid2op_evaluation_env(".", env_name)
    action_space=env.action_space

    # run tests
    test_grid2op_action_type_open_line_load(action_space)
    test_grid2op_action_type_close_line(action_space)
    test_grid2op_action_type_open_line(action_space)
    test_grid2op_action_type_close_coupling(action_space)
    test_grid2op_action_type_open_coupling(action_space)

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

    print("STARTING TESTS")
    test_overflow_graph_actions_filtered()
    test_overflow_graph_actions_filtered(check_with_action_description=False)
    test_grid2op_action_types()
    test_overflow_graph_construction()
    test_action_types()
    test_rules()
    print("ENDING TESTS")