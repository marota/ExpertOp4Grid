from grid2op.Action import TopologyAction
from grid2op.Reward import L2RPNReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Backend import PandaPowerBackend

config = {
    "backend": PandaPowerBackend,
    "action_class": TopologyAction,
    "observation_class": None,
    "reward_class": L2RPNReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "volagecontroler_class": None,
    "thermal_limits": [1740, 500, 500, 500, 500, 500, 500, 500, 88, 500, 500, 500,
                       500, 500, 500, 500, 500, 500, 500, 500],
}
