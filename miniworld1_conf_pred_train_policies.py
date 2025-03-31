num_iters = 4000
spec_num = 0
use_gpu = True

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import gymnasium
import miniworld

from spectrl.rl.ddpg.ddpg import DDPG

from conformal.all_paths_conformal_pred import all_paths_conformal_pred
from conformal.bucketed_conformal_pred import bucketed_conformal_pred
from conformal.nonconformity_score_graph import DIRLCumRewardScoreGraph, DIRLTimeTakenScoreGraph
from spectrl.hierarchy.construction import adj_list_from_task_graph, automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.spec_compiler import ev, seq, choose, alw
from spectrl.util.io import parse_command_line_options, save_log_info, save_object
from spectrl.util.rl import print_performance, get_rollout, ObservationWrapper
from spectrl.rl.ars import HyperParams
from spectrl.rl.ddpg import DDPGParams
from spectrl.envs.fetch import FetchPickAndPlaceEnv
import numpy as np
from numpy import linalg as LA

from spectrl.examples.rooms_envs import (
    GRID_PARAMS_LIST,
    MAX_TIMESTEPS,
    START_ROOM,
    FINAL_ROOM,
)
from spectrl.envs.rooms import RoomsEnv

render = False
folder = ''
itno = -1

log_info = []

def grip_near_object(err):
    def predicate(sys_state, res_state):
        dist = sys_state[:3] - (sys_state[3:6] + np.array([0., 0., 0.065]))
        dist = np.concatenate([dist, [sys_state[9] + sys_state[10] - 0.1]])
        return -LA.norm(dist) + err
    return predicate


above_corner1 = np.array([1.15, 1.0, 0.5])
above_corner2 = np.array([1.45, 1.0, 0.5])
corner1 = np.array([1.15, 1.0, 0.425])
corner2 = np.array([1.50, 1.05, 0.425])

# Specifications
spec1 = ev(grip_near_object(0.03))

specs = [spec1]

lb = [100., 100., 100., 100., 100., 100.]

env = gymnasium.make("Miniworld-CollectHealth-v0", size=16)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high
hyperparams = DDPGParams(state_dim, action_dim, action_bound,
                            minibatch_size=256, num_episodes=num_iters,
                            discount=0.95, actor_hidden_dim=256,
                            critic_hidden_dim=256, epsilon_decay=3e-6,
                            decay_function='linear', steps_per_update=100,
                            gradients_per_update=100, buffer_size=200000,
                            sigma=0.15, epsilon_min=0.3, target_noise=0.0003,
                            target_clip=0.003, warmup=1000)

agent = DDPG(hyperparams, use_gpu=use_gpu)
agent.train(env)
policy = agent.get_policy()

