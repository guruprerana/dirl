num_iters = 2
spec_num = 5
use_gpu = False

import os

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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

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


def hold_object(err):
    def predicate(sys_state, res_state):
        dist = sys_state[:3] - sys_state[3:6]
        dist2 = np.concatenate([dist, [sys_state[9] + sys_state[10] - 0.045]])
        return -LA.norm(dist2) + err
    return predicate


def object_in_air(sys_state, res_state):
    return sys_state[5] - 0.45


def object_at_goal(err):
    def predicate(sys_state, res_state):
        dist = np.concatenate([sys_state[-3:], [sys_state[9] + sys_state[10] - 0.045]])
        return -LA.norm(dist) + err
    return predicate


def gripper_reach(goal, err):
    '''
    goal: numpy array of dim (3,)
    '''
    def predicate(sys_state, res_state):
        return -LA.norm(sys_state[:3] - goal) + err
    return predicate


def object_reach(goal, err):
    '''
    goal: numpy array of dim (3,)
    '''
    def predicate(sys_state, res_state):
        return -LA.norm(sys_state[3:6] - goal) + err
    return predicate


above_corner1 = np.array([1.15, 1.0, 0.5])
above_corner2 = np.array([1.45, 1.0, 0.5])
corner1 = np.array([1.15, 1.0, 0.425])
corner2 = np.array([1.50, 1.05, 0.425])

# Specifications
spec1 = ev(grip_near_object(0.03))
spec2 = seq(spec1, ev(hold_object(0.03)))
spec3 = seq(spec2, ev(object_at_goal(0.05)))
spec4 = seq(seq(spec2, ev(object_in_air)), ev(object_at_goal(0.05)))
spec5 = seq(seq(spec2, ev(object_in_air)), ev(object_reach(above_corner1, 0.05)))
spec6 = seq(seq(spec2, ev(object_in_air)),
            choose(seq(ev(object_reach(above_corner1, 0.05)), ev(object_reach(corner1, 0.05))),
                   seq(ev(object_reach(above_corner2, 0.05)), ev(object_reach(corner2, 0.01)))))

specs = [spec1, spec2, spec3, spec4, spec5, spec6]

lb = [100., 100., 100., 100., 100., 100.]

env = ObservationWrapper(FetchPickAndPlaceEnv(), ['observation', 'desired_goal'],
                            relative=(('desired_goal', 0, 3), ('observation', 3, 6)))

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

print('\n**** Learning Policy for Spec {} ****'.format(spec_num))

_, abstract_reach = automaton_graph_from_spec(specs[spec_num])
print('\n**** Abstract Graph ****')
abstract_reach.pretty_print()

# Step 5: Learn policy
path_policies = abstract_reach.learn_all_paths(
    env,
    hyperparams,
    res_model=None,
    max_steps=40,
    render=render,
    neg_inf=-lb[spec_num],
    safety_penalty=-1,
    num_samples=1000,
    algo="ddpg",
    alpha=0,
    use_gpu=use_gpu
)

adj_list = adj_list_from_task_graph(abstract_reach.abstract_graph)
terminal_vertices = [i for i in range(len(adj_list)) if i in adj_list[i]]

time_taken_score_graph = DIRLTimeTakenScoreGraph(adj_list, path_policies)
n_samples = 2000
es = [0.2, 0.1, 0.05]
total_buckets = [5, 10, 20, 25, 50, 100]

data_time_taken = dict()
data_time_taken["metadata"] = {"es": es, "total_buckets": total_buckets, "scores": "cum-reward", "env": "16-rooms", "spec": spec_num, "n_samples": n_samples}

for e in es:
    e_data = dict()
    for buckets in total_buckets:
        bucket_data = dict()
        vbs = bucketed_conformal_pred(time_taken_score_graph, e, buckets, n_samples)
        min_path, min_path_scores = all_paths_conformal_pred(time_taken_score_graph, e, n_samples)
        vb = vbs.buckets[(terminal_vertices[0], buckets)]

        bucket_data["bucketed"] = {"path": vb.path, 
                                   "path_buckets": vb.path_buckets, 
                                   "path_score_quantiles": vb.path_score_quantiles, 
                                   "max_path_score_quantile": max(vb.path_score_quantiles)}
        bucket_data["all-paths"] = {"path": min_path, "min_path_scores": min_path_scores, "max_min_path_scores": max(min_path_scores)}
        e_data[buckets] = bucket_data
    data_time_taken[str(e)] = e_data

import json

# Convert the Python object to a JSON string
json_data = json.dumps(data_time_taken, indent=2)

# Store the JSON string in a file
with open("conformal_experiments_data/16rooms-spec13-time-taken.json", "w") as json_file:
    json_file.write(json_data)

cum_reward_score_graph = DIRLCumRewardScoreGraph(adj_list, path_policies)
n_samples = 2000
es = [0.2, 0.1, 0.05]
total_buckets = [5, 10, 20, 25, 50, 100]

data_time_taken = dict()
data_time_taken["metadata"] = {"es": es, "total_buckets": total_buckets, "scores": "cum-reward", "env": "9-rooms", "spec": spec_num, "n_samples": n_samples}

for e in es:
    e_data = dict()
    for buckets in total_buckets:
        bucket_data = dict()
        vbs = bucketed_conformal_pred(cum_reward_score_graph, e, buckets, n_samples)
        min_path, min_path_scores = all_paths_conformal_pred(cum_reward_score_graph, e, n_samples)
        vb = vbs.buckets[(terminal_vertices[0], buckets)]

        bucket_data["bucketed"] = {"path": vb.path, 
                                   "path_buckets": vb.path_buckets, 
                                   "path_score_quantiles": vb.path_score_quantiles, 
                                   "max_path_score_quantile": max(vb.path_score_quantiles)}
        bucket_data["all-paths"] = {"path": min_path, "min_path_scores": min_path_scores, "max_min_path_scores": max(min_path_scores)}
        e_data[buckets] = bucket_data
    data_time_taken[str(e)] = e_data

# Convert the Python object to a JSON string
json_data = json.dumps(data_time_taken, indent=2)

# Store the JSON string in a file
with open("conformal_experiments_data/16rooms-spec13-cum-reward.json", "w") as json_file:
    json_file.write(json_data)