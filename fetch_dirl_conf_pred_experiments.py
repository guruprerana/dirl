num_iters = 4000
spec_num = 5
use_gpu = True

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from conformal.all_paths_conformal_pred import all_paths_conformal_pred
from conformal.bucketed_conformal_pred import bucketed_conformal_pred
from conformal.calculate_coverage import calculate_coverage
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
import json

from spectrl.examples.rooms_envs import (
    GRID_PARAMS_LIST,
    MAX_TIMESTEPS,
    START_ROOM,
    FINAL_ROOM,
)
from spectrl.envs.rooms import RoomsEnv

import dill as pickle

with open("conformal_experiments_data/fetch-policies/path_policies.pkl", "rb") as f:
    path_policies = pickle.load(f)

with open("conformal_experiments_data/fetch-policies/adj_list.pkl", "rb") as f:
    adj_list = pickle.load(f)

with open("conformal_experiments_data/fetch-policies/terminal_vertices.pkl", "rb") as f:
    terminal_vertices = pickle.load(f)

# time_taken_score_graph = DIRLTimeTakenScoreGraph(adj_list, path_policies)
# n_samples = 2000
# n_samples_coverage = 1000
# es = [0.2, 0.1, 0.05]
# total_buckets = [5, 10, 20, 25, 50, 100]

# data_time_taken = dict()
# data_time_taken["metadata"] = {"es": es, "total_buckets": total_buckets, "scores": "cum-reward", "env": "16-rooms", "spec": spec_num, "n_samples": n_samples}

# for e in es:
#     e_data = dict()
#     for buckets in total_buckets:
#         bucket_data = dict()
#         vbs = bucketed_conformal_pred(time_taken_score_graph, e, buckets, n_samples)
#         min_path, min_path_scores = all_paths_conformal_pred(time_taken_score_graph, e, n_samples)
#         vb = vbs.buckets[(terminal_vertices[0], buckets)]

#         bucket_data["bucketed"] = {"path": vb.path, 
#                                    "path_buckets": vb.path_buckets, 
#                                    "path_score_quantiles": vb.path_score_quantiles, 
#                                    "max_path_score_quantile": max(vb.path_score_quantiles)}
#         bucket_data["all-paths"] = {"path": min_path, "min_path_scores": min_path_scores, "max_min_path_scores": max(min_path_scores)}

#         bucket_data["bucketed-coverage"] = calculate_coverage(
#             time_taken_score_graph, vb.path, vb.path_score_quantiles, n_samples_coverage
#         )
#         bucket_data["all-paths-coverage"] = calculate_coverage(
#             time_taken_score_graph, 
#             min_path, [max(min_path_scores) for _ in range(len(min_path))], 
#             n_samples_coverage,
#         )
#         e_data[buckets] = bucket_data
#     data_time_taken[str(e)] = e_data

# Convert the Python object to a JSON string
# json_data = json.dumps(data_time_taken, indent=2)

# Store the JSON string in a file
# with open("conformal_experiments_data/fetch-spec6-time-taken.json", "w") as json_file:
#     json_file.write(json_data)

cum_reward_score_graph = DIRLCumRewardScoreGraph(adj_list, path_policies)
n_samples = 2000
n_samples_coverage = 1000
es = [0.2, 0.1, 0.05]
total_buckets = [5, 10, 20, 25, 50, 100]

data_cum_reward = dict()
data_cum_reward["metadata"] = {"es": es, "total_buckets": total_buckets, "scores": "cum-reward", "env": "9-rooms", "spec": spec_num, "n_samples": n_samples}

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

        bucket_data["bucketed-coverage"] = calculate_coverage(
            cum_reward_score_graph, vb.path, vb.path_score_quantiles, n_samples_coverage
        )
        bucket_data["all-paths-coverage"] = calculate_coverage(
            cum_reward_score_graph, 
            min_path, 
            [max(min_path_scores) for _ in range(len(min_path)-1)], 
            n_samples_coverage,
        )
        e_data[buckets] = bucket_data
    data_cum_reward[str(e)] = e_data

# Convert the Python object to a JSON string
json_data = json.dumps(data_cum_reward, indent=2)

# Store the JSON string in a file
with open("conformal_experiments_data/fetch-spec6-cum-reward.json", "w") as json_file:
    json_file.write(json_data)

