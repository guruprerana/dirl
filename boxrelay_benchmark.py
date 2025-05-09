from conformal.bucketed_conformal_pred import bucketed_conformal_pred
import conformal.miniworld
import numpy as np
from conformal.miniworld.boxrelay import spec_graph, BoxRelay
from conformal.rl_task_graph import RLTaskGraph
import dill as pickle

# spec_graph = [
#     {
#         1: BoxRelay.Tasks.GOTO_LEFT_HALL_TARGET,
#     },
#     {
#         2: BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_ENTRY,
#     },
#     {
#         3: BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_TARGET,
#     },
#     {
#         4: BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_EXIT,
#     },
#     {
#         5: BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET,
#     },
#     {},
# ]

wandb_project_name = "boxrelayenv-agentview"
env_kwargs = {"view": "agent"}
task_graph = RLTaskGraph(spec_graph, "BoxRelay-v0", env_kwargs=env_kwargs, eval_env_kwargs=env_kwargs)

def train():
    # task_graph.train_all_edges(wandb_project_name, training_iters=500_000, final_policy_recordings=3, n_envs=1)
    task_graph.train_all_paths(wandb_project_name=wandb_project_name, n_samples=1000, training_iters=500_000, final_policy_recordings=3, n_envs=1)

def risk_min():
    task_graph.load_path_policies(subfolder=wandb_project_name)

    # scores = task_graph.sample_full_path([0, 1, 2, 3, 4, 5], 100)
    # count = 0
    # for score in scores:
    #     print(score)
    #     if max(score) == np.inf:
    #         count += 1

    # print(count)

    vbs = bucketed_conformal_pred(task_graph, 0.1, 10, 1000)

    # with open(f"./logs/{wandb_project_name}/task_graph.pkl", "wb") as f:
    #     pickle.dump(task_graph, f)

    vb = vbs.buckets[(5, 10)]

    data = {
        "path": vb.path, 
        "path_buckets": vb.path_buckets, 
        "path_score_quantiles": vb.path_score_quantiles, 
        "max_path_score_quantile": max(vb.path_score_quantiles),
    }

    print(data)

if __name__ == "__main__":
    train()
