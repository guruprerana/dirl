from conformal.bucketed_conformal_pred import bucketed_conformal_pred
import conformal.miniworld
from conformal.miniworld.boxrelay import spec_graph
from conformal.rl_task_graph import RLTaskGraph

wandb_project_name = "boxrelayenv-agentview"
env_kwargs = {"view": "agent"}
task_graph = RLTaskGraph(spec_graph, "BoxRelay-v0", env_kwargs=env_kwargs, eval_env_kwargs=env_kwargs)

def train():
    task_graph.train_all_edges(wandb_project_name, training_iters=200_000, final_policy_recordings=3, n_envs=1)

def risk_min():
    task_graph.load_edge_policies(subfolder=wandb_project_name)
    vbs = bucketed_conformal_pred(task_graph, 0.2, 20, 1000)
    vb = vbs.buckets[(5, 20)]

    data = {
        "path": vb.path, 
        "path_buckets": vb.path_buckets, 
        "path_score_quantiles": vb.path_score_quantiles, 
        "max_path_score_quantile": max(vb.path_score_quantiles),
    }

    print(data)

if __name__ == "__main__":
    risk_min()
