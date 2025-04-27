from conformal import miniworld
from conformal.miniworld import RiskyMiniworldEnv1
from conformal.rl_task_graph import RLTaskGraph

spec_graph = [
    {
        1: RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_BOTTOM,
        2: RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_TOP,
    },
    {
        3: RiskyMiniworldEnv1.Tasks.GOTO_RIGHT_HALL,
    },
    {
        3: RiskyMiniworldEnv1.Tasks.GOTO_RIGHT_HALL,
    },
    {},
]
wandb_project_name = "riskyminiworldenv1"

task_graph = RLTaskGraph(spec_graph, "RiskyMiniworldEnv1-v0")
task_graph.train_all_paths(wandb_project_name, 100, 500_000)
