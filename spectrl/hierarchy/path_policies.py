from typing import Any, Dict


class PathPolicy:
    """
    Inductive data structure to store policies associated with all paths in a task graph
    """

    def __init__(self, vertex: int, start_dist: Any) -> None:
        self.vertex = vertex
        self.edges: Dict[int, Any] = dict()
        self.policies: Dict[int, Any] = dict()
        self.path_policies: Dict[int, PathPolicy] = dict()
        self.start_dist: Any = start_dist
        self.reach_envs: Dict[int, Any] = dict()

    def add_edge(self, vertex: int, edge: Any):
        self.edges[vertex] = edge

    def add_policy(self, vertex: int, policy):
        self.policies[vertex] = policy
    
    def add_path_policy(self, vertex: int, path_policy):
        self.path_policies[vertex] = path_policy

    def add_reach_env(self, vertex: int, reach_env: Any):
        self.reach_envs[vertex] = reach_env
