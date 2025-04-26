from typing import Any, Callable, Dict, List, Tuple
from conformal.nonconformity_score_graph import NonConformityScoreGraph


class RLTaskGraph(NonConformityScoreGraph):
    def __init__(self, spec_graph: List[List[Tuple[int, str]]], env_fn: Callable):
        self.spec_graph = spec_graph
        adj_lists = [[v for (v, _) in edges] for edges in spec_graph]
        super().__init__(adj_lists)

        self.env_fn = env_fn
        self.path_policies: Dict[Tuple[int], Any] = dict()
        self.init_states: Dict[Tuple[int], List[Any]] = dict()

