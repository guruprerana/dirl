from typing import Any, Dict, List, Tuple
from conformal.nonconformity_score_graph import NonConformityScoreGraph


class RLTaskGraph(NonConformityScoreGraph):
    def __init__(self, spec_graph: List[List[Tuple[int, str]]], env_name: str):
        self.spec_graph = spec_graph
        adj_lists = [[v for (v, _) in edges] for edges in spec_graph]
        super().__init__(adj_lists)

        self.path_policies: Dict[Tuple[int], Any] = dict()
