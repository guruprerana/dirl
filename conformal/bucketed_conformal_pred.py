from typing import Dict, List, Tuple

from conformal.nonconformity_score_graph import NonConformityScoreGraph


class VertexBucket:
    def __init__(
        self,
        vertex: int,
        e: float,
        bucket: int,
        total_buckets: int,
        path: List[int],
        path_buckets: List[int],
        path_score_quantiles: List[float],
        path_samples: list,
    ) -> None:
        self.vertex = vertex
        self.e = e
        self.bucket = bucket
        self.total_buckets = total_buckets
        self.path = path
        self.path_buckets = path_buckets
        self.path_score_quantiles = path_score_quantiles
        self.path_samples = path_samples


class VertexBuckets:
    def __init__(
        self, n_vertices: int, e: float, total_buckets: int, n_samples: int
    ) -> None:
        self.n_vertices = n_vertices
        self.e = e
        self.total_buckets = total_buckets
        self.n_samples = n_samples
        self.buckets: Dict[Tuple[int, int], VertexBucket] = dict()


def bucketed_conformal_pred(
    score_graph: NonConformityScoreGraph, e: float, total_buckets: int, n_samples: int
) -> VertexBuckets:
    vb = VertexBuckets(len(score_graph.adj_lists), e, total_buckets, n_samples)
    
