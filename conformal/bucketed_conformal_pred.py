from typing import Dict, List, Tuple
import numpy as np

from conformal.nonconformity_score_graph import NonConformityScoreGraph


class VertexBucket:
    def __init__(
        self,
        vertex: int,
        bucket: int,
        path: List[int] = None,
        path_buckets: List[int] = None,
        path_score_quantiles: List[float] = None,
        path_samples: list = None,
    ) -> None:
        self.vertex = vertex
        self.bucket = bucket
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
    """
    DP implementation of the bucketed conformal prediction algorithm.
    """
    vbs = VertexBuckets(len(score_graph.adj_lists), e, total_buckets, n_samples)
    for i in range(total_buckets + 1):
        vbs.buckets[(0, i)] = VertexBucket(
            0, i, [0], [], [], [None for _ in range(n_samples)]
        )
    for layer in score_graph.dag_layers[1:]:
        # skip first layer which is just [0]
        for v in layer:
            for bucket in range(total_buckets + 1):
                vb = VertexBucket(v, bucket)
                vbs.buckets[(v, bucket)] = vb
                min_quantile = np.inf
                for pred in score_graph.rev_adj_lists[v]:
                    bucket_preds = range(bucket+1) if pred != 0 else [0]
                    # to get to vertex 0, we do not want to use any of the error param
                    for bucket_pred in bucket_preds:
                        pred_vb = vbs.buckets[(pred, bucket_pred)]
                        path_samples, scores = score_graph.sample_cached(
                            v,
                            n_samples,
                            pred_vb.path,
                            pred_vb.path_samples,
                        )
                        scores = sorted(scores)
                        rem_e = (bucket - bucket_pred) * (e / total_buckets)
                        quantile_index = min(n_samples - 1, int(np.ceil((1 - rem_e) * (n_samples + 1))))
                        quantile = scores[quantile_index]
                        if quantile <= min_quantile:
                            min_quantile = quantile
                            vb.path = [i for i in pred_vb.path] + [v]
                            vb.path_buckets = [i for i in pred_vb.path_buckets] + [
                                bucket - bucket_pred
                            ]
                            vb.path_score_quantiles = [
                                i for i in pred_vb.path_score_quantiles
                            ] + [quantile]
                            vb.path_samples = path_samples
    return vbs
