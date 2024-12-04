from typing import List, Tuple
import numpy as np

from conformal.nonconformity_score_graph import NonConformityScoreGraph


def all_paths_conformal_pred(
    score_graph: NonConformityScoreGraph, e: float, n_samples: int
) -> Tuple[Tuple[int], List[float]]:
    path_samples: dict[Tuple[int], list] = dict()
    path_scores: dict[Tuple[int], List[List[float]]] = dict()
    path_samples[(0,)] = [None for _ in range(n_samples)]
    path_scores[(0,)] = []

    stack: List[Tuple[int]] = [(0,)]

    while stack:
        path = stack.pop()
        for succ in score_graph.adj_lists[path[-1]]:
            if succ == path[-1]:
                continue
            samples, scores = score_graph.sample_cached(
                succ, n_samples, path, path_samples[path]
            )
            next_path = path + (succ,)
            path_samples[next_path] = samples
            path_scores[next_path] = [scores for scores in path_scores[path]] + [scores]

        del path_samples[path], path_scores[path]

    min_path = None
    min_path_quantile = np.inf
    min_path_scores = None
    for path in path_scores:
        score_maxes: List[Tuple[float, List[float]]] = list()
        for i in range(n_samples):
            sample_path_scores: List[float] = []
            for j in range(len(path)):
                sample_path_scores.append(path_scores[j][i])
            score_maxes.append((max(sample_path_scores), sample_path_scores))

        score_maxes = sorted(score_maxes, key=lambda t: t[0])
        quantile_index = int(np.ceil((1 - e) * (n_samples + 1)) / n_samples)
        max_score, scores = score_maxes[quantile_index]

        if max_score <= min_path_quantile:
            min_path = path
            min_path_quantile = max_score
            min_path_scores = scores

    return min_path, min_path_scores