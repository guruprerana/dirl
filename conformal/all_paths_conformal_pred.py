from typing import List, Tuple
import numpy as np

from conformal.nonconformity_score_graph import NonConformityScoreGraph


def all_paths_conformal_pred(
    score_graph: NonConformityScoreGraph, e: float, n_samples: int
) -> Tuple[Tuple[int], List[float]]:
    """
    Naive conformal prediction algorithm on the non-confirmity score graph
    that first builds n_samples traces over all paths in the graph
    to reach the terminal vertex and then runs split conformal prediction 
    on each path.

    Inputs:
        score_graph : NonConformityScoreGraph
        e : float (non-coverage rate)
        n_samples : int (number of sample traces to estimate quantile from along each path)

    Outputs:
        min_path : Tuple[int] (the path that achieves the minimum bound on the non-conformity score)
        min_path_scores : List[float] (the ~(1-e)th quantile of the minimum scores of the samples along this path)
    """
    path_samples: dict[Tuple[int], list] = dict()
    path_scores: dict[Tuple[int], List[List[float]]] = dict()
    path_samples[(0,)] = [None for _ in range(n_samples)]
    path_scores[(0,)] = []

    stack: List[Tuple[int]] = [(0,)]

    while stack:
        path = stack.pop()
        if len(score_graph.adj_lists[path[-1]]) == 0:
            continue
        for succ in score_graph.adj_lists[path[-1]]:
            if succ == path[-1]:
                continue
            samples, scores = score_graph.sample_cached(
                succ, n_samples, path, path_samples[path]
            )
            next_path = path + (succ,)
            path_samples[next_path] = samples
            path_scores[next_path] = [scores for scores in path_scores[path]] + [scores]
            stack.append(next_path)

        del path_samples[path], path_scores[path]

    min_path = None
    min_path_quantile = np.inf
    min_path_scores = None
    for path in path_scores:
        score_maxes: List[Tuple[float, List[float]]] = list()
        for i in range(n_samples):
            sample_path_scores: List[float] = []
            for j in range(len(path)-1):
                sample_path_scores.append(path_scores[path][j][i])
            score_maxes.append((max(sample_path_scores), sample_path_scores))

        score_maxes = sorted(score_maxes, key=lambda t: t[0])
        quantile_index = int(np.ceil((1 - e) * (n_samples + 1))) - 1 # -1 to account for 0 index
        # then make sure quantile_index is a valid index
        if quantile_index < 0:
            quantile_index = 0
        elif quantile_index >= n_samples:
            quantile_index = n_samples - 1
        max_score, scores = score_maxes[quantile_index]

        if max_score <= min_path_quantile:
            min_path = path
            min_path_quantile = max_score
            min_path_scores = scores

    return min_path, min_path_scores
