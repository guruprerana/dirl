from heapq import heappop, heappush
from typing import Dict, List, Tuple
import numpy as np

from spectrl.hierarchy.path_policies import PathPolicy
from spectrl.hierarchy.reachability import ReachabilityEnv
from spectrl.util.rl import get_rollout


class NonConformityScoreGraph:
    """
    Base class representing the non-conformity score graph.
    Allows sampling from distributions and evaluating the
    non-conformity scores.
    """

    def __init__(self, adj_lists: List[List[int]]) -> None:
        # adjacency lists of the DAG
        remove_loops(adj_lists)
        self.adj_lists = adj_lists
        self.rev_adj_lists = reverse_adj_list(self.adj_lists)
        self.dag_layers = dag_layers(self.adj_lists, self.rev_adj_lists)
        self.sample_cache: Dict[str, Tuple[list, List[float]]] = dict()
        self.samples_full_path_cache: Dict[str, List[List[float]]] = dict()

    def sample(
        self,
        target_vertex: int,
        n_samples: int,
        path: List[int],
        path_samples: list,
    ) -> Tuple[list, List[float]]:
        """
        Samples from the distribution induced on an edge by a path, i.e.,
        extends the path samples until the target_vertex.

        Also evaluates the non-conformity score on the samples.
        """
        raise NotImplementedError
    
    def sample_cached(
        self,
        target_vertex: int,
        n_samples: int,
        path: List[int],
        path_samples: list,
    ) -> Tuple[list, List[float]]:
        cache_string = str(path) + str(target_vertex)
        if cache_string in self.sample_cache:
            return self.sample_cache[cache_string]
        sample = self.sample(target_vertex, n_samples, path, path_samples)
        self.sample_cache[cache_string] = sample
        return sample
    
    def sample_full_path(
        self,
        path: List[int],
        n_samples: int,
    ) -> List[List[float]]:
        """
        Samples n_samples trajectories of non-conformity scores along specified path
        """
        trajectories_scores = [[] for _ in range(n_samples)]
        prev_samples = [None for _ in range(n_samples)]

        for i in range(1, len(path)):
            # starting from 1 because we sample on edges of the path
            prev_samples, scores = self.sample(path[i], n_samples, path[:i], prev_samples)
            for j in range(n_samples):
                trajectories_scores[j].append(scores[j])

        return trajectories_scores
    
    def sample_full_path_cached(
        self,
        path: List[int],
        n_samples: int,
    ) -> List[List[float]]:
        cache_string = str(path) + str(n_samples)
        if cache_string in self.samples_full_path_cache:
            return self.samples_full_path_cache[cache_string]
        sample = self.sample_full_path(path, n_samples)
        self.samples_full_path_cache[cache_string] = sample
        return sample
    

def remove_loops(adj_list: List[List[int]]) -> None:
    """
    Removes self loops on vertices from the adjacency lists
    """
    for i in range(len(adj_list)):
        if i in adj_list[i]:
            adj_list[i].remove(i)


def reverse_adj_list(adj_list: List[List[int]]) -> List[List[int]]:
    """
    Compute adjancency list of graph with reversed edges
    """
    reversed = [[] for _ in range(len(adj_list))]
    for i in range(len(adj_list)):
        for j in adj_list[i]:
            reversed[j].append(i)

    return reversed


def dag_layers(
    adj_list: List[List[int]], rev_adj_list: List[List[int]]
) -> List[List[int]]:
    """
    Partitions the vertices of a DAG into layers by the induced partial order
    """
    layers = [[0]]
    explored = set()
    queue = []
    heappush(queue, 0)

    while len(queue) > 0:
        v1 = heappop(queue)
        explored.add(v1)
        layer = []
        for v2 in adj_list[v1]:
            if v2 == v1:
                continue
            if all((pred in explored) for pred in rev_adj_list[v2]):
                layer.append(v2)
                heappush(queue, v2)
        if len(layer) > 0:
            layers.append(layer)

    return layers


class DIRLNonConformityScoreGraph(NonConformityScoreGraph):
    """
    Non-conformity score graphs for DIRL policies
    """

    def __init__(self, adj_lists: List[List[int]], path_policies: PathPolicy) -> None:
        super().__init__(adj_lists)
        self.path_policies = path_policies

    def sample(
        self,
        target_vertex: int,
        n_samples: int,
        path: List[int],
        path_samples: list,
    ) -> Tuple[list]:
        assert target_vertex in self.adj_lists[path[-1]]
        assert len(path_samples) == n_samples

        pp = self.path_policies.get_vertex_path_policy(path)
        scores: List[float] = []
        next_path_samples = []
        for init_state in path_samples:
            sarss = get_rollout(
                pp.reach_envs[target_vertex],
                pp.policies[target_vertex],
                False,
                init_state=init_state,
            )
            scores.append(self.compute_score(sarss, pp.reach_envs[target_vertex]))
            next_path_samples.append(pp.reach_envs[target_vertex].get_state())

        return path_samples, scores

    def compute_score(self, sarss: list, env: ReachabilityEnv) -> float:
        """
        Computes the non-conformity score on a given (s, a, r, s') trace.
        """
        raise NotImplementedError


class DIRLTimeTakenScoreGraph(DIRLNonConformityScoreGraph):
    """
    Non-conformity scores corresponding to time taken to complete reach objective.
    """

    def __init__(self, adj_lists: List[List[int]], path_policies: PathPolicy) -> None:
        super().__init__(adj_lists, path_policies)

    def compute_score(self, sarss: List, env: ReachabilityEnv) -> float:
        states = np.array([state for state, _, _, _ in sarss] + [sarss[-1][-1]])
        if env.cum_reward(states) <= 0:
            return np.inf
        return len(sarss)
    

class DIRLCumRewardScoreGraph(DIRLNonConformityScoreGraph):
    """
    Non-conformity scores corresponding to cumulative reward achieved.
    """

    def __init__(self, adj_lists: List[List[int]], path_policies: PathPolicy, cum_reward_type="normal") -> None:
        super().__init__(adj_lists, path_policies)
        self.cum_reward_type = cum_reward_type

    def compute_score(self, sarss: List, env: ReachabilityEnv) -> float:
        states = np.array([state for state, _, _, _ in sarss] + [sarss[-1][-1]])
        if self.cum_reward_type == "cum_safety_reward":
            return -env.cum_safety_reward(states)
        elif self.cum_reward_type == "cum_safety_reach_reward":
            return -env.cum_safety_reach_reward(states)
        return -env.cum_reward(states)
