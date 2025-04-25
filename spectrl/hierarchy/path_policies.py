from typing import Any, Dict, List

from spectrl.util.dist import FiniteDistribution
from spectrl.util.rl import get_rollout


class PathPolicy:
    """
    Inductive data structure to store policies associated with all paths in a task graph
    """

    def __init__(
        self,
        vertex: int,
        start_dist: Any,
        abstract_graph=None,
        env=None,
        hyperparams=None,
        algo="ars",
        res_model=None,
        max_steps=100,
        safety_penalty=-1,
        neg_inf=-10,
        alpha=0,
        num_samples=300,
        use_gpu=False,
        render=False,
        succ_thresh=0.0,
    ) -> None:
        """
        Parameters:
            env: gym.Env (with additional method, set_state: np.array -> NoneType)
            init_dist: Distribution (initial state distribution)
            hyperparams: HyperParams object (corresponding to the RL algo)
            algo: str (RL algorithm to use to learn policies for edges)
            res_model: Resource_Model (optional)
            safety_penalty: float (min penalty for violating constraints)
            neg_inf: float (large negative constant)
            num_samples: int (number of samples used to compute reach probabilities)
        """
        self.vertex = vertex
        self.edges: Dict[int, Any] = dict()
        self.policies: Dict[int, Any] = dict()
        self.path_policies: Dict[int, PathPolicy] = dict()
        self.start_dist: Any = start_dist
        self.reach_envs: Dict[int, Any] = dict()

        self.abstract_graph = abstract_graph
        self.env = env
        self.hyperparams = hyperparams
        self.algo = algo
        self.res_model = res_model
        self.max_steps = max_steps
        self.safety_penalty = safety_penalty
        self.neg_inf = neg_inf
        self.alpha = alpha
        self.num_samples = num_samples
        self.use_gpu = use_gpu
        self.render = render
        self.succ_thresh = succ_thresh

    def add_edge(self, vertex: int, edge: Any):
        self.edges[vertex] = edge

    def add_policy(self, vertex: int, policy):
        self.policies[vertex] = policy

    def add_path_policy(self, vertex: int, path_policy):
        self.path_policies[vertex] = path_policy

    def add_reach_env(self, vertex: int, reach_env: Any):
        self.reach_envs[vertex] = reach_env

    def train_edge(self, edge: Any) -> None:
        """
        Trains edge policy
        """

        if edge.target == self.vertex:
            return
        edge_policy, reach_env, log_info = edge.learn_policy(
            self.env,
            self.hyperparams,
            self.vertex,
            self.start_dist,
            self.algo,
            self.res_model,
            self.max_steps,
            self.safety_penalty,
            self.neg_inf,
            self.alpha,
            self.use_gpu,
            self.render,
        )

        final_states = []
        for _ in range(self.num_samples):
            get_rollout(reach_env, edge_policy, False)
            final_states.append(reach_env.get_state())

        pp_edge = PathPolicy(
            edge.target,
            FiniteDistribution(final_states),
            self.abstract_graph,
            self.env,
            self.hyperparams,
            self.vertex,
            self.start_dist,
            self.algo,
            self.res_model,
            self.max_steps,
            self.safety_penalty,
            self.neg_inf,
            self.alpha,
            self.use_gpu,
            self.render,
        )
        self.add_edge(edge.target, edge)
        self.add_policy(edge.target, edge_policy)
        self.add_path_policy(edge.target, pp_edge)
        self.add_reach_env(edge.target, reach_env)

    def get_vertex_path_policy(self, path: List[int]):
        """
        Traverse path in the PathPolicy object and return the PathPolicy
        object at the last vertex of the path.
        """
        pp = self
        for v in path[1:]:
            if v not in pp.path_policies:
                req_edge = None
                for edge in pp.abstract_graph[pp.vertex]:
                    if edge.target == v:
                        req_edge = edge
                        break

                if req_edge is None:
                    # edge to target vertex does not exist
                    raise KeyError
                
                self.train_edge(req_edge)
            
            pp = pp.path_policies[v]
        return pp
