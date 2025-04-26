from typing import Any, List, Optional
from gymnasium import spaces, utils
import copy

from miniworld.entity import Entity
from miniworld.miniworld import MiniWorldEnv

class RiskyMiniworld(MiniWorldEnv):
    def __init__(
            self, 
            init_states: Optional[List[Any]]=None,
            **kwargs
        ):
        self.init_states=init_states
        super().__init__(**kwargs)

    def reset(self, *, seed=None, options=None):
        if options and isinstance(options, dict):
            if "state" in options:
                self.set_env_state = options["state"]
        else:
            self.set_env_state = None
        obs, info = super().reset(seed=seed, options=options)
        info["env_state"] = self.get_env_state()
        return obs, info
    
    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        info["env_state"] = self.get_env_state()
        return obs, reward, termination, truncation, info

    def get_env_state(self) -> Any:
        raise NotImplementedError
