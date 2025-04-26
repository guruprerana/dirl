from enum import Enum
from typing import Any, List, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy

from miniworld.entity import Entity
from miniworld.miniworld import MiniWorldEnv

class RiskyMiniworld(MiniWorldEnv):
    def __init__(
            self, 
            init_states: Optional[List[Any]]=None,
            task_str: Optional[str]=None,
            **kwargs
        ):
        self.init_states=init_states
        super().__init__(**kwargs)
        self.task_str = task_str

    def reset(self, *, seed=None, options=None):
        if options and isinstance(options, dict):
            if "state" in options:
                self.set_env_state = options["state"]
        else:
            self.set_env_state = None
        obs, info = super().reset(seed=seed, options=options)
        info["env_state"] = self.get_env_state()
        info["loss_eval"] = self.get_loss_eval()
        return obs, info
    
    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        info["env_state"] = self.get_env_state()
        info["loss_eval"] = self.get_loss_eval()
        reward += self.get_reward()
        termination = self.eval_terminated() or termination
        return obs, reward, termination, truncation, info

    def get_env_state(self, **kwargs) -> Any:
        raise NotImplementedError
    
    def get_loss_eval(self, **kwargs) -> float:
        raise NotImplementedError
    
    def get_reward(self, **kwargs) -> float:
        raise NotImplementedError
    
    def eval_terminated(self, **kwargs) -> bool:
        raise NotImplementedError
    

class RiskyMiniworldEnv1(RiskyMiniworld):
    class Tasks(Enum):
        GOTO_MIDDLE_BOTTOM = "goto-middle-bottom"
        GOTO_MIDLE_TOP = "goto-middle-top"
        GOTO_RIGHT_HALL = "goto-right-hall"

    def __init__(self, max_episode_steps=300, **kwargs):
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)

        # Allow only movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        self.left_hall = self.add_rect_room(min_x=0, max_x=5, min_z=0, max_z=21)
        self.middle_bottom = self.add_rect_room(min_x=6, max_x=16, min_z=0, max_z=10)
        self.middle_top = self.add_rect_room(min_x=6, max_x=16, min_z=11, max_z=21)
        self.right_hall = self.add_rect_room(min_x=17, max_x=22, min_z=0, max_z=21)

        self.connect_rooms(self.left_hall, self.middle_bottom, min_z=4, max_z=6)
        self.connect_rooms(self.left_hall, self.middle_top, min_z=15, max_z=17)
        self.connect_rooms(self.middle_bottom, self.right_hall, min_z=4, max_z=6)
        self.connect_rooms(self.middle_top, self.right_hall, min_z=15, max_z=17)

        pos = np.array([2, 0, 10])
        dir = 0
        if self.set_env_state:
            pos = np.copy(self.set_env_state["agent"]["pos"])
            dir = self.set_env_state["agent"]["dir"]
            self.set_env_state = None
        elif self.init_states:
            state = self.np_random.choice(self.init_states, 1)
            if state:
                pos = np.copy(state["agent"]["pos"])
                dir = state["agent"]["dir"]

        self.place_agent(pos=pos, dir=dir)

    def get_env_state(self, **kwargs):
        return {
            "agent": {
                "pos": np.copy(self.agent.pos), 
                "dir": self.agent.dir,
            }
        }
    
    def get_reward(self, **kwargs):
        target_state = self.get_target_state()
        rew = -np.linalg.norm(np.array(self.agent.pos) - target_state)
        if -rew <= 1:
            rew += 10000
        return rew
    
    def get_loss_eval(self, **kwargs):
        return 0
    
    def eval_terminated(self, **kwargs):
        target_state = self.get_target_state()
        dist = np.linalg.norm(np.array(self.agent.pos) - target_state)
        if dist <= 1:
            return True
        return False
    
    def get_target_state(self):
        if self.task_str == RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_BOTTOM:
            return np.array([11, 0, 5])
        elif self.task_str == RiskyMiniworldEnv1.Tasks.GOTO_MIDLE_TOP:
            return np.array([11, 0, 15])
        elif self.task_str == RiskyMiniworldEnv1.Tasks.GOTO_RIGHT_HALL:
            return np.array([19, 0, 10])
        else:
            raise ValueError
        

gym.register("RiskyMiniworldEnv1-v0", RiskyMiniworldEnv1)
