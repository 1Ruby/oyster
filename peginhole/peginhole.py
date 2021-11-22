import numpy as np

from peginhole_env import *
from gym import spaces
from ..rlkit.envs import register_env

def flatten_obs(ob_dict):
    ob_list = []
    for key in ob_dict:
        ob_list.append(ob_dict[key].flatten())
    return np.concatenate(ob_list)

@register_env('peginhole')
class MultitaskPeginHole(PeginHole):
    def __init__(self, robots, n_tasks=4, randomize_tasks=False, **kwargs):
        super().__init__(robots, gripper_types=None, **kwargs)
        self._goal = self.peg_class
        self.num_tasks = 4
    
    def get_all_task_idx(self):
        return range(self.num_tasks)
    
    def reset_task(self, idx):
        if self.peg_class != idx:
            self.peg_class = idx
            self._load_model()
            self._postprocess_model()
            self._initialize_sim()
            self._reset_internal()
            self._observables = self._setup_observables()
        self._goal = self.peg_class
        self.reset()
        
    @property
    def observation_space(self):
        ob_dict = self.observation_spec()
        flat_ob = self._flatten_obs(ob_dict)
        
        high = np.inf * np.ones(flat_ob.size)
        low = -high
        return spaces.Box(low=low, high=high)
    
    @property
    def action_space(self):
        low, high = self.action_spec
        return spaces.Box(low=low, high=high)
    
    def reset(self):
        ob_dict = super().reset()
        return flatten_obs(ob_dict)
    
    def step(self, action):
        if self._check_success():
            action_zero = np.zeros_like(action)
            ob_dict, reward, done, info = super().step(action_zero)
        else:
            ob_dict, reward, done, info = super().step(action)
        return flatten_obs(ob_dict), reward, done, info
    
        