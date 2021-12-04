import numpy as np

from peginhole_env import *
from gym import spaces

class MultitaskPeginHole(PeginHole):
    def __init__(self, robots, n_tasks=10, randomize_tasks=False, **kwargs):
        super().__init__(robots, gripper_types=None, **kwargs)
        self._goal = self.peg_class
        self.num_tasks = 10
    
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
    
    def _flatten_obs(self, ob_dict):
        ob_list = []
        for key in ob_dict:
            ob_list.append(ob_dict[key].flatten())
        return np.concatenate(ob_list)
        
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
        return self._flatten_obs(ob_dict)
    
    def step(self, action):
        ob_dict, reward, done, info = super().step(action)
        if self._check_success():
            done = True
        return self._flatten_obs(ob_dict), reward, done, info
    
        