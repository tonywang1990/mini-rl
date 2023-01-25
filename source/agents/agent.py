from typing import Dict, List, Optional, Set, Tuple

from gym.spaces import Space
from typing import Any


class Agent(object):
    def __init__(self, state_space: Space, action_space: Space, discount_rate: float, epsilon: float, learning_rate: float, learning:float=True):
        self._state_space = state_space
        self._action_space = action_space
        self._discount_rate = discount_rate
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._learning = learning
        self._Q = None
        self._policy = None

    def sample_action(self, state: Any):
        raise NotImplementedError

    def control():
        raise NotImplementedError
    
    def reset(self):
        pass

    def pre_process(self):
        pass

    def post_process(self, state: Any, action: Any, reward: float, next_state: Any, terminal: bool, action_info: dict):
        pass
    