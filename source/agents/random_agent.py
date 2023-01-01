from typing import Dict, List, Optional, Set, Tuple

import gym
from gym.spaces import Space
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from typing import Any, Union
from pettingzoo.utils.env import AECEnv
import numpy as np
from source.agents.agent import Agent
import torch


class RandomAgent(Agent):
    def __init__(self, state_space: Space, action_space: Space, discount_rate: float, epsilon: float, learning_rate: float, learning: bool):
        super().__init__(state_space, action_space,
                         discount_rate, epsilon, learning_rate, learning)
        self._device = 'cpu'
        self._leraning = False

    def sample_action(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        if action_mask is not None:
            legal_actions = np.nonzero(action_mask)[0]
            action = np.random.choice(legal_actions)
        else:
            action = self._action_space.sample()
        return action

    def control(self, state: Any, action: Any, reward: float, new_state: Any, terminal: bool):
        raise NotImplementedError
