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
    def __init__(self, state_space: Space, action_space: Space, discount_rate: float, epsilon: float, learning_rate: float, learning:bool):
        self._state_space = state_space
        self._action_space = action_space
        self._discount_rate = discount_rate
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._learning = learning
        self._device='cpu'


    def sample_action(self, state:torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if action_mask is not None:
            legal_actions = np.nonzero(action_mask.numpy())[0]
            random_action = np.random.choice(legal_actions)
        else:
            random_action = self._action_space.sample()
        action = torch.tensor([[random_action]], device=self._device, dtype=torch.long)
        return action

    def control(self, state: Any, action: Any, reward: float, new_state: Any, terminal: bool):
        raise NotImplementedError