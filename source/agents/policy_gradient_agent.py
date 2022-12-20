import numpy as np
from collections import namedtuple, deque
from gym.spaces import Discrete, Box, Space
import random
import gym
from typing import Union, Optional
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import math

from source.agents.agent import Agent
from source import utils


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class DenseNet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DenseNet, self).__init__()
        width = 128
        self.layer1 = nn.Linear(n_observations, width)
        self.layer2 = nn.Linear(width, width)
        self.layer3 = nn.Linear(width, width)
        self.layer4 = nn.Linear(width, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return F.softmax(x, dim=-1)

class PolicyGradientAgent(Agent):
    def __init__(self, state_space: Space, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._rewards = []
        self._log_prob = []
        self._eps = np.finfo(np.float32).eps.item()
        self._device = 'cpu' #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f'using device: {self._device}')

        # Get number of actions from gym action space
        self._n_actions = action_space.n
        self._state_dim = state_space.sample().shape
        self._n_states = len(state_space.sample().flatten())

        self._policy_net = DenseNet(self._n_states, self._n_actions).to(self._device)
        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=self._learning_rate, amsgrad=True)
        #self._optimizer = optim.Adam(self._policy_net.parameters(), lr=self._learning_rate)

        self._step = 0
        self._debug = True

    def to_feature(self, data: np.ndarray) -> torch.Tensor:
        # Convert state into tensor and unsqueeze: insert a new dim into tensor (at dim 0): e.g. 1 -> [1] or [1] -> [[1]] 
        # state: np.array
        # returns: torch.Tensor of shape [1]
        if self._debug:
            assert isinstance(data, np.ndarray), f'data is not of type ndarray: {type(data)}'
        return torch.tensor(data.flatten(), dtype=torch.float32, device=self._device)

    def reset(self):
        del self._rewards[:]
        del self._log_prob[:]

    def sample_action(self, state: np.ndarray) -> int:
        # state: tensor of shape [n_states]
        # return: int
        state_tensor = self.to_feature(state) # [n_states]
        if self._debug:
            assert list(state_tensor.shape) == [self._n_states], f"state_tensor has wrong shape: {state_tensor.shape}"
        p_actions = self._policy_net(state_tensor) # [n_actions]
        if self._debug:
            assert list(p_actions.shape) == [self._n_actions], f"p_actions has wrong shape: {p_actions.shape}"
        dist = Categorical(p_actions)
        action = dist.sample()
        self._log_prob.append(dist.log_prob(action))
        return action.item()

    # Reference: https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
    def control(self):
        G = 0
        policy_loss = []
        returns = deque()
        # reconstruct returns from MC 
        for reward in self._rewards[::-1]:
            G = self._discount_rate * G + reward 
            returns.appendleft(G)  # insert left to maintain same order as _log_prob
        returns_tensor = torch.tensor(returns, device=self._device)
        # batch norm
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + self._eps)
        if self._debug:
            assert list(returns_tensor.shape) == [len(self._rewards)], f"returns_tensor has wrong shape: {returns_tensor.shape}"
        # calcuate loss term
        for R, log_prob in zip(returns_tensor, self._log_prob):
            policy_loss.append((-R*log_prob).view(1)) # reshape to allow concat
        # sum up loss to a single value
        policy_loss_tensor = torch.cat(policy_loss).sum() 
        # backprop
        self._optimizer.zero_grad()
        policy_loss_tensor.backward()
        self._optimizer.step()
        # reset
        self.reset()

    def play_episode(self, env: gym.Env, learning: Optional[bool] = True, epsilon: Optional[float] = None, learning_rate: Optional[float] = None, video_path: Optional[str] = None):
        if video_path is not None:
            video = VideoRecorder(env, video_path)
        state, info = env.reset()
        terminal = False
        total_reward, num_steps = 0, 0
        if learning_rate is not None:
            self._learning_rate = learning_rate
        while not terminal:
            action = self.sample_action(state)
            new_state, reward, terminal, truncated, info = env.step(action)
            self._rewards.append(reward)
            terminal = terminal or truncated
            state = new_state
            total_reward += reward
            num_steps += 1
            if video_path is not None:
                video.capture_frame()
        if learning:
            self.control()
        if video_path is not None:
            video.close()
        return total_reward, num_steps

def test_agent():
    agent = PolicyGradientAgent(Box(low=0, high=1, shape=[4,4,3]), Discrete(2), 1.0, 0.1, 1.0)
    for _ in range(5):
        state = agent._state_space.sample()
        _ = agent.sample_action(state)
    agent._rewards = [1] * 5
    agent.control()
    print('policy_gradient_agent_test passed!')
test_agent()