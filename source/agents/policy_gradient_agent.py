import numpy as np
from collections import namedtuple, deque
from gym.spaces import Discrete, Box, Space
import random
import gym
from typing import Union, Optional, Any
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import math

from source.agents.agent import Agent
from source.utils import utils
from source.net import DenseNet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyGradientAgent(Agent):
    def __init__(self, state_space: Space, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, net_params: dict):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._rewards = []
        self._log_prob = []
        self._eps = np.finfo(np.float32).eps.item()
        # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._device = 'cpu'
        print(f'using device: {self._device}')

        # Get number of actions from gym action space
        self._n_actions = action_space.n
        self._state_dim = state_space.sample().shape
        self._n_states = len(state_space.sample().flatten())

        self._policy_net = DenseNet(self._n_states, self._n_actions,
                                    net_params['width'], net_params['n_hidden'], softmax=True).to(self._device)
        self._optimizer = optim.AdamW(
            self._policy_net.parameters(), lr=self._learning_rate, amsgrad=True)
        #self._optimizer = optim.Adam(self._policy_net.parameters(), lr=self._learning_rate)

        self._step = 0
        self._debug = True

    def reset(self):
        del self._rewards[:]
        del self._log_prob[:]

    def sample_action(self, state: np.ndarray) -> int:
        # state: tensor of shape [n_states]
        # return: int
        state_tensor = utils.to_feature(state)  # [n_states]
        if self._debug:
            assert list(state_tensor.shape) == [
                self._n_states], f"state_tensor has wrong shape: {state_tensor.shape}"
        p_actions = self._policy_net(state_tensor)  # [n_actions]
        if self._debug:
            assert list(p_actions.shape) == [
                self._n_actions], f"p_actions has wrong shape: {p_actions.shape}"
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
            # insert left to maintain same order as _log_prob
            returns.appendleft(G)
        returns_tensor = torch.tensor(returns, device=self._device)
        # batch norm
        returns_tensor = (returns_tensor - returns_tensor.mean()
                          ) / (returns_tensor.std() + self._eps)
        if self._debug:
            assert list(returns_tensor.shape) == [
                len(self._rewards)], f"returns_tensor has wrong shape: {returns_tensor.shape}"
        # calcuate loss term
        for R, log_prob in zip(returns_tensor.detach(), self._log_prob):
            # reshape to allow concat
            policy_loss.append((-R*log_prob).view(1))
        # sum up loss to a single value
        policy_loss_tensor = torch.cat(policy_loss).sum()
        # backprop
        self._optimizer.zero_grad()
        policy_loss_tensor.backward()
        self._optimizer.step()
        # reset
        self.reset()

    def post_process(self, state: Any, action: Any, reward: float, next_state: Any, terminal: bool):
        self._rewards.append(reward)

def test_agent():
    agent = PolicyGradientAgent(Box(low=0, high=1, shape=[4, 4, 3]), Discrete(
        2), 1.0, 0.1, 1.0, {'width': 8, 'n_hidden': 1})
    for _ in range(5):
        state = agent._state_space.sample()
        _ = agent.sample_action(state)
    agent._rewards = [1] * 5
    agent.control()
    print('policy_gradient_agent_test passed!')


test_agent()
