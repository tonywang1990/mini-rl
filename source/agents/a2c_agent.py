import numpy as np
from collections import namedtuple, deque
from gym.spaces import Discrete, Box, Space
import random
import gym
from typing import Union, Optional, Any, Tuple
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import math

from source.agents.agent import Agent
from source.utils import utils
from source.net import DenseNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class A2CAgent(Agent):
    def __init__(self, state_space: Space, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, policy_lr: float, value_lr: float, net_params: list, tempreture: float):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._rewards = []
        self._log_prob = []
        self._state_value = []
        self._eps = np.finfo(np.float32).eps.item()
        # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._device = 'cpu'
        print(f'using device: {self._device}')
        self._policy_lr = policy_lr
        self._value_lr = value_lr

        # Get number of actions from gym action space
        self._n_actions = action_space.n
        self._state_dim = state_space.sample().shape
        self._n_states = len(state_space.sample().flatten())

        # Policy
        self._policy_net = DenseNet(self._n_states, self._n_actions,
                                    net_params, softmax=True, tempreture=tempreture).to(self._device)
        self._policy_optimizer = optim.AdamW(
            self._policy_net.parameters(), lr=self._policy_lr, amsgrad=True)
        #self._optimizer = optim.Adam(self._policy_net.parameters(), lr=self._learning_rate)

        # Value
        self._value_net = DenseNet(
            self._n_states, 1, net_params, softmax=False).to(self._device)
        self._value_optimizer = optim.AdamW(
            self._value_net.parameters(), lr=self._value_lr, amsgrad=True)

        self._step = 0
        self._debug = True
        #torch.autograd.set_detect_anomaly(True)

    def reset(self):
        del self._rewards[:]
        del self._log_prob[:]
        del self._state_value[:]

    def sample_action(self, state: np.ndarray, mask: Optional[np.ndarray]=None) -> Tuple[int, dict]:
        # state: tensor of shape [n_states]
        # return: int
        state_tensor = utils.to_feature(state)  # [n_states]
        p_actions = self._policy_net(state_tensor)  # [n_actions]
        if self._debug:
            assert list(state_tensor.shape) == [
                self._n_states], f"state_tensor has wrong shape: {state_tensor.shape}"
            assert list(p_actions.shape) == [
                self._n_actions], f"p_actions has wrong shape: {p_actions.shape}"
            assert ~np.isnan(p_actions.sum().item()), (p_actions, state_tensor)
            if mask is not None:
                assert list(p_actions.shape) == list(mask.shape), f"mask has the wrong shape: {mask.shape} != {p_actions.shape}"
        if mask is not None:
            p_actions = (p_actions + 1e-20) * torch.from_numpy(mask)
        dist = Categorical(p_actions)
        action = dist.sample()
        self._log_prob.append(dist.log_prob(action).view(1))
        return action.item(), dict(logp=dist.log_prob(action).view(1))

    def control(self) -> dict:
        if not self._learning: 
            return {}
        G = 0
        returns = deque()
        # reconstruct returns from MC
        for reward in self._rewards[::-1]:
            G = self._discount_rate * G + reward
            # insert left to maintain same order as _log_prob
            returns.appendleft(G)
        with torch.no_grad():
            returns_tensor = torch.tensor(returns, device=self._device)
            # batch norm
            returns_tensor = (returns_tensor - returns_tensor.mean()
                          ) / (returns_tensor.std() + self._eps)
        if self._debug:
            assert list(returns_tensor.shape) == [
                len(self._rewards)], f"returns_tensor has wrong shape: {returns_tensor.shape}"

        # Value Update
        #criterion = nn.SmoothL1Loss()
        state_value_tensor = torch.cat(self._state_value)
        #state_value_tensor = (state_value_tensor - state_value_tensor.mean()) / (state_value_tensor.std() + self._eps)
        assert returns_tensor.requires_grad == False and state_value_tensor.requires_grad == True
        assert ~np.isnan(returns_tensor.sum().item()) and ~np.isnan(state_value_tensor.sum().item())
        value_loss_tensor = ((returns_tensor - state_value_tensor)**2).mean() #pyre-fixme[58]
        # backprop
        self._value_optimizer.zero_grad()
        value_loss_tensor.backward()
        self._value_optimizer.step()

        # Policy Update
        log_prob_tensor = torch.cat(self._log_prob)
        advantage_tensor = (returns_tensor - state_value_tensor).detach()
        assert advantage_tensor.requires_grad == False and log_prob_tensor.requires_grad == True
        assert ~np.isnan(log_prob_tensor.mean().item())
        policy_loss_tensor = (-advantage_tensor * log_prob_tensor).mean()
        # backprop
        self._policy_optimizer.zero_grad()
        policy_loss_tensor.backward()
        self._policy_optimizer.step()

        # reset
        self.reset()

        return {'value_loss': value_loss_tensor.item(), 'policy_loss':policy_loss_tensor.item()}
    
    def post_process(self, state: Any, action: Any, reward: float, next_state: Any, terminal: bool, action_info: dict):
        state_tensor = utils.to_feature(state)  # [n_states]
        state_value = self._value_net(state_tensor)
        self._state_value.append(state_value)
        self._rewards.append(reward)

def test_agent():
    agent = A2CAgent(Box(low=0, high=1, shape=[4, 4, 3]), Discrete(
        2), 1.0, 0.1, None, 1.0, 1.0, [8], 1)
    for _ in range(5):
        state = agent._state_space.sample()
        action, info = agent.sample_action(state)
    #agent._rewards = [1] * 5
        agent.post_process(state, action, 1, None, None, action_info=info)
    agent.control()
    print('a2c_agent test passed!')


test_agent()
