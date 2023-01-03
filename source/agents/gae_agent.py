import numpy as np
from collections import namedtuple, deque, defaultdict
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


class GAEAgent(Agent):
    def __init__(self, state_space: Space, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, policy_lr: float, value_lr: float, net_params: dict, exp_average_discount: float):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._log_prob = []
        self._transitions = []
        self._eps = np.finfo(np.float32).eps.item()
        self._exp_average_discount = exp_average_discount
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
                                    net_params['width'], net_params['n_hidden'], softmax=True).to(self._device)
        self._policy_optimizer = optim.AdamW(
            self._policy_net.parameters(), lr=self._policy_lr, amsgrad=True)
        # self._optimizer = optim.Adam(self._policy_net.parameters(), lr=self._learning_rate)

        # Value
        self._value_net = DenseNet(
            self._n_states, 1, net_params['width'], net_params['n_hidden'], softmax=False).to(self._device)
        self._value_optimizer = optim.AdamW(
            self._value_net.parameters(), lr=self._value_lr, amsgrad=True)

        self._step = 0
        self._debug = True

    def reset(self):
        del self._log_prob[:]
        del self._transitions[:]


    def sample_action(self, state: np.ndarray) -> int:
        # state: tensor of shape [n_states]
        # return: int
        # predict actions, gradients required
        p_actions = self._policy_net(utils.to_feature(state))  # [n_actions]
        if self._debug:
            assert list(p_actions.shape) == [
                self._n_actions], f"p_actions has wrong shape: {p_actions.shape}"
        dist = Categorical(p_actions)
        action = dist.sample()
        self._log_prob.append(dist.log_prob(action).view(1))
        return action.item()

    def post_process(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool):
        # Calculate state 
        state_tensor = self._value_net(utils.to_feature(state))
        with torch.no_grad():
            next_state_tensor = self._value_net(utils.to_feature(next_state))
        self._transitions.append(utils.Transition(
            state_tensor, action, next_state_tensor, reward, terminal))

    def process_batch(self, batch_size:int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate advantage.
        adv_list = list()
        advantage = torch.tensor(0, dtype=torch.float)
        for trans in reversed(self._transitions):
            state_tensor, _, next_state_tensor, reward, terminal = trans
            delta = reward + self._discount_rate * \
                next_state_tensor * (1 - terminal) - state_tensor
            advantage = delta + self._discount_rate * \
                self._exp_average_discount * advantage * (1 - terminal)
            adv_list.append(advantage)
        adv_list.reverse()
        # Get batch size data.
        if batch_size != -1 and batch_size < len(adv_list):
            adv_list = adv_list[:batch_size]
            self._log_prob = self._log_prob[:batch_size]
        adv_tensor = torch.concat(adv_list)
        prob_tensor = torch.concat(self._log_prob)
        return adv_tensor, prob_tensor

    def control(self):
        advantage_tensor, log_prob_tensor = self.process_batch()
        # Value Update.
        value_loss_tensor = (advantage_tensor ** 2).sum()
        # backprop
        self._value_optimizer.zero_grad()
        value_loss_tensor.backward()
        self._value_optimizer.step()

        # Policy Update
        #print(log_prob_tensor.shape)
        assert log_prob_tensor.requires_grad == True and advantage_tensor.shape == log_prob_tensor.shape
        policy_loss_tensor = (-advantage_tensor.detach()
                              * log_prob_tensor).sum()
        # backprop
        self._policy_optimizer.zero_grad()
        policy_loss_tensor.backward()
        self._policy_optimizer.step()

        # reset
        self.reset()


def test_agent():
    agent = GAEAgent(Box(low=0, high=1, shape=[4, 1]), Discrete(
        2), 1.0, 0.1, None, 1.0, 1.0, {'width': 8, 'n_hidden': 1}, 0.5)
    for _ in range(2):
        state = agent._state_space.sample()
        _ = agent.sample_action(state)
        agent.post_process(np.array([1.0,2,3,4]), 1, 0.5, np.array([-1,-1,-1,-1]), False)
    # print(agent._memory)
    agent.control()
    print('policy_gradient_agent_test passed!')


test_agent()
