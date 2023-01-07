import numpy as np
from collections import namedtuple, deque
from gym.spaces import Discrete, Box, Space
import random
import gym
from typing import Union, Optional
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


class ActorCriticAgent(Agent):
    def __init__(self, state_space: Space, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, policy_lr: float, value_lr: float):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
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
        self._policy_net = DenseNet(
            self._n_states, self._n_actions, 32, 1, softmax=True).to(self._device)
        self._policy_optimizer = optim.AdamW(
            self._policy_net.parameters(), lr=self._policy_lr, amsgrad=True)
        #self._optimizer = optim.Adam(self._policy_net.parameters(), lr=self._learning_rate)

        # Value
        self._value_net = DenseNet(
            self._n_states, 1, 32, 1, softmax=False).to(self._device)
        self._value_optimizer = optim.AdamW(
            self._value_net.parameters(), lr=self._value_lr, amsgrad=True)

        self._step = 0
        self._debug = True

    def sample_action(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        # state: tensor of shape [n_states]
        # return: int
        state_tensor = utils.to_feature(state)  # [n_states]
        # policy
        p_actions = self._policy_net(state_tensor)  # [n_actions]
        dist = Categorical(p_actions)
        action = dist.sample()
        if self._debug:
            assert list(p_actions.shape) == [
                self._n_actions], f"p_actions has wrong shape: {p_actions.shape} != {[self._n_actions]}"
        return action.item(), dist.log_prob(action).view(1)

    def control(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool, action_log_prob: torch.Tensor):
        # calculate TD error
        state_tensor = utils.to_feature(state)  # [n_states]
        state_value_tensor = self._value_net(state_tensor)
        reward_tensor = torch.tensor([reward], device=self._device)
        # with torch.no_grad():
        if terminal:
            td_tensor = reward_tensor
        else:
            next_state_tensor = utils.to_feature(next_state)
            next_state_value_tensor = self._value_net(next_state_tensor)
            td_tensor = reward_tensor + self._discount_rate * next_state_value_tensor
        #td_tensor = td_tensor.detach()
        # Value Update
        criterion = nn.SmoothL1Loss()
        #state_value_tensor = torch.cat(self._state_value)
        #assert state_value_tensor.grad_fn is not None and td_tensor.grad_fn is None
        value_loss_tensor = criterion(td_tensor.detach(), state_value_tensor)
        # backprop
        self._value_optimizer.zero_grad()
        value_loss_tensor.backward()
        torch.nn.utils.clip_grad_value_(self._value_net.parameters(), 1000)
        self._value_optimizer.step()

        # Policy Update
        td_error_tensor = td_tensor - state_value_tensor
        policy_loss_tensor = -td_error_tensor.detach() * action_log_prob
        # backprop
        self._policy_optimizer.zero_grad()
        policy_loss_tensor.backward()
        torch.nn.utils.clip_grad_value_(self._policy_net.parameters(), 1000)
        self._policy_optimizer.step()

    def play_episode(self, env: gym.Env, learning: Optional[bool] = True, epsilon: Optional[float] = None, learning_rate: Optional[float] = None, video_path: Optional[str] = None):
        if video_path is not None:
            video = VideoRecorder(env, video_path)
        state, info = env.reset()
        terminal = False
        total_reward, num_steps = 0, 0
        if learning_rate is not None:
            self._learning_rate = learning_rate
        while not terminal:
            action, action_log_prob = self.sample_action(state)
            next_state, reward, terminal, truncated, info = env.step(action)
            if learning:
                self.control(state, action, reward, next_state,
                             terminal, action_log_prob)
            state = next_state
            terminal = terminal or truncated
            total_reward += reward
            num_steps += 1
            if video_path is not None:
                video.capture_frame()
        if video_path is not None:
            video.close()
        return total_reward, num_steps


def test_agent():
    agent = ActorCriticAgent(
        Box(low=0, high=1, shape=[4, 4, 3]), Discrete(2), 1.0, 0.1, None, 1.0, 1.0)
    state = agent._state_space.sample()
    action, prob = agent.sample_action(state)
    reward = 1
    next_state = agent._state_space.sample()
    agent.control(state, action, reward, next_state, False, prob)
    print('policy_gradient_agent_test passed!')


test_agent()
