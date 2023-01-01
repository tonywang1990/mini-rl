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


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class DenseNet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DenseNet, self).__init__()
        width = 32
        self.layer1 = nn.Linear(n_observations, width)
        self.layer2 = nn.Linear(width, width)
        self.layer3 = nn.Linear(width, width)
        self.layer4 = nn.Linear(width, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        #x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return F.softmax(x, dim=-1)

class PolicyGradientWithBaselineAgent(Agent):
    def __init__(self, state_space: Space, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, policy_lr: float, value_lr: float):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._rewards = []
        self._log_prob = []
        self._state_value = []
        self._eps = np.finfo(np.float32).eps.item()
        self._device = 'cpu' #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f'using device: {self._device}')
        self._policy_lr = policy_lr
        self._value_lr = value_lr

        # Get number of actions from gym action space
        self._n_actions = action_space.n
        self._state_dim = state_space.sample().shape
        self._n_states = len(state_space.sample().flatten())

        # Policy
        self._policy_net = DenseNet(self._n_states, self._n_actions).to(self._device)
        self._policy_optimizer = optim.AdamW(self._policy_net.parameters(), lr=self._policy_lr, amsgrad=True)
        #self._optimizer = optim.Adam(self._policy_net.parameters(), lr=self._learning_rate)
        
        # Value 
        self._value_net = DenseNet(self._n_states, 1).to(self._device)
        self._value_optimizer = optim.AdamW(self._value_net.parameters(), lr=self._value_lr, amsgrad=True)

        self._step = 0
        self._debug = True

    def reset(self):
        del self._rewards[:]
        del self._log_prob[:]
        del self._state_value[:]

    def sample_action(self, state: np.ndarray) -> int:
        # state: tensor of shape [n_states]
        # return: int
        state_tensor = utils.to_feature(state) # [n_states]
        if self._debug:
            assert list(state_tensor.shape) == [self._n_states], f"state_tensor has wrong shape: {state_tensor.shape}"
        p_actions = self._policy_net(state_tensor) # [n_actions]
        state_value = self._value_net(state_tensor)
        if self._debug:
            assert list(p_actions.shape) == [self._n_actions], f"p_actions has wrong shape: {p_actions.shape}"
        dist = Categorical(p_actions)
        action = dist.sample()
        self._log_prob.append(dist.log_prob(action).view(1))
        self._state_value.append(state_value)
        return action.item()
    
    def control(self):
        G = 0
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

        ## Value Update
        criterion = nn.SmoothL1Loss()
        state_value_tensor = torch.cat(self._state_value) 
        #state_value_tensor = (state_value_tensor - state_value_tensor.mean()) / (state_value_tensor.std() + self._eps)
        assert returns_tensor.requires_grad == False and state_value_tensor.requires_grad == True
        value_loss_tensor = criterion(returns_tensor.detach(), state_value_tensor)
        # backprop
        self._value_optimizer.zero_grad()
        value_loss_tensor.backward()
        self._value_optimizer.step()

        ## Policy Update
        log_prob_tensor = torch.cat(self._log_prob)
        advantage_tensor = (returns_tensor - state_value_tensor).detach()
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + self._eps)
        assert advantage_tensor.requires_grad == False and log_prob_tensor.requires_grad == True
        policy_loss_tensor = (-advantage_tensor * log_prob_tensor).sum()
        # backprop
        self._policy_optimizer.zero_grad()
        policy_loss_tensor.backward()
        self._policy_optimizer.step()

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
    agent = PolicyGradientWithBaselineAgent(Box(low=0, high=1, shape=[4,4,3]), Discrete(2), 1.0, 0.1, None, 1.0, 1.0)
    for _ in range(5):
        state = agent._state_space.sample()
        _ = agent.sample_action(state)
    agent._rewards = [1] * 5
    agent.control()
    print('policy_gradient_agent_test passed!')
test_agent()