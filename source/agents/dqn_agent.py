import numpy as np
from numpy.core.getlimits import inf
from collections import namedtuple, deque
from gym.spaces import Discrete, Box, Space
import random
import gym
from typing import Union, Optional
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import math

from source.value_function import ActionValue, LearnedActionValue
from source.agents.agent import Agent
from source import utils


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer_output = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer_output(x)

class Epsilon(object):
    def __init__(self):
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self._eps_start = 0.9
        self._eps_end = 0.05
        self._eps_decay = 1000
    def get(self, step: int) -> float:
        return self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1. * step / self._eps_decay)

    


class DQNAgent(Agent):
    def __init__(self, state_space: Space, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, batch_size: int, tau: float):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        # TAU is the update rate of the target network
        # LR is the learning rate of the AdamW optimizer
        self._batch_size = batch_size
        self._tau = tau
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get number of actions from gym action space
        n_actions = action_space.n
        n_states = len(state_space.sample())

        self._policy_net = DQN(n_states, n_actions).to(self._device)
        self._target_net = DQN(n_states, n_actions).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())

        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=self._learning_rate, amsgrad=True)
        self._memory = ReplayMemory(10000)

        self._step = 0
    
    def init_state(self, state: list) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)

    def sample_action(self, state: torch.Tensor):
        sample = random.random()
        eps_threshold = self._epsilon #self._epsilon.get(self._step)
        self._step += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self._policy_net(state).max(1)[1].view(1, 1).item()
        else:
            return torch.tensor([[self._action_space.sample()]], device=self._device, dtype=torch.long).item()

    def _optimize_model(self):
        if len(self._memory) < self._batch_size:
            return
        transitions = self._memory.sample(self._batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self._device, dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states) != 0:
            non_final_next_states = torch.cat(non_final_next_states)
        else:
            print('warning: no non_final_next_states')
            return
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self._policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self._batch_size, device=self._device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self._target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._discount_rate) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self._policy_net.parameters(), 100)
        self._optimizer.step()
    
    def _update_target_net(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self._target_net.state_dict()
        policy_net_state_dict = self._policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self._tau + target_net_state_dict[key]*(1-self._tau)
        self._target_net.load_state_dict(target_net_state_dict)

     # SARS(A) on policy control
    def control(self, state: torch.Tensor, action: int, reward: float, new_state: list, terminal: bool):
        action = torch.tensor([[action]], device=self._device)
        reward = torch.tensor([reward], device=self._device)
        if terminal:
            next_state = None
        else:
            next_state = torch.tensor(new_state, dtype=torch.float32, device=self._device).unsqueeze(0)
        # Store the transition in memory
        self._memory.push(state, action, next_state, reward)
        # Perform one step of the optimization (on the policy network)
        self._optimize_model()
        self._update_target_net()


