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

class DenseNet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DenseNet, self).__init__()
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
        self._device = 'cpu' #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f'using device: {self._device}')
        MEMORY_SIZE = 10000

        # Get number of actions from gym action space
        n_actions = action_space.n
        n_states = len(state_space.sample())

        self._policy_net = DenseNet(n_states, n_actions).to(self._device)
        self._target_net = DenseNet(n_states, n_actions).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())

        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=self._learning_rate, amsgrad=True)
        self._memory = ReplayMemory(MEMORY_SIZE)

        self._step = 0

    def init_state(self, state: list) -> torch.Tensor:
        # Convert state into tensor and unsqueeze: insert a new dim into tensor (at dim 0): e.g. 1 -> [1] or [1] -> [[1]] 
        # state: Int
        # returns: torch.Tensor of shape [1]
        # why increase dimension??
        return torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0) 

    def sample_action(self, state: torch.Tensor):
        # state: tensor of shape [1, n_states]
        # return: tensor of shape [1, n_actions]
        sample = random.random()
        eps_threshold = self._epsilon #self._epsilon.get(self._step)
        self._step += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self._policy_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[self._action_space.sample()]], device=self._device, dtype=torch.long)
        return action

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
            non_final_next_states = None

        state_batch = torch.cat(batch.state) # [batch_size, n_states]
        action_batch = torch.cat(batch.action) # [batch_size, 1]
        reward_batch = torch.cat(batch.reward) # [batch_size]

        # update Q_policy to minimize:
        # reward + discount_rate * Q_target(new_state, new_action) - Q_policy(state, action)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self._policy_net(state_batch).gather(1, action_batch) # [batch_size, 1] 

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self._batch_size, device=self._device) # [batch_size]
        if non_final_next_states is not None: 
            with torch.no_grad():
                next_state_values[non_final_mask] = self._target_net(non_final_next_states).max(1)[0]
        
        #next_state_values_new = torch.zeros(self._batch_size, device=self._device) # [batch_size]
        #print(batch.next_state)
        #next_state_batch = torch.cat(batch.next_state) # [batch_size]
        #next_state_values_new += self._target_net(next_state_batch).max(1)[0] * non_final_mask
        #assert torch.equal(next_state_values,next_state_values_new)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._discount_rate) + reward_batch # [batch_size]

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

    def control(self, state: torch.Tensor, action: torch.Tensor, reward: float, new_state: list, terminal: bool):
        reward = torch.tensor([reward], dtype=torch.float32, device=self._device)
        if terminal:
            next_state = None
        else:
            next_state = torch.tensor(new_state, dtype=torch.float32, device=self._device).unsqueeze(0)
        # Store the transition in memory
        self._memory.push(state, action, next_state, reward)
        # Perform one step of the optimization (on the policy network)
        self._optimize_model()
        self._update_target_net()


    def play_episode(self, env: gym.Env, learning: Optional[bool] = True, epsilon: Optional[float] = None, learning_rate: Optional[float] = None, video_path: Optional[str] = None):
        if video_path is not None:
            video = VideoRecorder(env, video_path)
        state, info = env.reset()
        terminal = False
        steps = 0
        total_reward = 0
        if epsilon is not None:
            self._epsilon = epsilon
        if learning_rate is not None:
            self._learning_rate = learning_rate
        while not terminal:
            state = self.init_state(state)
            action = self.sample_action(state)
            new_state, reward, terminal, truncated, info = env.step(action.item())
            total_reward += reward 
            terminal = terminal or truncated
            if learning:
                self.control(state, action, reward,
                             new_state, terminal)
            state = new_state
            steps += 1
            #if steps > 1000:
            #    terminal = True
            if video_path is not None:
                video.capture_frame()
        if video_path is not None:
            video.close()
        return total_reward, steps
