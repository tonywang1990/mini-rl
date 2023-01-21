from __future__ import annotations
import numpy as np
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


class DQNAgent(Agent):
    def __init__(self, state_space: Space, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, learning: bool, batch_size: int, tau: float, eps_decay: float, net_params:list, update_freq:int):
        super().__init__(state_space, action_space,
                         discount_rate, epsilon, learning_rate, learning)
        # TAU is the update rate of the target network
        # LR is the learning rate of the AdamW optimizer
        self._batch_size = batch_size
        self._tau = tau
        self._eps_decay = eps_decay
        self._update_freq = update_freq
        # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._device = 'cpu'
        print(f'using device: {self._device}')
        MEMORY_SIZE = 10000

        # Get number of actions from gym action space
        self._n_actions = action_space.n
        self._state_dim = state_space.shape
        self._n_states = np.prod(np.array(self._state_dim)).astype(int)

        print(net_params)
        self._policy_net = DenseNet(self._n_states, self._n_actions, net_params, softmax=False).to(self._device)
        #self._policy_net = DenseNet1(self._n_states, self._n_actions).to(self._device)
        self._target_net = DenseNet(self._n_states, self._n_actions, net_params, softmax=False).to(self._device)
        #self._target_net = DenseNet1(self._n_states, self._n_actions).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())

        self._optimizer = optim.AdamW(
            self._policy_net.parameters(), lr=self._learning_rate, amsgrad=True)
        self._memory = utils.ReplayMemory(MEMORY_SIZE)

        self._step = 0
        self._debug = True

    def to_feature(self, data: np.ndarray) -> torch.Tensor:
        # Convert state into tensor and unsqueeze: insert a new dim into tensor (at dim 0): e.g. 1 -> [1] or [1] -> [[1]]
        # state: np.array
        # returns: torch.Tensor of shape [1]
        if self._debug:
            assert isinstance(
                data, np.ndarray), f'data is not of type ndarray: {type(data)}'
        return torch.tensor(data.flatten(), dtype=torch.float32, device=self._device).unsqueeze(0)

    def to_array(self, tensor: torch.Tensor, shape: list) -> np.ndarray:
        return tensor.cpu().numpy().reshape(shape)

    def sample_action(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> Union[int, float]:
        # state: tensor of shape [1, n_states]
        # return: tensor of shape [1, n_actions]
        sample = random.random()
        # eps_threshold = self._epsilon #self._epsilon.get(self._step)
        self._epsilon = utils.epsilon(self._step, eps_start=self._epsilon, eps_decay=self._eps_decay)
        self._step += 1
        if sample > self._epsilon:
            state_tensor = utils.to_feature(state, device=self._device)
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_prob = self._policy_net(state_tensor)
                if action_mask is not None:
                    large = torch.finfo(action_prob.dtype).max
                    action_prob -= (1-action_mask) * large
                # -1 is the non batch dimension 
                action = action_prob.max(-1)[1].item()
        else:
            if action_mask is not None:
                legal_actions = np.nonzero(action_mask)[0]
                action = np.random.choice(legal_actions)
            else:
                action = self._action_space.sample()
        return action

    def _optimize_model(self) -> float:
        if len(self._memory) < self._batch_size:
            return 0.0
        transitions = self._memory.sample(self._batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = utils.Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)  # [batch_size, n_states]
        # [batch_size, 1] - additional dim needed to perform gather
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)  # [batch_size]
        next_state_batch = torch.cat(
            batch.next_state)  # [batch_size, n_states]
        terminal_batch = torch.cat(batch.terminal)  # [batch_size]
        # update Q_policy to minimize:
        # reward + discount_rate * Q_target(next_state, new_action) - Q_policy(state, action)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self._policy_net(
            state_batch).gather(1, action_batch)  # [batch_size, 1]
        if self._debug:
            assert list(state_action_values.shape) == [
                self._batch_size, 1], f' {list(state_action_values.shape)} != {[self._batch_size, 1]}'

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = self._target_net(next_state_batch).max(
            1)[0] * (~terminal_batch)  # [batch_size, 1]
        # print(self._target_net(next_state_batch).max(1)[0])
        # print(terminal_batch)
        # print(next_state_values)
        if self._debug:
            assert list(next_state_values.shape) == [
                self._batch_size], f' {list(next_state_values.shape)} != {[self._batch_size]}'

        # next_state_values_new = torch.zeros(self._batch_size, device=self._device) # [batch_size]
        # print(batch.next_state)
        # next_state_batch = torch.cat(batch.next_state) # [batch_size]
        #next_state_values_new += self._target_net(next_state_batch).max(1)[0] * non_final_mask
        #assert torch.equal(next_state_values,next_state_values_new)

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self._discount_rate) + reward_batch  # [batch_size]
        if self._debug:
            assert list(expected_state_action_values.shape) == [
                self._batch_size], f' {list(expected_state_action_values.shape)} != {[self._batch_size]}'
        # print(expected_state_action_values.unsqueeze(1))
        # xx

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self._policy_net.parameters(), 100)
        self._optimizer.step()

        return loss.item()

    def _update_target_net(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self._target_net.state_dict()
        policy_net_state_dict = self._policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                self._tau + target_net_state_dict[key]*(1-self._tau)
        self._target_net.load_state_dict(target_net_state_dict)

    def post_process(self, state: np.ndarray, action: Union[int,float], reward: float, next_state: np.ndarray, terminal: bool):
        state_tensor = utils.to_feature(state).unsqueeze(0)
        action_tensor = torch.tensor([[action]], dtype=torch.long, device=self._device)
        reward_tensor = torch.tensor(
            [reward], dtype=torch.float32, device=self._device)
        if terminal:
            next_state_tensor = torch.zeros(
                self._n_states, device=self._device).unsqueeze(0)
        else:
            next_state_tensor = utils.to_feature(next_state).unsqueeze(0)
        terminal_tensor = torch.tensor(
            [terminal], device=self._device, dtype=torch.bool)
        # Store the transition in memory
        self._memory.push(state_tensor, action_tensor, next_state_tensor,
                          reward_tensor, terminal_tensor)

    # Perform one step of the optimization (on the policy network)
    def control(self) -> dict:
        if not self._learning:
            return {}
        loss = 0
        for _ in range(self._update_freq):
            loss += self._optimize_model() / self._update_freq 
            self._update_target_net()
        return {'value_loss': loss} 
    
    def update_weights_from(self, agent: DQNAgent, tau:float = 0.005):
        utils.update_weights(source_net=agent._policy_net, target_net=self._policy_net, tau=tau)

def test_agent():
    agent = DQNAgent(state_space=Box(low=0, high=1, shape=[4, 5, 3]), action_space=Discrete(
        2), discount_rate=0.9, epsilon=0.1, learning_rate=1e-3, learning=True, batch_size=2, tau=0.001, eps_decay=1000, net_params=[128,128], update_freq=500)
    state = agent._state_space.sample()
    next_state = agent._state_space.sample()
    action = agent.sample_action(state)
    agent.post_process(state, action, 1.0, next_state, terminal=False)
    agent.control()
    #agent.control(state, action, 1.0, next_state, terminal=False)
    #np.testing.assert_equal(state, next_state)
    print('dqn_agent test passed!')


test_agent()
