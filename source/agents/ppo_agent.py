from __future__ import annotations

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
import scipy.signal


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer(object):
    def __init__(self, state_dim: int, action_dim: int, batch_size: int, discount_rate: float, gae_lambda: float):
        self._observation_buffer = np.zeros(
            (batch_size, state_dim), dtype=np.float32)
        assert action_dim == 1, 'only single dimension action is supported!'
        self._action_buffer = np.zeros((batch_size), dtype=np.float32)
        self._reward_buffer = np.zeros(batch_size, dtype=np.float32)
        self._advantage_buffer = np.zeros(batch_size, dtype=np.float32)
        self._log_prob_buffer = np.zeros(batch_size, dtype=np.float32)
        self._state_value_buffer = np.zeros(batch_size, dtype=np.float32)
        self._return_buffer = np.zeros(batch_size, dtype=np.float32)
        self._discount_rate = discount_rate
        self._gae_lambda = gae_lambda
        self._ptr, self._episode_start_ptr, self._batch_size = 0, 0, batch_size
        self._last_value = 0.0

    def store(self, observation: np.ndarray, action: int, reward: float, state_value: float, logp: float):
        if self._ptr >= self._batch_size:
            return
        #assert self._ptr < self._batch_size, "buffer is full!"
        self._observation_buffer[self._ptr] = observation
        self._action_buffer[self._ptr] = action
        self._reward_buffer[self._ptr] = reward
        self._state_value_buffer[self._ptr] = state_value
        self._log_prob_buffer[self._ptr] = logp
        self._ptr += 1

    def process_episode(self):
        path_slice = slice(self._episode_start_ptr, self._ptr)
        rewards = np.append(self._reward_buffer[path_slice], self._last_value)
        state_values = np.append(
            self._state_value_buffer[path_slice], self._last_value)
        self._last_value = 0.0
        # GAE advantage
        gae_delta = rewards[:-1] + self._discount_rate * \
            state_values[1:] - state_values[:-1]
        self._advantage_buffer[path_slice] = discount_cumsum(
            gae_delta, self._discount_rate * self._gae_lambda)
        #print(self._advantage_buffer[path_slice] )
        # return
        self._return_buffer[path_slice] = discount_cumsum(
            rewards, self._discount_rate)[:-1]

        self._episode_start_ptr = self._ptr

    def get(self) -> Optional[dict[str, torch.Tensor]]:
        if not self.is_full():
            return None
        # Reset buffer
        self._ptr, self._episode_start_ptr = 0, 0
        #assert self._ptr == self._batch_size, f'buffer not full: {self._ptr}'
        mean, std = np.mean(self._advantage_buffer), np.std(
            self._advantage_buffer)
        self._advantage_buffer = (self._advantage_buffer - mean) / std
        mean, std = np.mean(self._return_buffer), np.std(
            self._return_buffer)
        #self._return_buffer = (self._return_buffer - mean) / std
        data = dict(obs=self._observation_buffer, act=self._action_buffer,
                   ret=self._return_buffer, adv=self._advantage_buffer, logp=self._log_prob_buffer)
        return {k: torch.as_tensor(v) for k, v in data.items()}

    def is_full(self) -> bool:
        assert self._ptr <= self._batch_size, f'ptr overflowed {self._ptr}'
        return self._ptr == self._batch_size
    
    def set_last_value(self, last_value: float):
        self._last_value = last_value


def test_ppo_buffer():
    buf = PPOBuffer(4, 1, 8, 0.8, 0.9)
    data = buf.get()
    assert data is None
    for _ in range(8):
        buf.store([1, 1, 1, 1], 2, 0.1, 0.5, -15)
    buf.set_last_value(1.0)
    buf.process_episode()
    data = buf.get()
    np.testing.assert_allclose(data['adv'], [-1.0663, -0.9338, -0.7498, -0.4941, -0.1391,  0.3540,  1.0389,  1.9901], rtol=0.001)
    print('PPOBuffer test passed!')


class PPOAgent(Agent):
    # Default param values are from openai spinup
    def __init__(self, state_space: Space, action_space: Discrete, net_params: list, discount_rate: float = 0.99, epsilon: float = -1, learning_rate: float = -1, policy_lr: float = 3e-4, value_lr: float = 1e-4, gae_lambda: float = 0.97, clip_ratio: float = 0.1, num_updates: int = 80, batch_size: int = 1000):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._eps = np.finfo(np.float32).eps.item()
        self._clip_ratio = clip_ratio
        self._num_updates = num_updates
        # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._device = 'cpu'
        print(f'using device: {self._device}')
        self._policy_lr = policy_lr
        self._value_lr = value_lr
        self._target_kl = 0.01

        # Get number of actions from gym action space
        self._n_actions = action_space.n
        self._state_dim = state_space.sample().shape
        self._n_states = len(state_space.sample().flatten())

        self._buffer = PPOBuffer(state_dim=self._n_states, action_dim=1,
                                 batch_size=batch_size, discount_rate=discount_rate, gae_lambda=gae_lambda)

        # Policy
        self._policy_net = DenseNet(self._n_states, self._n_actions,
                                    net_params, softmax=True).to(self._device)
        self._policy_optimizer = optim.AdamW(
            self._policy_net.parameters(), lr=self._policy_lr, amsgrad=True)
        # self._optimizer = optim.Adam(self._policy_net.parameters(), lr=self._learning_rate)

        # Value
        self._value_net = DenseNet(
            self._n_states, 1, net_params, softmax=False).to(self._device)
        self._value_optimizer = optim.AdamW(
            self._value_net.parameters(), lr=self._value_lr, amsgrad=True)

        self._step = 0
        self._debug = True

    def sample_action(self, state: Union[torch.Tensor, np.ndarray], mask: Optional[np.ndarray] = None) -> Tuple[Union[bool, float, int], dict]:
        # state: tensor of shape [n_states]
        # return: int
        # predict actions, gradients required
        action_info = {}
        if isinstance(state, np.ndarray):
            state = utils.to_feature(state)
        with torch.no_grad():
            p_actions = self._policy_net(state)  # [n_actions]
            if self._debug:
                assert list(p_actions.shape) == [
                    self._n_actions], f"p_actions has wrong shape: {p_actions.shape}"
                if mask is not None:
                    assert list(p_actions.shape) == list(
                        mask.shape), f"mask has the wrong shape: {mask.shape} != {p_actions.shape}"
                    assert ~np.isnan(p_actions[0]), (p_actions, mask, state)
            if mask is not None:
                mp_actions = p_actions * torch.from_numpy(mask)
            else:
                mp_actions = p_actions
            dist = Categorical(mp_actions)
            action = dist.sample()
        action_info['logp'] = Categorical(p_actions).log_prob(action)
        return action.item(), action_info

    def post_process(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool, action_info: dict):
        with torch.no_grad():
            state_value = self._value_net(utils.to_feature(state))
        self._buffer.store(state.flatten(), action, reward, state_value.item(), action_info['logp'])  
        # bootstrap with the last state value if the end of buffer is not a terminal state.
        if self._buffer.is_full():
            if terminal: 
                last_value = 0
            else:
                with torch.no_grad():
                    last_value = self._value_net(utils.to_feature(next_state)).item()
            self._buffer.set_last_value(last_value)

    def compute_policy_loss(self, data: dict) -> Tuple[torch.Tensor, dict]:
        # data is Batched dataset: [batch_size, ...]
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        p_actions = self._policy_net(obs) 
        dist = Categorical(p_actions)
        logp = dist.log_prob(act)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self._clip_ratio, 1+self._clip_ratio) * adv
        loss = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        entropy = dist.entropy().mean().item()
        clipped = ratio.gt(1+self._clip_ratio) | ratio.lt(1-self._clip_ratio)
        clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        return loss, dict(approx_kl=approx_kl, entropy=entropy, clip_frac=clip_frac)

    def compute_value_loss(self, data: dict) -> torch.Tensor:
        # data is Batched dataset: [batch_size, ...]
        obs, returns = data['obs'], data['ret']
        state_value = self._value_net(obs)
        loss = ((torch.squeeze(state_value) - returns) ** 2).mean()
        return loss

    def control(self) -> Optional[dict]:
        # Process data of this episode in the buffer.
        self._buffer.process_episode()
        # Check if buffer is full.
        data = self._buffer.get()
        if data is None:
            return None
        # If buffer is full, start learning.
        #print('buffer full, start training...')
        # Value Update
        for _ in range(self._num_updates): #self._num_updates):
            value_loss = self.compute_value_loss(data)
            assert ~np.isnan(value_loss.item()), 'value loss is nan!' and value_loss.requires_grad == True
            # backprop
            self._value_optimizer.zero_grad()
            value_loss.backward()
            self._value_optimizer.step()

        num_policy_update = 0
        # Policy Update
        #policy_loss = torch.tensor(0)
        for _ in range(self._num_updates):
            policy_loss, info = self.compute_policy_loss(data)
            # If KL divergence is too large, the new policy is diverging from old policy, stop training since it could lead to unstable/bad udpates.
            if info['approx_kl'] > 1.5 * self._target_kl:
                #print(f'Early stopping at step {_} due to {mean_approx_kl} reaching max kl.')
                break
            num_policy_update += 1
            assert ~np.isnan(policy_loss.item()), 'policy loss is nan!'
            # backprop
            self._policy_optimizer.zero_grad()
            policy_loss.backward()
            self._policy_optimizer.step()

        return {'value_loss': value_loss.item(), 'policy_loss': policy_loss.item(), 'num_policy_udpate': num_policy_update}
 
    def update_weights_from(self, agent: PPOAgent, tau: float = 0.01):
        utils.update_weights(source_net=agent._value_net,
                             target_net=self._value_net, tau=tau)
        utils.update_weights(source_net=agent._policy_net,
                             target_net=self._policy_net, tau=tau)


def test_agent():
    batch_size=5
    agent = PPOAgent(state_space=Box(low=0, high=1, shape=[4, 1]), action_space=Discrete(
        2), net_params=[8], batch_size=batch_size)
    for _ in range(batch_size):
        state = agent._state_space.sample()
        action, info = agent.sample_action(state)
        agent.post_process(state,
                           action, 0.5, agent._state_space.sample(), False, info)
    # print(agent._memory)
    agent.control()
    print('ppo_agent passed!')


#test_ppo_buffer()
test_agent()
