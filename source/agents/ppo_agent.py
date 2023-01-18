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



class PPOAgent(Agent):
    # Default param values are from openai spinup
    def __init__(self, state_space: Space, action_space: Discrete, net_params: dict, discount_rate: float = 0.99, epsilon: float = -1, learning_rate: float = -1, policy_lr: float = 3e-4, value_lr: float=1e-4, gae_lambda: float=0.97, clip_ratio: float=0.1, num_updates: int=80, entropy_coeff:float = 0.01):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._log_probs = []
        self._transitions = []
        self._eps = np.finfo(np.float32).eps.item()
        self._gae_lambda = gae_lambda
        self._clip_ratio = clip_ratio
        self._num_updates = num_updates
        # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._device = 'cpu'
        print(f'using device: {self._device}')
        self._policy_lr = policy_lr
        self._value_lr = value_lr
        self._target_kl = 0.01 
        self._entropy_coeff = entropy_coeff

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
        del self._log_probs[:]
        del self._transitions[:]

    def sample_action(self, state: Union[torch.Tensor, np.ndarray], mask: Optional[np.ndarray]=None) -> Union[bool, float, int]:
        # state: tensor of shape [n_states]
        # return: int
        # predict actions, gradients required
        if isinstance(state, np.ndarray):
            state = utils.to_feature(state)
        with torch.no_grad():
            p_actions = self._policy_net(state)  # [n_actions]
            if self._debug:
                assert list(p_actions.shape) == [
                    self._n_actions], f"p_actions has wrong shape: {p_actions.shape}"
                if mask is not None:
                    assert list(p_actions.shape) == list(mask.shape), f"mask has the wrong shape: {mask.shape} != {p_actions.shape}"
                    assert ~np.isnan(p_actions[0]), (p_actions, mask, state)
            if mask is not None:
                mp_actions = p_actions * torch.from_numpy(mask)
            else:
                mp_actions = p_actions
            dist = Categorical(mp_actions)
            action = dist.sample()
        self._log_probs.append(Categorical(p_actions).log_prob(action))
        return action.item()

    def post_process(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool):
        # Calculate state
        self._transitions.append(utils.Transition(
            utils.to_feature(state), torch.tensor(action), utils.to_feature(next_state), reward, terminal))

    def process_batch(self, batch_size: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        # Calculate advantage.
        adv_list, imp_sample, kl_divs, entropy = list(), list(), list(), list()
        advantage = torch.tensor(0, dtype=torch.float)
        for trans in reversed(self._transitions):
            # Calculate advantage using GAE
            state_tensor, action_tensor, next_state_tensor, reward, terminal = trans
            with torch.no_grad():
                next_state_value_tensor = self._value_net(next_state_tensor)
            state_value_tensor = self._value_net(state_tensor)
            delta = reward + self._discount_rate * \
                next_state_value_tensor * (1 - terminal) - state_value_tensor
            advantage = delta + self._discount_rate * \
                self._gae_lambda * advantage * (1 - terminal)
            adv_list.append(advantage)
        adv_list.reverse()

        for trans, prev_log_prob in zip(self._transitions, self._log_probs):
            # Calculate importance sample
            state_tensor, action_tensor, *_ = trans
            p_actions = self._policy_net(state_tensor)
            dist = Categorical(p_actions)
            entropy.append(dist.entropy().view(1))
            log_prob = dist.log_prob(action_tensor) 
            imp_sample.append(torch.exp(log_prob - prev_log_prob).view(1))
            assert ~np.isnan(imp_sample[-1].item()), 'importance sampling ratio is nan!'
            kl_divs.append(prev_log_prob.item() - log_prob.item())

        # Get batch size data.
        if batch_size != -1 and batch_size < len(adv_list):
            adv_list = adv_list[:batch_size]
            imp_sample = imp_sample[:batch_size]
            entropy = entropy[:batch_size]
        adv_tensor = torch.concat(adv_list)
        importance_tensor = torch.concat(imp_sample)
        entropy_tensor = torch.concat(entropy)
        return adv_tensor, importance_tensor, entropy_tensor, np.mean(np.array(kl_divs)).astype(float)

    def control(self) -> dict:
        # Value Update
        for _ in range(self._num_updates):
            advantage_tensor, importance_tensor, entropy_tensor, mean_approx_kl = self.process_batch()
            value_loss_tensor = (advantage_tensor ** 2).mean() #pyre-fixme[58]
            assert ~np.isnan(value_loss_tensor.item()), 'value loss is nan!'
            # backprop
            self._value_optimizer.zero_grad()
            value_loss_tensor.backward()
            #torch.nn.utils.clip_grad_value_(self._value_net.parameters(), 100)
            self._value_optimizer.step()

        num_policy_update = 0
        # Policy Update
        for _ in range(self._num_updates):
            advantage_tensor, importance_tensor, entropy_tensor, mean_approx_kl = self.process_batch()
            #print('adv',  advantage_tensor, 'imp:', importance_tensor)
            clip_loss_tensor = (-torch.minimum(advantage_tensor.detach() * importance_tensor, torch.clip(
                importance_tensor, 1-self._clip_ratio, 1+self._clip_ratio) * advantage_tensor.detach())).mean()
            # max entropy 
            policy_loss_tensor = clip_loss_tensor - self._entropy_coeff * entropy_tensor.mean()
            # If KL divergence is too large, the new policy is diverging from old policy, stop training since it could lead to unstable/bad udpates.
            if mean_approx_kl > 1.5 * self._target_kl:
                #print(f'Early stopping at step {_} due to {mean_approx_kl} reaching max kl.')
                break
            num_policy_update += 1
            assert importance_tensor.requires_grad == True and advantage_tensor.shape == importance_tensor.shape
            assert ~np.isnan(policy_loss_tensor.item()), 'policy loss is nan!'
            # backprop
            self._policy_optimizer.zero_grad()
            policy_loss_tensor.backward()
            #torch.nn.utils.clip_grad_value_(self._policy_net.parameters(), 100)
            self._policy_optimizer.step()

        # reset
        self.reset()

        return {'value_loss': value_loss_tensor.item(), 'policy_loss': policy_loss_tensor.item(), 'entropy': entropy_tensor.mean().item(), 'num_policy_udpate': num_policy_update}
    
    def update_weights_from(self, agent: PPOAgent, tau:float = 0.01):
        utils.update_weights(source_net=agent._value_net, target_net=self._value_net, tau=tau)
        utils.update_weights(source_net=agent._policy_net, target_net=self._policy_net, tau=tau)


def test_agent():
    agent = PPOAgent(state_space=Box(low=0, high=1, shape=[4, 1]), action_space=Discrete(
        2), net_params={'width': 8, 'n_hidden': 1})
    for _ in range(5):
        state = agent._state_space.sample()
        _ = agent.sample_action(state)
        agent.post_process(np.array([1.0, 2, 3, 4]),
                           1, 0.5, np.array([-1, -1, -1, -1]), False)
    # print(agent._memory)
    agent.control()
    print('ppo_agent passed!')


test_agent()
