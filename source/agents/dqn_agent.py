import numpy as np
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


class DQNAgent(Agent):
    def __init__(self, state_space: Space, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, learning: bool, batch_size: int, tau: float, eps_decay: float):
        super().__init__(state_space, action_space,
                         discount_rate, epsilon, learning_rate, learning)
        # TAU is the update rate of the target network
        # LR is the learning rate of the AdamW optimizer
        self._batch_size = batch_size
        self._tau = tau
        self._eps_decay = eps_decay
        # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._device = 'cpu'
        print(f'using device: {self._device}')
        MEMORY_SIZE = 10000

        # Get number of actions from gym action space
        self._n_actions = action_space.n
        self._state_dim = state_space.shape
        self._n_states = np.prod(np.array(self._state_dim))

        self._policy_net = DenseNet(
            self._n_states, self._n_actions).to(self._device)
        self._target_net = DenseNet(
            self._n_states, self._n_actions).to(self._device)
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

    def sample_action(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # state: tensor of shape [1, n_states]
        # return: tensor of shape [1, n_actions]
        sample = random.random()
        # eps_threshold = self._epsilon #self._epsilon.get(self._step)
        self._epsilon = utils.epsilon(self._step, eps_decay=self._eps_decay)
        self._step += 1
        if sample > self._epsilon:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_prob = self._policy_net(state)
                if action_mask is not None:
                    large = torch.finfo(action_prob.dtype).max
                    action_prob -= (1-action_mask) * large
                action = action_prob.max(1)[1].view(1, 1)
        else:
            if action_mask is not None:
                legal_actions = np.nonzero(action_mask.numpy())[0]
                random_action = np.random.choice(legal_actions)
            else:
                random_action = self._action_space.sample()
            action = torch.tensor(
                [[random_action]], device=self._device, dtype=torch.long)
        assert list(action.shape) == [
            1, 1], f"{list(action.shape)} != {[1, 1]}"
        return action

    def _optimize_model(self):
        if len(self._memory) < self._batch_size:
            return
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
        # reward + discount_rate * Q_target(new_state, new_action) - Q_policy(state, action)

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

    def _update_target_net(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self._target_net.state_dict()
        policy_net_state_dict = self._policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                self._tau + target_net_state_dict[key]*(1-self._tau)
        self._target_net.load_state_dict(target_net_state_dict)

    def control(self, state: torch.Tensor, action: torch.Tensor, reward: float, new_state: torch.Tensor, terminal: bool):
        reward_tensor = torch.tensor(
            [reward], dtype=torch.float32, device=self._device)
        if terminal:
            next_state = torch.zeros(
                self._n_states, device=self._device).unsqueeze(0)
        else:
            next_state = new_state
        terminal_tensor = torch.tensor(
            [terminal], device=self._device, dtype=torch.bool)
        # Store the transition in memory
        self._memory.push(state, action, next_state,
                          reward_tensor, terminal_tensor)
        # Perform one step of the optimization (on the policy network)
        self._optimize_model()
        self._update_target_net()

    def play_episode(self, env: gym.Env, epsilon: Optional[float] = None, learning_rate: Optional[float] = None, video_path: Optional[str] = None):
        if video_path is not None:
            video = VideoRecorder(env, video_path)
        state, info = env.reset()
        state = self.to_feature(state)
        terminal = False
        steps = 0
        total_reward = 0
        if epsilon is not None:
            self._epsilon = epsilon
        if learning_rate is not None:
            self._learning_rate = learning_rate
        while not terminal:
            action = self.sample_action(state)
            new_state, reward, terminal, truncated, info = env.step(
                action.item())
            total_reward += reward
            terminal = terminal or truncated
            new_state = self.to_feature(new_state)
            if self._learning:
                self.control(state, action, reward,
                             new_state, terminal)
            state = new_state
            steps += 1
            # if steps > 1000:
            #    terminal = True
            if video_path is not None:
                video.capture_frame()
        if video_path is not None:
            video.close()
        return total_reward, steps


def test_agent():
    agent = DQNAgent(Box(low=0, high=1, shape=[4, 5, 3]), Discrete(
        2), 1.0, 0.1, 1.0, True, 2, 0.001, 1000)
    state = agent._state_space.sample()
    state_tensor = agent.to_feature(state)
    action = agent.sample_action(state_tensor)
    new_state = agent.to_array(state_tensor, agent._state_dim)
    np.testing.assert_equal(state, new_state)
    print('dqn_agent test passed!')


test_agent()
