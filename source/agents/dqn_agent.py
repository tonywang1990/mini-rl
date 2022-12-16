import numpy as np
from numpy.core.getlimits import inf
from collections import defaultdict
from gym.spaces import Discrete, Box, Space
import random
import gym
from typing import Union, Optional
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from source.value_function import ActionValue, LearnedActionValue
from source.agents.agent import Agent
from source import utils


class DQNAgent(Agent):
    def __init__(self, state_space: Space, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        # action values
        self._Q = LearnedActionValue(state_space, action_space)
        self._target_Q = LearnedActionValue(state_space, action_space)
        self._sync_freq = 1000
        self._experience = []
        self._step = 0
        self._exp_limit = 1000
        self._batch_size = 10

    def sample_action(self, state) -> int:
        action, value = self._Q.sample_action(state, self._epsilon)
        return action

    def experience_replay(self, samples: list) -> float:
        states, actions, returns = [], [], []
        for state, action, reward, new_state in samples:
            #print(self._step, self._target_Q._fitted)
            _, q_max = self._target_Q.sample_action(new_state, epsilon=0.0)
            states.append(state)
            actions.append(action)
            returns.append(reward + self._discount_rate * q_max)
        score = self._Q.fit(states, actions, returns, self._batch_size)
        return score

    # SARS(A) on policy control
    def control(self, state, action, reward, new_state, terminal):
        # record expereince
        if self._step <= self._exp_limit:
            self._experience.append((state, action, reward, new_state))
        else:
            self._experience[self._step % self._exp_limit] = (
                state, action, reward, new_state)

        # select action
        if terminal:
            new_action = None
        else:
            # choice action based on epsilon greedy policy
            new_action = self.sample_action(new_state)

        # Experience replay
        if self._step > self._batch_size:
            samples = random.sample(self._experience, self._batch_size)
            score = self.experience_replay(samples)

        # Sync target_Q with Q
        if self._step % self._sync_freq == 0:
            self._target_Q.copy_from(self._Q)

        self._step += 1
        return new_action


def test_sarsa_agent():
    np.random.seed(0)
    agent = DQNAgent(
        state_space=Box(low=0, high=1.0, shape=(5, 5), dtype=np.float32),
        action_space=Discrete(4),
        discount_rate=1.0,
        epsilon=0.0,
        learning_rate=1.0
    )
    agent._Q.weight = np.ones((26,))
    state = np.zeros((5, 5))
    state[0][0] = 1.0
    action = 1
    reward = 1.0
    new_state = np.zeros((5, 5))
    new_state[0][1] = 1.0
    #agent._Q.set(new_state, 2, 1)
    np.testing.assert_equal(agent._Q.get(state, action), 2)
    new_action = agent._Q.sample_action(new_state)
    np.testing.assert_equal(new_action, 3)
    np.testing.assert_equal(agent._Q.get(new_state, new_action), 4)
    # 1*(1 + 4 - 2) * [1, 0, ..., 0, 1]
    new_action = agent.control(state, action, reward, new_state, False)
    np.testing.assert_equal(agent._Q.get(state, action), 4 + 4)
    print("test_sarsa_agent passed!")

# test_sarsa_agent()
