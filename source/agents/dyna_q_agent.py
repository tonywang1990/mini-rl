import numpy as np
from numpy.core.getlimits import inf
from collections import defaultdict
from gym.spaces import Discrete
import random
import gym
from typing import Union
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from source.agents.agent import Agent
from source.utils import *


class DynaQAgent(Agent):
    def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, agent_type: str, planning_steps: int):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._agent_type = agent_type
        self._planning_steps = planning_steps
        # action values
        # np.full((state_space.n, action_space.n), 0.0)
        self._Q = np.random.rand(state_space.n, action_space.n)
        # environment model
        self._model = defaultdict(set)
        # policy
        self._policy = get_epsilon_greedy_policy_from_action_values(
            self._Q, self._epsilon)

    # get an action from policy
    def sample_action(self, state):
        return np.random.choice(len(self._policy[state]), p=self._policy[state])

    # update action value and policy
    def control(self, state, action, reward, new_state, terminal):
        # Learn from real experience
        self.learning(state, action, reward, new_state, terminal)
        # assuming non-determinsitc environment, storing all possible transitions 
        self._model[(state, action)].add((reward, new_state, terminal))
        # Learn from simulated experience
        self.planning(self._planning_steps)

        # update policy
        self._policy[state] = get_epsilon_greedy_policy_from_action_values(
            self._Q[state], self._epsilon)

    def learning(self, state, action, reward, new_state, terminal):
        # if new_state is a terminal state
        if terminal:
            self._Q[state][action] += self._learning_rate * \
                (reward - self._Q[state][action])
        else:
            if self._agent_type == 'q_learning':
                returns = np.max(self._Q[new_state])
            elif self._agent_type == 'expected_sarsa':
                returns = np.sum(self._Q[new_state] * self._policy[new_state])
            else:
                raise NotImplementedError
            self._Q[state][action] += self._learning_rate * \
                (reward + self._discount_rate *
                 returns - self._Q[state][action])

    def planning(self, n: int):
        for _ in range(n):
            key, val = random.choice(list(self._model.items()))
            state, action = key
            reward, new_state, terminal = random.choice(tuple(val))
            self.learning(state, action, reward, new_state, terminal)


def test_dyna_q_agent():
    np.random.seed(0)
    agent = DynaQAgent(
        state_space=Discrete(4),
        action_space=Discrete(4),
        discount_rate=1.0,
        epsilon=1.0,
        learning_rate=0.5,
        agent_type='q_learning',
        planning_steps=1
    )
    state = 1
    action = 1
    new_state = 2
    reward = 3.0
    agent._Q = np.full((4, 4), 0.0)
    agent._Q[state, action] = 10
    agent._Q[new_state, 3] = 5
    agent._policy[state] = np.full(4, 0.0)
    new_action = agent.control(state, action, reward, new_state, False)
    np.testing.assert_almost_equal(agent._Q[state, action], 8.5)
    print("test_dyna_q_agent passed!")


test_dyna_q_agent()
