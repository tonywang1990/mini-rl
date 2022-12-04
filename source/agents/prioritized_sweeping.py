import numpy as np
from numpy.core.getlimits import inf
from collections import defaultdict
from gym.spaces import Discrete
import random
import gym
from typing import Union
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from queue import PriorityQueue

from source.agents.agent import Agent
from source.utils import *


class PrioritizedSweepingAgent(Agent):
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
        self._queue = PriorityQueue()

    # get an action from policy
    def sample_action(self, state):
        return np.random.choice(len(self._policy[state]), p=self._policy[state])

    # update action value and policy
    def control(self, state, action, reward, new_state, terminal):
        # Learn from real experience
        update = self.learning(state, action, reward, new_state, terminal)
        # TODO: change to support non-determinstic environment
        self._model[(state, action)].add((reward, new_state, terminal))
        if update > 0:
            self._queue.put((update, (state, action)))
        # Learn from simulated experience
        self.planning(self._planning_steps)

        # update policy
        self._policy[state] = get_epsilon_greedy_policy_from_action_values(
            self._Q[state], self._epsilon)

    def learning(self, state, action, reward, new_state, terminal) -> float:
        # if new_state is a terminal state
        if terminal:
            update = reward - self._Q[state][action] 
            self._Q[state][action] += self._learning_rate * returns
        else:
            if self._agent_type == 'q_learning':
                returns = np.max(self._Q[new_state])
            elif self._agent_type == 'expected_sarsa':
                returns = np.sum(self._Q[new_state] * self._policy[new_state])
            else:
                raise NotImplementedError
            update =  reward + self._discount_rate * returns - self._Q[state][action]
            self._Q[state][action] += self._learning_rate * update
        return update
               

    def planning(self, n: int):
        for _ in range(n):
            key = self._queue.get()[1]
            if key not in self._model:
                return
            val = self._model[key]
            state, action = key
            reward, new_state, terminal = random.choice(tuple(val))
            self.learning(state, action, reward, new_state, terminal)


def test_prioritized_sweeping_agent():
    agent = PrioritizedSweepingAgent(
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
    reward = 13.0
    agent._Q = np.full((4, 4), 0.0)
    agent._Q[state, action] = 10
    agent._Q[new_state, 3] = 5
    agent._policy[state] = np.full(4, 0.0)
    new_action = agent.control(state, action, reward, new_state, False)
    np.testing.assert_almost_equal(agent._Q[state, action], 16)
    print("test_prioritized_sweeping_agent passed!")

test_prioritized_sweeping_agent()
