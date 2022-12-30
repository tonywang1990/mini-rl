import numpy as np
from collections import defaultdict
from gym.spaces import Discrete
import random
import gym
from typing import Union, List, Tuple, Optional

from source.agents.agent import Agent
from source.utils import utils
from source.model import Model


class RolloutAgent(Agent):
    def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, agent_type: str, planning_steps: int):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._agent_type = agent_type
        self._planning_steps = planning_steps
        # action values
        # np.full((state_space.n, action_space.n), 0.0)
        self._Q = np.random.rand(state_space.n, action_space.n)
        # environment model
        # policy
        self._policy = utils.get_epsilon_greedy_policy_from_action_values(
            self._Q, self._epsilon)
        self._model = Model()

    # get an action from policy
    def sample_action(self, state):
        action = self.decision_time_planning(state, self._planning_steps)
        return action

    def sample_policy(self, state):
        return np.random.choice(len(self._policy[state]), p=self._policy[state])

    def decision_time_planning(self, state: int, num_rollout: int):
        model = self._model
        #Q = np.zeros(self._action_space.n)
        Q = self._Q[state]
        for action in range(self._action_space.n): #pyre-fixme[16]
            if model.check_action(state, action) == False:
                continue
            action_values = []
            for _ in range(num_rollout):
                reward, new_state, terminal = model.step(state, action)
                if terminal:
                    discounted_return = reward
                else:
                    value, incomplete = self.rollout(new_state)
                    if incomplete:
                        continue
                    discounted_return = reward + \
                        self._discount_rate * value
                action_values.append(discounted_return)
            if len(action_values) != 0:
                Q[action] = np.mean(np.array(action_values))
        policy = utils.get_epsilon_greedy_policy_from_action_values(Q, self._epsilon)
        return np.random.choice(len(policy), p=policy)

    def rollout(self, state) -> Tuple[float, bool]:
        rewards = []
        terminal = False
        while terminal == False:
            action = self.sample_policy(state)
            # Anytime the episode is incomplete, return immediately
            if self._model.check_action(state, action) == False:
                return 0.0, True
            reward, new_state, terminal = self._model.step(state, action)
            state = new_state
            rewards.append(reward)
        discounted_return = 0.0
        for reward in rewards[::-1]:
            discounted_return = self._discount_rate * discounted_return + reward
        return discounted_return, False

    # update action value and policy
    def control(self, state, action, reward, new_state, terminal):
        # Learn from real experience
        self.learning(state, action, reward, new_state, terminal)
        # assuming non-determinsitc environment, storing all possible transitions
        self._model.update(state, action, reward, new_state, terminal)

        # update policy
        self._policy[state] = utils.get_epsilon_greedy_policy_from_action_values(
            self._Q[state], self._epsilon)

    def learning(self, state, action, reward, new_state, terminal, learning_rate: Optional[float] = None):
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
            key, val = random.choice(list(self._model._data.items()))
            state, action = key
            reward, new_state, terminal = random.choice(tuple(val))
            self.learning(state, action, reward, new_state, terminal)

def test_rollout_agent():
    agent = RolloutAgent(
        state_space=Discrete(4),
        action_space=Discrete(4),
        discount_rate=0.5,
        epsilon=0.0,
        learning_rate=0.5,
        agent_type='q_learning',
        planning_steps=1
    )
    model = Model()
    # state, action, reward, new_state, terminal
    model.update(0, 1, 2.5, 1, False)
    model.update(1, 0, 1.5, 2, False)
    model.update(2, 1, 3.0, 3, True)
    agent._model = model
    agent._policy = np.zeros(agent._policy.shape)
    agent._policy[0, 1] = 1.0
    agent._policy[1, 0] = 1.0
    agent._policy[2, 1] = 1.0
    # Test rollout mothod
    np.testing.assert_equal(agent.rollout(0)[0], 4)

    # Test rollout mothod
    agent._Q = np.zeros(agent._Q.shape)
    np.testing.assert_equal(agent.decision_time_planning(0, 3), 1)

# if '__name__' == '__main__':
test_rollout_agent()
