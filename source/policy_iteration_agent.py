import gym
import numpy as np
from numpy.core.getlimits import inf
from collections import defaultdict
from gym.spaces import Discrete
import random

from agent import Agent
from utils import *


class PolicyIterationAgent(Agent):
    def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float):
        super().__init__(state_space, action_space, discount_rate, epsilon=0.0, learning_rate=0.0)
        self._num_state = state_space.n
        self._num_action = action_space.n
        self._state_values = np.full((state_space.n), 0.0)
        self._policy = np.full(
            (state_space.n, action_space.n), 1.0 / action_space.n)
        self._discount_rate = discount_rate

    # Find state values.
    def policy_evaluation(self, env_dynamic: np.array, threshold):
        converged = False
        terminal_states = [self._num_state-1]
        while not converged:
            max_delta = 0.0
            for s in range(self._num_state):
                if s in terminal_states:
                    continue
                new_state_value = 0.0
                for a in range(self._num_action):
                    # for i-th possible env dynamic p(s', r | s, a):
                    for i in range(len(env_dynamic[s][a])):
                        new_state = env_dynamic[s][a][i][1]
                        reward = env_dynamic[s][a][i][2]
                        p_sr_sa = env_dynamic[s][a][i][0]
                        new_state_value += self._policy[s][a] * p_sr_sa * (
                            reward + self._discount_rate * self._state_values[new_state])
                max_delta = max(
                    abs(new_state_value - self._state_values[s]), max_delta)
                converged = max_delta < threshold
                # update state values
                self._state_values[s] = new_state_value

    # Find policy.
    def policy_improvement(self, env_dynamic: np.array) -> bool:
        policy_stable = True
        for s in range(self._num_state):
            action_values = []
            for a in range(self._num_action):
                # for i-th possible env dynamic p(s', r | s, a):
                action_value = 0.0
                for i in range(len(env_dynamic[s][a])):
                    new_state = env_dynamic[s][a][i][1]
                    reward = env_dynamic[s][a][i][2]
                    p_sr_sa = env_dynamic[s][a][i][0]
                    action_value += p_sr_sa * \
                        (reward + self._discount_rate *
                         self._state_values[new_state])
                action_values.append(action_value)
            new_optimal_actions = np.where(
                action_values == np.max(action_values))[0]
            old_optimal_actions = np.where(
                self._policy[s] == np.max(self._policy[s]))[0]
            if not np.array_equal(new_optimal_actions, old_optimal_actions):
                policy_stable = False
            self._policy[s] = [1/len(new_optimal_actions) if i in list(
                new_optimal_actions) else 0 for i in range(self._num_action)]
        return policy_stable

    def policy_iteration(self, env_dynamic: np.array):
        policy_stable = False
        step = 0
        while not policy_stable:
            self.policy_evaluation(env_dynamic, 0.01)
            policy_stable = self.policy_improvement(env_dynamic)
            step += 1
            if step % 10 == 0:
                print(f"step {step}: state value = {self._state_values[0]}")

    def sample_action(self, state: int) -> int:
        print('here')
        # Take greedy action from learned optimal policy.
        max_prob = np.max(self._policy[state])
        action = random.choice(np.where(self._policy[state] == max_prob)[
            0])  # Random break tie
        return action


def test_policy_evaluation():
    env = gym.make('CliffWalking-v0')
    agent = PolicyIterationAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        discount_rate=0.9
    )
    agent.policy_evaluation(env.P, 0.01)
    print("test_policy_evaluation passed")


def test_policy_improvement():
    env = gym.make('CliffWalking-v0')
    agent = PolicyIterationAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        discount_rate=0.9
    )
    stable = agent.policy_improvement(env.P)
    assert (agent._policy.reshape(4, 12, 4)[
            0][0] == 0.25).all(), "wrong policy value"
    print("test_policy_improvement passed")


def test_policy_iteration():
    env = gym.make('CliffWalking-v0')
    agent = PolicyIterationAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        discount_rate=0.9
    )
    agent.policy_iteration(env.P)
    np.testing.assert_almost_equal(
        agent._state_values[0], -7.7123207545039, err_msg="wrong state values!")
    assert (agent._policy.reshape(4, 12, 4)[0][0] == [
            0, 0.5, 0.5, 0]).all(), "wrong policy values!"
    print("test_policy_iteration passed")


test_policy_evaluation()
test_policy_improvement()
test_policy_iteration()
