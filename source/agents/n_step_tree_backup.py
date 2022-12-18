import numpy as np
from numpy.core.getlimits import inf
from collections import defaultdict
from gym.spaces import Discrete
import random
import gym
import sys
from typing import Union
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from source.agents.agent import Agent
from source.utils import *


class nStepTreeBackupAgent(Agent):
    def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, n: int):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._n = n
        # action values
        # np.full((state_space.n, action_space.n), 0.0)
        self._Q = np.random.rand(state_space.n, action_space.n)
        # policy
        self._behavior_policy = get_epsilon_greedy_policy_from_action_values(
            self._Q, self._epsilon)
        self._target_policy = get_epsilon_greedy_policy_from_action_values(
            self._Q)
        self.reset()

    def reset(self):
        self._actions = []
        self._states = []
        self._rewards = [0.0]

    def sample_action(self, state, policy_name: str) -> int:
        if policy_name == 'behavior':
            policy = self._behavior_policy
        elif policy_name == 'target':
            policy = self._target_policy
        else:
            raise NotImplemented
        action = np.random.choice(len(policy[state]), p=policy[state])
        self._actions.append(action)
        return action

    def record(self, state, reward: Optional[int] = None):
        self._states.append(state)
        if reward is not None:
            self._rewards.append(reward)

    def control(self, t: int, T: int):
        n = self._n
        # state that is visited at time step tao will be updated (it's n step before t)
        tao = t - n + 1
        if tao >= 0:
            # calculate returns G
            # Step 1: caculate end of step return (step t+1)
            if t + 1 >= T:
                G = self._rewards[T]
            else:
                # return of the last step = reward + all future returns dicounted
                G = self._rewards[t+1] + self._discount_rate * np.sum(
                    self._target_policy[self._states[t+1]] * self._Q[self._states[t+1]])
            # Step 2: recursively calculate returns backward from step t -> tao+1
            for k in range(min(t, T-1), tao+1, -1):
                leaves = 0
                # suming up the expected return from leaf nodes that wasn't taken at step k
                for i in range(self._action_space.n):
                    if i != self._actions[k]:
                        leaves += self._target_policy[self._states[k],
                                                      i] * self._Q[self._states[k], i]
                # add returns from nodes that was taken at step k
                G = self._rewards[k] + self._discount_rate * \
                    (leaves + G *
                     self._target_policy[self._states[k], self._actions[k]])
            # update Q value
            self._Q[self._states[tao], self._actions[tao]] += self._learning_rate * \
                (G - self._Q[self._states[tao], self._actions[tao]])
            # update policy of updated state
            self._target_policy[self._states[tao]] = get_epsilon_greedy_policy_from_action_values(
                self._Q[self._states[tao]])
            self._behavior_policy[self._states[tao]] = get_epsilon_greedy_policy_from_action_values(
                self._Q[self._states[tao]], self._epsilon)
        return tao == T-1

    def play_episode(self, env: gym.Env, learning: Optional[bool] = True, epsilon: Optional[float] = None, learning_rate: Optional[float] = None, video_path: Optional[str] = None, policy_type: Optional[str] = 'target') -> Tuple[float, int]:
        if video_path is not None:
            video = VideoRecorder(env, video_path)
        state, _ = env.reset()
        self.reset()
        self.record(state)
        t = 0
        T = sys.maxsize
        action = self.sample_action(state, policy_type)
        stop = False
        while not stop:
            if t < T:
                new_state, reward, terminal, truncated, info = env.step(action)
                self.record(new_state, reward)
                if terminal:
                    T = t + 1
                else:
                    action = self.sample_action(new_state, policy_type)
            stop = self.control(t, T)
            t += 1
            if video_path is not None:
                video.capture_frame()
        if video_path is not None:
            video.close()

        return reward, T-1


def test_n_step_tree_backup_agent():
    np.random.seed(0)
    agent = nStepTreeBackupAgent(
        state_space=Discrete(4),
        action_space=Discrete(4),
        discount_rate=1.0,
        epsilon=0.2,
        learning_rate=1.0,
        n=1
    )
    agent._states = [0, 0]
    agent._actions = [0, 1]
    agent._rewards = [0, 2]
    tao = 0
    G = agent._rewards[1] + agent._discount_rate * agent._Q[0, 1]
    Q = agent._Q[0, 0]
    Q += agent._learning_rate * (G - Q)
    done = agent.control(0, 10)
    np.testing.assert_equal(Q, agent._Q[0, 0])

    print("test_n_step_tree_backup_agent passed!")


test_n_step_tree_backup_agent()
