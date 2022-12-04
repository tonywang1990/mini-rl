import gym
import numpy as np
from numpy.core.getlimits import inf
from collections import defaultdict
from gym.spaces import Discrete
import random
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from source.agents.agent import Agent
from source.utils import *



class ValueIterationAgent(Agent):
    def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float):
        super().__init__(state_space, action_space, discount_rate, epsilon=0.0, learning_rate=0.0)
        self._num_state = state_space.n
        self._num_action = action_space.n
        self._state_values = np.full((state_space.n), 0.0)
        self._discount_rate = discount_rate

    def value_iteration(self, env_dynamic: np.array, threshold):
        converged = False
        terminal_states = [self._num_state-1]
        while not converged:
            max_delta = 0.0
            for s in range(self._num_state):
                if s in terminal_states:
                    continue
                new_action_values = []
                for a in range(self._num_action):
                    # for i-th possible env dynamic p(s', r | s, a):
                    new_action_value = 0.0
                    for i in range(len(env_dynamic[s][a])):
                        new_state = env_dynamic[s][a][i][1]
                        reward = env_dynamic[s][a][i][2]
                        p_sr_sa = env_dynamic[s][a][i][0]
                        new_action_value += p_sr_sa * \
                            (reward + self._discount_rate *
                             self._state_values[new_state])
                    new_action_values.append(new_action_value)
                new_state_value = max(new_action_values)
                max_delta = max(
                    abs(new_state_value - self._state_values[s]), max_delta)
                converged = max_delta < threshold
                # update state values
                self._state_values[s] = new_state_value

    def sample_action(self, state: int, env_dynamic: np.array) -> int:
        # Take greedy action according to the state values.
        action_values = []
        for action in env_dynamic[state]:
            action_value = 0
            for i in range(len(env_dynamic[state][action])):
                new_state = env_dynamic[state][action][i][1]
                reward = env_dynamic[state][action][i][2]
                p_sr_sa = env_dynamic[state][action][i][0]
                action_value += p_sr_sa * \
                    (reward + self._discount_rate *
                     self._state_values[new_state])
            action_values.append(action_value)
        action = random.choice(np.where(action_values == np.max(action_values))[
                               0])  # Random break tie
        return action

    def play_episode(self, env: gym.Env, learning: Optional[bool] = True, video_path: str = None):
        if video_path is not None:
            video = VideoRecorder(env, video_path)
        state, info = env.reset()
        terminal = False
        steps = 0
        while not terminal:
            action = self.sample_action(state, env.P)
            new_state, reward, terminal, _, info = env.step(action)
            if learning:
                self.control(state, action, reward, new_state, terminal)
            state = new_state
            steps += 1
            if video_path is not None:
                video.capture_frame()
        if video_path is not None:
            video.close()
        return reward, steps


def test_value_iteration():
    env = gym.make('CliffWalking-v0')
    agent = ValueIterationAgent(
        state_space=env.observation_space,
        action_space=env.action_space,
        discount_rate=1.0
    )
    agent.value_iteration(env.P, 0.01)
    np.testing.assert_almost_equal(
        agent._state_values[0], -14, err_msg="wrong state values!")
    assert agent._state_values[-1] == 0, "wrong state values!"

    action = agent.sample_action(0, env.P)
    print("test_value_iteration passed")


test_value_iteration()
