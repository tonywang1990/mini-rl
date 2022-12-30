import numpy as np
from collections import defaultdict
from gym.spaces import Discrete
import random
from typing import Union, Optional, Tuple
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from source.agents.agent import Agent
from source.utils import utils
from gym import Env


class BanditAgent(Agent):
    '''
    Multi-arm Bandit Agent.
    '''

    def __init__(self, observation_space: Discrete, action_space: Discrete, learning_rate: float, epsilon: bool, training: bool, initial_value: float):
        super().__init__(observation_space, action_space, 1.0, epsilon, learning_rate)
        self._training = training
        self._prev_action = None

        self._action_values = np.array([initial_value] * action_space.n)

    def take_action(self, observation: int, prev_reward: float, action_mask: Optional[np.ndarray] = None) -> int:
        # sample from action space
        # todo: add expsilon-greedy
        masked_action_values = self._action_values
        if action_mask is not None:
            masked_action_values = np.ma.masked_array(
                self._action_values, 1 - action_mask)
        action = np.argmax(masked_action_values).astype(int)

        if not self._training:
            return action

        # update action values: incremental implementation
        if self._prev_action is not None:
            prev_action_value = self._action_values[self._prev_action]
            self._action_values[self._prev_action] += self._learning_rate * \
                (prev_reward - prev_action_value)

        self._prev_action = action
        return action

    def play_episode(self, env: Env, epsilon: Optional[float] = None, learning_rate: Optional[float] = None, video_path: Optional[str] = None) -> Tuple[float, int]:
        observation, _ = env.reset()
        terminal = False
        total_reward = 0
        reward = 0
        steps = 0
        action_mask = None
        while not terminal:
            action = self.take_action(observation, reward, action_mask)
            observation, reward, terminal, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            terminal = terminal or truncated
            if steps > 1000:
                terminal = True
        # update with final reward at terminal
        # self.update_action_values(reward)
        return reward, steps


class ContextualBanditAgent(Agent):
    '''
    Multi-arm Contextual Bandit Agent.
    '''

    def __init__(self, observation_space: Discrete, action_space: Discrete, learning_rate: float, epsilon: bool, training: bool, initial_value: float):
        super().__init__(observation_space, action_space, 1.0, epsilon, learning_rate)
        self._action_space = action_space
        self._training = training
        self._prev_action = None
        self._prev_observation = None
        self._action_values = np.full(
            (observation_space.n, action_space.n), initial_value)

    def update_action_values(self, prev_reward: float):
        # update action values: incremental implementation
        prev_action_value = self._action_values[self._prev_observation][self._prev_action]
        self._action_values[self._prev_observation][self._prev_action] += self._learning_rate * (
            prev_reward - prev_action_value)

    def take_action(self, observation: int, prev_reward: float, action_mask: Optional[np.ndarray] = None) -> int:
        # sample from action space
        masked_action_values = self._action_values[observation]
        if random.random() < self._epsilon:
            if action_mask is not None:
                action = random.choice(np.where(action_mask == 1))
            else:
                action = random.choice(list(range(0, self._action_space.n)))
        else:
            if action_mask is not None:
                masked_action_values = np.ma.masked_array(
                    masked_action_values, 1 - action_mask)
            max_value = np.max(masked_action_values)
            action = random.choice(np.where(masked_action_values == max_value)[
                                   0])  # Random break tie

        if not self._training:
            return action

        if self._prev_action is not None:
            self.update_action_values(prev_reward)

        self._prev_action = action
        self._prev_observation = observation
        return action

    def play_episode(self, env: Env, epsilon: Optional[float] = None, learning_rate: Optional[float] = None, video_path: Optional[str] = None) -> Tuple[float, int]:
        observation, _ = env.reset()
        terminal = False
        total_reward = 0
        reward = 0
        steps = 0
        action_mask = None
        while not terminal:
            action = self.take_action(observation, reward, action_mask)
            observation, reward, terminal, truncated, info = env.step(action)
            terminal = terminal or truncated
            total_reward += reward
            steps += 1
        # update with final reward at terminal
        self.update_action_values(reward)
        return reward, steps


def test_bandit_agent():
    agent = BanditAgent(observation_space=Discrete(4), action_space=Discrete(
        5), learning_rate=0.5, epsilon=0, training=True, initial_value=5.0)
    assert agent._action_values.shape == (
        5,), f"{agent._action_values.shape} not correct"
    agent._prev_action = 1
    action = agent.take_action(
        observation=1, prev_reward=10, action_mask=np.array([0, 0, 1, 0, 0]))
    assert action == 2, 'Wrong greedy action selected'
    assert agent._action_values[
        1] == 7.5, f'Wrong action avalue update: {agent._action_values[1]}'
    print("bandit agent pass test.")


def test_contexual_bandit_agent():
    agent = ContextualBanditAgent(observation_space=Discrete(4), action_space=Discrete(
        5), learning_rate=0.5, epsilon=0, training=True, initial_value=5.0)
    assert agent._action_values.shape == (
        4, 5), f"{agent._action_values.shape} not correct"
    agent._prev_observation = 2
    agent._prev_action = 1
    action = agent.take_action(
        observation=1, prev_reward=10, action_mask=np.array([0, 0, 1, 1, 0]))
    assert isinstance(action, np.int64), "Wrong action data type"
    assert (action == 2) or (action == 3), 'Wrong greedy action selected'
    assert agent._action_values[2][
        1] == 7.5, f'Wrong action avalue update: {agent._action_values[2][1]}'
    print("contextual bandit agent pass test.")


test_bandit_agent()
test_contexual_bandit_agent()
