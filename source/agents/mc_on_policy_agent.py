import numpy as np
from collections import defaultdict
from gym.spaces import Discrete
import random
import gym
from typing import Union
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from source.agents.agent import Agent
from source.utils import utils

class OnPolicyMonteCarloAgent(Agent):
  def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float, epsilon: float):
    super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate=1.0) 
    self._policy = np.full((state_space.n, action_space.n), 1.0 / action_space.n) 
    self._action_values = np.full((state_space.n, action_space.n), 0.0) 
    self._returns = defaultdict(lambda: defaultdict(list))
    self._epsilon = epsilon
  
  def update(self, episode: list):
    G = 0
    for i, (state, action, reward) in enumerate(episode[::-1]):
      for s,a,_ in episode[:i]:
        if s == state and a == action:
          continue
      G = self._discount_rate * G + reward 
      self._returns[state][action].append(G)
      self._action_values[state][action] = np.mean(self._returns[state][action])
      # Randomly break tie
      optimal_action = random.choice(np.where(self._action_values[state] == np.max(self._action_values[state]))[0])
      num_actions = len(self._policy[state])
      self._policy[state] = np.full(num_actions, self._epsilon / num_actions)
      self._policy[state][optimal_action] += 1 - self._epsilon
  
def test_on_policy_monte_carlo_agent():
  agent = OnPolicyMonteCarloAgent(
    state_space=Discrete(4), 
    action_space=Discrete(4), 
    discount_rate=0.9,
    epsilon=0.1
  )
#  episode = play_episode(agent, env)
  episode = [(0,1,0), (2,2,1)]
  agent.update(episode)
  assert agent._returns[0][1] == [0.9]
  assert agent._returns[2][2] == [1.0]
  assert agent._action_values[0][1] == 0.9
  assert agent._action_values[2][2] == 1.0
  episode = [(0,1,0)]
  agent.update(episode)
  assert agent._returns[0][1] == [0.9, 0.0]
  assert agent._action_values[0][1] == 0.45
  print("test_on_policy_monte_carlo_agent passed!")

test_on_policy_monte_carlo_agent() 