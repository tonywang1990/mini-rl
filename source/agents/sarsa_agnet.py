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

class SarsaAgent(Agent):
  def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float, epsilon:float, learning_rate:float):
    super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate) 
    self._epsilon = epsilon
    self._learning_rate = learning_rate 
    # action values
    self._Q = np.random.rand(state_space.n, action_space.n) #np.full((state_space.n, action_space.n), 0.0) 
    # policy
    self._policy = get_epsilon_greedy_policy_from_action_values(self._Q, self._epsilon)

  def sample_action(self, state):
    return np.random.choice(len(self._policy[state]), p = self._policy[state])

  # SARS(A) on policy control
  def control(self, state, action, reward, new_state, terminal):
    if terminal:
      self._Q[state][action] += self._learning_rate * (reward - self._Q[state][action])
      new_action = None
    else:
      # choice action based on epsilon greedy policy
      new_action = self.sample_action(new_state)
      # update Q value
      self._Q[state][action] += self._learning_rate * (reward + self._discount_rate * self._Q[new_state][new_action] - self._Q[state][action])
    # update policy
    self._policy[state] = get_epsilon_greedy_policy_from_action_values(self._Q[state], self._epsilon) 
    return new_action 

def test_sarsa_agent():
  np.random.seed(0)
  agent = SarsaAgent(
    state_space=Discrete(4), 
    action_space=Discrete(4), 
    discount_rate=1.0,
    epsilon=1.0,
    learning_rate=1.0
  )
  state = 1
  action = 1
  agent._Q[state, action] = 10
  agent._policy[state] = np.full(4, 0.0)
  agent._policy[state, action] = 1.0
  reward = 1.0
  new_state = 1
  new_action = agent.control(state, action, reward, new_state, False)
  np.testing.assert_almost_equal(agent._Q[state,action], 11)
  print("test_sarsa_agent passed!")
  
test_sarsa_agent() 