import numpy as np
from numpy.core.getlimits import inf
from collections import defaultdict
from gym.spaces import Discrete
import random
import gym
from typing import Union
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from source.value_function import TabularActionValue
from source.agents.agent import Agent
from source.utils import *

class SarsaAgent(Agent):
  def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float, epsilon:float, learning_rate:float):
    super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate) 
    self._epsilon = epsilon
    self._learning_rate = learning_rate 
    # action values
    self._Q = TabularActionValue(state_space.n, action_space.n)
    # policy
    self._policy = get_epsilon_greedy_policy_from_action_values(self._Q.weight, self._epsilon)

  def sample_action(self, state):
    return np.random.choice(len(self._policy[state]), p = self._policy[state])

  # SARS(A) on policy control
  def control(self, state, action, reward, new_state, terminal):
    if terminal:
      Q_value = self._Q.get(state,action) + self._learning_rate * (reward - self._Q.get(state,action))
      self._Q.set(state, action, Q_value)
      new_action = None
    else:
      # choice action based on epsilon greedy policy
      new_action = self.sample_action(new_state)
      # update Q value
      Q_value = self._Q.get(state, action) + self._learning_rate * (reward + self._discount_rate * self._Q.get(new_state,new_action) - self._Q.get(state,action))
      self._Q.set(state, action, Q_value)
    # update policy
    self._policy = get_epsilon_greedy_policy_from_action_values(self._Q.weight, self._epsilon) 
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
  agent._Q.set(state, action,10)
  agent._policy[state] = np.full(4, 0.0)
  agent._policy[state, action] = 1.0
  reward = 1.0
  new_state = 1
  new_action = agent.control(state, action, reward, new_state, False)
  np.testing.assert_almost_equal(agent._Q.get(state,action), 11)
  print("test_sarsa_agent passed!")
  
test_sarsa_agent() 