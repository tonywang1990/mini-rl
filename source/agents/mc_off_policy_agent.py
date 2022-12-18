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

class OffPolicyMonteCarloAgent(Agent):
  def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float):
    super().__init__(state_space, action_space, discount_rate, epsilon=0.0, learning_rate=1.0) 
    # action values
    self._Q = np.full((state_space.n, action_space.n), 1.0 / action_space.n)# np.random.rand(state_space.n, action_space.n)
    self._C = np.full((state_space.n, action_space.n), 0.0) 
    # Target policy
    self._policy = get_epsilon_greedy_policy_from_action_values(self._Q)

  def prediction(self, behavior_policy: np.array, episode: list):
    G = 0
    W = 1
    for i, (state, action, reward) in enumerate(episode[::-1]):
      if W == 0:
        break
      G = self._discount_rate * G + reward 
      self._C[state, action] += W
      self._Q[state, action] += W / self._C[state, action] * (G - self._Q[state, action])
      W *= self._policy[state, action] / behavior_policy[state, action]
      
  def control(self, behavior_policy: np.array, episode: list):
    num_action = self._action_space.n
    G = 0
    W = 1
    for i, (state, action, reward) in enumerate(episode[::-1]):
      if W == 0:
        break
      G = self._discount_rate * G + reward 
      # update action value
      self._C[state, action] += W
      self._Q[state, action] += W / self._C[state, action] * (G - self._Q[state, action])
      # update policy
      self._policy[state] = np.full(num_action, 0.0)
      self._policy[state, np.argmax(self._Q[state])] = 1.0
      # update W
      W *= self._policy[state][action] / behavior_policy[state][action]
  
def test_prediction_off_policy_monte_carlo_agent():
  agent = OffPolicyMonteCarloAgent(
    state_space=Discrete(4), 
    action_space=Discrete(4), 
    discount_rate=0.9,
  )
#  episode = play_episode(agent, env)
  episode = [(0,1,0), (2,2,1)]
  #rand_policy = np.random.rand(4,4)
  #behavior_policy = rand_policy / rand_policy.sum(axis=1, keepdims=True)
  behavior_policy = get_epsilon_greedy_policy_from_action_values(agent._Q, 0.1)
  agent.prediction(behavior_policy, episode)
  assert agent._Q[2][2] == 1.0
  episode = [(0,1,0)]
  print("test_prediction_off_policy_monte_carlo_agent passed!")
  
def test_control_off_policy_monte_carlo_agent():
  agent = OffPolicyMonteCarloAgent(
    state_space=Discrete(4), 
    action_space=Discrete(4), 
    discount_rate=0.9,
  )
#  episode = play_episode(agent, env)
  episode = [(0,1,0), (2,2,1)]
  behavior_policy = agent._policy
  agent.control(behavior_policy, episode)
  assert agent._Q[0][1] == 0.9
  assert agent._Q[2][2] == 1.0
  assert agent._policy[2][2] == 1.0
  episode = [(0,1,0)]
  print("test_control_off_policy_monte_carlo_agent passed!") 

test_prediction_off_policy_monte_carlo_agent() 
test_control_off_policy_monte_carlo_agent() 