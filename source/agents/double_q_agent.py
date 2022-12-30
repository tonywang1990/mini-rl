import numpy as np
from collections import defaultdict
from gym.spaces import Discrete
import random
import gym
from typing import Union, Optional
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from source.agents.agent import Agent
from source.utils import utils

class DoubleQAgent(Agent):
  def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float, epsilon:float, learning_rate:float, learning:bool=True):
    super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate) 
    # action values
    self._Q1 = np.random.rand(state_space.n, action_space.n) #np.full((state_space.n, action_space.n), 0.0) 
    self._Q2 = np.random.rand(state_space.n, action_space.n) #np.full((state_space.n, action_space.n), 0.0) 
    # policy
    self._policy = utils.get_epsilon_greedy_policy_from_action_values(self._Q1+self._Q2, self._epsilon)

  # get an action from policy
  def sample_action(self, state):
    return np.random.choice(len(self._policy[state]), p = self._policy[state])

  # update action value and policy 
  def control(self, state, action, reward, new_state, terminal):
    if random.random() < 0.5:
      Q1 = self._Q1
      Q2 = self._Q2
    else:
      Q1 = self._Q2
      Q2 = self._Q1

    if terminal:
      Q1[state][action] += self._learning_rate * (reward - Q1[state][action])
    else:
      # update Q value
      returns = Q2[new_state, np.argmax(Q1[new_state])]
      Q1[state][action] += self._learning_rate * (reward + self._discount_rate * returns - Q1[state][action])
    # update policy
    self._policy[state] = utils.get_epsilon_greedy_policy_from_action_values(Q1[state]+Q2[state], self._epsilon) 
  
def test_double_q_agent():
  np.random.seed(0)
  agent = DoubleQAgent(
    state_space=Discrete(4), 
    action_space=Discrete(4), 
    discount_rate=1.0,
    epsilon=1.0,
    learning_rate=1.0,
  )
  state = 1
  action = 1
  agent._Q1[state, action] = 5
  agent._Q2[state, action] = 5
  agent._policy[state] = np.full(4, 0.0)
  agent._policy[state, action] = 1.0
  reward = 1.0
  new_state = 1
  new_action = agent.control(state, action, reward, new_state, False)
  np.testing.assert_almost_equal(agent._Q1[state,action] + agent._Q2[state,action], 11)
  print("test_double_q_agent passed!")
  
test_double_q_agent() 