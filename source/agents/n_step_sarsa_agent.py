import numpy as np
from collections import defaultdict
from gym.spaces import Discrete
import random
import gym
import sys
from typing import Union, Optional, Tuple
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from source.agents.agent import Agent
from source.utils import utils

class nStepSarsaAgent(Agent):
  def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float, epsilon:float, learning_rate:float, n:int):
    super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate) 
    self._n = n
    # action values
    self._Q = np.random.rand(state_space.n, action_space.n) #np.full((state_space.n, action_space.n), 0.0) 
    # policy
    self._policy = utils.get_epsilon_greedy_policy_from_action_values(self._Q, self._epsilon)
    self.reset()

  def reset(self):
    self._actions = []
    self._states = []
    self._rewards = [0.0]

  def sample_action(self, state) -> int:
    action = np.random.choice(len(self._policy[state]), p = self._policy[state])
    self._actions.append(action)
    return action 

  def record(self, state, reward: Optional[float] = None):
    self._states.append(state)
    if reward is not None:
      self._rewards.append(reward)
  #pyre-fixme[14] 
  def control(self, t: int, T: int):
    n = self._n
    # state that is visited at time step tao will be updated (it's n step before t)
    tao = t - n + 1
    if tao >= 0:
      # calculate returns G
      G = 0
      for i in range(tao+1, min(tao+n, T)+1):
        G += (self._discount_rate ** (i-tao-1)) * self._rewards[i]
      # if episode has not finished, add the approximation of all future discoutned return using Q
      if tao + n < T:
        G += (self._discount_rate ** n) * self._Q[self._states[tao+n], self._actions[tao+n]]
      # update Q value 
      self._Q[self._states[tao], self._actions[tao]] += self._learning_rate* (G - self._Q[self._states[tao], self._actions[tao]])
      # update policy of updated state 
      self._policy[self._states[tao]] = utils.get_epsilon_greedy_policy_from_action_values(self._Q[self._states[tao]], self._epsilon) 
    return tao == T-1
  
  def play_episode(self, env: gym.Env, learning: Optional[bool] = True, epsilon: Optional[float] = None, learning_rate: Optional[float] = None, video_path: Optional[str] = None)->Tuple[float, int]:
    if video_path is not None:
      video = VideoRecorder(env, video_path)
    if epsilon is not None:
      self._epsilon = epsilon
    if learning_rate is not None:
      self._learning_rate = learning_rate

    state, info = env.reset()
    self.reset()
    self.record(state)
    t = 0
    reward = 0
    T = sys.maxsize
    action = self.sample_action(state)
    stop = False
    while not stop:
      if t < T: 
        new_state, reward, terminal, _, info = env.step(action)
        self.record(new_state, reward)
        if terminal:
          T = t + 1
        else:
          action = self.sample_action(new_state)
      if learning:
        stop = self.control(t, T)
      else:
        stop = (t >= T)
      t += 1
    return reward, T-1


def test_n_step_sarsa_agent():
  np.random.seed(0)
  agent = nStepSarsaAgent(
    state_space=Discrete(4), 
    action_space=Discrete(4), 
    discount_rate=1.0,
    epsilon=0.2,
    learning_rate=1.0,
    n=1
  )
  agent._states = [0, 0]
  agent._actions = [0, 1]
  agent._rewards = [0,2]
  tao = 0
  G = agent._rewards[1] + agent._discount_rate * agent._Q[0, 1]
  Q = agent._Q[0,0] 
  Q += agent._learning_rate* (G - Q)
  done = agent.control(0,10)
  np.testing.assert_equal(Q, agent._Q[0,0])

  print("test_n_step_sarsa_agent passed!")
  
test_n_step_sarsa_agent() 