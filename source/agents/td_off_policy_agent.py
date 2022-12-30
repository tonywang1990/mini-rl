import numpy as np
from gym.spaces import Discrete

from source.agents.agent import Agent
from source.utils import utils

class TDOffPolicyAgent(Agent):
  def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float, epsilon:float, learning_rate:float, agent_type: str):
    super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate) 
    # action values
    self._Q = np.random.rand(state_space.n, action_space.n) #np.full((state_space.n, action_space.n), 0.0) 
    # policy
    self._policy = utils.get_epsilon_greedy_policy_from_action_values(self._Q, self._epsilon)
    self._agent_type = agent_type

  # get an action from policy
  def sample_action(self, state):
    return np.random.choice(len(self._policy[state]), p = self._policy[state])

  # update action value and policy 
  def control(self, state, action, reward, new_state, terminal):
    if terminal:
      self._Q[state][action] += self._learning_rate * (reward - self._Q[state][action])
    else:
      # update Q value
      if self._agent_type == 'q_learning':
        returns = np.max(self._Q[new_state])
      elif self._agent_type == 'expected_sarsa':
        returns = np.sum(self._Q[new_state] * self._policy[new_state])
      else:
        raise NotImplementedError
      self._Q[state][action] += self._learning_rate * (reward + self._discount_rate * returns - self._Q[state][action])
    # update policy
    self._policy[state] = utils.get_epsilon_greedy_policy_from_action_values(self._Q[state], self._epsilon) 

def test_td_off_policy_agent():
  np.random.seed(0)
  agent = TDOffPolicyAgent(
    state_space=Discrete(4), 
    action_space=Discrete(4), 
    discount_rate=1.0,
    epsilon=1.0,
    learning_rate=1.0,
    agent_type = "q_learning"
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
  print("test_td_off_policy_agent passed!")
  
test_td_off_policy_agent() 