import numpy as np
from numpy.core.getlimits import inf
from collections import defaultdict
from gym.spaces import Discrete
import random
import gym
from typing import Union
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from agent import Agent
from utils import *


class TreeSearchAgent(Agent):
    def __init__(self, state_space: Discrete, action_space: Discrete, discount_rate: float, epsilon: float, learning_rate: float, agent_type: str, planning_steps: int):
        super().__init__(state_space, action_space, discount_rate, epsilon, learning_rate)
        self._agent_type = agent_type
        self._planning_steps = planning_steps
        # action values
        # np.full((state_space.n, action_space.n), 0.0)
        self._Q = np.random.rand(state_space.n, action_space.n)
        # environment model
        self._model = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # policy
        self._policy = get_epsilon_greedy_policy_from_action_values(
            self._Q, self._epsilon)

    # get an action from policy
    def sample_action(self, state):
        if random.random() < self._epsilon:
            action = random.choice([i for i in range(self._action_space.n)])
        else:
            action, _ = self.tree_search(state, self._planning_steps)
        return action
    
    def sample_policy(self, state):
        return np.random.choice(len(self._policy[state]), p=self._policy[state])

    def tree_search(self, state: int, steps: int) -> Tuple[int, float]:
        #policy_action = self.sample_policy(state)
        greedy_action = np.argmax(self._Q[state])
        default = (greedy_action, self._Q[state][greedy_action])
        #print(self._model[state].keys())
        if state not in self._model or steps == 0:
            ## TODO: should we use epsilon greedy here?
            return default
        best_action_value = None
        best_action = None
        for action in range(self._action_space.n):
        #for action in self._model[state]:
            if action not in self._model[state]:
                action_value = self._Q[state][action] 
            else:
                action_value = 0
                total_count = 0
                for (reward, new_state, terminal), count in self._model[state][action].items():
                    if terminal:
                        action_value += reward * count
                    else:
                        _, new_action_value = self.tree_search(new_state, steps-1)
                        action_value += (reward + self._discount_rate * new_action_value) * count
                    total_count += count
                # Expected action value
                action_value /= total_count
            if best_action_value is None or best_action_value < action_value:
                best_action_value = action_value
                best_action = action
        return best_action, best_action_value 

    # update action value and policy
    def control(self, state, action, reward, new_state, terminal):
        # Learn from real experience
        self.learning(state, action, reward, new_state, terminal)
        # assuming non-determinsitc environment, storing all possible transitions 
        self._model[state][action][(reward, new_state, terminal)] += 1
        # Learn from simulated experience
        #self.planning(self._planning_steps)

        # update policy
        self._policy[state] = get_epsilon_greedy_policy_from_action_values(
            self._Q[state], self._epsilon)

    def learning(self, state, action, reward, new_state, terminal, learning_rate: Optional[float] = None):
        # if new_state is a terminal state
        if terminal:
            self._Q[state][action] += self._learning_rate * \
                (reward - self._Q[state][action])
        else:
            if self._agent_type == 'q_learning':
                returns = np.max(self._Q[new_state])
            elif self._agent_type == 'expected_sarsa':
                returns = np.sum(self._Q[new_state] * self._policy[new_state])
            else:
                raise NotImplementedError
            self._Q[state][action] += self._learning_rate * \
                (reward + self._discount_rate *
                 returns - self._Q[state][action])

    def planning(self, n: int):
        for _ in range(n):
            key, val = random.choice(list(self._model.items()))
            state, action = key
            reward, new_state, terminal = random.choice(tuple(val))
            self.learning(state, action, reward, new_state, terminal)


def test_tree_search_agent():
    agent = TreeSearchAgent(
        state_space=Discrete(4),
        action_space=Discrete(4),
        discount_rate=1.0,
        epsilon=1.0,
        learning_rate=0.5,
        agent_type='q_learning',
        planning_steps=0
    )
    state = 1
    action = 1
    new_state = 2
    reward = 3.0
    agent._Q = np.full((4, 4), 0.0)
    agent._Q[state, action] = 10
    agent._Q[new_state, 3] = 5
    agent._policy[state] = np.full(4, 0.0)
    #agent.sample_action(state)
    new_action = agent.control(state, action, reward, new_state, False)
    np.testing.assert_almost_equal(agent._Q[state, action], 9)
    print("test_tree_search_agent passed!")

#if '__name__' == '__main__':
test_tree_search_agent()