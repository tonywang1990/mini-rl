from base64 import b64encode
from typing import Dict, List, Optional, Set, Tuple

from collections import namedtuple, deque, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from gym import Env
from tqdm import tqdm
import math
import torch
import random
from source.agents.agent import Agent
from pettingzoo.utils.env import AECEnv

# Utils

## General


def evaluate_agent(agent: Agent, env: Env, num_episode: int = 100000, epsilon: float = 0.0, threshold: float = 0.0):
    total_reward = 0
    successful_episode = 0
    ## Set learning to false
    agent._learning=False
    for _ in tqdm(range(num_episode)):
        reward, _ = agent.play_episode(env, epsilon=epsilon)
        if reward > threshold:
            successful_episode += 1
        total_reward += reward
    return total_reward / num_episode, successful_episode / num_episode


def estimate_success_rate(agent: Agent, env: Env, num_episode: int = 100000, epsilon: float = 0.0, threshold: float = 0.0):
    return evaluate_agent(agent, env, num_episode, epsilon, threshold)[0]


def create_decay_schedule(num_episodes: int, value_start: float = 0.9, value_decay: float = .9999, value_min: float = 0.05):
    # get 20% value at 50% espisode
    value_decay = 0.2 ** (1/(0.5 * num_episodes))
    return [max(value_min, (value_decay**i)*value_start) for i in range(num_episodes)]


def epsilon(step: int, eps_start: float = 0.9, eps_end: float = 0.05, eps_decay: float = 1000) -> float:
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)

# Tabular


def get_epsilon_greedy_policy_from_action_values(action_values: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    optimal_actions = np.argmax(action_values, axis=-1)
    num_actions = action_values.shape[-1]
    policy = np.full(action_values.shape, epsilon / num_actions)
    if optimal_actions.ndim == 0:
        policy[optimal_actions] += 1.0 - epsilon
    elif optimal_actions.ndim == 1:
        for i, j in enumerate(optimal_actions):
            policy[i, j] += 1.0 - epsilon
    else:
        raise NotImplementedError
    return policy


def get_state_values_from_action_values(action_values: np.ndarray, policy: Optional[np.ndarray] = None) -> np.ndarray:
    if policy is None:
        # assume greedy policy
        policy = get_epsilon_greedy_policy_from_action_values(action_values)
    state_values = np.sum(action_values * policy, axis=1)
    return state_values


# Visualization
def render_mp4(videopath: str) -> str:
    """
    Gets a string containing a b4-encoded version of the MP4 video
    at the specified path.
    """
    mp4 = open(videopath, 'rb').read()
    base64_encoded_mp4 = b64encode(mp4).decode()
    return f'<video width=400 controls><source src="data:video/mp4;' \
        f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'


def plot_history(history: list[float], smoothing: bool = True):
    num_episode = len(history)
    plt.figure(0, figsize=(16, 4))
    plt.title("average reward per step")
    history_smoothed = [
        np.mean(np.array(history[max(0, i-num_episode//10): i+1])) for i in range(num_episode)]
    plt.plot(history, 'o', alpha=0.2)
    if smoothing:
        plt.plot(history_smoothed, linewidth=5)


def show_state_action_values(agent: Agent, game: str):
    # Plot the action values.
    # cliff walking
    if game == 'cliff_walking':
        shape = (4, 12, 4)
    # frozen lake
    elif game == 'frozen_lake_4x4':
        # small
        shape = (4, 4, 4)
    elif game == 'frozen_lake_8x8':
        # large
        shape = (8, 8, 4)
    else:
        raise NotImplemented

    direction = {
        0: "LEFT",
        1: "DOWN",
        2: "RIGHT",
        3: "UP"
    }
    #actions = np.argmax(agent._policy, axis=1)
    #actions = actions.reshape(shape[:2])
    #named_actions = np.chararray(actions.shape, itemsize=4)
    #map = [[""] * shape[1]] * shape[0]
    # for idx, val in np.ndenumerate(actions):
    #    named_actions[idx] = direction[val]
    #    #map[idx[0]][idx[1]] = direction[val]
    # print(named_actions)

    # Action values
    plt.figure(1, figsize=(16, 4))
    action_values = agent._Q.reshape(shape)
    num_actions = action_values.shape[-1]
    plt.suptitle("action_values (Q)")
    for i in range(num_actions):
        plt.subplot(1, num_actions, i+1)
        plt.title(f"{i}, {direction[i]}")
        plt.imshow(action_values[:, :, i])
        for (y, x), label in np.ndenumerate(action_values[:, :, i]):
            plt.text(x, y, round(label, 2), ha='center', va='center')

        plt.colorbar(orientation='vertical')
        # print(action_values[:,:,i])

    # State values
    plt.figure(2)
    state_values = get_state_values_from_action_values(agent._Q, agent._policy)
    values = state_values.reshape(shape[:2])
    plt.imshow(values)
    for (j, i), label in np.ndenumerate(values):
        plt.text(i, j, round(label, 5), ha='center', va='center')
    plt.colorbar(orientation='vertical')
    plt.title("state_values")


# Pytorch
def to_feature(data: np.ndarray, device: str = 'cpu', debug: bool = True) -> torch.Tensor:
    # Convert state into tensor and unsqueeze: insert a new dim into tensor (at dim 0): e.g. 1 -> [1] or [1] -> [[1]]
    # state: np.array
    # returns: torch.Tensor of shape [1]
    assert isinstance(
        data, np.ndarray), f'data is not of type ndarray: {type(data)}'
    return torch.tensor(data.flatten(), dtype=torch.float32, device=device)


def to_array(tensor: torch.Tensor, shape: list) -> np.ndarray:
    return tensor.cpu().numpy().reshape(shape)


# DQN
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Pettingzoo.classsic environment


def play_multiagent_episode(agent_dict: Dict[str, Agent], env: AECEnv) -> Tuple[defaultdict, defaultdict]:
    env.reset()
    for agent in agent_dict.values():
        agent.reset()
    steps, total_reward = defaultdict(int), defaultdict(float)
    history = defaultdict(list)

    for agent_id in env.agent_iter():
        agent = agent_dict[agent_id]
        # Make observation
        observation, reward, terminal, truncated, info = env.last()
        if observation is None:
            continue
        # Select action
        if terminal or truncated:
            action = None
        else:
            action = agent.sample_action(observation['observation'], observation['action_mask'])
        env.step(action)
        # Train the agent
        if agent._learning and agent_id in history:
            prev_ob, prev_action = history[agent_id][-1]
            agent.control(prev_ob, prev_action, reward, observation['observation'],
                          terminal)
        history[agent_id].append((observation['observation'], action))
        # bookkeeping
        total_reward[agent_id] += reward
        steps[agent_id] += 1
        # if steps > 1000:
        #    terminal = True
    return total_reward, steps


def evaluate_multiagent(agent_dict: Dict[str, Agent], env: AECEnv, num_episode: int = 100000, epsilon: float = 0.0, threshold: float = 0.0):
    for agent in agent_dict.values():
        agent._epsilon = 0.0
    total_reward = defaultdict(float)
    successful_episode = 0
    for _ in tqdm(range(num_episode)):
        rewards, steps = play_multiagent_episode(
            agent_dict, env)  # ,epsilon=epsilon_schedule[i])
        for agent, reward in rewards.items():
            total_reward[agent] += reward
    return total_reward, num_episode
