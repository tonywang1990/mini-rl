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
import gym
from source.agents.agent import Agent
from pettingzoo.utils.env import AECEnv
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import json

# Utils

## General


def evaluate_agent(agent: Agent, env: Env, num_episode: int = 100000, epsilon: float = 0.0, threshold: float = 0.0):
    total_reward = 0
    successful_episode = 0
    ## Set learning to false
    agent._learning=False
    for _ in tqdm(range(num_episode)):
        reward, _ = play_episode(agent, env, epsilon=epsilon)
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


def epsilon(step: int, eps_start: float = 0.9, eps_end: float = 0.00, eps_decay: float = 1000) -> float:
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)

def eps():
    return np.finfo(np.float32).eps.item()

def normalize(tensor: torch.Tensor):
    return (tensor - tensor.mean()
                          ) / (tensor.std() + eps())

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

metric_metadata = {'default': {'format': '-', 'smooth':True}, 'episode_len': {'skip':True}}

def plot_logs(logs: list[defaultdict], window: int = 50):
    data = defaultdict(lambda: defaultdict(list))
    num_episode = len(logs)
    for ep in logs:
        for agent_id, metrics in ep.items():
            for name, metric in metrics.items():
                data[agent_id][name].append(metric)
    idx = 0
    for agent_id, metrics in data.items():
        plt.figure(idx, figsize=(16,4))
        plt.title(agent_id)
        for name, metric in metrics.items():
            #if name in metric_metadata and metric_metadata[name]['smooth']:
            if name == 'episode_len': 
                continue
            metric = apply_smooth(metric, window)
            plt.plot(metric, linewidth=3, label=name)
        plt.legend()
        idx+=1

def apply_smooth(data: list, window:int = 10) -> list:
    length = len(data)
    return [np.mean(np.array(data[max(0, i-length//window): i+1])) for i in range(length)]

def plot_history(rewards: list[float], smoothing: bool = True):
    num_episode = len(rewards)
    plt.figure(0, figsize=(16, 4))
    plt.title("average reward per step")
    history_smoothed = [
        np.mean(np.array(rewards[max(0, i-num_episode//10): i+1])) for i in range(num_episode)]
    plt.plot(rewards, 'o', alpha=0.2)
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

# gym environment
def play_episode(agent: Agent, env: gym.Env, epsilon: Optional[float] = None, learning_rate: Optional[float] = None, video_path: Optional[str] = None) -> Tuple[float, int]:
    if video_path is not None:
        video = VideoRecorder(env, video_path)
    state, info = env.reset()
    stop = False
    steps = 0
    total_reward = 0
    if epsilon is not None:
        agent._epsilon = epsilon
    if learning_rate is not None:
        agent._learning_rate = learning_rate
    while not stop:
        action = agent.sample_action(state)
        new_state, reward, terminal, truncated, info = env.step(action)
        agent.post_process(state, action, reward, new_state, terminal)
        state = new_state
        stop = terminal or truncated
        # bookkeeping
        total_reward += reward 
        steps += 1
        if video_path is not None:
            video.capture_frame()
    if agent._learning:
        agent.control()
    if video_path is not None:
        video.close()
    return total_reward, steps

def create_shuffled_agent(agent_dict: Dict[str, Agent], shuffle: bool) -> Dict[str, str]:
    agent_names = list(agent_dict.keys())
    if shuffle:
        shuffled = dict(zip(agent_names, random.sample(agent_names, len(agent_names))))
        return shuffled
    else:
        return dict(zip(agent_names, agent_names))

# Pettingzoo.classsic environment
# TODO: randomize agent so that different agent take turns to start first.
def play_multiagent_episode(agent_dict: Dict[str, Agent], env: AECEnv, shuffle: bool = False, debug: bool = False) -> defaultdict:
    if debug: 
        # In debug mode, we fix the random behavior so it's the same sequence for every episode
        np.random.seed(101)
    shuffled = create_shuffled_agent(agent_dict, shuffle)
    env.reset()
    for agent in agent_dict.values():
        agent.reset()
    logs = defaultdict(lambda: defaultdict(float))
    history = defaultdict(list)

    for agent_id in env.agent_iter():
        agent = agent_dict[shuffled[agent_id]]
        # Make observation
        observation, reward, terminal, truncated, info = env.last()
        assert observation is not None
        # Select action
        if terminal or truncated:
            action = None
        else:
            action = agent.sample_action(observation['observation'], observation['action_mask'])
        env.step(action)
        # Train the agent
        if shuffled[agent_id] in history:
            prev_ob, prev_action = history[shuffled[agent_id]][-1]
            agent.post_process(prev_ob, prev_action, reward, observation['observation'],
                          terminal)
        history[shuffled[agent_id]].append((observation['observation'], action))
        # bookkeeping
        logs[shuffled[agent_id]]['reward'] += reward
        logs[shuffled[agent_id]]['episode_len'] += 1
        # if steps > 1000:
        #    terminal = True
    for agent_id, agent in agent_dict.items():
        loss_dict = agent.control()
        if len(loss_dict) > 0:
            logs[shuffled[agent_id]] |= loss_dict

    return logs


def evaluate_multiagent(agent_dict: Dict[str, Agent], env: AECEnv, num_episode: int = 100000, epsilon: float = 0.0, threshold: float = 0.0):
    for agent in agent_dict.values():
        agent._epsilon = 0.0
    total_reward = defaultdict(float)
    for _ in tqdm(range(num_episode)):
        rewards, steps, vloss, ploss = play_multiagent_episode(
            agent_dict, env, False)  # ,epsilon=epsilon_schedule[i])
        for agent, reward in rewards.items():
            total_reward[agent] += reward
    return total_reward, num_episode
