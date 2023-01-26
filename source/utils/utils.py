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
import copy

# Utils

# General

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))

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


metric_metadata = {'default': {'format': '-',
                               'smooth': True}, 'episode_len': {'skip': True}}


def apply_smooth(data: list, window: int) -> list:
    length = len(data)
    return [np.mean(np.array(data[i:i+window])) for i in range(length-window)]


def plot_history(rewards: list[float], smoothing: bool = True):
    num_episode = len(rewards)
    plt.figure(0, figsize=(16, 4))
    plt.title("average reward per step")
    history_smoothed = [
        np.mean(np.array(rewards[max(0, i-num_episode//10): i+1])) for i in range(num_episode)]
    plt.plot(rewards, 'o', alpha=0.2)
    if smoothing:
        plt.plot(history_smoothed, linewidth=5)


def plot_training_logs(metrics: defaultdict):
    idx = 0
    plt.figure(idx, figsize=(16, 4))
    plt.title('training')
    for name, metric in metrics.items():
        # if name in metric_metadata and metric_metadata[name]['smooth']:
        if name in ['episode_len', 'num_policy_udpate']:
            continue
        plt.plot(metric, linewidth=3, label=name)
    plt.legend()

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


def update_weights(source_net: torch.nn.Module, target_net: torch.nn.Module, tau: float):
    source_state_dict = source_net.state_dict()
    target_stste_dict = target_net.state_dict()
    for key in target_stste_dict:
        target_stste_dict[key] = source_state_dict[key] * \
            tau + target_stste_dict[key]*(1-tau)
    target_net.load_state_dict(target_stste_dict)

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
        action, action_info = agent.sample_action(state)
        new_state, reward, terminal, truncated, info = env.step(action)
        agent.post_process(state, action, reward,
                           new_state, terminal, action_info)
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


def evaluate_agent(agent: Agent, env: Env, num_episode: int = 100000, epsilon: float = 0.0, threshold: float = 0.0):
    total_reward = 0
    successful_episode = 0
    # Set learning to false
    agent._learning = False
    for _ in tqdm(range(num_episode)):
        reward, _ = play_episode(agent, env, epsilon=epsilon)
        if reward > threshold:
            successful_episode += 1
        total_reward += reward
    return total_reward / num_episode, successful_episode / num_episode


# Pettingzoo.classsic environment
def create_shuffled_agent(agent_dict: Dict[str, Agent], shuffle: bool) -> Dict[str, str]:
    agent_names = list(agent_dict.keys())
    if shuffle:
        shuffled = dict(
            zip(agent_names, random.sample(agent_names, len(agent_names))))
        return shuffled
    else:
        return dict(zip(agent_names, agent_names))


def play_multiagent_episode(agent_dict: Dict[str, Agent], env: AECEnv, shuffle: bool = False, debug: bool = False) -> defaultdict:
    if debug:
        # In debug mode, we fix the random behavior so it's the same sequence for every episode
        np.random.seed(101)
    shuffled = create_shuffled_agent(agent_dict, shuffle)
    env.reset()
    for agent in agent_dict.values():
        agent.reset()
    logs = defaultdict(lambda: defaultdict(float))
    prev_epsisode = {}

    for agent_id in env.agent_iter():
        agent = agent_dict[shuffled[agent_id]]
        # Make observation
        observation, reward, terminal, truncated, info = env.last()
        assert observation is not None
        # Select action
        if terminal or truncated:
            action, action_info = None, {}
        else:
            action, action_info = agent.sample_action(
                observation['observation'], observation['action_mask'])
        env.step(action)
        # Post process
        if shuffled[agent_id] in prev_epsisode:
            prev_ob, prev_action, prev_action_info = prev_epsisode[shuffled[agent_id]]
            agent.post_process(prev_ob, prev_action, reward, observation['observation'],
                               terminal, prev_action_info)
        prev_epsisode[shuffled[agent_id]] = (
            observation['observation'], action, action_info)
        # bookkeeping
        logs[shuffled[agent_id]]['reward'] += reward
        logs[shuffled[agent_id]]['episode_len'] += 1
    for agent_id, agent in agent_dict.items():
        loss_dict = agent.control()
        if loss_dict is not None and len(loss_dict) > 0:
            logs[shuffled[agent_id]] |= loss_dict

    return logs


def duel_training(env: AECEnv, agent_dict: dict, num_epoch: int, num_episode: int, self_play: bool, shuffle: bool, verbal: bool, debug: bool) -> defaultdict:
    if self_play:
        assert agent_dict['player_1'] is not None
        agent_dict['player_2'] = copy.deepcopy(agent_dict['player_1'])
        agent_dict['player_2']._learning = False
    if verbal:
        print('agents:', agent_dict)
    stats = defaultdict(list)
    for i in range(num_epoch):
        logging = defaultdict(lambda: defaultdict(list))
        for _ in tqdm(range(num_episode)):
            logs = play_multiagent_episode(
                agent_dict, env, shuffle=shuffle, debug=False)
            for agent_id, log in logs.items():
                for name, metric in log.items():
                    logging[agent_id][name].append(metric)
        if self_play:
            agent_dict['player_2'].update_weights_from(
                agent_dict['player_1'], tau=1.0)
        output = ""
        for name, metric in logging['player_1'].items():
            stats[name].append(np.mean(np.array(metric)))
            if name == 'reward':
                output += f"win: {metric.count(1)}, lose: {metric.count(-1)}, draw: {metric.count(0.0)}, "
        output += ", ".join([f"{k}: {v[-1]:.5f}" for k, v in stats.items()])
        if verbal:
            print(f"epoch: {i}, {output}")
    plot_training_logs(stats)
    return stats
