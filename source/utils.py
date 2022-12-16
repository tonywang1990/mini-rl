from base64 import b64encode
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gym import Env
from tqdm import tqdm

from source.agents.agent import Agent

# Utils


def render_mp4(videopath: str) -> str:
    """
    Gets a string containing a b4-encoded version of the MP4 video
    at the specified path.
    """
    mp4 = open(videopath, 'rb').read()
    base64_encoded_mp4 = b64encode(mp4).decode()
    return f'<video width=400 controls><source src="data:video/mp4;' \
        f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'

# TODO: extend this function to take ActionValue as input
def get_epsilon_greedy_policy_from_action_values(action_values: np.array, epsilon: Optional[float] = 0.0) -> np.array:
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


def get_state_values_from_action_values(action_values: np.array, policy: Optional[np.array] = None) -> np.array:
    if policy is None:
        # assume greedy policy
        policy = get_epsilon_greedy_policy_from_action_values(action_values)
    state_values = np.sum(action_values * policy, axis=1)
    return state_values


def plot_history(history: list):
    num_episode = len(history)
    plt.figure(0, figsize=(16, 4))
    plt.title("average reward per step")
    history_smoothed = [
        np.mean(history[max(0, i-num_episode//10): i+1]) for i in range(num_episode)]
    plt.plot(history, 'o', alpha=0.2)
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
    #for idx, val in np.ndenumerate(actions):
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


def estimate_success_rate(agent: Agent, env: Env, num_episode: int = 100000, epsilon: float = 0.0):
    total_reward = 0
    for _ in tqdm(range(num_episode)):
        reward, _ = agent.play_episode(env, learning=False, epsilon=epsilon)
        total_reward += reward 
    return total_reward / num_episode


def create_decay_schedule(num_episodes: int, value_start: float = 0.9, value_decay: float = .9999, value_min: float = 0.05):
    # get 20% value at 50% espisode
    value_decay = 0.2 ** (1/(0.5 * num_episodes))
    return [max(value_min, (value_decay**i)*value_start) for i in range(num_episodes)]
