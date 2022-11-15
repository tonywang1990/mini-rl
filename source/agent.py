from typing import Dict, List, Optional, Set, Tuple

import gym
from gym.spaces import Space
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class Agent(object):
    def __init__(self, state_space: Space, action_space: Space, discount_rate: float):
        self._state_spacee = state_space
        self._action_space = action_space
        self._discount_rate = discount_rate
        self._prev_action = None
        self._prev_state = None

    def sample_action(self, state: int):
        pass

    def control(self, state: int, action: int, reward: float, new_state: int, terminal: bool):
        pass

    def play_episode(self, env: gym.Env, learning: Optional[bool] = True, video: Optional[VideoRecorder] = None):
        state, info = env.reset()
        terminal = False
        steps = 0
        while not terminal:
            action = self.sample_action(state)
            new_state, reward, terminal, _, info = env.step(action)
            if learning:
                self.control(state, action, reward, new_state, terminal)
            state = new_state
            steps += 1
            if video is not None:
                video.capture_frame()
        return reward, steps
