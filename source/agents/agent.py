from typing import Dict, List, Optional, Set, Tuple

import gym
from gym.spaces import Space
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class Agent(object):
    def __init__(self, state_space: Space, action_space: Space, discount_rate: float, epsilon: float, learning_rate: float):
        self._state_spacee = state_space
        self._action_space = action_space
        self._discount_rate = discount_rate
        self._epsilon = epsilon
        self._learning_rate = learning_rate

    def sample_action(self, state: int):
        raise NotImplementedError

    def control(self, state: int, action: int, reward: float, new_state: int, terminal: bool):
        raise NotImplementedError
    
    def play_episode(self, env: gym.Env, learning: Optional[bool] = True, epsilon: Optional[float] = None, learning_rate: Optional[float] = None, video_path: Optional[str] = None) -> Tuple[float, int]:
        if video_path is not None:
            video = VideoRecorder(env, video_path)
        state, info = env.reset()
        terminal = False
        steps = 0
        total_reward = 0
        if epsilon is not None:
            self._epsilon = epsilon
        if learning_rate is not None:
            self._learning_rate = learning_rate
        while not terminal:
            action = self.sample_action(state)
            new_state, reward, terminal, truncated, info = env.step(action)
            total_reward += reward 
            terminal = terminal or truncated
            if learning:
                self.control(state, action, reward,
                             new_state, terminal)
            state = new_state
            steps += 1
            #if steps > 1000:
            #    terminal = True
            if video_path is not None:
                video.capture_frame()
        if video_path is not None:
            video.close()
        return total_reward, steps
