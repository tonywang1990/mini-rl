from typing import Dict, List, Optional, Set, Tuple

import gym
from gym.spaces import Space
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from typing import Any


class Agent(object):
    def __init__(self, state_space: Space, action_space: Space, discount_rate: float, epsilon: float, learning_rate: float, learning:float=True):
        self._state_space = state_space
        self._action_space = action_space
        self._discount_rate = discount_rate
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._learning = learning
        self._Q = None
        self._policy = None

    def sample_action(self, state: Any):
        raise NotImplementedError

    def control(self, state: Any, action: Any, reward: float, new_state: Any, terminal: bool):
        raise NotImplementedError
    
    def reset(self):
        pass

    def pre_process(self):
        pass

    def post_process(self):
        pass
    
    def play_episode(self, env: gym.Env, epsilon: Optional[float] = None, learning_rate: Optional[float] = None, video_path: Optional[str] = None) -> Tuple[float, int]:
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
            if self._learning:
                self.control(state, action, reward,
                             new_state, terminal)
            state = new_state
            steps += 1
            if video_path is not None:
                video.capture_frame()
        if video_path is not None:
            video.close()
        return total_reward, steps
