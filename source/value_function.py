from __future__ import annotations

from base64 import b64encode
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from gym import Env
from gym.spaces import Discrete, Space, Box
from tqdm import tqdm
from copy import deepcopy

from source.agents.agent import Agent
from sklearn.neural_network import MLPRegressor
import random


class Featurizer(object):
    def __init__(self):
        pass

    def featurize(self, input):
        pass


class IdentityFeaturizer(Featurizer):
    def __init__(self):
        pass

    def featurize(self, input):
        return input


# Only support discrete action space!
# get: Q(s, a)
# set: Q.update(s, a, value)
class ActionValue(object):
    def __init__(self, state_space: Space, action_space: Discrete, featurizer: Optional[Featurizer] = None):
        assert isinstance(
            action_space, Discrete), f"Only Discrete type is supported for action_space but {type(action_space)} given"
        self._state_space = state_space
        self._action_space = action_space
        if featurizer is None:
            self._featurizer = IdentityFeaturizer()
        else:
            self._featurizer = featurizer

    def featurize(self, value: np.array) -> np.array:
        return self._featurizer.featurize(value)

    def get(self, state, action):
        raise NotImplementedError

    def set(self, state, action, value):
        raise NotImplementedError

    def sample_action(self, state: Any, epsilon: Optional[float] = 0) -> Tuple[int, float]:
        if np.random.rand() < epsilon:
            action = np.random.choice(self._action_space.n)
            return action, self.get(state, action)
        values = []
        for action in range(self._action_space.n):
            values.append(self.get(state, action))
        return np.argmax(values), np.max(values)


class TabularActionValue(ActionValue):
    def __init__(self, state_space: Discrete, action_space: Discrete):
        super().__init__(state_space, action_space)
        weight_shape = (state_space.n, action_space.n)
        self.init_weight(weight_shape, 'random')

    def get(self, state: int, action: int):
        return self._weight[state][action]

    def set(self, state: int, action: int, value: float):
        self._weight[state][action] = value

    def init_weight(self, shape: tuple, type: str):
        if type == 'zero':
            self._weight = np.zeros((shape), dtype=float)
        elif type == 'random':
            self._weight = np.random.rand(*shape)
        else:
            raise NotImplemented


class LinearActionValue(ActionValue):
    def __init__(self, state_space: Space, action_space: Discrete, featurizer: Optional[Featurizer] = None):
        super().__init__(state_space, action_space, featurizer)
        sample_state, sample_action = self._state_space.sample(), self._action_space.sample()
        weight_shape = self._feature_vec(sample_state, sample_action).shape
        self.init_weight(weight_shape, 'random')

    def _feature_vec(self, state: np.array, action: int) -> np.array:
        state_feature = self.featurize(state)
        flattened = np.reshape(state_feature, (-1))
        return np.append(flattened, action)

    def _model(self, feature_vec: np.array) -> float:
        return np.dot(feature_vec, self._weight)

    def get(self, state: np.array, action: int):
        feature_vec = self._feature_vec(state, action)
        return self._model(feature_vec)

    def get_gradient(self, state: np.array, action: int):
        feature_vec = self._feature_vec(state, action)
        return feature_vec

    def init_weight(self, shape: tuple, type: str):
        if type == 'zero':
            self._weight = np.zeros((shape), dtype=float)
        elif type == 'random':
            self._weight = np.random.rand(*shape)
        else:
            raise NotImplemented

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value: np.array):
        assert value.shape == self._weight.shape, f"Unmatch shape! weight: {self._weight.shape}, input: {value.shape}"
        self._weight = value


class LearnedActionValue(ActionValue):
    def __init__(self, state_space: Space, action_space: Discrete, featurizer: Optional[Featurizer] = None):
        super().__init__(state_space, action_space, featurizer)
        self._model = MLPRegressor(hidden_layer_sizes=(
            8, 8, 8, 4, 2), random_state=1, max_iter=500, learning_rate_init=1)
        self._fitted = False

    def _feature_vec(self, state: np.array, action: int) -> np.array:
        flattened = np.reshape(state, (-1))
        return np.append(flattened, action)

    def get(self, state: np.array, action: int) -> float:
        if self._fitted == False:
            return random.uniform(-100, 0)
        feature = self._feature_vec(state, action)
        feature = feature.reshape(1, feature.shape[0])
        return self._model.predict(feature)[0]

    def set(self, state: np.array, action: int, value: float):
        feature = self._feature_vec(state, action)
        feature = feature.reshape(1, feature.shape[0])
        self._model.partial_fit(feature, np.array([value]))
        self._fitted = True

    def fit(self, state: list, action: list, value: list, batch_size: int) -> float:
        state = np.vstack(state).reshape(batch_size, (-1))
        action = np.array(action).reshape(batch_size, 1)
        value = np.array(value).reshape(batch_size,)
        assert state.shape[0] == action.shape[0] and action.shape[0] == value.shape[
            0], f"shape doesn't match! {state.shape}, {action.shape}, {value.shape}"
        feature = np.concatenate([state, action], axis=1)
        self._model.partial_fit(feature, value)
        self._fitted = True
        return self._model.score(feature, value)

    def copy_from(self, action_value: LearnedActionValue):
        self._model = deepcopy(action_value._model)
        self._fitted = action_value._fitted


def test_learned_action_value():
    state = np.ones((5, 8))
    action = np.ones((5,))
    value = np.ones((5,))
    Q = LearnedActionValue(Box(low=0.0, high=1.0, shape=(8, )), Discrete(4))
    Q.fit(state, action, value, 5)
    #np.testing.assert_almost_equal(Q.get(np.ones(8,), 1), [0.642], decimal=3)


test_learned_action_value()


def test_linear_action_value():
    Q = LinearActionValue(Box(low=0.0, high=10.0, shape=(5, 2)), Discrete(4))
    np.testing.assert_equal(Q.weight.shape, (11,))
    Q.weight = np.ones((11,))
    # Model
    value = Q.get(np.ones((5, 2)), 2)
    np.testing.assert_equal(value, 12)
    # get_gradient
    grad = Q.get_gradient(np.ones((5, 2)), 2)
    np.testing.assert_equal(grad, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]))
    print('test_linear_action_value passed!')


def test_tablular_action_value():
    Q = TabularActionValue(Discrete(5), Discrete(5))
    Q.set(3, 2, 10)
    np.testing.assert_equal(Q.get(3, 2), 10)
    print('test_tablular_action_value passed!')


test_linear_action_value()
test_tablular_action_value()
