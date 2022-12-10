from base64 import b64encode
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from gym import Env
from tqdm import tqdm

from source.agents.agent import Agent

# TODO: possibly need a feature class for transforming raw state -> feature


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

# get: Q(s, a)
# set: Q.update(s, a, value)


class ActionValue(object):
    def __init__(self, weight_shape: tuple, featurizer: Optional[Featurizer] = None, init: Optional[str] = 'random'):
        if init == 'zero':
            self._weight = np.zeros((weight_shape), dtype=float)
        elif init == 'random':
            self._weight = np.random.rand(*weight_shape)
        else:
            raise NotImplemented

        if featurizer is None:
            self._featurizer = IdentityFeaturizer()
        else:
            self._featurizer = featurizer

    def featurize(self, value: np.array) -> np.array:
        return self._featurizer.featurize(value)

    def get(self, state, action):
        raise NotImplementedError

    def get_gradient(self, state, action):
        raise NotImplementedError

    def set(self, state, action, value):
        raise NotImplementedError

    # Weights
    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value: np.array):
        assert value.shape == self._weight.shape, f"Unmatch shape! weight: {self._weight.shape}, input: {value.shape}"
        self._weight = value


class TabularActionValue(ActionValue):
    def __init__(self, state_dim: int, action_dim: int):
        weight_shape = (state_dim, action_dim)
        super().__init__(weight_shape, init='random')

    def get(self, state: int, action: int):
        return self._weight[state][action]

    def set(self, state: int, action: int, value: float):
        self._weight[state][action] = value


class LinearActionValue(ActionValue):
    def __init__(self, weight_shape: tuple[int]):
        super().__init__(weight_shape, init='random')

    def model(self, state_feature: np.array, action: Union[int, float]) -> float:
        flattened = np.reshape(state_feature, (-1))
        feature = np.append(flattened, action)
        return np.dot(feature, self._weight)

    def get(self, state: np.array, action: Union[int, float]):
        state_feature = self.featurize(state)
        return self.model(state_feature, action)

    def get_gradient(self, state: np.array, action: Union[int, float]):
        state_feature = self.featurize(state)
        flattened = np.reshape(state_feature, (-1))
        feature = np.append(flattened, action)
        return feature


def test_linear_action_value():
    Q = LinearActionValue((11,))
    Q.weight = np.ones((11,))
    # Model
    value = Q.model(np.ones((5, 2)), 2)
    np.testing.assert_equal(value, 12)
    # get
    value = Q.get(np.ones((5, 2)), 2)
    np.testing.assert_equal(value, 12)
    # get_gradient
    grad = Q.get_gradient(np.ones((5, 2)), 2)
    np.testing.assert_equal(grad, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]))
    print('test_linear_action_value passed!')

def test_tablular_action_value():
    Q = TabularActionValue(5,5)
    Q.set(3,2, 10)
    np.testing.assert_equal(Q.get(3,2), 10)
    print('test_tablular_action_value passed!')


test_linear_action_value()
test_tablular_action_value()
