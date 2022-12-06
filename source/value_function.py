from base64 import b64encode
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from gym import Env
from tqdm import tqdm

from source.agents.agent import Agent

#TODO: possibly need a feature class for transforming raw state -> feature

class ValueFunction(object):
    def __init__(self, feature_dim: int):
        self._feature_dim = feature_dim
        pass
    # d v(s,w) = 
    def get_value(self, feature):
        pass
    def get_gradient(self, feature):
        pass

class LinearValueFunction(ValueFunction):
    def __init__(self, feature_dim: int):
        super().__init(self, feature_dim)
        self._weight = np.zeros((self._feature_dim), dtype = float)
    # d v(s,w) = X(s) * w.T
    def get_value(self, feature: np.array) -> float:
        assert feature.shape == self._weight.shape, f"feature shape {feature.shape} doesn't match weight share {self._weight.shape}"
        return np.dot(self._weight, feature)
    # d v(s,w) / d w = X(s)
    def get_gradient(self, feature):
        return feature
    @property
    def weight(self):
        return self._weight
    @weight.setter
    def weight(self, value):
        self._weight = value

def test_linear_value_function():
    model = LinearValueFunction(5)
    model.weight = np.array([1,1,1,1,1])
    np.testing.assert_equal(model.get_value(np.array([2,2,2,2,2])), 10)
    np.testing.assert_equal(model.weight, np.array([1,1,1,1,1]))
    print('linear value fuction test passed!')
        