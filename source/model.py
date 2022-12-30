from base64 import b64encode
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
from gym import Env
from tqdm import tqdm

from source.agents.agent import Agent


class Model(object):
    def __init__(self):
        self._data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self._updated = defaultdict(lambda: defaultdict(lambda: bool))
        self._probs = defaultdict(lambda: defaultdict(lambda: []))
        self._outcomes = defaultdict(lambda: defaultdict(lambda: []))

    def check_state(self, state: int) -> bool:
        if state not in self._data:
            return False
        return True

    def check_action(self, state: int, action: int) -> bool:
        if self.check_state(state) == False:
            return False
        if action not in self._data[state]:
            return False
        return True

    def update(self, state, action, reward, new_state, terminal):
        self._data[state][action][(reward, new_state, terminal)] += 1
        self._updated[state][action] = True

    def step(self, state: int, action: int) -> Tuple[float, int, bool]:
        if self.check_action(state, action) == False:
            return 0, -1, True
        if self._updated[state][action] == True:
            total_count = np.sum(list(self._data[state][action].values()))
            probs = []
            outcomes = []
            for rnt, count in self._data[state][action].items():
                probs.append(count / total_count)
                outcomes.append(rnt)
            self._probs[state][action] = probs
            self._outcomes[state][action] = outcomes
            self._updated[state][action] = False
        choice = np.random.choice(
            len(self._probs[state][action]), p=self._probs[state][action])
        reward, new_state, terminal = self._outcomes[state][action][choice]
        return reward, new_state, terminal


def test_model():
    model = Model()
    # state, action, reward, new_state, terminal
    model.update(0, 1, 2.5, 1, False)
    model.update(1, 0, 1.5, 2, False)
    model.update(2, 1, 3.0, 3, True)
    # Test basic step
    reward, new_state, terminal = model.step(0, 1)
    np.testing.assert_equal(reward, 2.5)
    np.testing.assert_equal(new_state, 1)
    np.testing.assert_equal(terminal, False)
    reward, new_state, terminal = model.step(1, 0)
    np.testing.assert_equal(reward, 1.5)
    np.testing.assert_equal(new_state, 2)
    np.testing.assert_equal(terminal, False)
    reward, new_state, terminal = model.step(2, 1)
    np.testing.assert_equal(reward, 3.0)
    np.testing.assert_equal(new_state, 3)
    np.testing.assert_equal(terminal, True)
    # Test stocasitic behavior
    model.update(0, 1, 0, 4, False)
    model.update(0, 1, 0, 4, False)
    model.update(0, 1, 0, 4, False)
    state_counter = [0] * 5
    for _ in range(10000):
        _, new_state, _ = model.step(0, 1)
        state_counter[new_state] += 1
    np.testing.assert_almost_equal(
        state_counter[4] / state_counter[1], 3, decimal=1)
    print('test_model passed!')


test_model()
