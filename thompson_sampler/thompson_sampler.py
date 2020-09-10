from typing import Optional, List, Union
from copy import deepcopy

import numpy as np


class ThompsonSampler:
    """Class for performimg thompson sampling based on a beta distribution for the bernoulli case"""
    def __init__(self, num_arms: int, priors: Optional[dict] = None, thetas: Optional[List[float]] = None):
        self._num_arms = num_arms
        self._posteriors = {}
        if priors is None:
            for i in range(num_arms):
                self._posteriors[i] = {'alpha': 1.0, 'beta': 1.0}
        else:
            assert len(priors) == num_arms, 'num_arms has to be correspond to size of priors'
            for i, prior in enumerate(priors):
                self._posteriors[i] = prior

        if thetas is None:
            self._thetas = np.random.random(num_arms)
        else:
            assert len(thetas) == num_arms
            self._thetas = thetas

        self._regrets = []
        self._posteriors_over_time = [self._posteriors]

    def run(self, steps: int):
        for i in range(steps):
            idx = self._sample_arm_idx()
            reward = self._get_reward(idx)
            self._update_parameters(idx, reward)
            self._store_regret(idx)
            self._posteriors_over_time.append(deepcopy(self._posteriors))

    def _sample_arm_idx(self) -> int:
        """Sample index of bandit arm from a beta distribution"""
        alphas = []
        betas = []
        for i, param in self._posteriors.items():
            alphas.append(param.get('alpha'))
            betas.append(param.get('beta'))
        sampled_thetas = np.random.beta(alphas, betas)
        idx_max = np.argmax(sampled_thetas)
        return idx_max

    def _get_reward(self, idx: int) -> int:
        reward = 1 if np.random.rand() <= self._thetas[idx] else 0
        return reward

    def _update_parameters(self, idx, reward):
        """Update beta distribution parameters"""
        self._posteriors[idx]['alpha'] += reward
        self._posteriors[idx]['beta'] += 1 - reward

    def _store_regret(self, idx):
        """Store possible loss which results from choosing a suboptimal arm"""
        self._regrets.append(np.max(self._thetas) - self._thetas[idx])

    @property
    def get_posteriors_over_time(self) -> List[dict]:
        return self._posteriors_over_time

    @property
    def get_regrets(self) -> List[float]:
        return self._regrets

    @property
    def get_thetas(self) -> Union[np.array, list]:
        return self._thetas







