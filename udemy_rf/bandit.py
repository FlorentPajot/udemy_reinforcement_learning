import logging
from typing import List, Tuple

from numpy import random
from numpy import argmax
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

BanditConf = Tuple[float, float]

class Bandit(object):

    def __init__(self, reward_probability, reward):
        self.p = reward_probability
        self.r = reward

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, probability: float):
        if probability >= 0 and probability <= 1:
            self._p = probability
        else:
            raise ValueError('You have to provide a reward probability between 0 and 1')

    def pull(self):
        "Pull the bandit once"
        if random.binomial(1, self._p):
            return self.r
        else:
            return 0

    def __repr__(self):
        return f"Bandit({round(self._p, 2) * 100}% , {self.r})"

    def __str__(self):
        return f"Bandit : {round(self._p, 2) * 100}% / {self.r}"

class Experiment(object):

    def __init__(self, n_runs: int, n_bandits: int, algo: str = None,  **kwargs):
        if n_runs > 0:
            self.n_runs = n_runs
        else:
            raise ValueError("Number on runs hs to be greater than 0")
        if n_bandits > 0:
            self.n_bandits = n_bandits
        else:
            raise ValueError("Number of bandits hs to be greater than 0")
        if algo in ['epsilon-greedy']:
            self.algo = algo
            if algo == 'epsilon-greedy':
                if 'epsilon' in kwargs:
                    self.epsilon = kwargs['epsilon']
                else:
                    logger.warning("Automatically set epsilon to 0.7")
                    self.epsilon = 0.5
            else:
                raise NotImplementedError
        else:
            logger.warning("Default experiment algo set: epsilon-greedy")
            self.algo = 'epsilon-greedy'
            self.epsilon = 0.5

        self.bandits = []
        self.means = []
        self.trials = []

    def get_bandits(self, n_bandits: int, bandits_conf: List[BanditConf] = None):
        if bandits_conf:
            if len(bandits_conf) == n_bandits:
                if isinstance(bandits_conf, List[BanditConf]):
                    for p, r in bandits_conf:
                        self.bandits.append(Bandit(p, r))
                else:
                    raise algoError("bandits_conf expecting List[Tuple(float, float)]")
            else:
                raise ValueError("Provided bdanits_conf doesn't match n_bandits")
        else:
            for _ in range(n_bandits):
                self.bandits.append(Bandit(random.random(), random.randint(0,100)))

        self.means = [0] * n_bandits
        self.trials = [0] * n_bandits
        logger.info(self.bandits)

    def update(self, index: int, reward: int):
        if self.algo == 'epsilon-greedy':
            if self.trials[index] == 0:
                self.means[index] = reward
            else:
                self.means[index] = self.means[index] * (1 - 1 / self.trials[index]) + reward / self.trials[index]
            self.trials[index] += 1
        else:
            raise NotImplementedError

    def run(self):
        if self.algo == 'epsilon-greedy':
            if random.binomial(1, self.epsilon):
                # act greedily
                bandit_index = argmax(self.means)
            else:
                # explore
                bandit_index = random.randint(0, len(self.bandits))
            reward = self.bandits[bandit_index].pull()
            self.update(bandit_index, reward)
            return reward
        else:
            raise NotImplementedError

    def play(self):
        if len(self.bandits) == 0:
            logger.info("Initialize bandits")
            self.get_bandits(self.n_bandits)
        else:
            logger.info("Reset metrics")
            self.means = [0] * n_bandits
            self.trials = [0] * n_bandits

        rewards = []
        trials = []
        for _ in range(self.n_runs):
            rewards.append(self.run())
            trials.append([v for v in self.trials])

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(
            title_text=f"Multi-armed bandit : {self.bandits}"
        )

        x = list(range(len(rewards)))
        for b in range(len(self.bandits)):
            t = [v[b] for v in trials]
            fig.add_trace(go.Scatter(x=x, y=t, mode='lines', name=f'bandit-{b}'), secondary_y=True)

        fig.add_trace(go.Scatter(x=x, y=[sum(rewards[:i]) for i in range(len(rewards))], mode='lines', name=f'total-rewards'))

        # Set x-axis title
        fig.update_xaxes(title_text="<b>Trial index</b>")

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Cumulative rewards</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Number of trials</b>", secondary_y=True)
        fig.show()

    def __str__(self):
        return f"""Experiment with parameters: {self.n_runs} runs / {self.algo}
        """


if __name__ == '__main__':
    exp = Experiment(1000, 3, algo='epsilon-greedy', epsilon=0.9)
    print(exp)
    exp.play()
