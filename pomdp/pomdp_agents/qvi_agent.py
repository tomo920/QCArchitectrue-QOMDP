import sys

import numpy as np

from .agent import Agent
from ..pomdp_solvers.quantum.value_function import get_best_action
from ..pomdp_solvers.quantum.vi import update_value_function
from ..pomdp_solvers.quantum.calculation import state_update

class QVIAgent(Agent):
    '''
    Quantum Value iteration Agent class.
    Value finction is represented by set of upsilon matrices eta,
    V(|s>) = max{ <s|upsilon|s> | upsilon in eta },
    agent updates value function by updating eta.
    Agent infers quantum state using observed observation.

    Args
    ----------
    config: dictionary
        configuration.
    env: environment
    '''

    def __init__(self, config, env):
        super().__init__(config)

        self.state_dim = env.state_dim
        self.action_set = env.action_set
        self.observation_set = env.observation_set

        # model of environment
        self.A = env.A
        self.R = env.R

        self.horizon = config.horizon
        self.gamma = config.gamma

        # initialize
        self.initialize_value_update()

        self.iterations = 0


    def initialize_value_update(self):
        self.eta = self.initialize_value_function()

    def initialize_value_function(self):
        '''
        Initialize set of upsilon matrices.
        '''

        eta_init = []

        # initialize upsilon matrix set
        if self.config.init_value_function_type == 'zero':
            eta_init.append( {'a':-1, 'v':np.zeros((self.state_dim, self.state_dim)).astype(np.complex128)} )
        else:
            print('unsupported initialization type')
            sys.exit()

        return eta_init

    def learn(self):
        '''
        Update value function by updating set of upsilon matrices.
        '''

        self.eta = update_value_function(self.eta, self.state_dim, self.A, self.gamma,
                                         self.action_set, self.observation_set, self.R)

        self.iterations += 1

    def initialize(self):
        """
        Initialize quantum state of the system.
        """

        self.state = np.zeros((self.state_dim, 1)).astype(np.complex128)
        self.state[0] = 1

    def policy(self):
        """
        Calculate policy in current state.

        Args
        ----------

        Returns
        ----------
        pi: list
            pi[a] is the probability of choosing action a in current state.
        """

        best_action = get_best_action(self.state, self.eta)
        # deterministic policy only sample best action
        pi = np.zeros(len(self.action_set))
        pi[best_action] = 1.

        return pi

    def update(self, action, observation):
        """
        Update quantum state of the system.

        Args
        ----------
        action: int
            Action agent executed.
        observation: int
            Observation agent observed.

        Returns
        ----------
        """

        self.state = state_update(self.state, action, observation, self.A)

    def get_model(self):
        return None

    def save_model(self, path):
        np.save('{}/eta.npy'.format(path), self.eta)

    def load_model(self, path):
        self.eta = np.load('{}/eta.npy'.format(path), allow_pickle=True)
