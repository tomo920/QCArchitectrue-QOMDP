import sys

import numpy as np

from .agent import Agent
from pomdp_solvers.value_iteration import belief_update, update_value_function, get_best_action

class VIAgent(Agent):
    '''
    Value iteration Agent class.

    Args
    ----------
    config: dictionary
        configuration.
    env: environment
    '''

    def __init__(self, config, env):
        super().__init__(config)

        self.state_set = env.state_set
        self.state_num = env.state_num
        self.action_set = env.action_set
        self.action_num = env.action_num
        self.observation_set = env.observation_set
        self.observation_num = env.observation_num

        # model of environment
        self.p_t = env.p_t
        self.p_o = env.p_o
        self.r_sa = env.r_sa

        self.horizon = config.horizon
        self.gamma = config.gamma

        # initialize value function
        self.initialize_value_update()

        self.iterations = 0

    def initialize_value_update(self):
        self.nu = self.initialize_value_function()

    def initialize_value_function(self):
        '''
        Initialize set of alpha vectors.

        Args
        ----------

        Returns
        ----------
        nu_init: list
            Initial set of alpha vectors.
        '''

        nu_init = []  # set of alpha vectors

        # initialize alpha vector set
        if self.config.init_value_function_type == 'zero':
            nu_init.append( {'a':-1, 'v':np.zeros(self.state_num)} )
        else:
            print('unsupported initialization type')
            sys.exit()

        return nu_init

    def learn(self):
        self.nu = update_value_function(self.nu, self.p_t, self.p_o, self.r_sa, self.gamma,
                                        self.state_set, self.action_set, self.observation_set)
        self.iterations += 1

    def initialize_belief_state(self):
        '''
        Initialize belief state.

        Args
        ----------

        Returns
        ----------
        b_init: list
            Initial belief state.
        '''

        if self.config.init_belief_type == 'uniform':
            b_init = np.ones(self.state_num) / self.state_num
        else:
            print('unsupported initialization type')
            sys.exit()

        return b_init

    def initialize(self):
        '''
        Initialize agent.

        Args
        ----------

        Returns
        ----------
        '''

        self.belief_state = self.initialize_belief_state()

    def policy(self):
        best_action = get_best_action(self.belief_state, self.nu)
        # deterministic policy only sample best action
        pi = np.zeros(self.action_num)
        pi[best_action] = 1.

        return pi

    def update_belief_state(self, belief_state, action, observation):
        """
        Update belief state using action agent executed and
        observation agent observed.

        Args
        ----------
        belief_state: list
            Belief state before update
        action: int
            Action agent executed.
        observation: int
            Observation agent observed.

        Returns
        ----------
        belief_state_: list
            Belief state after update
        """

        # update belief state
        belief_state_ = belief_update(belief_state, action, observation,
                                      self.p_t, self.p_o, self.state_set)

        return belief_state_

    def update(self, action, observation):
        """
        Update agent.

        Args
        ----------
        action: int
            Action agent executed.
        observation: int
            Observation agent observed.

        Returns
        ----------
        """

        # update belief state
        self.belief_state = self.update_belief_state(self.belief_state, action, observation)

    def get_model(self):
        return None

    def save_model(self, path):
        return None

    def load_model(self, path):
        return None
