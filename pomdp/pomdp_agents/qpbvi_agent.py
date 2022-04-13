import numpy as np

from .agent import Agent
from .qvi_agent import QVIAgent
from ..pomdp_solvers.quantum.pbvi import point_based_value_update, state_point_expand
from ..pomdp_solvers.quantum.calculation import get_observation_probability, state_update

class QPBVIAgent(QVIAgent):
    '''
    Quantum Point-based Value iteration Agent class.
    Value function is represented by set of upsilon matrices eta,
    V(|s>) = max{ <s|upsilon|s> | upsilon in eta },
    agent updates value function by updating eta over some finite set of state points.
    Agent infers quantum state using observed observation.

    Args
    ----------
    config: dictionary
        configuration.
    env: environment
    '''

    def __init__(self, config, env):
        super().__init__(config, env)
    
    def initialize_value_update(self):
        self.S = self.initialize_state_point()
        self.eta = self.initialize_value_function()

    def initialize_state_point(self):
        '''
        Initialize set of state points.
        '''

        S_init = []

        # initialize state point set
        init_state_point = np.zeros((self.state_dim, 1)).astype(np.complex128)
        init_state_point[0] = 1
        S_init.append(init_state_point)

        if self.config.init_belief_type == 'expand':
            while len(S_init) < self.config.init_point_set_num:
                S_init = state_point_expand(S_init, self.action_set, self.observation_set, self.A, 
                                            self.config.belief_choice_metric)

        if self.config.init_belief_type == 'explore':
            while len(S_init) < self.config.init_point_set_num:
                # simulate environment and collect reachable belief points
                state = init_state_point
                for _ in range(self.config.max_step):
                    action = np.random.choice(self.action_set)
                    p_o = get_observation_probability(state, action, self.A, self.observation_set)
                    observation = np.random.choice(self.observation_set, p = p_o)
                    # execute one step simulation
                    state = state_update(state, action, observation, self.A)
                    S_init.append(state)

        return S_init

    def learn(self):
        '''
        Update value function by updating set of upsilon matrices.
        Interleave value update with state point expansion.
        '''

        if self.iterations > 0:
            self.S = state_point_expand(self.S, self.action_set, self.observation_set, self.A, 
                                        self.config.belief_choice_metric)

        for h in range(self.config.horizon):
            self.eta = point_based_value_update(self.S, self.eta, self.state_dim, self.A, self.gamma,
                                                self.action_set, self.observation_set, self.R)

        self.iterations += 1

    def save_model(self, path):
        np.save('{}/eta.npy'.format(path), self.eta)
        np.save('{}/S.npy'.format(path), self.S)

    def load_model(self, path):
        self.eta = np.load('{}/eta.npy'.format(path), allow_pickle=True)
        self.S = np.load('{}/S.npy'.format(path), allow_pickle=True)
