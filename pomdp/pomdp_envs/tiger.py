import numpy as np

from env import Env

class TigerEnv(Env):
    '''
    Tiger Environment class.
    State 0 -> tiger is behind right door.
          1 -> tiger is behind left door.
    Action 0 -> listen
           1 -> open right door
           2 -> open left door
    Observation 0 -> sound from right
                1 -> sound from left

    Args
    ----------
    config: dictionary
        configuration.
    '''

    def __init__(self, config):
        super().__init__(config)

        # define state space
        self.state_set = [0, 1]
        self.state_num = 2

        # define action space
        self.action_set = [0, 1, 2]
        self.action_num = 3

        # define observation space
        self.observation_set = [0, 1]
        self.observation_num = 2

        self.max_step = config.max_step

        # model of environment
        self.p_t = self.get_transition_probability()
        self.p_o = self.get_observation_probability()
        self.r_sa = self.get_reward_function()

    def get_transition_probability(self):
        p_t = np.array([[[1., 0.5, 0.5], [0., 0.5, 0.5]],
                        [[0., 0.5, 0.5], [1., 0.5, 0.5]]])

        return p_t

    def get_observation_probability(self):
        p_o = np.array([[[0.75, 0.25], [0.5, 0.5], [0.5, 0.5]],
                        [[0.25, 0.75], [0.5, 0.5], [0.5, 0.5]]])

        return p_o

    def get_reward_function(self):
        r_sa = np.array([[-1., -25., 10.],
                         [-1., 10., -25.]])

        return r_sa

    def init_state(self):
        '''
        Initialize tiger position to right door(0) or left door(1).
        '''

        self.state = np.random.choice(self.state_set)

    def transition_state(self, state, action):
        next_state = np.random.choice( self.state_set, p = self.p_t[:, state, action] )
        return next_state

    def get_observation(self, action, next_state):
        observation = np.random.choice( self.observation_set, p = self.p_o[:, action, next_state] )
        return observation

    def get_reward(self, state, action):
        reward = self.r_sa[state][action]
        return reward

    def reset(self):
        self.init_state()
        self.steps = 0
        self.total_reward = 0.

    def is_done(self, action):
        if action == 0:
            return False
        else:
            return True

    def step(self, action):
        reward = self.get_reward(self.state, action)
        self.state = self.transition_state(self.state, action)
        observation = self.get_observation(action, self.state)
        done = self.is_done(action)

        self.steps += 1
        self.total_reward += reward

        return observation, reward, done
