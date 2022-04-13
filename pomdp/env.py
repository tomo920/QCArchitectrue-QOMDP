class Env():
    def __init__(self, config):
        self.config = config

    def init_state(self):
        raise NotImplementedError()

    def transition_state(self, state, action):
        raise NotImplementedError()

    def get_observation(self, action, next_state):
        raise NotImplementedError()

    def get_reward(self, state, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def is_done(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()
