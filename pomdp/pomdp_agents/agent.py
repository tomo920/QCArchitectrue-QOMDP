class Agent():
    '''
    Base POMDP Agent class.
    '''

    def __init__(self, config):
        self.config = config

    def initialize(self):
        raise NotImplementedError()

    def policy(self, history):
        raise NotImplementedError()

    def update(self, transitions):
        raise NotImplementedError()

    def get_model(self):
        raise NotImplementedError()

    def save_model(self, path):
        raise NotImplementedError()

    def load_model(self, path):
        raise NotImplementedError()
