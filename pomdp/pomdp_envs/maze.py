import numpy as np
import sys

from env import Env

legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]), np.array([4.0, 1.0]), np.array([5.0, 1.0]),
                np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 2.0]),
                np.array([1.0, 3.0]), np.array([3.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([4.0, 4.0]),
                np.array([2.0, 5.0]), np.array([3.0, 5.0]), np.array([4.0, 5.0]), np.array([5.0, 5.0])]
illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]), np.array([6.0, 0.0]),
                np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([6.0, 1.0]),
                np.array([0.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0]), np.array([6.0, 2.0]),
                np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([6.0, 3.0]),
                np.array([0.0, 4.0]), np.array([3.0, 4.0]), np.array([5.0, 4.0]), np.array([6.0, 4.0]),
                np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([6.0, 5.0]),
                np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([6.0, 6.0])]
legal_states = [state.tostring() for state in legal_states]
illegal_states = [state.tostring() for state in illegal_states]

observation_list = {}
observation_list[legal_states[0]] = 9
observation_list[legal_states[1]] = 2
observation_list[legal_states[2]] = 11
observation_list[legal_states[3]] = 4
observation_list[legal_states[4]] = 9
observation_list[legal_states[5]] = 10
observation_list[legal_states[6]] = 6
observation_list[legal_states[7]] = 6
observation_list[legal_states[8]] = 7
observation_list[legal_states[9]] = 8
observation_list[legal_states[10]] = 4
observation_list[legal_states[11]] = 1
observation_list[legal_states[12]] = 5
observation_list[legal_states[13]] = 6
observation_list[legal_states[14]] = 1
observation_list[legal_states[15]] = 2
observation_list[legal_states[16]] = 3
observation_list[legal_states[17]] = 4

o_num = 11

start_state = np.array([1.0, 3.0])
goal_state = np.array([5.0, 3.0])

legal_action_list = {}
legal_action_list[1] = [0, 3]
legal_action_list[2] = [0, 1]
legal_action_list[3] = [0, 1, 3]
legal_action_list[4] = [1]
legal_action_list[5] = [1, 2]
legal_action_list[6] = [2, 3]
legal_action_list[7] = [0]
legal_action_list[8] = [0, 1, 2, 3]
legal_action_list[9] = [0, 2]
legal_action_list[10] = [1, 3]
legal_action_list[11] = [0, 1, 2]

'''
action 0 -> right
action 1 -> left
action 2 -> up
action 3 -> down
'''
action_list = [np.array([1.0, 0.0]),
               np.array([-1.0, 0.0]),
               np.array([0.0, 1.0]),
               np.array([0.0, -1.0])]

class PoMaze(Env):
    def __init__(self, config):
        super().__init__(config)
        self.max_step = config.max_step

        if config.is_action_discrete:
            self.action_num = len(action_list)
            self.action_set = list(range(self.action_num))
        else:
            print('This environment is discrete environment.')
            sys.exit()

    def init_history(self):
        self.history =[]

    def update_history(self, a, o, s_, r):
        self.history.append([a, o, s_, r])

    def get_observation(self):
        state = self.state.tostring()
        o = observation_list[state]
        return o

    def reset(self):
        self.state = start_state
        self.init_history()
        self.steps = 0

    def step(self, action):
        self.change_state(action)
        self.steps += 1

        observation = self.get_observation()

        if self.check_goal():
            reward = 100.
            done = True
        elif self.steps >= self.max_step:
            reward = 0.
            done = True
        else:
            reward = 0.
            done = False

        self.update_history(action, observation, self.state, reward)

        return observation, reward, done

    def change_state(self, action):
        c_state = self.state
        self.state = self.state+action_list[action]
        if not self.check_legal():
            self.state = c_state

    def check_legal(self):
        state = self.state.tostring()
        if state in legal_states:
            return True
        else:
            return False

    def check_goal(self):
        return self.state[0] == goal_state[0] and self.state[1] == goal_state[1]

    def save_result(self, save_path):
        np.save('{}/history.npy'.format(save_path), self.history)
        np.save('{}/step.npy'.format(save_path), self.steps)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state