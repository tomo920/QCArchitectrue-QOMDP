import os
import sys

import numpy as np

from quantumcircuit.quantum_circuit_design import QuantumCircuitDesignEnv

from quantumcircuit.qc import s_base, get_tensor_product

class StatePreparation(QuantumCircuitDesignEnv):
    '''
    State preparation task is to get gates making the state of system target state.
    State preparation task is defined by QOMDP.

    Reward is the fidelity of system state and target state.

    Args
    ----------
    config: dictionary
        configuration.
    '''

    def __init__(self, config):
        super().__init__(config)
    
    def define_task(self):
        # set target state
        if self.config.target_state == 'one':
            self.s_target = get_tensor_product([s_base[1] for _ in range(self.bn_c)])
        elif self.config.target_state == 'random':
            s = np.random.uniform(-1.0, 1.0, (2**self.bn_c, 1)) + 1j * np.random.uniform(-1.0, 1.0, (2**self.bn_c, 1))
            self.s_target = s / np.sqrt(np.sum(np.abs(s)**2))
        elif self.config.target_state == 'load':
            self.s_target = np.load('{}/target.npy'.format(os.path.dirname(__file__)), allow_pickle=True)
        elif self.config.target_state == 'bell':
            if self.bn_c == 2:
                self.s_target = (np.kron(s_base[0], s_base[0]) + np.kron(s_base[1], s_base[1])) / np.sqrt(2)
            else:
                print('Bell state is for 2 qubits')
                sys.exit()
        elif self.config.target_state == 'ghz':
            if self.bn_c > 2:
                self.s_target = (get_tensor_product([s_base[0]]*self.bn_c) + get_tensor_product([s_base[1]]*self.bn_c)) / np.sqrt(2)
            else:
                print('GHZ state is for qubits > 2 qubits')
                sys.exit()

    def get_reward_operator(self):
        '''
        Reward operator is R[a_t, a_t+1],
        reward is calculated by r_t+1 = <s_t|R[a_t, a_t+1]|s_t>.
        Reward of state preparation is fidelity, 
        R[a_t, a_t+1] = ∑_o A[a_t, a_t+1][o]†|s_target><s_target|A[a_t, a_t+1][o]

        Args
        ----------

        Returns
        ----------
        R: dictionary
            Set of reward operators.
            R[a] is reward operator matrix of executing action a.
        '''

        rho_target = np.dot(self.s_target, self.s_target.conj().T)

        R = {}
        for a in self.action_set:
            R_a = np.zeros((self.state_dim, self.state_dim)).astype(np.complex128)
            for o in self.observation_set:
                R_a += np.dot(np.dot(self.A[a][o].conj().T, rho_target), self.A[a][o])
            R.setdefault(a, R_a)

        return R

    def calculate_fidelity(self, s, s_target):
        """
        Calculate fidelity of |s> and |s_target>.
        Fidelity = |<s|s_target>|^2.
        """

        return np.abs(np.dot(s.conj().T, s_target))**2

    def evaluate(self, state):
        # calculate fidelity
        fidelity = self.calculate_fidelity(state, self.s_target)

        self.update_eval_history(fidelity)

        reward = fidelity
        # check episode done
        if fidelity > self.config.fidelity_threshold or self.steps >= self.max_step:
            done = True
        else:
            done = False

        return reward, done

    def save_task(self, save_path):
        np.save('{}/target.npy'.format(save_path), self.s_target)
