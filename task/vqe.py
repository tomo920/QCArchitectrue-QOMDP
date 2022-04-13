import os
import sys

import numpy as np

from quantumcircuit.quantum_circuit_design import QuantumCircuitDesignEnv

from .hamiltonian import get_hamiltonian

class VQE(QuantumCircuitDesignEnv):
    '''
    VQE class is to get gates make expectation value of hamiltonian minimum value.
    VQE class is defined by Quantum circuit design environment.

    Reward is calculated by -1 * expectation value.

    Args
    ----------
    config: dictionary
        configuration.
    '''

    def __init__(self, config):
        super().__init__(config)
    
    def define_task(self):
        # define hamiltonian
        self.hamiltonian = get_hamiltonian(self.config.molecule, self.config.bond_length)

    def get_reward_operator(self):
        '''
        Reward operator is R[a_t, a_t+1],
        reward is calculated by r_t+1 = <s_t|R[a_t, a_t+1]|s_t>.
        Reward of vqe is calculated by -1 * expectation value, 
        R[a_t, a_t+1] = ∑_o A[a_t, a_t+1][o]† Hamiltonian A[a_t, a_t+1][o]

        Args
        ----------

        Returns
        ----------
        R: dictionary
            Set of reward operators.
            R[a] is reward operator matrix of executing action a.
        '''

        R = {}
        for a in self.action_set:
            R_a = np.zeros((self.state_dim, self.state_dim)).astype(np.complex128)
            for o in self.observation_set:
                R_a += -np.dot(np.dot(self.A[a][o].conj().T, self.hamiltonian), self.A[a][o])
            R.setdefault(a, R_a)

        return R

    def calculate_expvalue(self, s, hamiltonian):
        """
        Calculate expectation value of hamiltonian with respect to state s.
        expval = <s|hamiltonian|s>.
        """

        return np.dot(s.conj().T, np.dot(hamiltonian, s)).real

    def evaluate(self, state):
        # calculate expectation value
        expval = self.calculate_expvalue(state, self.hamiltonian)

        self.update_eval_history(expval)

        reward = -expval
        # check episode done
        if expval < self.config.expval_threshold or self.steps >= self.max_step:
            done = True
        else:
            done = False

        return reward, done

    def save_task(self, save_path):
        np.save('{}/hamiltonian.npy'.format(save_path), self.hamiltonian)
