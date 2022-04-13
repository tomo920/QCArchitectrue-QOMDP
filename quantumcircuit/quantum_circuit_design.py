import os
import sys
import copy

import numpy as np
from numpy import pi
import itertools

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, execute, Aer, IBMQ
from qiskit.circuit import Qubit

from pomdp.env import Env
from quantumcircuit.circuit_config import Circuit_config, check_unitary_constraints

from quantumcircuit.qc import (
    s_base, 
    M, 
    R, 
    get_gate_matrix, 
    get_tensor_product, 
    enum_control_bn, 
    enum_swap_bn, 
    enum_toffoli_bn,
    enum_cswap_bn,
    apply_gate
)
from calculation import convert_radix

class QuantumCircuitDesignEnv(Env):
    '''
    Quantum Circuit Design Environment class.
    Quantum Circuit Design is defined by QOMDP framework.
    In each step, agent execute action which applies unitary in the circuit, 
    then agent get observation by meaurement of ancilla.

        -----H---U_o(a_t)----------------------M---|0>--      ancilla
        ------------|---------H---U_o(a_t+1)---M---|0>--      ancilla
    ....            |                |         |         ....
        --U_a(a_t)--.----U_a(a_t+1)--.---------|--------      system
                                               |
                                         o_t+1, o_t+2

    This figure is example for measurement_step = 2.

    The goal of the agent is to get policy maximizing future expected reward.
    Reward is task specific value calculated by using quantum state.

    Args
    ----------
    config: dictionary
        configuration.
    '''

    def __init__(self, config):
        super().__init__(config)

        # steps to apply gates before measurement
        self.m_step = config.measurement_step

        # system qubits
        self.bn_c = config.qubit_c
        self.state_dim = 2**self.bn_c

        # ancilla qubits
        self.bn_o_step = config.qubit_o
        self.bn_o = self.bn_o_step * self.m_step

        self.max_step = config.max_step

        # define action space
        self.define_action_space()

        # define observation space
        self.define_observation_space()

        # target state
        self.define_task()

        # model of environment
        self.A = self.get_transition_operator()
        self.R = self.get_reward_operator()

        # qiskit parameter
        self.backend = Aer.get_backend('statevector_simulator')
    
    def define_unitary_action_interface(self):
        '''
        Define what unitary is applied by action in each step.

        Args
        ----------

        Returns
        ----------
        a2unitary: dictionary
            Set of unitaries applied by action.
            a2unitary[a] is list of gates applied by executing action a.
        action_num: int
            Number of actions.
        '''

        rotation_discrete_num = self.config.rotation_discrete_num
        rotation_angle = Circuit_config['rotation_angle']
        # rotation angle list
        rotation_list = np.linspace(-rotation_angle, rotation_angle, rotation_discrete_num)

        # single rotation gate
        single_rotation_gate = Circuit_config['single_rotation_gate']

        gate_type_list = Circuit_config['gate_type_list']

        # gate of each gate type
        single_gate_list = copy.deepcopy(Circuit_config['single_gate_list'])
        single_gate_list.extend(list(itertools.product(single_rotation_gate, rotation_list)))
        target_gate_list = Circuit_config['target_gate_list']

        gate_set_all = []
        # enumerate all types of unitary appplied in system
        for u in itertools.product(gate_type_list, repeat=self.bn_c):
            if not check_unitary_constraints(u):
                continue

            gate_type_bn = {}
            [gate_type_bn.setdefault(type, []) for type in gate_type_list]

            # store index of each gate 
            [gate_type_bn[gate].append(i) for i, gate in enumerate(u)]

            gate_set = []
            for gate in gate_type_list:
                bn_type = gate_type_bn[gate]
                if not bn_type:
                    continue

                if gate == 'single':
                    single_type_list = itertools.product(single_gate_list, repeat=len(bn_type))
                    single_gate_set = [[[t, i] for t, i in zip(t_list, bn_type)] for t_list in single_type_list]
                    gate_set.append(single_gate_set)
                elif gate == 'target':
                    target_type_list = list(itertools.product(target_gate_list, repeat=len(bn_type)))
                    target_bn_list = enum_control_bn(gate_type_bn['cont_targ'], bn_type)
                    target_t_b_set = itertools.product(target_type_list, target_bn_list)
                    target_gate_set = [[[t, b] for t, b in zip(type, bn)] for type, bn in target_t_b_set]
                    gate_set.append(target_gate_set)
                elif gate == 'cz':
                    cz_bn_list = enum_swap_bn(bn_type)
                    cz_gate_set = [[['cz', b] for b in bn] for bn in cz_bn_list]
                    gate_set.append(cz_gate_set)
                elif gate == 'swap':
                    swap_bn_list = enum_swap_bn(bn_type)
                    swap_gate_set = [[['swap', b] for b in bn] for bn in swap_bn_list]
                    gate_set.append(swap_gate_set)
                elif gate == 'toffoli':
                    toffoli_bn_list = enum_toffoli_bn(gate_type_bn['cont_toffoli'], bn_type)
                    toffoli_gate_set = [[['toffoli', b] for b in bn] for bn in toffoli_bn_list]
                    gate_set.append(toffoli_gate_set)
                elif gate == 'cswap':
                    cswap_bn_list = enum_cswap_bn(gate_type_bn['cont_swap'], bn_type)
                    cswap_gate_set = [[['cswap', b] for b in bn] for bn in cswap_bn_list]
                    gate_set.append(cswap_gate_set)

            gate_set_all.extend(list(itertools.product(*gate_set)))
        
        action_num = len(gate_set_all)

        # convert action to list of unitary
        a2unitary = {}
        for a in range(action_num):

            uni_list = []

            # apply gate in ancilla
            if self.config.ancilla_gate in single_rotation_gate:
                theta_anc = eval(self.config.ancilla_rotation_angle)
                uni_list.append({'qubit': ['ancilla'], 'gate': self.config.ancilla_gate, 'inds': [None], 'theta': theta_anc})
            else:
                uni_list.append({'qubit': ['ancilla'], 'gate': self.config.ancilla_gate, 'inds': [None], 'theta': None})

            # convert action to set of gates applied in system
            for gate in gate_set_all[a]:
                for g in gate:
                    if type(g[0]) == tuple:
                        uni_list.append({'qubit': ['system'], 'gate': g[0][0], 'inds': [g[1]], 'theta': g[0][1]})
                    elif g[0] in single_gate_list:
                        uni_list.append({'qubit': ['system'], 'gate': g[0], 'inds': [g[1]], 'theta': None})
                    else:
                        uni_list.append({'qubit': ['system']*len(g[1]), 'gate': g[0], 'inds': g[1], 'theta': None})

            # apply control gate in system-ancilla
            for i in range(self.bn_c):
                qubit = ['system', 'ancilla']
                inds = [[i], None]
                uni_list.append({'qubit': qubit, 'gate': 'cx', 'inds': inds, 'theta': None})
            
            a2unitary.setdefault(a, uni_list)
        
        return a2unitary, action_num

    def define_action_space(self):
        '''
        Define action space. Aciton is set of multi step actions.
        Action is the gate and the parameter of the gate.
        In discrete type, action is the index of the gate and the dsicreted rotation angle.

        Args
        ----------
        '''

        if self.config.is_action_discrete:
            self.a2unitary, self.action_num = self.define_unitary_action_interface()

            # Set of actions for multi step
            self.action_set = list(range(self.action_num**self.m_step))

            # convert action to multi step actions
            self.a2ma = convert_radix
        else:
            print('continuous action space is not supported yet')
            sys.exit()

    def define_observation_space(self):
        '''
        Define observation space.
        Observation is measurement outcomes of ancilla.
        When the state of ancilla after measurement is |o_0>|o_1>...|o_n-1>,
        observation is o_0 * 2**(n-1) + o_1 * 2**(n-2) + ... + o_n-1 * 2**0.

        Args
        ----------
        '''

        self.observation_set = list(range(2**self.bn_o))

        # observation gate U_o
        self.u_o_gate = self.config.observation_gate

        self.o2bin = lambda o: bin(o)[2:].zfill(self.bn_o)
        self.bin2o = lambda bin: int(bin, 2)

    def define_task(self):
        return NotImplementedError("You have to implement goal of task.")

    def save_task(self, save_path):
        return NotImplementedError("You have to implement saving task setting method.")

    def get_transition_operator(self):
        '''
        Transition operator is A[a_t, a_t+1][o_t+1, o_t+2],
        |s_t+2> = A[a_t, a_t+1][o_t+1, o_t+2]|s_t>/sqrt(p(o_t+1, o_t+2||s_t>, a_t, a_t+1)).

        |0>    -----H---U_o(a_t)----------------------M---|0>--          ancilla
        |0>    ------------|---------H---U_o(a_t+1)---M---|0>--          ancilla
                           |                |         |
        |s_t>  --U_a(a_t)--.----U_a(a_t+1)--.---------|-------- |s_t+2>  system
                                                      |
                                                o_t+1, o_t+2

        Args
        ----------

        Returns
        ----------
        A: dictionary
            Set of transition operators.
            A[a][o] is transition operator matrix of executing action a and
            getting observation o.
        '''

        backend = Aer.get_backend('unitary_simulator')
        
        I_c = get_tensor_product([get_gate_matrix('i') for _ in range(self.bn_c)])
        # basis states for state space of ancilla
        e = {}
        for o in self.observation_set:
            e_o = get_tensor_product([s_base[int(m)] for m in self.o2bin(o)])
            e.setdefault(o, e_o)

        A = {}
        for a in self.action_set:
            A.setdefault(a, {})
            # get unitary applied by action a
            self.init_quantum_circuit()
            self.apply_unitary(a)
            U_a = execute(self.qc, backend).result().get_unitary(self.qc)
            for o in self.observation_set:
                # get kraus operator
                A_a_o = np.dot(np.dot(np.kron(I_c, e[o].conj().T), U_a), np.kron(I_c, e[0]))
                A[a].setdefault(o, A_a_o)

        return A

    def get_reward_operator(self):
        return NotImplementedError("You have to implement reward operator.")

    def evaluate(self, state):
        return NotImplementedError("You have to implement evaluate method.")

    def init_history(self):
        self.history = []

    def init_eval_history(self):
        self.eval_list = []

    def update_history(self, a, o, s_, r, m):
        self.history.append([a, o, s_, r, m])

    def update_eval_history(self, evaluation):
        self.eval_list.append(evaluation)

    def apply_unitary(self, action):
        '''
        Apply unitary defined by action in the circuit.

        Args
        ----------
        action: int
            Excuted action.

        Returns
        ----------
        '''

        # convert action to multi action
        ma = self.a2ma(action, self.action_num, self.m_step)

        for i in range(self.m_step):
            uni_list = self.a2unitary[ma[i]]

            for unitary in uni_list:
                # get quantum register used to aplly gate
                qubit = [self.get_qr(name, ind, i) for name, ind in zip(unitary['qubit'], unitary['inds'])]
                # apply gate
                apply_gate(self.qc, unitary['gate'], qubit, unitary['theta'])

    def init_quantum_circuit(self, previous_state=None):
        self.qr_c = QuantumRegister(self.bn_c, 'q_c')
        self.qr_o = QuantumRegister(self.bn_o, 'q_o')
        self.cr = ClassicalRegister(self.bn_o)
        self.qc = QuantumCircuit(self.qr_o, self.qr_c, self.cr)
        if previous_state is not None:
            self.qc.set_statevector(previous_state)
    
    def get_qr(self, name, inds, m_step=None):
        '''
        Get quantum register.

        Args
        ----------
        name: str
            Name of type of system to get register.
        inds: list
            Indices in the bits of the name.
        m_step: int
            Number of step in multi step execution.

        Returns
        ----------
        quantum register object in the circuit.
        '''

        if name == 'system':
            qr = self.qr_c if inds is None else self.qr_c[inds]
        elif name == 'ancilla':
            # ancilla indices used in this step
            anc_inds = list(range(m_step*self.bn_o_step, (m_step+1)*self.bn_o_step))
            qr_o_step = self.qr_o[anc_inds]
            qr = qr_o_step if inds is None else [qr_o_step[ind] for ind in inds]
        return qr

    def run_circuit_one_step(self, action):
        '''
        Run one step in the circuit.

        Args
        ----------
        action: int
            Excuted action.

        Returns
        ----------
        s_: list
            Quantum state after one step execution.
        o: str
            Meaurement outcomes of ancilla qubits.
        '''

        self.apply_unitary(action)

        self.qc.measure(self.qr_o, self.cr)

        self.qc.reset(self.qr_o)

        result = execute(self.qc, self.backend).result()

        s_  = result.get_statevector(self.qc).reshape((-1, 1))
        o = list(result.get_counts(self.qc).keys())[0]

        return s_, o

    def reset(self):
        self.init_quantum_circuit()
        self.init_history()
        self.init_eval_history()
        self.steps = 0

    def step(self, action):
        s_, o = self.run_circuit_one_step(action)
        self.steps += self.m_step

        # initialize quantum circuit with previous state
        self.init_quantum_circuit(previous_state=s_)

        observation = self.bin2o(o)

        # state of system
        s_system = s_.reshape((-1, 2**self.bn_o))[:,0].reshape(-1, 1)

        # evaluation
        reward, done = self.evaluate(s_system)

        # save this transition
        self.update_history(action, observation, s_system, reward, o)

        return observation, reward[0][0], done
    
    def save_model(self, save_path):
        np.save('{}/A.npy'.format(save_path), self.A)
        np.save('{}/R.npy'.format(save_path), self.R)
        np.save('{}/a2unitary.npy'.format(save_path), self.a2unitary)
        self.save_task(save_path)

    def save_result(self, save_path):
        np.save('{}/history.npy'.format(save_path), self.history)
        np.save('{}/step.npy'.format(save_path), self.steps)
        np.save('{}/eval_history.npy'.format(save_path), self.eval_list)
