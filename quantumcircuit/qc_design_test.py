from itertools import permutations
import numpy as np
import argparse

import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from quantum_circuit_design import QuantumCircuitDesignEnv
from quantum_circuit_design_no_measurement import QCDesignNoMeasurementEnv
from qc import get_gate_matrix, get_tensor_product
from pomdp.pomdp_solvers.quantum.calculation import state_update

def run_circuit(env, config):
    a_history = []
    o_history = []

    # execute in circuit
    env.reset()
    while True:
        a = np.random.choice(env.action_set)
        observation, _, done = env.step(a)
        a_history.append(a)
        o_history.append(observation)
        if done:
            break
    s_circuit = env.history[-1][2]

    return s_circuit, a_history, o_history, env.steps

def update_state(state_dim, step, a_history, o_history, A):
    # state update using model
    s_ = np.zeros((state_dim, 1)).astype(np.complex128)
    s_[0] = 1
    for i in range(step):
        s_ = state_update(s_, a_history[i], o_history[i], A)
    return s_

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set parameters.')

    parser.add_argument('target_state', type=str,
                        help='Target state for circuit design'
                             '[one]'
                             '[ghz]',
                        choices=['one', 'ghz'])
    parser.add_argument('max_step', type=int, help='Max number of steps in one episode')

    parser.add_argument('--seed', default=123, type=int, help='Random seed')
    parser.add_argument('--fidelity_threshold',  default=0.99, type=float, help='Episode ends when obtained this fidelity')
    parser.add_argument('--is_action_discrete', action='store_true')

    # for Quantum Circuit
    parser.add_argument('--qubit_c', default=3, type=int, help='Number of qubits controlled by POMDP algorithm')
    parser.add_argument('--qubit_o', default=3, type=int, help='Number of qubits for measurement')
    parser.add_argument('--measurement_step', default=1, type=int, help='Number of steps to execute untill next measurement')
    parser.add_argument('--rotation_discrete_num', default=12, type=int, help='Number of parts of the rotation angle of optimized gate divided when action discrete type')
    parser.add_argument('--ancilla_gate', default='h', type=str,
                         help='Gate applied in ancilla before applying control gate'
                              '[h]'
                              '[rx]'
                              '[ry]'
                              '[rz]',
                         choices=['h', 'rx', 'ry', 'rz'])
    parser.add_argument('--ancilla_rotation_angle', default='pi/2.5', type=str,
                         help='Rotation angle of gate applied in ancilla before applying control gate')
    parser.add_argument('--observation_gate', default='x', type=str,
                        help='Contorl Gate applied before measurement'
                             '[cx]'
                             '[ch]'
                             '[crx]'
                             '[cry]'
                             '[crz]',
                        choices=['x', 'h', 'rx', 'ry', 'rz'])

    config = parser.parse_args()

    env = QuantumCircuitDesignEnv(config)

    env_no_measurement = QCDesignNoMeasurementEnv(config)

    # check action space

    print('action_num is {}'.format(len(env.action_set)))

    for a in env.action_set:
        print('action', a)
        ma = env.a2ma(a, env.action_num, env.m_step)
        for i in range(config.measurement_step):
            print('multi_step = {}'.format(i))
            print(env.a2unitary[ma[i]][1:-config.qubit_c])
        env.reset()
        env.run_circuit_one_step(a)
        print(env.qc)


    # check completeness

    I = get_tensor_product([get_gate_matrix('i') for _ in range(env.bn_c)])

    for a in env.action_set:
        sum = np.zeros((env.state_dim, env.state_dim)).astype(np.complex128)
        for o in env.observation_set:
            sum += np.dot(env.A[a][o].conj().T, env.A[a][o])
        if np.sum(np.abs(I - sum)**2) > 1e-7:
            print('error')
            sys.exit()
    
    print('completeness is OK.')

    # check environment
    for a in env.action_set:
        print('unitary')
        print(env_no_measurement.a2unitary[a])
        print(env.a2unitary[a][1:-env.bn_c])

    # check whether measurement affects the state of system

    for a in env.action_set:

        # only for 2 observations
        if np.sum(np.abs(env.A[a][0] - env.A[a][1])) < 1e-7:
            # print('not same operator')
            print('same operator')
            sys.exit()
        else:
            for _ in range(10000):
                s = np.random.uniform(-1.0, 1.0, (2**env.bn_c, 1)) + 1j * np.random.uniform(-1.0, 1.0, (2**env.bn_c, 1))
                s = s / np.sqrt(np.sum(np.abs(s)**2))

                p_0 = np.dot(np.dot(env.A[a][0], s).conj().T, np.dot(env.A[a][0], s)).real[0][0]
                p_1 = np.dot(np.dot(env.A[a][1], s).conj().T, np.dot(env.A[a][1], s)).real[0][0]

                if np.abs(p_0 - p_1) < 1e-7:
                    # print('p_0 is different from p_1')
                    print('p_0 is same as p_1')
                    sys.exit()
                
                print('probability')
                print(p_0)
                print(state_update(s, a, 0, env.A))
                print(p_1)
                print(state_update(s, a, 1, env.A))

                if np.sum(np.abs(env_no_measurement.A[a] - env.A[a][0] / np.sqrt(p_0))) < 1e-7:
                    # print('not same as unitary')
                    print('same as unitary')
                    sys.exit()
                
                if np.sum(np.abs(env_no_measurement.A[a] - env.A[a][1] / np.sqrt(p_1))) < 1e-7:
                    # print('not same as unitary')
                    print('same as unitary')
                    sys.exit()


    # check inference by model

    np.random.seed(config.seed)
    s_circuit_multi, a_history_multi, o_history_multi, step_multi = run_circuit(env, config)


    s_model = update_state(env.state_dim, int(step_multi/config.measurement_step), a_history_multi, o_history_multi, env.A)

    print('circuit - model in sequantial')
    print(np.sum(np.abs(s_circuit_multi - s_model)))
