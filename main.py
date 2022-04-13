import argparse
import numpy as np
import os
import pickle


if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Set parameters.')

     parser.add_argument('pomdp_algorithm', type=str,
                         help='POMDP algorithm to optimize Quantum Circuit'
                              '[value iteration]'
                              '[point-based value iteration]',
                         choices=['value_iteration', 'point_based_value_iteration'])
     parser.add_argument('task', type=str,
                         help='Task to solve by pomdp'
                              '[state_preparation]'
                              '[vqe]',
                         choices=['state_preparation', 'vqe'])
     parser.add_argument('--target_state', type=str,
                         help='Target state for circuit design'
                              '[one]'
                              '[random]'
                              '[ghz]',
                         choices=['one', 'random', 'load', 'bell', 'ghz'])
     parser.add_argument('--molecule', type=str,
                         help='Molecule for vqe'
                              '[h2]'
                              '[n2]'
                              '[li2]'
                              '[hhe]'
                              '[h2o]'
                              '[lih]'
                              '[beh2]',
                         choices=['h2', 'n2', 'li2', 'hhe', 'h2o', 'lih', 'beh2'])
     parser.add_argument('--bond_length', type=float, help='Length between atoms')
     parser.add_argument('max_step', type=int, help='Max number of steps in one episode')
     parser.add_argument('save_dir', type=str, help='Name of directory to save result')
     parser.add_argument('--seed', default=123, type=int, help='Random seed')
     parser.add_argument('--episode_num', default=1000, type=int, help='Number of episodes in one epoch')
     parser.add_argument('--gamma',  default=0.95, type=float, help='Discount factor')
     parser.add_argument('--fidelity_threshold',  default=0.99, type=float, help='Episode ends when obtained this fidelity')
     parser.add_argument('--expval_threshold',  type=float, help='Episode ends when obtained this expectation value')
     parser.add_argument('--is_action_discrete', action='store_true')
     parser.add_argument('--only_final_reward', action='store_true')

     # for Quantum Circuit
     parser.add_argument('--qubit_c', default=3, type=int, help='Number of oscillator qubits')
     parser.add_argument('--qubit_o', default=3, type=int, help='Number of ancilla qubits')
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


     # for value iteration
     parser.add_argument('--horizon', default=10, type=int, help='Planning horizon in value iteration')
     parser.add_argument('--iteration_num', default=10, type=int, help='Number of iterations learn method is executed')
     parser.add_argument('--init_belief_type', default='uniform', type=str,
                         help='Initial belief state'
                              '[one]'
                              '[uniform]'
                              '[expand]'
                              '[explore]',
                         choices=['one', 'uniform', 'expand', 'explore'])
     parser.add_argument('--init_point_set_num', default=10, type=int, help='Number of belief points of initialize belief points set')
     parser.add_argument('--init_value_function_type', default='zero', type=str,
                         help='Initial value function'
                              '[zero]',
                         choices=['zero'])
     parser.add_argument('--belief_choice_metric', default='l2norm', type=str,
                         help='Metric used in belief point choice'
                              '[l1norm]'
                              '[l2norm]'
                              '[fubini-study]'
                              '[fidelity]'
                              '[bures]',
                         choices=['l1norm', 'l2norm', 'fubini-study', 'fidelity', 'bures'])


     config = parser.parse_args()

     v_config = vars(config)
     config_text = ''
     for k in v_config:
          config_text += '{0}: {1}\n'.format(k, v_config[k])

     # seed
     np.random.seed(config.seed)

     # result save directory
     save_path = os.path.abspath("./result/{}".format(config.save_dir))
     if not os.path.exists(save_path):
          os.makedirs(save_path)
          with open('{}/config.pkl'.format(save_path), mode='wb') as f:
               pickle.dump(config, f)

     # Quantum circuit design Environment
     if config.task == 'state_preparation':
          from task.state_preparation import StatePreparation
          env = StatePreparation(config)
     elif config.task == 'vqe':
          from task.vqe import VQE
          env = VQE(config)

     env.save_model(save_path)

     # POMDP Agent
     if config.pomdp_algorithm == 'value_iteration':
          from pomdp.pomdp_agents.qvi_agent import QVIAgent
          from run_vi import run_value_iteration
          agent = QVIAgent(config, env)
          run = run_value_iteration
     elif config.pomdp_algorithm == 'point_based_value_iteration':
          from pomdp.pomdp_agents.qpbvi_agent import QPBVIAgent
          from run_vi import run_value_iteration
          agent = QPBVIAgent(config, env)
          run = run_value_iteration

     run(config,
          env,
          agent,
          save_path)
