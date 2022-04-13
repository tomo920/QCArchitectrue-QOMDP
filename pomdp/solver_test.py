import argparse
import numpy as np

from pomdp_envs.tiger import TigerEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set parameters.')

    parser.add_argument('pomdp_algorithm', type=str,
                        help='POMDP algorithm to optimize Quantum Circuit'
                             '[value iteration]',
                        choices=['value_iteration'])
    parser.add_argument('max_step', type=int, help='Max number of steps in one episode')
    parser.add_argument('--seed', default=123, type=int, help='Random seed')
    parser.add_argument('--episode_num', default=1000, type=int, help='Number of episodes in one epoch')
    parser.add_argument('--gamma',  default=0.95, type=float, help='Discount factor')

    # for value iteration
    parser.add_argument('--horizon', default=10, type=int, help='Planning horizon in value iteration')
    parser.add_argument('--init_belief_type', default='uniform', type=str,
                        help='Initial belief state'
                             '[uniform]',
                        choices=['uniform'])
    parser.add_argument('--init_value_function_type', default='zero', type=str,
                        help='Initial value function'
                             '[zero]',
                        choices=['zero'])


    config = parser.parse_args()

    # seed
    np.random.seed(config.seed)

    # Tiger Environment
    env = TigerEnv(config)

    # POMDP Agent
    from pomdp_agents.vi_agent import VIAgent
    agent = VIAgent(config, env)


    # run algorithm
    for i in range(config.horizon):
        print('Horizon = {}'.format(i))
        result_list = []

        # value backup
        agent.learn()

        # execute episode
        for i in range(config.episode_num):
          env.reset()
          agent.initialize()

          while True:
               pi = agent.policy()
               action = np.random.choice(env.action_set, p = pi)
               next_observation, reward, done = env.step(action)
               agent.update(action, next_observation)
               if done:
                    print('Episode ends. Total reward = {}.'.format(env.total_reward))
                    result_list.append(env.total_reward)
                    break

        print('All episodes end. Average reward = {}.'.format(np.average(result_list)))
