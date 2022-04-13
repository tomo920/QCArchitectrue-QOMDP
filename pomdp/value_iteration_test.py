import argparse
import numpy as np
import matplotlib.pyplot as plt

from pomdp_envs.tiger import TigerEnv

if __name__ == '__main__':
    test_steps = 3

    parser = argparse.ArgumentParser(description='Set parameters.')

    parser.add_argument('pomdp_algorithm', type=str,
                        help='POMDP algorithm to test'
                             '[value iteration]',
                        choices=['value_iteration'])
    parser.add_argument('test_part', type=str,
                        help='Test part'
                             '[belief_state]'
                             '[value_function]',
                        choices=['belief_state', 'value_function'])
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

    if config.test_part == 'belief_state':
        # belief state list for step t
        belief_list_t = []
        # initialize belief state
        agent.initialize()
        belief_list_t.append(agent.belief_state)

        # test all combinations
        for t in range(test_steps):
            print('---------------------------------------------------------')
            print('step {0}: initial belief {1}'.format(t, belief_list_t))

            # belief state list for step t+1
            belief_list_t_ = []
            for belief_t in belief_list_t:
                for a in env.action_set:
                    for o in env.observation_set:
                        # belief update by action a and observation
                        belief_t_ = agent.update_belief_state(belief_t, a, o)

                        # added updated belief state if the belief is new
                        is_append = True
                        for b_t_ in belief_list_t_:
                            if np.array_equal(b_t_, belief_t_):
                                is_append = False
                                break
                        if is_append:
                            belief_list_t_.append(belief_t_)

                        print('belief state: {0} action: {1} observation: {2} -> next_belief_state: {3}'.format(belief_t, a, o, belief_t_))

            # update belief state list
            belief_list_t = belief_list_t_
    elif config.test_part == 'value_function':
        # test planning part
        from pomdp_solvers.value_iteration import update_value_function

        nu_list = []
        nu = agent.initialize_value_function()
        nu_list.append(nu)

        print('------------------planning---------------------')

        for _ in range(agent.horizon):
            nu = update_value_function(nu, agent.p_t, agent.p_o, agent.r_sa, agent.gamma,
                                       agent.state_set, agent.action_set, agent.observation_set)
            nu_list.append(nu)

        # plot
        for i, nu in enumerate(nu_list):
            print('horizon = {}'.format(i))
            fig = plt.figure(figsize=(10,8),dpi=200)
            ax = fig.add_subplot(111)
            l_0 = True
            l_1 = True
            l_2 = True
            for alpha in nu:
                if alpha['a'] == 0:
                    color = "green"
                    if l_0:
                        ax.plot([0., 1.], [alpha['v'][1], alpha['v'][0]], color = color, label = 'a = a{}'.format(alpha['a']))
                        l_0 = False
                elif alpha['a'] == 1:
                    color = "blue"
                    if l_1:
                        ax.plot([0., 1.], [alpha['v'][1], alpha['v'][0]], color = color, label = 'a = a{}'.format(alpha['a']))
                        l_1 = False
                elif alpha['a'] == 2:
                    color = "red"
                    if l_2:
                        ax.plot([0., 1.], [alpha['v'][1], alpha['v'][0]], color = color, label = 'a = a{}'.format(alpha['a']))
                        l_2 = False
                else:
                    # initial alpha vector
                    color = "black"
                ax.plot([0., 1.], [alpha['v'][1], alpha['v'][0]], color = color)
                print('V(b|a{0}) = {1}*p0 + {2}*p1'.format(alpha['a'], alpha['v'][0], alpha['v'][1]))
            ax.grid(axis='both')
            plt.xlim(0.,1.)
            ax.set_xlabel("p0", fontsize=20)
            ax.set_ylabel("V(b|a)", fontsize=20)
            ax.legend(fontsize=30)
            plt.savefig('t_{}.jpg'.format(i), bbox_inches='tight', pad_inches=0)
