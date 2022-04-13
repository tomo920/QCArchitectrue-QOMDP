import numpy as np
import os
import time

from execute import execute

def run_value_iteration(config,
                        env,
                        agent,
                        save_path):
    '''
    Run value iteration algorithm
    '''

    for i in range(config.iteration_num):
        print('Iteration = {}'.format(i+1))

        result_path = os.path.join(save_path, "iteration_{}".format(i+1))
        os.makedirs(result_path)

        plan_start_time = time.time()

        # update agent
        agent.learn()

        planning_time = time.time() - plan_start_time
        
        # save data
        agent.save_model(result_path)
        np.save('{}/planning_time.npy'.format(result_path), planning_time)

        # execute circuit design
        execute(config,
                env,
                agent,
                result_path)
