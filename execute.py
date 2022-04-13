import numpy as np
import os

def execute(config,
            env,
            agent,
            save_path):
    '''
    Execute learning or test
    '''

    for i in range(config.episode_num):
        env.reset()
        agent.initialize()

        while True:
            pi = agent.policy()
            action = np.random.choice(env.action_set, p = pi)
            next_observation, reward, done = env.step(action)
            agent.update(action, next_observation)
            if done:
                print('Episode ends. Steps to goal = {0}. Reward = {1}'.format(env.steps, reward))
                episode_save_path = os.path.join(save_path, "episode_{}".format(i))
                os.makedirs(episode_save_path)
                env.save_result(episode_save_path)
                break
