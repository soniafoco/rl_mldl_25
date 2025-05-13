"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env


def main():

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    params = [0.1, 0.3, 0.5, 0.7, 1.0]

    for param in params: 
        print(f"\n==== Training with param = {param} ====")
        train_env = gym.make('CustomHopper-source-v0', param=param)

        model = SAC("MlpPolicy", train_env, verbose=0)
        model.learn(total_timesteps=100_000)

        model_path = f"sac_hopper_param_{param:.2f}"
        model.save(model_path)
        
        model = SAC("MlpPolicy", train_env, verbose=1)

        model.learn(total_timesteps=100000)

        model.save("sac_hopper")

        #If we want to use deletion and reloading
        """
            del model
            model = SAC.load("sac_hopper")
        """


        #EVALUATION
        obs = train_env.reset()

        cumulative_reward = 0
        i = 0
            
        while i < 10:

            action, _states = model.predict(obs)
            obs, reward, done, info = train_env.step(action)
            cumulative_reward += reward
            #train_env.render()
            if done: 
                i += 1
                print(f"Cumulative reward of episode {i}: {cumulative_reward}")
                cumulative_reward = 0
                obs = train_env.reset()
            
            

if __name__ == '__main__':
    main()


   