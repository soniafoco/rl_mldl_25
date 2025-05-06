"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



def main():
    train_env = gym.make('CustomHopper-source-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #
    

    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_cartpole")

    model = PPO.load("ppo_cartpole")

    obs = train_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = train_env.step(action)
        train_env.render("human")

if __name__ == '__main__':
    main()