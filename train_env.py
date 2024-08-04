from stable_baselines3 import PPO
import os
from Stable_env_no_graphics import CarEnv

models_dir = "models/PPO/Speed_Reward"
logdir = "logs"
#model_path = models_dir + "/160000.zip"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = CarEnv(do_render=True, speed_reward=True)
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

#model = PPO.load(model_path, env)

TIMESTEPS = 10000

i = 1
while True:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="Speed_Reward")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    i += 1