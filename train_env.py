from stable_baselines3.common.env_util import make_vec_env
from Stable_env_no_graphics import CarEnv
from stable_baselines3 import PPO
from glob import glob
import os



run_name = "accel-0.11_decel-0.97_rotation-5"
models_dir = "models/PPO/" + run_name
logdir = "logs/" + run_name
model_path = ''

models = glob(f"{models_dir}/*.zip")

if models:
    models.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    model_path = models[-1]
    print("Previous model found")
else:
    print("No previous models found")
#model_path = models_dir + "/160000.zip"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

def make_env():
    env = CarEnv(do_render=False, speed_reward=True, action_limit=1000)
    env.reset()
    return env

env = make_vec_env(make_env, n_envs=8)

TIMESTEPS = 10000
i = 1

if model_path:
    print("Loading", model_path)
    #model = PPO.load(model_path, env)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cpu")
    model.set_parameters(model_path)

    i = int(model_path.split("\\")[-1].split(".")[0]) // TIMESTEPS + 1
else:
    print("Creating new model")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cpu")

while True:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="Speed_Reward", progress_bar=True)
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    i += 1