from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from car import CarAgent, ChaserAgent
from track import Track
from glob import glob
from math import ceil
import os


if __name__ == "__main__":
    run_name = "chaser_4"
    models_dir = "models/PPO/" + run_name
    logdir = "logs/" + run_name
    model_path = ""

    base_model = "accel-0.11_decel-0.97_rotation-5_view-400_col-3_2"

    models = glob(f"{models_dir}/*.zip")

    if models:
        models.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
        model_path = models[-1]
        print("Previous model found")
    else:
        print("No previous models found")
    # model_path = models_dir + "/160000.zip"

    models_dir_base = "models/PPO/" + base_model

    models = glob(f"{models_dir_base}/*.zip")

    if models:
        models.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
        model_path_base = models[-1]

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    track = Track()

    def make_env():
        target = CarAgent(track=track, do_render=False, draw_lines=False)
        env = ChaserAgent(
            track=track,
            target=target,
            do_render=False,
            training=True,
            action_limit=2000,
            number_of_collisions=3,
        )
        env.reset()
        return env

    n_timesteps = 100000
    n_steps = 2048
    n_envs = 12

    n_steps_p_batch = ceil(n_timesteps / (n_steps * n_envs)) * n_steps * n_envs

    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    
    i = 1

    if model_path:
        print("Loading", model_path)
        model = PPO.load(model_path, env, device="cpu")
        # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cpu")
        # model.set_parameters(model_path)

        i = int(model_path.split("\\")[-1].split(".")[0]) // n_timesteps + 1
    else:
        print("Creating new model")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cpu")

        if model_path_base:
            print("Loading base model", model_path_base)
            model.set_parameters(model_path_base)

    while True:
        model.learn(
            total_timesteps=n_timesteps,
            reset_num_timesteps=False,
            tb_log_name="Speed_Reward",
            progress_bar=True,
        )
        print("Saving model")
        model.save(f"{models_dir}/{n_steps_p_batch * i}")
        i += 1
