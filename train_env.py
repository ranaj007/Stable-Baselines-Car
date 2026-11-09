from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from Stable_env_no_graphics import CarEnv
from stable_baselines3 import PPO
from glob import glob
import os


if __name__ == "__main__":
    # Changing the reward makes this a different experiment: use a new run_name rather than
    # resuming a model (and value function) trained on the old reward.
    run_name = "accel-0.11_decel-0.97_rotation-5_view-400_col-3_2_2026_reward_scaled"

    REWARD_SCALE = 0.01  # rewards of about -10..+107 per step become about -0.1..+1.07
    CRASH_PENALTY = 100  # a crash step's reward becomes -100 before scaling (-1 after); was -10
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
        env = CarEnv(
            do_render=False,
            speed_reward=True,
            action_limit=1000,
            number_of_collisions=3,
            reward_scale=REWARD_SCALE,
            crash_penalty=CRASH_PENALTY,
        )
        env.reset()
        return env

    env = make_vec_env(make_env, n_envs=16, vec_env_cls=SubprocVecEnv)

    TIMESTEPS = 100000
    BATCH_SIZE = 64  # SB3 default is 64; larger batches make each PPO update much faster but risks finding local optima
    i = 1

    if model_path:
        print("Loading", model_path)
        model = PPO.load(model_path, env, device="cpu", batch_size=BATCH_SIZE)
        #model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cpu")
        #model.set_parameters(model_path)

    else:
        print("Creating new model")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cpu", batch_size=BATCH_SIZE)

    while True:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="Speed_Reward", progress_bar=True)
        print("Saving model")
        model.save(f"{models_dir}/{model.num_timesteps}")