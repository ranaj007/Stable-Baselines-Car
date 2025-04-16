from stable_baselines3 import PPO
from car import CarAgent
from track import Track
from glob import glob

run_name = "accel-0.11_decel-0.97_rotation-5_view-400_col-3_2"
models_dir = "models/PPO/" + run_name

models = glob(f"{models_dir}/*.zip")

if models:
    models.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    model_path = models[-1]

track = Track(do_render=True, show_fps=True)

envs = [CarAgent(track=track, do_render=True, training=False) for _ in range(3)]

models = [PPO.load(model_path, env, device="cpu") for env in envs]

obs_array = [env.reset() for env in envs]

while True:
    track.new_frame()

    for i, env in enumerate(envs):
        action, _ = models[i].predict(obs_array[i])
        obs, reward, done, info = env.step(action)
        obs_array[i] = obs
        if done:
            obs_array[i] = env.reset()

    track.render()