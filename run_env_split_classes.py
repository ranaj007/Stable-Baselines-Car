from car import CarAgent, ChaserAgent
from stable_baselines3 import PPO
from pathlib import Path
from track import Track
from glob import glob

run_name = "accel-0.11_decel-0.97_rotation-5_view-400_col-3_2"
models_dir = "models/PPO/" + run_name

models = glob(f"{models_dir}/*.zip")

if models:
    models.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    model_path = models[-1]

track = Track(do_render=True, show_fps=True)

envs = [CarAgent(track=track, do_render=True, draw_lines=False) for _ in range(1)]

#models = [PPO.load(model_path, env, device="cpu") for env in envs]

#obs_array = [env.reset() for env in envs]

run_name = "chaser_2"
models_dir = "models/PPO/" + run_name

models = glob(f"{models_dir}/*.zip")

if models:
    models.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    model_path = models[-1]

model_path = f"{models_dir}/12400000"

env2 = ChaserAgent(
    track=track, target=envs[0], do_render=True
)

model2 = PPO.load(model_path, env2, device="cpu")
obs2 = env2.reset()

while True:
    track.new_frame()

    #for i, env in enumerate(envs):
        #action, _ = models[i].predict(obs_array[i])
        #obs, reward, done, info = env.step(action)
        #obs_array[i] = obs
        #if done:
            #obs_array[i] = env.reset()
    
    action, _ = model2.predict(obs2)
    obs2, reward, done, info = env2.step(action)
    if done:
        obs2 = env2.reset()

    track.render()