from car import CarAgent, ChaserAgent
from stable_baselines3 import PPO
from track import Track
from glob import glob

track = Track(do_render=True, show_fps=True)

target_env = CarAgent(track=track, do_render=True, draw_lines=False)

run_name = "chaser_4"
models_dir = "models/PPO/" + run_name

models = glob(f"{models_dir}/*.zip")

if models:
    models.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    model_path = models[-1]

model_path = f"{models_dir}/50872320"

env = ChaserAgent(
    track=track, target=target_env, do_render=True
)

model = PPO.load(model_path, env, device="cpu")
obs = env.reset()

while True:
    track.new_frame()

    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

    track.render()