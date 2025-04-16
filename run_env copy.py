from stable_baselines3 import PPO
from car import CarAgent
from track import Track
from glob import glob

models_dir = "models/PPO/Speed_Reward_5"
model_path = models_dir + "/1070000.zip"
# first attempt - best PPO 2950000.zip

run_name = "accel-0.11_decel-0.97_rotation-5_view-400_col-3_2"
models_dir = "models/PPO/" + run_name
model_path = ''

models = glob(f"{models_dir}/*.zip")

if models:
    models.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    model_path = models[-1]

track = Track(do_render=True, show_fps=True)

env = CarAgent(
    track=track,
    do_render=True,
    training=False
)

model = PPO.load(model_path, env, device="cpu")

env2 = CarAgent(
    track=track,
    do_render=True,
    training=False
)

model2 = PPO.load(model_path, env2, device="cpu")

while True:
    obs = env.reset()
    obs2 = env2.reset()
    action = 0
    done = False
    while not done:
        img = track.new_frame()
        env.new_frame(img)
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        env2.new_frame(img)
        action2, _ = model2.predict(obs2)
        obs2, reward2, done2, info2 = env2.step(action2)

        track.render()