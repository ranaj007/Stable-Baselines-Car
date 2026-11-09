from stable_baselines3 import PPO
from car import CarAgent
from track import Track
from glob import glob

models_dir = "models/PPO/Speed_Reward_5"
model_path = models_dir + "/1070000.zip"
# first attempt - best PPO 2950000.zip

run_name = "accel-0.11_decel-0.97_rotation-5_view-400_col-3_2_2026_reward_scaled"
models_dir = "models/PPO/" + run_name
model_path = ''

models = glob(f"{models_dir}/*.zip")

if models:
    models.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    model_path = models[-1]

model_path = f"{models_dir}/2621440.zip"

track = Track(do_render=True, show_fps=True)

# Live tuning: a / d / r select acceleration, deceleration or rotation; + / - change it
env = CarAgent(
    track=track,
    training=False,
    draw_lines=False,
    do_render=True,
    render_text=True,
    speed_reward=True,
    number_of_collisions=3,
)

model = PPO.load(model_path, env, device="cpu")

while True:
    obs = env.reset()
    done = False
    while not done:
        track.new_frame()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)  # the car draws itself onto the frame
        env.handle_key(track.render())
