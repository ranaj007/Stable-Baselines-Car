from Stable_env_no_graphics import CarEnv
from stable_baselines3 import PPO
from glob import glob

models_dir = "models/PPO/Speed_Reward_5"
model_path = models_dir + "/1070000.zip"
# first attempt - best PPO 2950000.zip

run_name = "accel-0.1_decel-0.99"
models_dir = "models/PPO/" + run_name
model_path = ''

models = glob(f"{models_dir}/*.zip")

if models:
    models.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    model_path = models[-1]


env = CarEnv(training=False, draw_lines=True)

model = PPO.load(model_path, env)

while True:
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)