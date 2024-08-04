from stable_baselines3 import PPO
import os
from Stable_env_no_graphics import CarEnv

models_dir = "models/PPO/Default_Parameters"
model_path = models_dir + "/2950000.zip"

# first attempt - best PPO 2950000.zip

env = CarEnv()

model = PPO.load(model_path, env)

while True:
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)