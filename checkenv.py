from stable_baselines3.common.env_checker import check_env
from Stable_env_no_graphics import CarEnv

env = CarEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)