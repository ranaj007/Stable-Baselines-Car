from Stable_env_no_graphics import CarEnv

import cProfile
import pstats

env = CarEnv()
env.reset()

with cProfile.Profile() as pr:
    env.step(0)

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()