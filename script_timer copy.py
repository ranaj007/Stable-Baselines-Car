from car import CarAgent
from track import Track

import cProfile
import pstats

track = Track(do_render=True, show_fps=True)
img = track.new_frame()

env = CarAgent(
    track=track,
    do_render=True,
    training=False
)
env.reset()

img = track.new_frame()
env.new_frame(img)
env.step(0)
track.render()

with cProfile.Profile() as pr:
    for _ in range(60*10):
        img = track.new_frame()
        env.new_frame(img)
        env.step(0)
        track.render()

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats("profile_stats.prof")