import os
import random
from glob import glob

from stable_baselines3 import PPO

from car import CarAgent
from track import Track

MODELS_ROOT = "models/PPO"

# main car: the model being trained (step=None picks the latest checkpoint)
MAIN_RUN = "accel-0.11_decel-0.97_rotation-5_view-400_col-3_2_2026_reward_scaled"
MAIN_STEP = 2621440

# reference car: an older model to compare against
REFERENCE_RUN = "accel-0.11_decel-0.97_rotation-5_view-400_col-3_2"
REFERENCE_STEP = None

# car body colours, as (B, G, R) for OpenCV
MAIN_COLOR = (0, 0, 255)  # red
REFERENCE_COLOR = (255, 0, 0)  # blue

RENDER = True # whether to render the race (render is capped at 60 fps)
LAPS = 10  # laps to win a race

def find_model(run_name: str, step: int | None = None) -> str:
    """Path to a run's checkpoint at `step`, or its latest checkpoint if step is None."""
    models_dir = f"{MODELS_ROOT}/{run_name}"

    if step is not None:
        path = f"{models_dir}/{step}.zip"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    models = glob(f"{models_dir}/*.zip")
    if not models:
        raise FileNotFoundError(f"no models in {models_dir}")
    return max(models, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def make_car(track: Track, render_text: bool, color: tuple[int, int, int]) -> CarAgent:
    return CarAgent(
        track=track,
        color=color,
        training=False,
        draw_lines=False,
        do_render=RENDER,
        render_text=render_text,
        speed_reward=True,
        number_of_collisions=3,
    )


if __name__ == "__main__":
    track = Track(do_render=RENDER, show_fps=True)

    # Live tuning (main car only): a / d / r select acceleration, deceleration or rotation; + / - change it
    env = make_car(track, render_text=True, color=MAIN_COLOR)
    reference_env = make_car(track, render_text=False, color=REFERENCE_COLOR)

    model = PPO.load(find_model(MAIN_RUN, MAIN_STEP), env, device="cpu")
    reference_model = PPO.load(find_model(REFERENCE_RUN, REFERENCE_STEP), reference_env, device="cpu")

    main_crashes = reference_crashes = 0
    main_wins = reference_wins = 0
    races = 0

    while True:
        # Both cars start from the same state: reset() seeds `random` before choosing the start
        seed = random.randrange(2**32)
        obs = env.reset(seed=seed)
        reference_obs = reference_env.reset(seed=seed)
        done = reference_done = False
        race_over = False

        # The race restarts when either car crashes or finishes LAPS laps. The finish line is
        # the first reward gate after the shared starting point.
        while not race_over:
            if RENDER:
                track.new_frame()

            # Main car takes its action
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)

            # Reference car takes its action
            reference_action, _ = reference_model.predict(reference_obs)
            reference_obs, _, reference_done, _ = reference_env.step(reference_action)

            # terminated = hit a wall or left the screen (not a time limit)
            main_crashes += env.terminated
            reference_crashes += reference_env.terminated

            # a car that crashed on this step can't win; if both finish on the same step, both win
            main_finished = env.laps_completed >= LAPS and not env.terminated
            reference_finished = reference_env.laps_completed >= LAPS and not reference_env.terminated
            main_wins += main_finished
            reference_wins += reference_finished

            race_over = done or reference_done or main_finished or reference_finished

            if not RENDER:
                continue

            track.draw_scoreboard(
                cars=[("New", MAIN_COLOR), ("Reference", REFERENCE_COLOR)],
                stats={
                    "Lap": [f"{min(e.laps_completed + 1, LAPS)}/{LAPS}" for e in (env, reference_env)],
                    "Wins": [main_wins, reference_wins],
                    "Crashes": [main_crashes, reference_crashes],
                },
            )
            env.handle_key(track.render())

        races += 1
        if not RENDER:
            print(
                f"Race {races}: wins {main_wins}-{reference_wins} (main-reference), "
                f"crashes {main_crashes}-{reference_crashes}"
            )
