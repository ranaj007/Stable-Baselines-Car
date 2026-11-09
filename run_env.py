from stable_baselines3 import PPO
from car import CarAgent
from track import Track
from glob import glob
import os

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
        do_render=True,
        render_text=render_text,
        speed_reward=True,
        number_of_collisions=3,
    )


if __name__ == "__main__":
    track = Track(do_render=True, show_fps=True)

    # Live tuning (main car only): a / d / r select acceleration, deceleration or rotation; + / - change it
    env = make_car(track, render_text=True, color=MAIN_COLOR)
    reference_env = make_car(track, render_text=False, color=REFERENCE_COLOR)

    model = PPO.load(find_model(MAIN_RUN, MAIN_STEP), env, device="cpu")
    reference_model = PPO.load(find_model(REFERENCE_RUN, REFERENCE_STEP), reference_env, device="cpu")

    while True:
        obs = env.reset()
        reference_obs = reference_env.reset()
        done = False

        while not done:
            track.new_frame()

            # Main car takes its action
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)

            # Reference car takes its action
            reference_action, _ = reference_model.predict(reference_obs)
            reference_obs, _, reference_done, _ = reference_env.step(reference_action)

            # Reset the reference car if it crashes
            if reference_done:
                reference_obs = reference_env.reset()

            env.handle_key(track.render())
