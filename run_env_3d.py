"""3D view of a race between two trained models (prototype).

The race works the same as in run_env.py: both cars start from the same state, a crash
restarts both cars, and the first car to finish LAPS laps wins. The simulation is still the
2D CarAgent; this file only draws its state in 3D using Ursina (pip install ursina).

Models, colours and LAPS are taken from run_env.py, so both views race the same cars.

Controls:
    c           cycle the camera: chase main car, chase reference car, overhead
    a / d / r   pick acceleration, deceleration or rotation for live tuning (main car)
    h / p       pick the chase camera's heading lag or position lag
    + / -       change the picked setting
    escape      quit
"""

import random

import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from ursina import (
    AmbientLight,
    DirectionalLight,
    Entity,
    Mesh,
    Sky,
    Text,
    Texture,
    Ursina,
    Vec2,
    Vec3,
    application,
    camera,
    color,
    lerp,
    window,
)
from ursina import time as ursina_time

from car import CarAgent
from run_env import (
    LAPS,
    MAIN_COLOR,
    MAIN_RUN,
    MAIN_STEP,
    REFERENCE_COLOR,
    REFERENCE_RUN,
    REFERENCE_STEP,
    find_model,
)
from track import Track

PIXEL_SCALE = 0.1  # world units per track pixel: the 1200 x 1000 track becomes 120 x 100
WALL_HEIGHT = 1.0
SIM_HZ = 60  # simulation steps per second, matching run_env.py's 60 FPS cap
CAMERA_MODES = ("Main", "Reference", "Overhead")

# chase camera placement, in world units relative to the car
CHASE_DISTANCE = 7
CHASE_HEIGHT = 3
CHASE_LOOK_AHEAD = 4
# Smoothing time constants in seconds (higher = smoother but laggier). The policy often flicks
# left/right every step (5 degrees each), so the camera follows a smoothed heading instead.
# Both can be changed while running (h / p, then + / -); these are the starting values.
HEADING_LAG = 0.05
POSITION_LAG = 0.1

# live-tuned camera settings: key -> (name, attribute on Race3D, step per + / - press)
CAMERA_TUNING = {
    "h": ("Heading lag", "heading_lag", 0.05),
    "p": ("Position lag", "position_lag", 0.01),
}
PLUS_KEYS = ("=", "+", "shift+=", "num_add", "num_plus")
MINUS_KEYS = ("-", "num_subtract", "num_minus")


def to_world(x: float, y: float, height: float = 0.0) -> Vec3:
    """Track pixel (x right, y down) to world position (x right, y up, z forward)."""
    return Vec3(x * PIXEL_SCALE, height, -y * PIXEL_SCALE)


def heading_to_rotation_y(car_rot: float) -> float:
    """CarAgent's Car_Rot (degrees, counter-clockwise on screen from +x) to Ursina's
    rotation_y (degrees, clockwise from above, 0 = +z)."""
    return 90 - car_rot


def forward_vector(car_rot: float) -> Vec3:
    rad = np.radians(car_rot)
    return Vec3(np.cos(rad), 0, np.sin(rad))


def bgr_to_color(bgr: tuple[int, int, int]):
    b, g, r = bgr
    return color.rgb32(r, g, b)


def lerp_angle(a: float, b: float, t: float) -> float:
    """Interpolate between two angles in degrees the short way round."""
    diff = (b - a + 180) % 360 - 180
    return a + diff * t


def smoothing_factor(dt: float, lag: float) -> float:
    """Fraction of the way to move towards a target this frame, for exponential smoothing
    that behaves the same at any frame rate. A lag of 0 means no smoothing."""
    if lag <= 0:
        return 1
    return 1 - np.exp(-dt / lag)


def wall_mesh(points: np.ndarray) -> Mesh:
    """A closed ring of wall quads along a border polyline, alternating red and white
    like kerbs so speed and distance are easy to judge."""
    vertices, triangles, colors = [], [], []
    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]
        base = len(vertices)
        vertices += [
            to_world(x0, y0, 0),
            to_world(x1, y1, 0),
            to_world(x1, y1, WALL_HEIGHT),
            to_world(x0, y0, WALL_HEIGHT),
        ]
        triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        segment_color = color.rgb32(200, 30, 30) if i % 2 else color.rgb32(235, 235, 235)
        colors += [segment_color] * 4
    return Mesh(vertices=vertices, triangles=triangles, colors=colors)


def ground_texture(track: Track) -> Texture:
    """The 2D track image (without cars) as a texture for the ground plane."""
    bgr = track.new_frame().copy()
    rgb = bgr[:, :, ::-1]
    # explicit RGBA: an RGB image comes through Ursina as fully transparent
    return Texture(Image.fromarray(np.ascontiguousarray(rgb)).convert("RGBA"))


class Car3D:
    """A CarAgent, its model, and the box that draws it."""

    def __init__(self, track: Track, name: str, run: str, step: int | None, bgr: tuple[int, int, int]):
        self.name = name
        # predict() doesn't need an env, so load the model on its own and read the collision
        # count from its observation size (5 + 8 view lines x collisions) to build a matching car
        self.model = PPO.load(find_model(run, step), device="cpu")
        number_of_collisions = (self.model.observation_space.shape[0] - 5) // 8
        self.env = CarAgent(
            track=track,
            training=False,
            do_render=False,  # the 3D view draws the car; skip the 2D drawing
            speed_reward=True,
            number_of_collisions=number_of_collisions,
        )
        self.obs = None
        self.done = False
        self.crashes = 0
        self.wins = 0

        # body, cabin and windscreen; 1 unit long like the 10-pixel 2D car
        self.entity = Entity(model="cube", color=bgr_to_color(bgr), scale=(0.55, 0.3, 1.0))
        Entity(parent=self.entity, model="cube", color=color.rgb32(30, 30, 30),
               scale=(0.8, 0.9, 0.45), position=(0, 0.75, -0.1))
        Entity(parent=self.entity, model="cube", color=color.rgb32(80, 140, 230),
               scale=(0.75, 0.6, 0.05), position=(0, 0.75, 0.13))

        self.prev_pos = self.pos = np.zeros(2)
        self.prev_rot = self.rot = 0.0
        self.draw_rot = 0.0  # heading as drawn this frame (between sim steps)

    def reset(self, seed: int):
        self.obs = self.env.reset(seed=seed)
        self.done = False
        self._remember_state(snap=True)

    def step(self):
        self._remember_state()
        action, _ = self.model.predict(self.obs)
        self.obs, _, self.done, _ = self.env.step(action)
        self.pos, self.rot = self.env.Car_Pos.copy(), float(self.env.Car_Rot)

    def _remember_state(self, snap: bool = False):
        if snap:
            self.pos, self.rot = self.env.Car_Pos.copy(), float(self.env.Car_Rot)
        self.prev_pos, self.prev_rot = self.pos, self.rot

    def draw(self, t: float):
        """Place the box between the last two simulation states (t = 0..1) for smooth motion."""
        x, y = self.prev_pos + (self.pos - self.prev_pos) * t
        self.draw_rot = lerp_angle(self.prev_rot, self.rot, t)
        self.entity.position = to_world(x, y, 0.15)
        self.entity.rotation_y = heading_to_rotation_y(self.draw_rot)

    @property
    def finished(self) -> bool:
        return self.env.laps_completed >= LAPS and not self.env.terminated


class Race3D(Entity):
    """Runs the race at SIM_HZ and draws it every frame. Ursina calls update() and input()."""

    def __init__(self):
        super().__init__()
        self.track = Track()

        width, height = self.track.SCREEN_WIDTH * PIXEL_SCALE, self.track.SCREEN_HEIGHT * PIXEL_SCALE
        Entity(model="plane", texture=ground_texture(self.track), scale=(width, 1, height),
               position=(width / 2, 0, -height / 2))
        Entity(model=wall_mesh(self.track.inner_lines), double_sided=True)
        Entity(model=wall_mesh(self.track.outer_lines), double_sided=True)
        self.finish_line = Entity(model="cube", color=color.white, scale=(0.1, 0.02, 1))

        self.cars = [
            Car3D(self.track, "Main", MAIN_RUN, MAIN_STEP, MAIN_COLOR),
            Car3D(self.track, "Reference", REFERENCE_RUN, REFERENCE_STEP, REFERENCE_COLOR),
        ]

        # black reads over the road, the red and white walls, and the sky
        self.scoreboard = Text(font="VeraMono.ttf", origin=(0.5, -0.5), scale=1.1, color=color.black,
                               position=window.bottom_right + Vec2(-0.03, 0.03))
        self.status = Text(font="VeraMono.ttf", origin=(-0.5, 0.5), scale=0.9, color=color.black,
                           position=window.top_left + Vec2(0.03, -0.03))

        self.camera_mode = 0
        self.camera_heading = 0.0  # smoothed heading the chase camera sits behind
        self.heading_lag = HEADING_LAG
        self.position_lag = POSITION_LAG
        self.camera_tuning = None  # key of the camera setting + / - change, or None for the car's
        self.snap_camera = True  # jump straight into place instead of swinging across the track
        self.sim_time = 0.0
        self.new_race()

    def new_race(self):
        # both cars start from the same state: reset() seeds `random` before choosing the start
        seed = random.randrange(2**32)
        for car in self.cars:
            car.reset(seed)
        self.snap_camera = True  # the cars have jumped to a new start

        # the finish line is the first gate after the start (see CarAgent.laps_completed)
        (x0, y0), (x1, y1) = self.cars[0].env.next_gate
        a, b = to_world(x0, y0, 0.02), to_world(x1, y1, 0.02)
        self.finish_line.position = (a + b) / 2
        self.finish_line.scale_x = max((b - a).length(), 0.1)
        self.finish_line.look_at(b, up=Vec3(0, 1, 0))
        self.finish_line.rotation_y -= 90  # look_at points +z at b; the line runs along x

    def sim_step(self):
        for car in self.cars:
            car.step()

        for car in self.cars:
            car.crashes += car.env.terminated
            car.wins += car.finished

        if any(car.done or car.finished for car in self.cars):
            self.new_race()

    def update(self):
        # fixed-rate simulation, so the race plays out the same whatever the frame rate
        self.sim_time += min(ursina_time.dt, 0.25)
        while self.sim_time >= 1 / SIM_HZ:
            self.sim_step()
            self.sim_time -= 1 / SIM_HZ

        t = self.sim_time * SIM_HZ
        for car in self.cars:
            car.draw(t)

        self.place_camera()
        self.draw_text()

    def place_camera(self):
        snap, self.snap_camera = self.snap_camera, False
        dt = ursina_time.dt
        mode = CAMERA_MODES[self.camera_mode]

        if mode == "Overhead":
            width, height = self.track.SCREEN_WIDTH * PIXEL_SCALE, self.track.SCREEN_HEIGHT * PIXEL_SCALE
            target = Vec3(width / 2, 0, -height / 2)
            position = target + Vec3(0, 210, -75)
        else:
            car = self.cars[CAMERA_MODES.index(mode)]
            # ease the camera's heading towards the car's, so quick left/right flicks average out
            heading_t = 1 if snap else smoothing_factor(dt, self.heading_lag)
            self.camera_heading = lerp_angle(self.camera_heading, car.draw_rot, heading_t) % 360
            forward = forward_vector(self.camera_heading)
            target = car.entity.position + forward * CHASE_LOOK_AHEAD
            position = car.entity.position - forward * CHASE_DISTANCE + Vec3(0, CHASE_HEIGHT, 0)

        position_t = 1 if snap else smoothing_factor(dt, self.position_lag)
        camera.position = lerp(camera.position, position, position_t)
        camera.look_at(target, up=Vec3(0, 1, 0))  # Ursina defaults to the camera's own up, which drifts

    def draw_text(self):
        names = [car.name for car in self.cars]
        rows = {
            "Lap": [f"{min(car.env.laps_completed + 1, LAPS)}/{LAPS}" for car in self.cars],
            "Wins": [car.wins for car in self.cars],
            "Crashes": [car.crashes for car in self.cars],
        }
        widths = [max(len(name), *(len(str(v[i])) for v in rows.values())) for i, name in enumerate(names)]
        label_width = max(len(label) for label in rows)

        def line(label, values):
            return label.ljust(label_width) + "".join(f"  {str(v).rjust(w)}" for v, w in zip(values, widths))

        self.scoreboard.text = "\n".join([line("", names)] + [line(k, v) for k, v in rows.items()])

        main = self.cars[0].env
        self.status.text = (
            f"Camera: {CAMERA_MODES[self.camera_mode]}  (c to change)\n"
            f"{self.tuning_label()}  (a/d/r/h/p, +/-)"
        )

    def tuning_label(self) -> str:
        """The setting + / - currently change, and its value."""
        if self.camera_tuning is None:
            main = self.cars[0].env
            return f"{main.alter_mode}: {main._tuned_value():.2f}"
        name, attribute, _ = CAMERA_TUNING[self.camera_tuning]
        return f"{name}: {getattr(self, attribute):.2f}s"

    def input(self, key):
        if key == "c":
            self.camera_mode = (self.camera_mode + 1) % len(CAMERA_MODES)
            self.snap_camera = True
        elif key == "escape":
            application.quit()
        elif key in ("a", "d", "r"):
            self.camera_tuning = None
            self.cars[0].env.handle_key(ord(key))
        elif key in CAMERA_TUNING:
            self.camera_tuning = key
        elif key in PLUS_KEYS or key in MINUS_KEYS:
            direction = 1 if key in PLUS_KEYS else -1
            if self.camera_tuning is None:
                self.cars[0].env.handle_key(ord("+" if direction > 0 else "-"))
            else:
                _, attribute, step = CAMERA_TUNING[self.camera_tuning]
                value = getattr(self, attribute) + direction * step
                setattr(self, attribute, round(max(0.0, value), 4))  # 0 = no smoothing


def build_app() -> Ursina:
    app = Ursina(title="Car Go Vroom 3D", borderless=False)
    window.color = color.rgb32(120, 170, 230)
    Sky()
    DirectionalLight(shadows=False).look_at(Vec3(1, -2, 1))
    AmbientLight(color=color.rgba(1, 1, 1, 0.4))
    Race3D()
    return app


if __name__ == "__main__":
    build_app().run()
