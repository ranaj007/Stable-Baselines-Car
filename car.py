from gymnasium import spaces
import calculations as calc
import gymnasium as gym
from track import Track
import numpy as np
import cv2 as cv
import random

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


class CarAgent(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        track: Track,
        do_render: bool = False,
        draw_lines: bool = False,
        render_text: bool = False,
        action_limit: int = 0,
        speed_reward: bool = False,
        training: bool = False,
        number_of_view_lines: int = 8,
        number_of_collisions: int = 3,
    ):
        super(CarAgent, self).__init__()
        self.action_space = spaces.Discrete(6)

        N_CHANNELS = 5 + 8 * number_of_collisions
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(N_CHANNELS,), dtype=np.float32
        )

        inner_lines, outer_lines, reward_gates_coords, start_positions = (
            track.get_borders()
        )

        self.img = track.new_frame()
        self.SCREEN_WIDTH = self.img.shape[1]
        self.SCREEN_HEIGHT = self.img.shape[0]
        self.start_positions = start_positions
        self.inner_lines = inner_lines
        self.outer_lines = outer_lines
        self.reward_gates_coords = reward_gates_coords

        self.View_Line = [0, 0, 0, 0, 0, 0, 0, 0]
        self.next_gate = []
        self.total_score = 0
        self.action_cntr = None

        self.do_render = do_render
        self.draw_lines = draw_lines
        self.render_text = render_text
        self.action_limit = action_limit
        self.wall_penalty = True
        self.speed_reward = speed_reward
        self.training = training
        self.number_of_collisions = number_of_collisions
        self.number_of_view_lines = number_of_view_lines

        self.Car_Accel = 0.11
        self.Car_Accel_Dec = 0.97
        self.Car_Rot_Speed = 5
        self.Car_Line_Length = 400

        self.Car_Pos = np.array([0, 0], dtype=np.float32)
        self.Car_Vel = np.array([0, 0], dtype=np.float32)
        self.Car_Rot = 0

        self.avg_speed = 0
        self.avg_speed_samples = 100

    def step(self, action: int):
        self.foward = False
        self.back = False
        self.left = False
        self.right = False

        if action == 0:
            self.foward = True
        elif action == 1:
            self.back = True
        elif action == 2:
            self.left = True
        elif action == 3:
            self.right = True
        elif action == 4:
            self.foward = True
            self.left = True
        elif action == 5:
            self.foward = True
            self.right = True
        # elif action == 6:
        #  self.reward -= 5
        #  pass

        observation = self.on_update()

        info = {}

        if self.training:
            return observation, self.reward, self.done, False, info

        return observation, self.reward, self.done, info

    def reset(self):
        self.foward = False
        self.back = False
        self.left = False
        self.right = False

        self.reward = 0
        self.total_score = 0
        self.done = False

        starting_state = random.choice(self.start_positions)

        self.Car_Pos = np.array(starting_state[0].copy(), dtype=np.float32)
        self.Car_Pos[0] += random.randint(-15, 15)
        self.Car_Pos[1] += random.randint(-15, 15)
        self.Car_Rot = starting_state[1]
        self.Car_Rot += random.randint(-15, 15)
        self.gate_progress = starting_state[2]

        self.Car_Vel = np.array(
            calc.calc_line_length((0, 0), random.uniform(0, 2), self.Car_Rot),
            dtype=np.float32,
        )
        # self.Car_Vel = np.array(starting_state[3].copy(), dtype=np.float32)

        self.avg_speed = 0

        self.start = [self.Car_Pos, self.Car_Rot, self.gate_progress, self.Car_Vel]

        self.past_reward_dist = 200
        self.chng_dist = 0
        self.action_cntr = 0

        temp_vel = self.Car_Vel.copy()
        self.Car_Vel = np.multiply(self.Car_Vel, 0)
        observation = self.on_update()
        self.Car_Vel = temp_vel

        if self.training:
            return observation, {}

        return observation  # reward, done, info can't be included

    def render(self):
        # draw car
        car_window = calc.calc_line_length(self.Car_Pos, 7, self.Car_Rot, integers=True)
        car_back = calc.calc_line_length(
            self.Car_Pos, 5, self.Car_Rot + 180, integers=True
        )
        car_front = calc.calc_line_length(self.Car_Pos, 5, self.Car_Rot, integers=True)
        for i in range(4):
            car_front_left_wheel = calc.calc_line_length(
                self.Car_Pos, 4, self.Car_Rot + 45 + 90 * i, integers=True
            )
            cv.line(
                self.img,
                calc.to_int(self.Car_Pos),
                car_front_left_wheel,
                BLACK,
                thickness=3,
            )

        cv.line(
            self.img,
            car_front,
            car_window,
            BLUE,
            thickness=3,
        )
        cv.line(
            self.img,
            car_back,
            car_front,
            RED,
            thickness=3,
        )

        # draw view lines
        if self.draw_lines:
            for i in range(len(self.View_Line)):
                cv.line(
                    self.img,
                    calc.to_int(self.Car_Pos),
                    calc.to_int(self.View_Line[i]),
                    BLUE,
                )

            for i in range(0, len(self.collisions), self.number_of_collisions):
                collision = calc.calc_line_length(
                    self.Car_Pos,
                    self.collisions[i],
                    self.Car_Rot + (360 / self.number_of_view_lines) * i,
                )
                collision = calc.to_int(collision)
                #cv.circle(self.img, collision, 5, color=(0, 255, 0))

        # draw gate
        cv.line(self.img, self.next_gate[0], self.next_gate[1], GREEN)

        if self.render_text:
            # draw text
            cv.putText(
                self.img,
                f"Action Counter: {self.action_cntr}",
                (100, 350),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                BLACK,
                1,
            )
            cv.putText(
                self.img,
                f"Total Score: {self.total_score:.1f}",
                (100, 400),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                BLACK,
                1,
            )
            cv.putText(
                self.img,
                f"Car Position: {self.Car_Pos[0]:.1f}, {self.Car_Pos[1]:.1f}",
                (100, 450),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                BLACK,
                1,
            )
            cv.putText(
                self.img,
                f"Average Speed: {self.avg_speed:.2f}",
                (100, 500),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                BLACK,
                1,
            )

    def draw_car_lines(
        self,
        num_o_lines: int,
        line_length: float,
        collision_objects: list,
        number_of_collisions: int = 1,
        draw: bool = False,
    ):
        distances = np.full(
            num_o_lines * number_of_collisions, line_length, dtype=np.float32
        )
        for i in range(num_o_lines):
            self.View_Line[i] = calc.calc_line_length(
                self.Car_Pos,
                line_length,
                self.Car_Rot + (360 / num_o_lines) * i,
            )

            collision1 = calc.detect_collisions(
                self.Car_Pos,
                self.Car_Pos,
                self.View_Line[i],
                collision_objects[0],
                self.Car_Line_Length,
                number_of_collisions,
            )
            collision2 = calc.detect_collisions(
                self.Car_Pos,
                self.Car_Pos,
                self.View_Line[i],
                collision_objects[1],
                self.Car_Line_Length,
                number_of_collisions,
            )

            collisions = np.concatenate((collision1, collision2), axis=0)

            collisions.sort()
            collisions = collisions[:number_of_collisions]

            distances[i * number_of_collisions : (i + 1) * number_of_collisions] = (
                collisions
            )
            if self.draw_lines and draw:
                for j in range(number_of_collisions):
                    collision = calc.calc_line_length(
                        self.Car_Pos,
                        collisions[j],
                        self.Car_Rot + (360 / num_o_lines) * i,
                    )
                    collision = calc.to_int(collision)
                    cv.circle(self.img, collision, 5, color=(0, 0, 255))
        return distances

    def get_observation(self):
        temp_collisions = self.collisions.copy()
        for idx in range(len(temp_collisions)):
            temp_collisions[idx] = (temp_collisions[idx] - self.Car_Line_Length / 2) / (
                self.Car_Line_Length / 2
            )

        # reward_index = [(self.reward_dist.index(min(self.reward_dist))-(len(self.reward_dist))/2)/(len(self.reward_dist))/2]
        # temp_list = self.reward_dist.tolist()
        # reward_index = [(temp_list.index(min(temp_list))-(len(temp_list))/2)/(len(temp_list))/2]
        gate_angle = calc.get_angle_to_gate(self.Car_Pos, self.next_gate)

        return np.array(
            temp_collisions.tolist()
            + [(gate_angle - 180) / 180]
            + [(min(self.reward_dist) - 7.5) / self.Car_Line_Length]
            + np.divide(self.Car_Vel.copy(), 4).tolist()
            + [(self.Car_Rot - 180) / 180],
            dtype=np.float32,
        )

    def on_update(self, steps: int = 2):
        self.action_cntr += 1
        self.reward = 0

        if self.left:
            self.Car_Rot -= self.Car_Rot_Speed
            self.Car_Rot = self.Car_Rot % 360

        if self.right:
            self.Car_Rot += self.Car_Rot_Speed
            self.Car_Rot = self.Car_Rot % 360

        for i in range(steps):
            self.Car_Pos = np.add(self.Car_Pos, self.Car_Vel)

            self.Car_Vel = np.multiply(self.Car_Vel, self.Car_Accel_Dec)

            if False:
                car_dec = -self.Car_Vel

                if car_dec[0] < 0:
                    car_dec[0] = max(car_dec[0], -self.Car_Accel_Dec)
                else:
                    car_dec[0] = min(car_dec[0], self.Car_Accel_Dec)

                if car_dec[1] < 0:
                    car_dec[1] = max(car_dec[1], -self.Car_Accel_Dec)
                else:
                    car_dec[1] = min(car_dec[1], self.Car_Accel_Dec)

                self.Car_Vel = np.add(self.Car_Vel, car_dec)

            # TODO: add avg_speed to rewards
            self.avg_speed = calc.get_average_speed(
                self.avg_speed_samples, self.avg_speed, self.Car_Vel
            )

            # self.Car_Vel[0] = round(self.Car_Vel[0], 3)
            # self.Car_Vel[1] = round(self.Car_Vel[1], 3)

            if np.sqrt(self.Car_Vel.dot(self.Car_Vel)) <= 1e-3:
                self.Car_Vel = np.multiply(self.Car_Vel, 0)

            if self.foward:
                self.Car_Vel = np.add(
                    self.Car_Vel,
                    calc.calc_line_length((0, 0), self.Car_Accel, self.Car_Rot),
                )

            if self.back:
                self.Car_Vel = np.subtract(
                    self.Car_Vel,
                    calc.calc_line_length((0, 0), self.Car_Accel, self.Car_Rot),
                )
                self.reward -= 5

            # calc and draw edge detection lines
            self.collisions = self.draw_car_lines(
                self.number_of_view_lines,
                self.Car_Line_Length,
                [self.inner_lines, self.outer_lines],
                self.number_of_collisions,
                i==steps-1
            )

            # calc and draw reward gate detection lines
            self.next_gate = [
                self.reward_gates_coords[self.gate_progress],
                self.reward_gates_coords[self.gate_progress + 1],
            ]
            self.reward_dist = self.draw_car_lines(
                self.number_of_view_lines,
                self.Car_Line_Length,
                [self.next_gate, self.next_gate],
            )

            if min(self.reward_dist) <= 7.5:
                self.reward += 100
                self.gate_progress += 2
                if self.gate_progress > len(self.reward_gates_coords) - 1:
                    self.gate_progress = 0
                self.past_reward_dist = 200
            else:
                self.reward += 0

            if self.speed_reward:
                self.reward += np.sqrt(self.Car_Vel.dot(self.Car_Vel))

            for dist in self.collisions:
                if dist <= 5:
                    if self.wall_penalty:
                        self.reward = -10
                    self.done = True

            if (
                self.Car_Pos[0] < 0
                or self.Car_Pos[0] > self.SCREEN_WIDTH
                or self.Car_Pos[1] < 0
                or self.Car_Pos[1] > self.SCREEN_HEIGHT
            ):
                self.done = True

            if i == steps - 1 and self.do_render:
                #if self.action_cntr > 3000:
                    #self.render()
                if self.do_render:
                    self.render()

            if self.action_limit and self.action_cntr >= self.action_limit:
                self.done = True

            self.total_score += self.reward

        observation = self.get_observation()

        if np.any(observation > 1) or np.any(observation < -1):
            print("WARNING: Observation out of bounds:", observation)
            observation = np.clip(observation, -1, 1)
        return observation
