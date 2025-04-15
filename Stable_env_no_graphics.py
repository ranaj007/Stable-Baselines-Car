from gymnasium import spaces
import calculations as calc
import gymnasium as gym
import numpy as np
import cv2 as cv
import random
import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


class CarEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        do_render=False,
        draw_lines=False,
        action_limit=0,
        speed_reward=False,
        training=True,
        number_of_collisions=3,
    ):
        super(CarEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(6)
        # Example for using image as input (channel-first; channel-last also works):
        N_CHANNELS = 5 + 8 * number_of_collisions
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(N_CHANNELS,), dtype=np.float32
        )

        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 1000
        self.img = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)

        self.inner_lines = []
        self.outer_lines = []
        self.reward_gates_coords = []
        self.start_positions = []
        self.next_gate = []
        self.bad_starts = []
        self.View_Line = [0, 0, 0, 0, 0, 0, 0, 0]
        self.total_score = 0

        self.do_render = do_render
        self.draw_lines = draw_lines
        self.action_limit = action_limit
        self.wall_penalty = True
        self.speed_reward = speed_reward
        self.training = training
        self.number_of_collisions = number_of_collisions

        self.last_frame_time = time.time()

        self.Load_borders(
            self.inner_lines,
            self.outer_lines,
            self.reward_gates_coords,
            self.start_positions,
        )

        self.inner_lines = np.array(self.inner_lines)
        self.outer_lines = np.array(self.outer_lines)
        self.reward_gates_coords = np.array(self.reward_gates_coords)
        self.start_positions = np.array(self.start_positions, dtype=object)

        self.alter_mode = "Acceleration"

        self.Car_Accel = 0.11  # 0.2
        self.Car_Accel_Dec = 0.97
        self.Car_Rot_Speed = 5
        self.Car_Line_Length = 400

        self.avg_speed = 0
        self.avg_speed_samples = 100

    def step(self, action):
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
        # print(f'{action}')
        # print(f'{self.reward=}')

        if self.training:
            return observation, self.reward, self.done, False, info

        return observation, self.reward, self.done, info

    def reset(self, seed=None, options=None):
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

    def render(self, mode="human", temp=True):
        self.img = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.img += 85

        # draw track
        cv.fillPoly(self.img, pts=[self.outer_lines], color=(128, 128, 128))
        cv.fillPoly(self.img, pts=[self.inner_lines], color=(85, 85, 85))
        cv.polylines(self.img, [self.outer_lines], True, BLACK)
        if temp:
            cv.polylines(self.img, [self.inner_lines], True, BLACK)
        else:
            cv.polylines(self.img, [self.inner_lines], True, BLUE)

        # draw car
        car_window = calc.calc_line_length(
            self.Car_Pos, 7, self.Car_Rot, integers=True
        )
        car_back = calc.calc_line_length(
            self.Car_Pos, 5, self.Car_Rot + 180, integers=True
        )
        car_front = calc.calc_line_length(
            self.Car_Pos, 5, self.Car_Rot, integers=True
        )
        for i in range(4):
            car_front_left_wheel = calc.calc_line_length(
                self.Car_Pos, 4, self.Car_Rot + 45 + 90*i, integers=True
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

        # draw gate
        cv.line(self.img, self.next_gate[0], self.next_gate[1], GREEN)

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

        if self.alter_mode == "Acceleration":
            var = self.Car_Accel
        elif self.alter_mode == "Deceleration":
            var = self.Car_Accel_Dec
        elif self.alter_mode == "Rotation":
            var = self.Car_Rot_Speed

        cv.putText(
            self.img,
            f"{self.alter_mode}: {var:.2f}",
            (100, 550),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            BLACK,
            1,
        )
        
        if False:
            fps = 1 / (time.time() - self.last_frame_time)
            self.last_frame_time = time.time()
            cv.putText(
                self.img,
                f"FPS: {fps:.0f}",
                (100, 600),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                BLACK,
                1,
            )

        # show image
        cv.imshow("Car Go Vroom", self.img)
        key = cv.waitKey(1)

        if key == ord("a"):
            self.alter_mode = "Acceleration"
        elif key == ord("d"):
            self.alter_mode = "Deceleration"
        elif key == ord("r"):
            self.alter_mode = "Rotation"

        change = 0
        if key == ord("+"):
            change = 0.01
        elif key == ord("-"):
            change = -0.01

        if self.alter_mode == "Acceleration":
            self.Car_Accel += change
        elif self.alter_mode == "Deceleration":
            self.Car_Accel_Dec += change
        elif self.alter_mode == "Rotation":
            self.Car_Rot_Speed += change

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------

    def Load_borders(self, inner_lines, outer_lines, reward_gates, starting_conditions):
        with open("Game_Boarders_Inner_Resized.txt") as file1:
            temp_list = file1.readlines()
            for i in range(1, len(temp_list), 2):
                inner_lines.append([int(temp_list[i - 1]), int(temp_list[i])])

        with open("Game_Boarders_Outer_Resized.txt") as file1:
            temp_list = file1.readlines()
            for i in range(1, len(temp_list), 2):
                outer_lines.append([int(temp_list[i - 1]), int(temp_list[i])])

        with open("Reward_Gates.txt") as file1:
            temp_list = file1.readlines()
            for i in range(1, len(temp_list), 2):
                reward_gates.append([int(temp_list[i - 1]), int(temp_list[i])])

        with open("Starting_States.txt") as file1:
            temp_list = file1.readlines()
            temp_list = [int(i) for i in temp_list]
            for i in range(3, len(temp_list), 4):
                starting_conditions.append(
                    [
                        [temp_list[i - 3], temp_list[i - 2]],
                        temp_list[i - 1],
                        temp_list[i],
                    ]
                )

    """
    def Load_bad_states(self, starting_conditions):
        with open("Bad_States.txt") as file1:
            temp_list = file1.readlines()
            temp_list = [float(i) for i in temp_list]
            for i in range(5, len(temp_list), 6):
                starting_conditions.append([[temp_list[i-5], temp_list[i-4]], temp_list[i-3], temp_list[i-2], [temp_list[i-1], temp_list[i]]])
    """

    def draw_car_lines(self, num_o_lines, line_length, collision_objects, number_of_collisions: int = 1):
        distances = np.full(num_o_lines * number_of_collisions, line_length, dtype=np.float32)
        for i in range(num_o_lines):
            self.View_Line[i] = calc.calc_line_length(
                self.Car_Pos,
                line_length,
                self.Car_Rot + (360 / num_o_lines) * i,
            )

            collision1 = calc.detect_collisions(
                self.Car_Pos, self.Car_Pos, self.View_Line[i], collision_objects[0], self.Car_Line_Length, number_of_collisions
            )
            collision2 = calc.detect_collisions(
                self.Car_Pos, self.Car_Pos, self.View_Line[i], collision_objects[1], self.Car_Line_Length, number_of_collisions
            )

            collisions = np.concatenate((collision1, collision2), axis=0)

            collisions.sort()
            collisions = collisions[:number_of_collisions]

            distances[i * number_of_collisions : (i + 1) * number_of_collisions] = collisions
        return distances

    def get_observation(self):
        temp_collisions = self.collisions.copy()
        for idx in range(len(temp_collisions)):
            temp_collisions[idx] = (temp_collisions[idx] - self.Car_Line_Length / 2) / (self.Car_Line_Length / 2)

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

                self.Car_Vel = np.add(
                        self.Car_Vel,
                        car_dec
                    )

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
                8, self.Car_Line_Length, [self.inner_lines, self.outer_lines], self.number_of_collisions
            )

            # calc and draw reward gate detection lines
            self.next_gate = [
                self.reward_gates_coords[self.gate_progress],
                self.reward_gates_coords[self.gate_progress + 1],
            ]
            self.reward_dist = self.draw_car_lines(
                8, self.Car_Line_Length, [self.next_gate, self.next_gate]
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

            if self.do_render and self.action_cntr > 3000:
                self.render()

            if self.do_render:
                self.render(temp=i != 0)
                time.sleep(0.01)

            if self.action_limit and self.action_cntr >= self.action_limit:
                self.done = True

            self.total_score += self.reward

        observation = self.get_observation()
        
        if np.any(observation > 1) or np.any(observation < -1):
            print("WARNING: Observation out of bounds:", observation)
            observation = np.clip(observation, -1, 1)
        return observation
