import gym
from gym import spaces
import math
import random
import time
import numpy as np
import cv2 as cv
from numba import jit

class CarEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, do_render=False, action_limit=0, speed_reward=False):
        super(CarEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(6)
        # Example for using image as input (channel-first; channel-last also works):
        N_CHANNELS = 13
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(N_CHANNELS,), dtype=np.float32)

        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 1000
        self.img = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH,3), dtype=np.uint8)

        self.inner_lines = []
        self.outer_lines = []
        self.reward_gates_coords = []
        self.start_positions = []
        self.next_gate = []
        self.bad_starts = []
        self.View_Line = [0, 0, 0, 0, 0, 0, 0, 0]
        self.total_score = 0

        self.do_render = do_render
        self.action_limit = action_limit
        self.wall_penalty = True
        self.speed_reward = speed_reward

        self.Load_borders(self.inner_lines, self.outer_lines, self.reward_gates_coords, self.start_positions)

        self.inner_lines = np.array(self.inner_lines)
        self.outer_lines = np.array(self.outer_lines)
        self.reward_gates_coords = np.array(self.reward_gates_coords)
        self.start_positions = np.array(self.start_positions, dtype = object)

        self.Car_Speed = 3
        #self.Car_Accel = 0.09
        #self.Car_Accel_Dec = 0.96
        self.Car_Accel = 0.2
        #self.Car_Accel_Dec = 0.9
        self.Car_Accel_Dec = 0.95
        self.Car_Rot_Speed = 3
        self.Car_Line_Length = 200

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
        #print(f'{action}')
        #print(f'{self.reward=}')
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
        self.Car_Pos[0] += random.randint(-15,15)
        self.Car_Pos[1] += random.randint(-15,15)
        self.Car_Rot = starting_state[1]
        self.Car_Rot += random.randint(-15,15)
        self.gate_progress = starting_state[2]

        self.Car_Vel = np.array(self.calc_line_length(0,0,random.uniform(0, 2), self.Car_Rot), dtype=np.float32)
        #self.Car_Vel = np.array(starting_state[3].copy(), dtype=np.float32)

        self.start = [self.Car_Pos, self.Car_Rot, self.gate_progress, self.Car_Vel]

        self.past_reward_dist = 200
        self.chng_dist = 0
        self.action_cntr = 0

        temp_vel = self.Car_Vel.copy()
        self.Car_Vel = np.multiply(self.Car_Vel, 0)
        observation = self.on_update()
        self.Car_Vel = temp_vel


        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        WHITE = (0,0,0)
        self.img = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH,3), dtype=np.uint8)
        self.img += 85

        # draw track
        cv.fillPoly(self.img, pts = [self.outer_lines], color = (128, 128, 128))
        cv.fillPoly(self.img, pts = [self.inner_lines], color = (85, 85, 85))
        cv.polylines(self.img, [self.outer_lines], True, WHITE)
        cv.polylines(self.img, [self.inner_lines], True, WHITE)

        # draw car
        center = (int(self.Car_Pos[0]), int(self.Car_Pos[1]))
        cv.circle(self.img, center, 3, (0, 0, 255), -1)

        # draw view lines
        if False:
            for i in range(len(self.View_Line)):
                cv.line(self.img, (int(self.Car_Pos[0]), int(self.Car_Pos[1])), (int(self.View_Line[i][0]), int(self.View_Line[i][1])), (0, 0, 255))

        # draw gate
        cv.line(self.img, self.next_gate[0], self.next_gate[1], (0, 255, 0))

        # draw text
        cv.putText(self.img, str(self.action_cntr), (100, 350), cv.FONT_HERSHEY_SIMPLEX, 1, WHITE, 1)
        cv.putText(self.img, str(self.total_score), (100, 400), cv.FONT_HERSHEY_SIMPLEX, 1, WHITE, 1)

        # show image
        cv.imshow("Car Go Vroom", self.img)
        cv.waitKey(1)



# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

    def Load_borders(self, inner_lines, outer_lines, reward_gates, starting_conditions):
        with open("Game_Boarders_Inner_Resized.txt") as file1:
            temp_list = file1.readlines()
            for i in range(1, len(temp_list), 2):
               inner_lines.append([int(temp_list[i-1]), int(temp_list[i])])

        with open("Game_Boarders_Outer_Resized.txt") as file1:
            temp_list = file1.readlines()
            for i in range(1, len(temp_list), 2):
               outer_lines.append([int(temp_list[i-1]), int(temp_list[i])])

        with open("Reward_Gates.txt") as file1:
            temp_list = file1.readlines()
            for i in range(1, len(temp_list), 2):
                reward_gates.append([int(temp_list[i-1]), int(temp_list[i])])

        with open("Starting_States.txt") as file1:
            temp_list = file1.readlines()
            temp_list = [int(i) for i in temp_list]
            for i in range(3, len(temp_list), 4):
                starting_conditions.append([[temp_list[i-3], temp_list[i-2]], temp_list[i-1], temp_list[i]])
    '''
    def Load_bad_states(self, starting_conditions):
        with open("Bad_States.txt") as file1:
            temp_list = file1.readlines()
            temp_list = [float(i) for i in temp_list]
            for i in range(5, len(temp_list), 6):
                starting_conditions.append([[temp_list[i-5], temp_list[i-4]], temp_list[i-3], temp_list[i-2], [temp_list[i-1], temp_list[i]]])
    '''

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        #a = "Return X and Y"
        #b = "Return X and Y"

        line_length = calc_dist(*line2)
        point1_dist = calc_dist(self.Car_Pos, line2[0])
        point2_dist = calc_dist(self.Car_Pos, line2[1])
        if point1_dist > line_length and point2_dist > line_length:
            if point1_dist > 210 and point2_dist > 210:
                #a = "Return None"
                return None

        def det(a, b):
           return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
          return None

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        
        big_x = min(max(line1[0][0], line1[1][0]), max(line2[0][0], line2[1][0]))
        small_x = max(min(line1[0][0], line1[1][0]), min(line2[0][0], line2[1][0]))

        big_y = min(max(line1[0][1], line1[1][1]), max(line2[0][1], line2[1][1]))
        small_y = max(min(line1[0][1], line1[1][1]), min(line2[0][1], line2[1][1]))

        if x > big_x or x < small_x or y > big_y or y < small_y:
           #b = "Return None"
           return None
        else:
           return x, y

        def to_int(Point1):
            return (int(Point1[0]), int(Point1[1]))

        if False:
            WHITE = (255,255,255)
            img = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH,3), dtype=np.uint8)
            center = (int(self.Car_Pos[0]), int(self.Car_Pos[1]))
            cv.circle(img, center, 3, (0, 0, 255), -1)
            points = tuple(map(to_int, line1))
            cv.line(img, points[0], points[1], (255, 0, 0))
            points = tuple(map(to_int, line2))
            cv.line(img, points[0], points[1], (0, 0, 255))
            cv.putText(img, "My method: " + a, (100, 350), cv.FONT_HERSHEY_SIMPLEX, 1, WHITE, 1)
            cv.putText(img, "Th method: "+ b, (100, 400), cv.FONT_HERSHEY_SIMPLEX, 1, WHITE, 1)
            cv.putText(img, f"{point1_dist=}", (100, 300), cv.FONT_HERSHEY_SIMPLEX, 1, WHITE, 1)
            cv.putText(img, f"{point2_dist=}", (100, 250), cv.FONT_HERSHEY_SIMPLEX, 1, WHITE, 1)
            cv.imshow("Car Go Vroom", img)
            cv.waitKey()

        if b == "Return None":
           return None
        else:
           return x, y


    def detect_collisions(self, Point1, Point2, object1, inc = 1): # inc = 1 for continuious lines, inc = 2 for reward gates
        distances = np.full(6, 200, dtype=np.float32)
        a = 0
        for idx in range(1, len(object1), inc):
            coords = self.line_intersection((Point1, Point2), (object1[idx-1], object1[idx]))
            if coords != None:
                distances[a] = calc_dist(Point1, coords)
                a += 1
        return min(distances)

    def calc_dist(self, Point1, Point2):
        x_diff = (Point1[0] - Point2[0])
        y_diff = (Point1[1] - Point2[1])

        dist = (x_diff**2 + y_diff**2)**0.5

        return dist

    def calc_line_length(self, x_offset, y_offset, base_line_length, angle_degrees):
        angle_radians = -math.radians(angle_degrees)
        XY = [x_offset + base_line_length*math.cos(angle_radians), y_offset + base_line_length*math.sin(angle_radians)]
        return XY

    def draw_car_lines(self, num_o_lines, line_length, collision_objects):
        distances = np.full(num_o_lines, line_length, dtype=np.float32)
        for i in range(num_o_lines):
            self.View_Line[i] = self.calc_line_length(self.Car_Pos[0], self.Car_Pos[1], line_length, self.Car_Rot + (360/num_o_lines)*i)
            
            collision1 = self.detect_collisions(self.Car_Pos, self.View_Line[i], collision_objects[0])
            collision2 = self.detect_collisions(self.Car_Pos, self.View_Line[i], collision_objects[1])

            distances[i] = min(collision1, collision2)
        return distances

    def get_observation(self):
        temp_collisions = self.collisions.copy()
        for idx in range(len(temp_collisions)):
            temp_collisions[idx] = (temp_collisions[idx]-105) / 100

        #reward_index = [(self.reward_dist.index(min(self.reward_dist))-(len(self.reward_dist))/2)/(len(self.reward_dist))/2]
        #temp_list = self.reward_dist.tolist()
        #reward_index = [(temp_list.index(min(temp_list))-(len(temp_list))/2)/(len(temp_list))/2]
        gate_angle = self.get_angle_to_gate()

        return np.array(temp_collisions.tolist() + [(gate_angle-180)/180] + [(min(self.reward_dist)-7.5)/200] + np.divide(self.Car_Vel.copy(), 4).tolist() + [(self.Car_Rot-180)/180], dtype=np.float32)

    def get_angle_to_gate(self):
        center_pos = np.add(self.next_gate[0], np.divide(np.subtract(self.next_gate[1], self.next_gate[0]), 2))

        dist_vector = np.subtract(self.Car_Pos, center_pos)
        dist_vector[0] *= -1

        absolute_angle = math.atan2(dist_vector[1], dist_vector[0])
        absolute_angle = 180*absolute_angle/math.pi

        if absolute_angle < 0:
            absolute_angle += 360
        
        return absolute_angle

    def on_update(self):
        self.action_cntr += 1
        if self.left == True:
            self.Car_Rot -= self.Car_Rot_Speed
            self.Car_Rot = self.Car_Rot % 360

        if self.right == True:
            self.Car_Rot += self.Car_Rot_Speed
            self.Car_Rot = self.Car_Rot % 360

        self.Car_Pos = np.add(self.Car_Pos, self.Car_Vel)

        self.Car_Vel = np.multiply(self.Car_Vel, self.Car_Accel_Dec)


        #self.Car_Vel[0] = round(self.Car_Vel[0], 3)
        #self.Car_Vel[1] = round(self.Car_Vel[1], 3)
        
        if np.sqrt(self.Car_Vel.dot(self.Car_Vel)) <= 1e-3:
            self.Car_Vel = np.multiply(self.Car_Vel, 0)
        

        if self.foward == True:
            self.Car_Vel = np.add(self.Car_Vel, self.calc_line_length(0, 0, self.Car_Accel, self.Car_Rot))

        if self.back == True:
            self.Car_Vel = np.subtract(self.Car_Vel, self.calc_line_length(0, 0, self.Car_Accel, self.Car_Rot))

         # calc and draw edge detection lines
        self.collisions = self.draw_car_lines(8, self.Car_Line_Length, [self.inner_lines, self.outer_lines])

        # calc and draw reward gate detection lines
        self.next_gate = [self.reward_gates_coords[self.gate_progress], self.reward_gates_coords[self.gate_progress+1]]
        self.reward_dist = self.draw_car_lines(8, self.Car_Line_Length, [self.next_gate, self.next_gate])

        if min(self.reward_dist) <= 7.5:
            self.reward = 100
            self.gate_progress += 2
            if self.gate_progress > len(self.reward_gates_coords) - 1:
                self.gate_progress = 0
            self.past_reward_dist = 200
        else:
            self.reward = 0

        if self.speed_reward:
            self.reward += np.sqrt(self.Car_Vel.dot(self.Car_Vel))

        for dist in self.collisions:
            if dist <= 5:
                if self.wall_penalty:
                    self.reward = -10
                self.done = True
                

        if self.do_render and self.action_cntr > 3000:
            self.render()

        if self.action_limit and self.action_cntr >= self.action_limit:
            self.done = True

        self.total_score += self.reward
        return self.get_observation()

@jit(nopython=True)
def calc_dist(Point1, Point2):
    x_diff = Point1[0] - Point2[0]
    y_diff = Point1[1] - Point2[1]

    dist = math.sqrt(x_diff**2 + y_diff**2)

    return dist