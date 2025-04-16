import numpy as np
import cv2 as cv
import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


class Track():

    def __init__(
        self,
        do_render: bool = False,
        show_fps: bool = False,
    ):
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 1000
        self.img = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)

        self.inner_lines = []
        self.outer_lines = []
        self.reward_gates_coords = []
        self.start_positions = []
        self.next_gate = []
        self.bad_starts = []

        self.do_render = do_render
        self.show_fps = show_fps

        self.last_frame_time = time.time()

        self.Load_borders()

        self.inner_lines = np.array(self.inner_lines)
        self.outer_lines = np.array(self.outer_lines)
        self.reward_gates_coords = np.array(self.reward_gates_coords)
        self.start_positions = np.array(self.start_positions, dtype=object)


    def Load_borders(self):
        with open("Game_Boarders_Inner_Resized.txt") as file1:
            temp_list = file1.readlines()
            for i in range(1, len(temp_list), 2):
                self.inner_lines.append([int(temp_list[i - 1]), int(temp_list[i])])

        with open("Game_Boarders_Outer_Resized.txt") as file1:
            temp_list = file1.readlines()
            for i in range(1, len(temp_list), 2):
                self.outer_lines.append([int(temp_list[i - 1]), int(temp_list[i])])

        with open("Reward_Gates.txt") as file1:
            temp_list = file1.readlines()
            for i in range(1, len(temp_list), 2):
                self.reward_gates_coords.append([int(temp_list[i - 1]), int(temp_list[i])])

        with open("Starting_States.txt") as file1:
            temp_list = file1.readlines()
            temp_list = [int(i) for i in temp_list]
            for i in range(3, len(temp_list), 4):
                self.start_positions.append(
                    [
                        [temp_list[i - 3], temp_list[i - 2]],
                        temp_list[i - 1],
                        temp_list[i],
                    ]
                )

    def get_borders(self):
        return self.inner_lines, self.outer_lines, self.reward_gates_coords, self.start_positions

    def new_frame(self) -> np.ndarray:
        self.img.fill(0)
        self.img += 85

        # draw track
        cv.fillPoly(self.img, pts=[self.outer_lines], color=(128, 128, 128))
        cv.fillPoly(self.img, pts=[self.inner_lines], color=(85, 85, 85))
        cv.polylines(self.img, [self.outer_lines], True, BLACK)
        cv.polylines(self.img, [self.inner_lines], True, BLACK)

        return self.img
    

    def render(self):
        fps = 1 / (time.time() - self.last_frame_time)

        target_fps = 60
        target_spf = 1 / target_fps

        spf = 1 / fps
        sleep_time = max(0, target_spf - spf)  # Ensure sleep_time is non-negative
        time.sleep(sleep_time)
        
        fps = 1 / (time.time() - self.last_frame_time)
        self.last_frame_time = time.time()

        if self.show_fps:
            cv.putText(
                self.img,
                f"FPS: {fps:.0f}",
                (100, 600),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                BLACK,
                1,
            )

        cv.imshow("Car Go Vroom", self.img)
        cv.waitKey(1)


    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
