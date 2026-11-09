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
        target_fps: int = 60,
    ):
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 1000
        self.img = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)

        self.target_spf = 1 / target_fps

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

    def draw_scoreboard(
        self,
        scores: list[tuple[str, tuple[int, int, int], int]],
        title: str = "Crashes",
        margin: int = 20,
        line_height: int = 40,
    ):
        """Draw a scoreboard in the bottom-right corner of the current frame.

        scores: one (name, colour, count) per row; the colour, as (B, G, R), is shown as a swatch.
        margin: gap in pixels between the scoreboard and the window's right and bottom edges.
        Call after new_frame() and before render().
        """
        font, scale, thickness = cv.FONT_HERSHEY_SIMPLEX, 1, 1
        swatch, gap = 20, 10
        rows = [f"{name}: {count}" for name, _, count in scores]

        # size the block from the text so it stays inside the window however long the names get
        title_width = cv.getTextSize(title, font, scale, thickness)[0][0]
        row_width = max((cv.getTextSize(r, font, scale, thickness)[0][0] for r in rows), default=0)
        width = max(title_width, swatch + gap + row_width)
        descent = cv.getTextSize(title, font, scale, thickness)[1]

        x = self.SCREEN_WIDTH - margin - width
        bottom_y = self.SCREEN_HEIGHT - margin - descent  # baseline of the last row
        y = bottom_y - line_height * len(rows)  # baseline of the title

        cv.putText(self.img, title, (x, y), font, scale, BLACK, thickness)
        for i, ((_, color, _), text) in enumerate(zip(scores, rows)):
            line_y = y + line_height * (i + 1)
            cv.rectangle(self.img, (x, line_y - swatch), (x + swatch, line_y), color, thickness=-1)
            cv.putText(self.img, text, (x + swatch + gap, line_y), font, scale, BLACK, thickness)


    def render(self) -> int:
        """Show the frame at up to target_fps. Returns the key pressed (-1 if none),
        so it can be passed to CarAgent.handle_key for live tuning."""
        spf = (time.time() - self.last_frame_time)
        fps = 1 / spf if spf > 0 else 0

        sleep_time = max(0, self.target_spf - spf)  # Ensure sleep_time is non-negative
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
        return cv.waitKey(1)


    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
