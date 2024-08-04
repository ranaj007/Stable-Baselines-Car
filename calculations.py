from numba import jit
import numpy as np
import math

@jit(nopython=True)
def calc_dist(Point1: np.ndarray, Point2: np.ndarray) -> float:
    x_diff = Point1[0] - Point2[0]
    y_diff = Point1[1] - Point2[1]

    dist = math.sqrt(x_diff**2 + y_diff**2)

    return dist

@jit(nopython=True)
def calc_line_length(x_offset: float, y_offset: float, base_line_length: float, angle_degrees: float) -> np.ndarray:
    angle_radians = -math.radians(angle_degrees)
    XY = np.array([x_offset + base_line_length*math.cos(angle_radians), y_offset + base_line_length*math.sin(angle_radians)])
    return XY

@jit(nopython=True)
def det(a: np.ndarray, b: np.ndarray) -> float:
    return a[0] * b[1] - a[1] * b[0]

@jit(nopython=True)
def line_intersection(Car_Pos: np.ndarray, line1: np.ndarray, line2: np.ndarray) -> np.ndarray | None:
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    #a = "Return X and Y"
    #b = "Return X and Y"

    line_length = calc_dist(*line2)
    point1_dist = calc_dist(Car_Pos, line2[0])
    point2_dist = calc_dist(Car_Pos, line2[1])
    if point1_dist > line_length and point2_dist > line_length:
        if point1_dist > 210 and point2_dist > 210:
            #a = "Return None"
            return None


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
        return np.array([x, y])

    def to_int(Point1):
        return (int(Point1[0]), int(Point1[1]))

    """
    if False:
        WHITE = (255,255,255)
        img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH,3), dtype=np.uint8)
        center = (int(Car_Pos[0]), int(Car_Pos[1]))
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
    """
    
@jit(nopython=True)
def detect_collisions(Car_Pos: np.ndarray, Point1: np.ndarray, Point2: np.ndarray, object1: np.ndarray, inc: int = 1) -> float: # inc = 1 for continuious lines, inc = 2 for reward gates
    distances = np.full(6, 200, dtype=np.float32)
    a = 0
    for idx in range(1, len(object1), inc):
        coords = line_intersection(Car_Pos, (Point1, Point2), (object1[idx-1], object1[idx]))
        if coords is not None:
            distances[a] = calc_dist(Point1, coords)
            a += 1
    return min(distances)

@jit(nopython=True)
def get_angle_to_gate(Car_Pos: np.ndarray, next_gate: np.ndarray) -> float:
    center_pos = np.add(next_gate[0], np.divide(np.subtract(next_gate[1], next_gate[0]), 2))

    dist_vector = np.subtract(Car_Pos, center_pos)
    dist_vector[0] *= -1

    absolute_angle = math.atan2(dist_vector[1], dist_vector[0])
    absolute_angle = 180*absolute_angle/math.pi

    if absolute_angle < 0:
        absolute_angle += 360
    
    return absolute_angle