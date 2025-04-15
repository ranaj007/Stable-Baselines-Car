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
def _calc_line_length(
    origin: tuple[float, float], base_line_length: float, angle_degrees: float
) -> np.ndarray:
    x_offset, y_offset = origin
    angle_radians = -math.radians(angle_degrees)
    x = x_offset + base_line_length * math.cos(angle_radians)
    y = y_offset + base_line_length * math.sin(angle_radians)
    return np.array([x, y], dtype=np.float64)


@jit(nopython=True)
def to_int(XY: np.ndarray) -> np.ndarray:
    return np.array([int(XY[0]), int(XY[1])], dtype=np.int32)


def calc_line_length(
    origin: tuple[float, float],
    base_line_length: float,
    angle_degrees: float,
    integers: bool = False,
) -> np.ndarray:
    XY = _calc_line_length(origin, base_line_length, angle_degrees)
    if integers:
        return to_int(XY)
    return XY


@jit(nopython=True)
def det(a: np.ndarray, b: np.ndarray) -> float:
    return a[0] * b[1] - a[1] * b[0]


@jit(nopython=True)
def line_intersection(
    Car_Pos: np.ndarray, line1: np.ndarray, line2: np.ndarray, max_dist: float
) -> np.ndarray | None:
    """
    Calculates the intersection point between two line segments, with an early rejection
    optimization based on distance from the car.

    Parameters:
    -----------
        Car_Pos : np.ndarray
            The position of the car, used to skip far-away segments for performance.
        line1 : tuple of np.ndarray
            The first line segment, typically the ray being cast (Point1, Point2).
        line2 : tuple of np.ndarray
            The second line segment to test against (e.g., wall or gate).
        max_dist : float
            The maximum distance to consider for the intersection check.

    Returns:
    --------
        np.ndarray or None
            The (x, y) coordinates of the intersection point if one exists within bounds,
            otherwise None.
    """
    # Calculate the differences in x and y coordinates for both lines
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    # a = "Return X and Y"
    # b = "Return X and Y"

    line_length = calc_dist(*line2)
    point1_dist = calc_dist(Car_Pos, line2[0])
    point2_dist = calc_dist(Car_Pos, line2[1])
    if point1_dist > line_length and point2_dist > line_length:
        if point1_dist > max_dist + 10 and point2_dist > max_dist + 10:
            # a = "Return None"
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
        # b = "Return None"
        return None
    else:
        return np.array([x, y])


@jit(nopython=True)
def detect_collisions(
    Car_Pos: np.ndarray,
    Point1: np.ndarray,
    Point2: np.ndarray,
    object1: np.ndarray,
    max_dist: float,
    number_of_collisions: int = 1,
    inc: int = 1,
) -> np.ndarray:  # inc = 1 for continuious lines, inc = 2 for reward gates
    """
    Casts a ray from Point1 to Point2 and checks for intersections with segments
    in the given object (e.g., walls or gates), returning the closest intersection distance.

    Parameters:
        Car_Pos : np.ndarray
            Position of the car, used by the underlying line_intersection function to skip distant lines.
        Point1 : np.ndarray
            Starting point of the ray (typically the car's position).
        Point2 : np.ndarray
            End point of the ray (defines direction and range).
        object1 : np.ndarray
            A list of points defining the segments to test for collision (every pair defines a line).
        max_dist : float
            Maximum distance to consider for collision detection. If no collision is found within this distance,
            the function returns this value.
        number_of_collisions : int, optional
            Number of closest collisions to return (default is 1).
        inc : int, optional
            Step increment to interpret line segments from object1 (default is 1 for continuous walls,
            2 for paired gates).

    Returns:
        np.ndarray
            An array of distances to the closest intersections, sorted in ascending order.
            If no intersections are found, returns an array filled with max_dist.
    """
    distances = np.full(12, max_dist, dtype=np.float32)
    a = 0
    for idx in range(1, len(object1), inc):
        coords = line_intersection(
            Car_Pos, (Point1, Point2), (object1[idx - 1], object1[idx]), max_dist
        )
        if coords is not None:
            distances[a] = calc_dist(Point1, coords)
            a += 1
    distances.sort()
    return distances[:number_of_collisions]


@jit(nopython=True)
def get_angle_to_gate(Car_Pos: np.ndarray, next_gate: np.ndarray) -> float:
    center_pos = np.add(
        next_gate[0], np.divide(np.subtract(next_gate[1], next_gate[0]), 2)
    )

    dist_vector = np.subtract(Car_Pos, center_pos)
    dist_vector[0] *= -1

    absolute_angle = math.atan2(dist_vector[1], dist_vector[0])
    absolute_angle = 180 * absolute_angle / math.pi

    if absolute_angle < 0:
        absolute_angle += 360

    return absolute_angle


@jit(nopython=True)
def get_average_speed(
    number_of_samples: int, average_speed: float, Car_Vel: np.ndarray
) -> float:
    average_speed -= average_speed / number_of_samples
    average_speed += math.sqrt(Car_Vel[0] ** 2 + Car_Vel[1] ** 2) / number_of_samples
    return average_speed
