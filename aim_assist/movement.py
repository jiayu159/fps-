import math
import time
from .config import SENSITIVITY, MAX_MOVE_DISTANCE, screen_width, CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS, SCREEN_CENTER_X, SCREEN_CENTER_Y, CENTER_THRESHOLD, DEBUG_MODE

MOVE_COOLDOWN = 0.016
last_move_time = 0.0


def can_move_now():
    global last_move_time
    current_time = time.time()
    return (current_time - last_move_time) >= MOVE_COOLDOWN


def mark_moved():
    global last_move_time
    last_move_time = time.time()


def apply_sensitivity_adjustment(dx, dy, stabilizer=None, current_time=None, target_pos=None, is_target_lost=False):
    distance = math.hypot(dx, dy)
    scaled_distance = distance * SENSITIVITY

    if scaled_distance < 1:
        return dx, dy

    scale_factor = scaled_distance / distance
    scaled_dx = dx * scale_factor
    scaled_dy = dy * scale_factor

    scaled_dx, scaled_dy = validate_movement(scaled_dx, scaled_dy)

    if stabilizer and current_time and target_pos and not is_target_lost:
        stabilizer.update_stabilization(target_pos, current_time)
        stabilized_dx, stabilized_dy = stabilizer.get_stabilized_movement(scaled_dx, scaled_dy, current_time)
        return stabilized_dx, stabilized_dy

    return scaled_dx, scaled_dy


def validate_movement(dx, dy):
    if abs(dy) > abs(dx) * 3:
        if DEBUG_MODE and abs(dy) > 50:
            print(f"可疑垂直移动: dx={dx:.1f}, dy={dy:.1f}, 比例={abs(dy / dx) if dx != 0 else 'inf'}")

    max_vertical_move = 60
    if abs(dy) > max_vertical_move:
        scale = max_vertical_move / abs(dy)
        dy = dy * scale
        dx = dx * scale
        if DEBUG_MODE:
            print(f"垂直移动限制: 原始dy={dy:.1f}, 缩放后={dy * scale:.1f}")

    if dy < -40:
        if DEBUG_MODE:
            print(f"可疑上拉: dy={dy:.1f}")
        dy = max(dy, -60)

    return dx, dy


def is_in_circle(x, y, center_x, center_y, radius):
    dx = x - center_x
    dy = y - center_y
    return dx * dx + dy * dy <= radius * radius


def get_circle_bounding_box(center_x, center_y, radius):
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(screen_width, center_x + radius)
    y2 = min(screen_height, center_y + radius)
    return (x1, y1, x2 - x1, y2 - y1)


def is_already_aimed(target_pos, distance_threshold=CENTER_THRESHOLD):
    dx = target_pos[0] - SCREEN_CENTER_X
    dy = target_pos[1] - SCREEN_CENTER_Y
    distance_sq = dx * dx + dy * dy
    return distance_sq <= distance_threshold * distance_threshold
