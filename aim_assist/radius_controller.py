import math
from . import config


def update_radius(target_pos=None, no_target_frames=0):
    if target_pos is not None:
        dx = target_pos[0] - config.SCREEN_CENTER_X
        dy = target_pos[1] - config.SCREEN_CENTER_Y
        dist = math.hypot(dx, dy)
        desired = int(dist * 1.8)
        desired = max(config.MIN_CIRCLE_RADIUS, min(config.CIRCLE_RADIUS, desired))
        config.current_radius = config.current_radius + max(1, (desired - config.current_radius) // 4)
    else:
        if no_target_frames > 60:
            config.current_radius = max(config.MIN_CIRCLE_RADIUS, config.current_radius - 2)
    return config.current_radius


def get_bounding_box(center_x, center_y, radius):
    import pyautogui
    sw, sh = pyautogui.size()
    r = max(config.CIRCLE_RADIUS, radius)
    x1 = max(0, center_x - r)
    y1 = max(0, center_y - r)
    x2 = min(sw, center_x + r)
    y2 = min(sh, center_y + r)
    return (x1, y1, x2 - x1, y2 - y1)
