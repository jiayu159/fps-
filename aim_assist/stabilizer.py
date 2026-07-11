import math
from .config import AIM_STABILIZATION_TIME, STABILIZATION_MOVEMENT_SCALE, MIN_STABILIZATION_FRAMES


class AimStabilizer:
    def __init__(self):
        self.stabilization_start_time = 0
        self.is_stabilizing = False
        self.stabilization_frame_count = 0
        self.last_target_pos = None

    def update_stabilization(self, target_pos, current_time):
        if target_pos is None:
            self.is_stabilizing = False
            self.stabilization_frame_count = 0
            self.last_target_pos = None
            return False

        if self.last_target_pos is None:
            self.is_stabilizing = False
            self.last_target_pos = target_pos
            return False

        dx = target_pos[0] - self.last_target_pos[0]
        dy = target_pos[1] - self.last_target_pos[1]
        distance = math.hypot(dx, dy)

        if distance > 20:
            self.is_stabilizing = False
            self.stabilization_frame_count = 0
            self.stabilization_start_time = 0
        else:
            if not self.is_stabilizing:
                self.is_stabilizing = True
                self.stabilization_start_time = current_time
                self.stabilization_frame_count = 1
            else:
                self.stabilization_frame_count += 1

        self.last_target_pos = target_pos
        return self.is_stabilizing

    def get_stabilized_movement(self, dx, dy, current_time):
        if not self.is_stabilizing or self.stabilization_frame_count < MIN_STABILIZATION_FRAMES:
            return dx, dy

        stabilization_time = current_time - self.stabilization_start_time

        if stabilization_time < AIM_STABILIZATION_TIME:
            progress = stabilization_time / AIM_STABILIZATION_TIME
            scale_factor = STABILIZATION_MOVEMENT_SCALE * (1.0 - progress * 0.5)
            return dx * scale_factor, dy * scale_factor

        return 0, 0

    def reset(self):
        self.is_stabilizing = False
        self.stabilization_frame_count = 0
        self.stabilization_start_time = 0
        self.last_target_pos = None
