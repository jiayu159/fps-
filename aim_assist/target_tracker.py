import math
from .config import SCREEN_CENTER_X, SCREEN_CENTER_Y, DEBUG_MODE, MAX_LOST_FRAMES


class TargetTracker:
    def __init__(self, max_lost_frames=MAX_LOST_FRAMES, switch_distance_ratio=0.6):
        self.current_target = None
        self.current_standing = False
        self.lost_count = 0
        self.max_lost_frames = max_lost_frames
        self.switch_ratio = switch_distance_ratio

    def update(self, detections):
        screen_center = (SCREEN_CENTER_X, SCREEN_CENTER_Y)

        if not detections:
            self.lost_count += 1
            if self.current_target is not None:
                if self.lost_count <= self.max_lost_frames:
                    return {'screen_pos': self.current_target, 'conf': 0.0, 'lost': True}
                if DEBUG_MODE:
                    print(f"目标丢失(超过{self.max_lost_frames}帧)")
                self.current_target = None
            return None

        self.lost_count = 0

        if self.current_target is None:
            return self._pick_closest(detections, screen_center)

        tracked = self._find_match(detections)
        if tracked is not None:
            self.current_target = tracked['screen_pos']
            self.current_standing = tracked.get('standing', False)
            return tracked

        better = self._find_significantly_better(detections, screen_center)
        if better is not None:
            if DEBUG_MODE:
                print("切换到更优目标")
            self.current_target = better['screen_pos']
            self.current_standing = better.get('standing', False)
            return better

        return self._pick_closest(detections, screen_center)

    def _find_match(self, detections):
        best = None
        best_dist = float('inf')
        cx, cy = self.current_target
        for det in detections:
            dx = det['screen_pos'][0] - cx
            dy = det['screen_pos'][1] - cy
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best = det
        if best_dist <= 300 * 300:
            return best
        return None

    def _find_significantly_better(self, detections, screen_center):
        cx, cy = screen_center
        current_dist = math.hypot(self.current_target[0] - cx, self.current_target[1] - cy)
        current_standing = self.current_standing
        best = None
        best_dist = float('inf')
        for det in detections:
            dx = det['screen_pos'][0] - cx
            dy = det['screen_pos'][1] - cy
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best = det
        if best is None:
            return None
        if not current_standing and best.get('standing', False):
            if best_dist < current_dist * max(self.switch_ratio, 0.9):
                return best
        if best_dist < current_dist * self.switch_ratio:
            return best
        return None

    def _pick_closest(self, detections, screen_center):
        cx, cy = screen_center
        standing = [d for d in detections if d.get('standing', False)]
        pool = standing if standing else detections
        best = None
        best_dist = float('inf')
        for det in pool:
            dx = det['screen_pos'][0] - cx
            dy = det['screen_pos'][1] - cy
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best = det
        if best is not None:
            self.current_target = best['screen_pos']
            self.current_standing = best.get('standing', False)
        return best

    def reset(self):
        self.current_target = None
        self.current_standing = False
        self.lost_count = 0
