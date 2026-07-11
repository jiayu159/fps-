import cv2
import torch
from .config import TARGET_CLASS, MIN_CONFIDENCE, device, CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS, screen_width, DEBUG_MODE


def is_in_circle(x, y, center_x, center_y, radius):
    dx = x - center_x
    dy = y - center_y
    return dx * dx + dy * dy <= radius * radius


def detect_humans_fixed(frame, region, model):
    x, y, width, height = region

    if frame is None or frame.size == 0:
        return []

    try:
        with torch.no_grad():
            results = model(frame,
                            verbose=False,
                            classes=[TARGET_CLASS],
                            conf=MIN_CONFIDENCE,
                            imgsz=416,
                            device=device)

        all_results = []

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                boxes_np = boxes.xyxy.cpu().numpy()
                confs_np = boxes.conf.cpu().numpy()

                for box, conf in zip(boxes_np, confs_np):
                    x1, y1, x2, y2 = map(int, box)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    x1 = max(0, min(x1, frame.shape[1] - 1))
                    y1 = max(0, min(y1, frame.shape[0] - 1))
                    x2 = max(0, min(x2, frame.shape[1] - 1))
                    y2 = max(0, min(y2, frame.shape[0] - 1))

                    body_height = y2 - y1
                    if body_height <= 0:
                        continue

                    head_ratio = 0.2
                    chest_offset = body_height * head_ratio
                    chest_y = y1 + chest_offset

                    center_x = (x1 + x2) // 2

                    chest_y = max(y1, min(int(chest_y), y2))

                    screen_x = center_x + x
                    screen_y = chest_y + y

                    screen_x = max(0, min(screen_x, screen_width - 1))
                    screen_y = max(0, min(screen_y, screen_width - 1))

                    if is_in_circle(screen_x, screen_y, CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS):
                        all_results.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, chest_y),
                            'screen_pos': (screen_x, screen_y),
                            'conf': conf
                        })

        return all_results

    except Exception as e:
        if DEBUG_MODE:
            print(f"检测错误: {e}")
        return []


def select_target(detections):
    if not detections:
        return None

    detections = [d for d in detections if d['conf'] > MIN_CONFIDENCE]

    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    min_distance_sq = float('inf')
    best_target = None

    for det in detections:
        dx = det['screen_pos'][0] - screen_center_x
        dy = det['screen_pos'][1] - screen_center_y
        distance_sq = dx * dx + dy * dy

        if distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            best_target = det

    return best_target
