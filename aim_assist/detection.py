import cv2
import math
import torch
import numpy as np
from .config import TARGET_CLASS, MIN_CONFIDENCE, CENTER_CONF_THRESHOLD, device, CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS, screen_width, screen_height, DEBUG_MODE, USE_FP16


def is_in_circle(x, y, center_x, center_y, radius):
    dx = x - center_x
    dy = y - center_y
    return dx * dx + dy * dy <= radius * radius


def preprocess_frame(frame, region, model):
    x, y, width, height = region
    if frame is None or frame.size == 0:
        return None, None

    orig_h, orig_w = frame.shape[:2]
    img = cv2.resize(frame, (416, 416))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if USE_FP16:
        img = img.half()
    img = img.unsqueeze(0)
    return img, (x, y, width, height)


def _dynamic_threshold(distance):
    if distance >= CIRCLE_RADIUS:
        return MIN_CONFIDENCE
    if distance <= 0:
        return CENTER_CONF_THRESHOLD
    ratio = distance / CIRCLE_RADIUS
    return CENTER_CONF_THRESHOLD + (MIN_CONFIDENCE - CENTER_CONF_THRESHOLD) * ratio


def detect_humans_fixed(frame, region, model):
    x, y, width, height = region

    if frame is None or frame.size == 0:
        return []

    try:
        with torch.no_grad():
            results = model(frame, verbose=False, classes=[TARGET_CLASS], conf=CENTER_CONF_THRESHOLD, imgsz=640, device=device, half=True)

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
                    screen_y = max(0, min(screen_y, screen_height - 1))

                    if not is_in_circle(screen_x, screen_y, CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS):
                        continue

                    distance = math.hypot(screen_x - CIRCLE_CENTER_X, screen_y - CIRCLE_CENTER_Y)
                    if conf < _dynamic_threshold(distance):
                        continue

                    all_results.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, chest_y),
                        'screen_pos': (screen_x, screen_y),
                        'conf': conf,
                        'standing': body_height > (x2 - x1)
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
    cx, cy = screen_width // 2, screen_height // 2
    best = min(detections, key=lambda d: (d['screen_pos'][0]-cx)**2 + (d['screen_pos'][1]-cy)**2, default=None)
    return best
