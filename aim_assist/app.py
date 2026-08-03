import sys
import os
import time
import ctypes
import cv2
import numpy as np
import torch
import pyautogui
import pydirectinput
import math
import keyboard
from . import config as _cfg

from ultralytics import YOLO

from .config import (
    device, screen_width, screen_height, SCREEN_CENTER_X, SCREEN_CENTER_Y,
    TARGET_CLASS, MIN_CONFIDENCE, SENSITIVITY, CENTER_THRESHOLD,
    CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS,
    MAX_MOVE_DISTANCE, PROCESSING_INTERVAL,
    DXCAM_AVAILABLE, DEBUG_MODE, UPPER_BODY_RATIO, AIM_STABILIZATION_TIME,
    USE_FP16
)
from .capture import DXCamCapture
from .detection import detect_humans_fixed, is_in_circle
from .stabilizer import AimStabilizer
from .target_tracker import TargetTracker
from .movement import get_circle_bounding_box
from .movement import (
    apply_sensitivity_adjustment, validate_movement,
    is_already_aimed, can_move_now, mark_moved, get_distance_scale
)
from .input_handler import start_mouse_listener, stop_mouse_listener, adjust_sensitivity


dxcam_capture = None
model = None

pydirectinput.PAUSE = 0.01
pydirectinput.FAILSAFE = False


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def load_model():
    global model
    print(f"使用设备: {device}")
    model_path = resource_path('yolo11s.pt')
    model = YOLO(model_path).to(device)
    model.fuse()
    if USE_FP16:
        model.model.half()
        print("已启用FP16推理加速")
    print(f"屏幕分辨率: {screen_width}x{screen_height}")
    return model


def cleanup_resources():
    global dxcam_capture
    print("正在清理资源...")

    stop_mouse_listener()

    if dxcam_capture:
        try:
            dxcam_capture.stop()
        except:
            pass

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except:
            pass

    time.sleep(0.1)
    print("资源清理完成")


def verify_coordinate_system():
    print("=== 坐标系验证 ===")
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    print(f"屏幕中心: ({screen_center_x}, {screen_center_y})")

    test_points = [
        (screen_center_x, screen_center_y),
        (screen_center_x + 100, screen_center_y + 100),
        (screen_center_x - 100, screen_center_y - 100)
    ]

    for i, (x, y) in enumerate(test_points):
        print(f"点{i}: ({x}, {y})")
        if is_in_circle(x, y, CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS):
            print(f"  在圆形区域内")
        else:
            print(f"  不在圆形区域内")

    print("=================\n")


def main():
    global dxcam_capture, model

    load_model()

    print("程序启动，按numlock键切换功能开关，按住鼠标左键瞄准，按Alt+C组合键停止")
    print(f"当前灵敏度: {_cfg.SENSITIVITY:.2f} (使用↑/↓键调整)")
    print(f"中心阈值: {CENTER_THRESHOLD}px (小于此值认为已瞄准)")
    print(f"瞄准稳定时间: {AIM_STABILIZATION_TIME}秒")
    print(f"瞄准部位: 胸部以上 (上半身比例: {UPPER_BODY_RATIO * 100}%)")
    print(f"检测区域: 以({CIRCLE_CENTER_X},{CIRCLE_CENTER_Y})为中心, 半径{CIRCLE_RADIUS}px")

    verify_coordinate_system()

    aim_stabilizer = AimStabilizer()
    target_tracker = TargetTracker()

    game_x, game_y, game_width, game_height = get_circle_bounding_box(
        CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS
    )
    print(f"检测区域外接矩形: x={game_x}, y={game_y}, width={game_width}, height={game_height}")

    if DXCAM_AVAILABLE:
        dxcam_capture = DXCamCapture(
            region=(game_x, game_y, game_width, game_height),
            target_fps=60
        )
        if not dxcam_capture.start():
            print("DXCam启动失败，将使用备用截图方法")
            dxcam_capture = None
    else:
        print("DXCam不可用，将使用pyautogui截图")

    frame_count = 0
    inference_count = 0
    start_time = time.time()
    last_fps_print = time.time()
    last_process_time = 0

    aimed_count = 0

    start_mouse_listener()

    print("3秒后准备就绪...")
    time.sleep(3)

    try:
        while True:
            current_time = time.time()

            current_numlock = keyboard.is_pressed('NumLock')
            if current_numlock and not _cfg.last_alt_state:
                _cfg.scan_enabled = not _cfg.scan_enabled
                print(f"扫描功能已{'启用' if _cfg.scan_enabled else '禁用'}")
            _cfg.last_alt_state = current_numlock

            if keyboard.is_pressed("up") or keyboard.is_pressed("down"):
                adjust_sensitivity()

            if _cfg.scan_enabled and _cfg.aim_active and (current_time - last_process_time >= PROCESSING_INTERVAL):
                last_process_time = current_time
                frame = None

                if dxcam_capture and dxcam_capture.is_running:
                    frame = dxcam_capture.get_frame()
                else:
                    try:
                        screenshot = pyautogui.screenshot(
                            region=(game_x, game_y, game_width, game_height)
                        )
                        frame = np.array(screenshot)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"截图失败: {e}")
                        time.sleep(0.01)
                        continue

                if frame is not None:
                    frame_count += 1
                    detections = detect_humans_fixed(
                        frame, (game_x, game_y, game_width, game_height), model
                    )
                    inference_count += 1

                    selected_target = target_tracker.update(detections)

                    if selected_target:
                        if selected_target.get('lost'):
                            aim_stabilizer.reset()
                            aimed_count = 0
                            continue

                        if is_already_aimed(selected_target['screen_pos'], CENTER_THRESHOLD):
                            aimed_count += 1
                            if aimed_count > 10:
                                time.sleep(0.5)
                            continue
                        else:
                            aimed_count = 0
                            aim_stabilizer.reset()

                        if can_move_now():
                            target_x, target_y = selected_target['screen_pos']

                            current_x, current_y = pyautogui.position()

                            dx = target_x - SCREEN_CENTER_X
                            dy = target_y - SCREEN_CENTER_Y

                            if DEBUG_MODE and (abs(dx) > 100 or abs(dy) > 100):
                                print(f"大距离移动: 目标({target_x},{target_y}), 中心({SCREEN_CENTER_X},{SCREEN_CENTER_Y}), "
                                      f"偏移(dx={dx:.1f}, dy={dy:.1f})")

                            dx, dy = apply_sensitivity_adjustment(
                                dx, dy, aim_stabilizer, current_time, (target_x, target_y), False
                            )
                            dx, dy = validate_movement(dx, dy)

                            distance = math.hypot(dx, dy)
                            scale = get_distance_scale(distance)
                            dx *= scale
                            dy *= scale
                            distance = math.hypot(dx, dy)

                            target_dist = math.hypot(
                                target_x - SCREEN_CENTER_X, target_y - SCREEN_CENTER_Y
                            )
                            if target_dist > 0 and distance > target_dist:
                                ratio = target_dist / distance
                                dx *= ratio
                                dy *= ratio
                                distance = target_dist

                            if abs(dy) > 0 and target_y < SCREEN_CENTER_Y:
                                if DEBUG_MODE:
                                    print(f"向上移动: 目标y={target_y}, 中心y={SCREEN_CENTER_Y}, dy={dy:.1f}")

                            if 1 < distance <= MAX_MOVE_DISTANCE:
                                try:
                                    target_mouse_x = current_x + dx
                                    target_mouse_y = current_y + dy

                                    target_mouse_x = max(0, min(target_mouse_x, screen_width - 1))
                                    target_mouse_y = max(0, min(target_mouse_y, screen_width - 1))

                                    move_duration = 0.05

                                    if DEBUG_MODE and distance > 50:
                                        print(f"移动: 从({current_x},{current_y})到({target_mouse_x:.0f},{target_mouse_y:.0f}), "
                                              f"距离={distance:.1f}px")

                                    pydirectinput.moveTo(int(target_mouse_x), int(target_mouse_y), duration=move_duration)
                                    time.sleep(0.03)

                                except Exception as e:
                                    print(f"鼠标移动错误: {str(e)}")

            else:
                aim_stabilizer.reset()
                aimed_count = 0
                time.sleep(0.01)

            if current_time - last_fps_print >= 5.0:
                elapsed = current_time - start_time
                capture_fps = dxcam_capture.get_fps() if dxcam_capture else 0
                print(f"[性能] 截图FPS={capture_fps:.0f} | "
                      f"推理帧={inference_count} | "
                      f"总帧={frame_count} | "
                      f"运行={elapsed:.0f}s")
                last_fps_print = current_time

            if keyboard.is_pressed('alt') and keyboard.is_pressed('c'):
                print("\n检测到停止快捷键")
                break

    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_resources()
        print("\n程序已停止")
