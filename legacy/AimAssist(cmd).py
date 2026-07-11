import cv2
import numpy as np
import time
import keyboard
import win32gui
import win32con
import win32api
import pyautogui
import pydirectinput
import os
from ultralytics import YOLO
import torch
from pynput import mouse
import random
import math
import sys
import threading
from collections import deque
import dxcam

# 设置环境变量，禁止pip版本检查
os.environ['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'

DXCAM_AVAILABLE = True


def resource_path(relative_path):
    """获取资源的绝对路径。用于PyInstaller打包后定位资源文件"""
    try:
        # PyInstaller创建的临时文件夹
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# 检查GPU可用性
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载YOLOv8模型
model_path = resource_path('yolov8n.pt')  # 使用资源路径函数
model = YOLO(model_path).to(device)       # 加载模型
model.fuse()

# 获取屏幕尺寸
screen_width, screen_height = pyautogui.size()
print(f"屏幕分辨率: {screen_width}x{screen_height}")

# 常量缓存
SCREEN_CENTER_X = screen_width // 2
SCREEN_CENTER_Y = screen_height // 2

# 目标类别 (person)
TARGET_CLASS = 0

# 程序状态
scan_enabled = False  # 控制扫描开关
aim_active = False    # 鼠标左键控制瞄准激活
last_alt_state = False  # 记录上一次状态

# 优化参数
MAX_MOVE_DISTANCE = 2000    # 最大有效移动距离(像素)
MIN_CONFIDENCE = 0.6        # 最低置信度阈值
SENSITIVITY = 0.4           # 灵敏度参数 (0.1-1.0)
CENTER_THRESHOLD = 10       # 中心点阈值(像素)，小于此值认为已瞄准

# 瞄准稳定性参数
AIM_STABILIZATION_TIME = 0.5  # 瞄准稳定时间(秒)
STABILIZATION_MOVEMENT_SCALE = 0.2  # 稳定阶段的移动缩放因子(0-1)
MIN_STABILIZATION_FRAMES = 10    # 最小稳定帧数要求

# 圆形检测区域参数
CIRCLE_CENTER_X = screen_width // 2       # 圆形区域中心X坐标（设为屏幕中心）
CIRCLE_CENTER_Y = screen_height // 2      # 圆形区域中心Y坐标（设为屏幕中心）
CIRCLE_RADIUS = 400          # 圆形区域半径

# 防止连续移动
last_move_time = 0
MOVE_COOLDOWN = 0.03  # 调整冷却时间到约33Hz，提高稳定性

# 性能优化参数
PROCESSING_INTERVAL = 0.03  # 处理间隔约33Hz，提高稳定性

# 全局变量声明
mouse_listener = None
dxcam_capture = None

# ---------- 瞄准状态跟踪 ----------
class AimStabilizer:
    """瞄准稳定器，用于平滑瞄准和增加停留时间"""
    def __init__(self):
        self.stabilization_start_time = 0
        self.is_stabilizing = False
        self.stabilization_frame_count = 0
        self.last_target_pos = None
        
    def update_stabilization(self, target_pos, current_time):
        """更新稳定状态"""
        if target_pos is None:
            self.is_stabilizing = False
            self.stabilization_frame_count = 0
            self.last_target_pos = None
            return False
            
        # 检查是否是新目标
        if self.last_target_pos is None:
            self.is_stabilizing = False
            self.last_target_pos = target_pos
            return False
            
        # 计算与上次目标的距离
        dx = target_pos[0] - self.last_target_pos[0]
        dy = target_pos[1] - self.last_target_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 如果目标移动过大，重置稳定状态
        if distance > 20:  # 如果目标移动超过20像素
            self.is_stabilizing = False
            self.stabilization_frame_count = 0
            self.stabilization_start_time = 0
        else:
            # 如果距离很小，开始或继续稳定
            if not self.is_stabilizing:
                self.is_stabilizing = True
                self.stabilization_start_time = current_time
                self.stabilization_frame_count = 1
            else:
                self.stabilization_frame_count += 1
                
        self.last_target_pos = target_pos
        return self.is_stabilizing
        
    def get_stabilized_movement(self, dx, dy, current_time):
        """获取稳定后的移动向量"""
        if not self.is_stabilizing or self.stabilization_frame_count < MIN_STABILIZATION_FRAMES:
            return dx, dy
            
        # 计算稳定时间
        stabilization_time = current_time - self.stabilization_start_time
        
        # 如果还在稳定阶段，减少移动幅度
        if stabilization_time < AIM_STABILIZATION_TIME:
            # 使用缓动函数，随着时间推移逐渐减小移动幅度
            progress = stabilization_time / AIM_STABILIZATION_TIME
            # 指数衰减，开始移动大，后期移动小
            scale_factor = STABILIZATION_MOVEMENT_SCALE * (1.0 - progress * 0.5)
            return dx * scale_factor, dy * scale_factor
            
        # 稳定阶段结束，返回0移动
        return 0, 0
        
    def reset(self):
        """重置稳定器"""
        self.is_stabilizing = False
        self.stabilization_frame_count = 0
        self.stabilization_start_time = 0
        self.last_target_pos = None

# ---------- 目标跟踪与锁定管理器 ----------
class TargetManager:
    def __init__(self, iou_threshold=0.3, lock_frames=15, max_history=50):
        """
        目标管理：为每个目标分配ID，跟踪其位置、置信度、大小和停留时间。
        iou_threshold: 同一目标在两帧之间的最小IoU
        lock_frames: 锁定目标后保持锁定的帧数
        """
        self.iou_threshold = iou_threshold
        self.lock_frames = lock_frames
        self.next_id = 1
        self.tracked_targets = {}  # id -> { 'bbox', 'center', 'screen_pos', 'conf', 'stay_frames', 'last_seen' }
        self.locked_target_id = None
        self.lock_counter = 0
        
        # 稳定性相关
        self.stabilization_start_time = 0
        self.is_stabilizing = False
        self.stabilization_frame_count = 0
        self.last_target_pos = None
        self.aim_stabilization_time = 0.5  # 瞄准稳定时间(秒)
        self.stabilization_movement_scale = 0.2  # 稳定阶段的移动缩放因子(0-1)
        self.min_stabilization_frames = 10    # 最小稳定帧数要求

    def update(self, detections, current_time):
        """
        输入当前帧检测到的所有目标(detections列表，每个元素包含bbox, center, screen_pos, conf)
        更新跟踪列表，并返回当前应该锁定的最佳目标（或None）
        """
        if not detections:
            # 没有检测到目标，清空锁定
            self.locked_target_id = None
            self.lock_counter = 0
            self.reset_stabilization()
            return None

        # 1. 计算每个检测与现有跟踪目标的IoU，进行匹配
        matched_det_ids = set()
        updated_tracked = {}

        # 复制现有跟踪目标，先全部标记为未匹配
        for tid, tinfo in self.tracked_targets.items():
            tinfo['matched'] = False

        # 为每个检测寻找最佳匹配的跟踪目标
        for det in detections:
            best_iou = 0
            best_tid = None
            for tid, tinfo in self.tracked_targets.items():
                iou = self._compute_iou(det['bbox'], tinfo['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                # 匹配成功，更新该目标的信息
                matched_det_ids.add(id(det))  # 使用对象id作为临时标记
                tinfo = self.tracked_targets[best_tid]
                tinfo['bbox'] = det['bbox']
                tinfo['center'] = det['center']
                tinfo['screen_pos'] = det['screen_pos']
                tinfo['conf'] = det['conf']
                tinfo['stay_frames'] = tinfo.get('stay_frames', 0) + 1
                tinfo['last_seen'] = current_time
                tinfo['matched'] = True
                updated_tracked[best_tid] = tinfo
            else:
                # 未匹配，作为新目标
                new_id = self.next_id
                self.next_id += 1
                updated_tracked[new_id] = {
                    'bbox': det['bbox'],
                    'center': det['center'],
                    'screen_pos': det['screen_pos'],
                    'conf': det['conf'],
                    'stay_frames': 1,
                    'last_seen': current_time,
                    'matched': True
                }

        # 2. 对于未匹配的跟踪目标，如果超过一定时间未出现，则移除
        for tid, tinfo in self.tracked_targets.items():
            if not tinfo.get('matched', False):
                # 超过0.5秒未出现则认为消失
                if current_time - tinfo['last_seen'] > 0.5:
                    continue
                # 否则保留，但stay_frames不变
                updated_tracked[tid] = tinfo

        self.tracked_targets = updated_tracked

        # 3. 计算每个跟踪目标的得分，并选择最佳目标
        best_target_info = None
        best_score = -float('inf')

        for tid, tinfo in self.tracked_targets.items():
            # 得分函数：综合考虑置信度、距离中心距离、目标大小、停留时间
            # 距离中心距离（归一化）
            dx = tinfo['screen_pos'][0] - SCREEN_CENTER_X
            dy = tinfo['screen_pos'][1] - SCREEN_CENTER_Y
            distance = math.hypot(dx, dy)
            distance_norm = distance / (CIRCLE_RADIUS * 0.8)  # 归一化因子

            # 目标大小（边界框面积），归一化到0~1之间（假设最大面积为0.1*屏幕面积）
            bbox = tinfo['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            max_area = screen_width * screen_height * 0.1
            area_norm = min(1.0, area / max_area)

            # 停留时间加分（停留越久越优先，最高加0.3）
            stay_bonus = min(0.3, tinfo['stay_frames'] / 50.0)

            # 置信度
            conf = tinfo['conf']

            # 综合得分：置信度权重高，距离惩罚，面积加分，停留加分
            score = conf * 1.5 - distance_norm * 0.8 + area_norm * 0.3 + stay_bonus

            if score > best_score:
                best_score = score
                best_target_info = (tid, tinfo)

        # 4. 目标锁定逻辑
        if best_target_info is None:
            self.locked_target_id = None
            self.lock_counter = 0
            self.reset_stabilization()
            return None

        best_tid, best_tinfo = best_target_info

        if self.locked_target_id is not None and self.lock_counter > 0:
            # 当前有锁定目标，检查锁定目标是否仍然存在
            if self.locked_target_id in self.tracked_targets:
                # 仍然存在，继续锁定
                self.lock_counter -= 1
                return self.tracked_targets[self.locked_target_id]
            else:
                # 锁定目标消失，立即切换到最佳目标
                self.locked_target_id = best_tid
                self.lock_counter = self.lock_frames
                return best_tinfo
        else:
            # 无锁定或锁定已过期，切换到最佳目标
            self.locked_target_id = best_tid
            self.lock_counter = self.lock_frames
            return best_tinfo
    
    def update_stabilization(self, target_pos, current_time):
        """更新稳定状态"""
        if target_pos is None:
            self.reset_stabilization()
            return False
            
        # 检查是否是新目标
        if self.last_target_pos is None:
            self.is_stabilizing = False
            self.last_target_pos = target_pos
            return False
            
        # 计算与上次目标的距离
        dx = target_pos[0] - self.last_target_pos[0]
        dy = target_pos[1] - self.last_target_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 如果目标移动过大，重置稳定状态
        if distance > 20:  # 如果目标移动超过20像素
            self.reset_stabilization()
        else:
            # 如果距离很小，开始或继续稳定
            if not self.is_stabilizing:
                self.is_stabilizing = True
                self.stabilization_start_time = current_time
                self.stabilization_frame_count = 1
            else:
                self.stabilization_frame_count += 1
                
        self.last_target_pos = target_pos
        return self.is_stabilizing
        
    def get_stabilized_movement(self, dx, dy, current_time):
        """获取稳定后的移动向量"""
        if not self.is_stabilizing or self.stabilization_frame_count < self.min_stabilization_frames:
            return dx, dy
            
        # 计算稳定时间
        stabilization_time = current_time - self.stabilization_start_time
        
        # 如果还在稳定阶段，减少移动幅度
        if stabilization_time < self.aim_stabilization_time:
            # 使用缓动函数，随着时间推移逐渐减小移动幅度
            progress = stabilization_time / self.aim_stabilization_time
            # 指数衰减，开始移动大，后期移动小
            scale_factor = self.stabilization_movement_scale * (1.0 - progress * 0.5)
            return dx * scale_factor, dy * scale_factor
            
        # 稳定阶段结束，返回0移动
        return 0, 0
        
    def reset_stabilization(self):
        """重置稳定器"""
        self.is_stabilizing = False
        self.stabilization_frame_count = 0
        self.stabilization_start_time = 0
        self.last_target_pos = None

    def _compute_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0


# ---------- DXCam 高性能截图（优化到100fps） ----------
class DXCamCapture:
    """
    高性能截图类，使用DXCam进行游戏截图，优化到100fps
    """
    def __init__(self, region, target_fps=100):
        self.region = region  # (left, top, width, height)
        self.target_fps = target_fps
        self.dxcam_region = (region[0], region[1], region[0] + region[2], region[1] + region[3])
        self.camera = None
        self.is_running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0
        self.frame_queue = deque(maxlen=2)  # 双缓冲，避免锁竞争

    def start(self):
        if not DXCAM_AVAILABLE:
            print("DXCam不可用，使用备用截图方法")
            return False
        try:
            self.camera = dxcam.create(output_idx=0, output_color="BGRA")
            if self.camera is None:
                print("无法创建DXCam实例")
                return False
            self.camera.start(target_fps=self.target_fps, region=self.dxcam_region)
            self.is_running = True
            self.start_time = time.perf_counter()
            self.frame_count = 0
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            print(f"DXCam截图器已启动 - 区域: {self.region} - 目标FPS: {self.target_fps}")
            return True
        except Exception as e:
            print(f"启动DXCam失败: {e}")
            return False

    def _capture_loop(self):
        while self.is_running:
            try:
                frame = self.camera.get_latest_frame()
                if frame is not None:
                    self.frame_queue.append(frame)
                    self.frame_count += 1
                    elapsed = time.perf_counter() - self.start_time
                    if elapsed > 0:
                        self.fps = self.frame_count / elapsed
            except Exception as e:
                print(f"截图错误: {e}")
                time.sleep(0.001)

    def get_frame(self):
        """获取最新帧，返回BGR格式的numpy数组（零拷贝优化）"""
        if self.frame_queue:
            frame = self.frame_queue.pop()
            if frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return None

    def get_raw_frame(self):
        """获取原始BGRA格式帧（用于需要原始数据的场景）"""
        if self.frame_queue:
            return self.frame_queue.pop()
        return None

    def get_fps(self):
        return self.fps

    def stop(self):
        self.is_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass
        if self.frame_count > 0:
            print(f"DXCam已停止 - 平均FPS: {self.fps:.1f}")

    def __del__(self):
        self.stop()


# ---------- 异步推理类（高性能优化） ----------
class AsyncDetector:
    def __init__(self, model, region, target_fps=60):
        self.model = model
        self.region = region  # (x, y, width, height)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.latest_detections = []
        self.detections_lock = threading.Lock()
        self.running = False
        self.thread = None
        self.capture = None  # 将在start时设置
        self.inference_count = 0
        self.inference_time = 0

    def start(self, capture):
        self.capture = capture
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        print(f"异步推理线程已启动，目标帧率: {self.target_fps}")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _inference_loop(self):
        last_time = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            elapsed = now - last_time
            
            if elapsed < self.frame_interval:
                sleep_time = self.frame_interval - elapsed - 0.001
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue
            
            last_time = now

            if self.capture is None:
                continue

            frame = self.capture.get_frame()
            if frame is None:
                continue

            try:
                start_infer = time.perf_counter()
                detections = detect_humans_optimized(frame, self.region, self.model)
                infer_time = time.perf_counter() - start_infer
                
                with self.detections_lock:
                    self.latest_detections = detections
                
                self.inference_count += 1
                self.inference_time += infer_time
                
            except Exception as e:
                print(f"推理错误: {e}")

    def get_detections(self):
        with self.detections_lock:
            return self.latest_detections if self.latest_detections else []
    
    def get_stats(self):
        if self.inference_count > 0:
            avg_time = (self.inference_time / self.inference_count) * 1000
            return f"推理次数: {self.inference_count}, 平均耗时: {avg_time:.1f}ms"
        return "无推理数据"


# ---------- 检测函数（高性能优化版） ----------
def detect_humans_optimized(frame, region, model):
    """高性能人体检测 - 优化推理速度"""
    x, y, width, height = region
    if frame is None or frame.size == 0:
        return []

    try:
        with torch.no_grad():
            results = model(frame,
                            verbose=False,
                            classes=[TARGET_CLASS],
                            conf=MIN_CONFIDENCE,
                            imgsz=320,
                            device=device,
                            half=True if device == 'cuda' else False,
                            augment=False)

        all_results = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                boxes_np = boxes.xyxy.cpu().numpy()
                confs_np = boxes.conf.cpu().numpy()
                
                for i in range(len(boxes_np)):
                    box = boxes_np[i]
                    conf = confs_np[i]
                    
                    x1, y1, x2, y2 = box.astype(int)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    x1 = max(0, min(x1, frame.shape[1]-1))
                    y1 = max(0, min(y1, frame.shape[0]-1))
                    x2 = max(0, min(x2, frame.shape[1]-1))
                    y2 = max(0, min(y2, frame.shape[0]-1))
                    
                    body_height = y2 - y1
                    if body_height <= 0:
                        continue
                    
                    head_ratio = 0.2
                    chest_offset = body_height * head_ratio
                    chest_y = y1 + chest_offset
                    center_x = (x1 + x2) >> 1  # 位运算替代除法
                    chest_y = max(y1, min(int(chest_y), y2))
                    
                    screen_x = center_x + x
                    screen_y = chest_y + y
                    screen_x = max(0, min(screen_x, screen_width - 1))
                    screen_y = max(0, min(screen_y, screen_height - 1))
                    
                    if is_in_circle(screen_x, screen_y, CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS):
                        all_results.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, chest_y),
                            'screen_pos': (screen_x, screen_y),
                            'conf': float(conf)
                        })
        return all_results
    except Exception as e:
        return []


def detect_humans_fixed(frame, region, model):
    """兼容性检测函数 - 调用优化版本"""
    return detect_humans_optimized(frame, region, model)


def apply_sensitivity_adjustment(dx, dy, stabilizer=None, current_time=None, target_pos=None, is_target_lost=False):
    """应用灵敏度调整到移动向量"""
    # 计算移动距离
    distance = math.sqrt(dx**2 + dy**2)
    
    # 基础灵敏度缩放
    scaled_distance = distance * SENSITIVITY
    
    # 如果缩放后距离为0，则返回原始值
    if scaled_distance < 1:
        return dx, dy
    
    # 计算缩放因子
    scale_factor = scaled_distance / distance
    
    # 应用基础缩放
    scaled_dx = dx * scale_factor
    scaled_dy = dy * scale_factor
    
    # 验证移动向量
    scaled_dx, scaled_dy = validate_movement(scaled_dx, scaled_dy)
    
    # 如果提供了稳定器，应用稳定逻辑
    if stabilizer and current_time and target_pos and not is_target_lost:
        # 更新稳定状态
        stabilizer.update_stabilization(target_pos, current_time)
        
        # 获取稳定后的移动
        stabilized_dx, stabilized_dy = stabilizer.get_stabilized_movement(scaled_dx, scaled_dy, current_time)
        return stabilized_dx, stabilized_dy
    
    return scaled_dx, scaled_dy


def is_in_circle(x, y, center_x, center_y, radius):
    dx = x - center_x
    dy = y - center_y
    return dx*dx + dy*dy <= radius*radius


def get_circle_bounding_box(center_x, center_y, radius):
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(screen_width, center_x + radius)
    y2 = min(screen_height, center_y + radius)
    return (x1, y1, x2 - x1, y2 - y1)


def validate_movement(dx, dy):
    """验证移动向量是否合理"""
    max_vertical_move = 300
    if abs(dy) > abs(dx) * 3:
        max_vertical_move = 60
    if abs(dy) > max_vertical_move:
        scale = max_vertical_move / abs(dy)
        dy = dy * scale
        dx = dx * scale
    if dy < -40:
        dy = max(dy, -60)
    return dx, dy


def cleanup_resources():
    """清理所有资源"""
    global mouse_listener, dxcam_capture, async_detector
    print("正在清理资源...")
    if async_detector:
        async_detector.stop()
    if mouse_listener and mouse_listener.is_alive():
        try:
            mouse_listener.stop()
        except:
            pass
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


def on_click(x, y, button, pressed):
    global aim_active
    if button == mouse.Button.left:
        aim_active = pressed


# 启动鼠标监听器
mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()


def adjust_sensitivity():
    global SENSITIVITY
    if keyboard.is_pressed("up"):
        SENSITIVITY = min(1.0, SENSITIVITY + 0.05)
        print(f"灵敏度增加至: {SENSITIVITY:.2f}")
        time.sleep(0.2)
    elif keyboard.is_pressed("down"):
        SENSITIVITY = max(0.1, SENSITIVITY - 0.05)
        print(f"灵敏度减少至: {SENSITIVITY:.2f}")
        time.sleep(0.2)


def can_move_now():
    global last_move_time
    current_time = time.time()
    return (current_time - last_move_time) >= MOVE_COOLDOWN


def is_already_aimed(target_pos, distance_threshold=CENTER_THRESHOLD):
    dx = target_pos[0] - SCREEN_CENTER_X
    dy = target_pos[1] - SCREEN_CENTER_Y
    return dx*dx + dy*dy <= distance_threshold*distance_threshold


def verify_coordinate_system():
    print("=== 坐标系验证 ===")
    print(f"屏幕中心: ({SCREEN_CENTER_X}, {SCREEN_CENTER_Y})")
    test_points = [(SCREEN_CENTER_X, SCREEN_CENTER_Y),
                   (SCREEN_CENTER_X+100, SCREEN_CENTER_Y+100),
                   (SCREEN_CENTER_X-100, SCREEN_CENTER_Y-100)]
    for i, (x, y) in enumerate(test_points):
        print(f"点{i}: ({x}, {y})", end=' ')
        if is_in_circle(x, y, CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS):
            print("在圆形区域内")
        else:
            print("不在圆形区域内")
    print("=================\n")


def main():
    global scan_enabled, aim_active, last_alt_state, SENSITIVITY, last_move_time, dxcam_capture, async_detector

    print("程序启动，按numlock键切换功能开关，按住鼠标左键瞄准，按Alt+C组合键停止")
    print(f"当前灵敏度: {SENSITIVITY:.2f} (使用↑/↓键调整)")
    print(f"中心阈值: {CENTER_THRESHOLD}px (小于此值认为已瞄准)")
    print(f"检测区域: 以({CIRCLE_CENTER_X},{CIRCLE_CENTER_Y})为中心, 半径{CIRCLE_RADIUS}px的圆形区域")

    verify_coordinate_system()

    pydirectinput.PAUSE = 0.01
    pydirectinput.FAILSAFE = False

    game_x, game_y, game_width, game_height = get_circle_bounding_box(CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS)
    print(f"检测区域外接矩形: x={game_x}, y={game_y}, width={game_width}, height={game_height}")

    # 初始化DXCam截图器 - 目标100fps
    if DXCAM_AVAILABLE:
        dxcam_capture = DXCamCapture(region=(game_x, game_y, game_width, game_height), target_fps=100)
        if not dxcam_capture.start():
            print("DXCam启动失败，将使用备用截图方法")
            dxcam_capture = None
    else:
        print("DXCam不可用，将使用pyautogui截图")
        dxcam_capture = None

    # 初始化异步检测器 - 60fps推理
    async_detector = None
    if dxcam_capture:
        async_detector = AsyncDetector(model, (game_x, game_y, game_width, game_height), target_fps=80)
        async_detector.start(dxcam_capture)
    else:
        print("警告：未使用DXCam，检测将同步进行，可能影响性能")

    # 初始化目标管理器
    target_manager = TargetManager(iou_threshold=0.3, lock_frames=15)
    
    # 初始化瞄准稳定器
    aim_stabilizer = AimStabilizer()

    aimed_count = 0
    last_process_time = time.perf_counter()
    loop_count = 0
    fps_start_time = time.perf_counter()
    current_fps = 0

    print("3秒后准备就绪...")
    time.sleep(3)

    try:
        while True:
            loop_start = time.perf_counter()
            current_time = time.perf_counter()

            # FPS计算
            loop_count += 1
            if current_time - fps_start_time >= 1.0:
                current_fps = loop_count / (current_time - fps_start_time)
                loop_count = 0
                fps_start_time = current_time

            # 检测NumLock键切换开关
            current_alt_pressed = keyboard.is_pressed('NumLock')
            if current_alt_pressed and not last_alt_state:
                scan_enabled = not scan_enabled
                print(f"扫描功能已{'启用' if scan_enabled else '禁用'}")
                if scan_enabled and dxcam_capture:
                    cam_fps = dxcam_capture.get_fps()
                    print(f"当前截图FPS: {cam_fps:.1f}")
            last_alt_state = current_alt_pressed

            # 调整灵敏度
            if keyboard.is_pressed("up") or keyboard.is_pressed("down"):
                adjust_sensitivity()

            if scan_enabled and aim_active and (current_time - last_process_time >= PROCESSING_INTERVAL):
                last_process_time = current_time

                # 获取最新检测结果（异步）
                if async_detector:
                    detections = async_detector.get_detections()
                else:
                    # 同步回退
                    try:
                        screenshot = pyautogui.screenshot(region=(game_x, game_y, game_width, game_height))
                        frame = np.array(screenshot)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        detections = detect_humans_optimized(frame, (game_x, game_y, game_width, game_height), model)
                    except Exception as e:
                        detections = []

                # 使用目标管理器选择最佳目标（含得分和锁定）
                selected_target = target_manager.update(detections, current_time)

                if selected_target:
                    # 检查目标是否已经在中心附近
                    if is_already_aimed(selected_target['screen_pos'], CENTER_THRESHOLD):
                        aimed_count += 1
                        
                        # 当稳定瞄准时，轻微增加冷却时间让瞄准更稳定
                        if aimed_count > 10:  # 连续瞄准超过10帧
                            pass  # 轻微延迟，增加稳定性
                        
                        # 跳过移动操作，保持当前位置
                        continue
                    else:
                        aimed_count = 0  # 重置连续瞄准计数
                        aim_stabilizer.reset()  # 重置稳定器
                    
                    # 检查是否可以移动
                    if can_move_now():
                        # 获取目标屏幕坐标
                        target_x, target_y = selected_target['screen_pos']
                        
                        # 获取当前鼠标位置
                        cursor = win32api.GetCursorPos()
                        current_x, current_y = cursor[0], cursor[1]
                        
                        # 计算需要移动的距离（目标与屏幕中心的偏移）
                        dx = target_x - SCREEN_CENTER_X
                        dy = target_y - SCREEN_CENTER_Y
                        
                        # 应用灵敏度调整和稳定逻辑
                        dx, dy = apply_sensitivity_adjustment(dx, dy, aim_stabilizer, current_time, (target_x, target_y), False)
                        
                        # 再次验证移动向量
                        dx, dy = validate_movement(dx, dy)
                        
                        # 计算实际移动距离
                        distance = math.sqrt(dx**2 + dy**2)
                        
                        # 如果距离非常接近，减少移动幅度
                        if distance < 20:
                            dx *= 0.5
                            dy *= 0.5
                        
                        # 如果距离在有效范围内
                        if 1 < distance <= MAX_MOVE_DISTANCE:
                            try:
                                # 计算目标鼠标位置（当前鼠标位置 + 偏移）
                                target_mouse_x = current_x + dx
                                target_mouse_y = current_y + dy
                                
                                # 最终边界检查
                                target_mouse_x = max(0, min(target_mouse_x, screen_width - 1))
                                target_mouse_y = max(0, min(target_mouse_y, screen_height - 1))
                                
                                # 一次性移动到目标位置（带平滑过渡）
                                move_duration = 0.05
                                
                                pydirectinput.moveTo(int(target_mouse_x), int(target_mouse_y), duration=move_duration)
                                time.sleep(move_duration + 0.03)  # 确保移动完成
                                
                                last_move_time = current_time
                            except Exception as e:
                                pass
            elif not (scan_enabled and aim_active):
                aimed_count = 0
                aim_stabilizer.reset()  # 重置稳定器

            # 退出检测
            if keyboard.is_pressed('alt') and keyboard.is_pressed('c'):
                print("\n检测到停止快捷键")
                break
            
            # 精确的帧率控制
            loop_elapsed = time.perf_counter() - loop_start
            target_frame_time = 1.0 / 120  # 目标120Hz主循环
            if loop_elapsed < target_frame_time:
                sleep_time = target_frame_time - loop_elapsed - 0.001
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if async_detector:
            print(f"\n{async_detector.get_stats()}")
        cleanup_resources()
        print("\n程序已停止")


if __name__ == "__main__":
    # 全局变量声明
    async_detector = None
    main()