import cv2
import numpy as np
import time
import keyboard
import win32gui
import win32con
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

# 调试模式设置
DEBUG_MODE = False  # 设为True开启调试信息，False关闭

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
model_path = resource_path('yolov8m.pt')  # 使用资源路径函数
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
SENSITIVITY = 0.4       # 灵敏度参数 (0.1-1.0)
CENTER_THRESHOLD = 10      # 中心点阈值(像素)，小于此值认为已瞄准

# 瞄准稳定性参数
AIM_STABILIZATION_TIME = 0.5  # 瞄准稳定时间(秒)，增加这个值可以让瞄准更稳定
STABILIZATION_MOVEMENT_SCALE = 0.2  # 稳定阶段的移动缩放因子(0-1)
MIN_STABILIZATION_FRAMES = 10    # 最小稳定帧数要求

# 上半身比例 (从头部到胸部)
UPPER_BODY_RATIO = 0.5       # 上半身占整个身体高度的比例

# 圆形检测区域参数
CIRCLE_CENTER_X = screen_width // 2       # 圆形区域中心X坐标（设为屏幕中心）
CIRCLE_CENTER_Y = screen_height // 2      # 圆形区域中心Y坐标（设为屏幕中心）
CIRCLE_RADIUS = 200          # 圆形区域半径

# 防止连续移动
last_move_time = 0
MOVE_COOLDOWN = 0.016  # 冷却时间

# 性能优化参数
PROCESSING_INTERVAL = 0.016 # 处理间隔增加，提高帧率稳定性



# 全局变量声明
mouse_listener = None
dxcam_capture = None

# 瞄准状态跟踪
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

class DXCamCapture:
    """
    高性能截图类，使用DXCam进行游戏截图
    """
    def __init__(self, region, target_fps=45):  # 降低目标FPS以减少CPU占用
        """
        初始化DXCam截图器
        
        参数：
        region: (left, top, width, height) 截图区域
        target_fps: 目标截图帧率
        """
        self.region = region  # (left, top, width, height)
        self.target_fps = target_fps
        
        # 转换区域格式：DXCam使用(left, top, right, bottom)
        self.dxcam_region = (
            region[0], 
            region[1], 
            region[0] + region[2], 
            region[1] + region[3]
        )
        
        # 初始化变量
        self.camera = None
        self.is_running = False
        self.frame_buffer = deque(maxlen=3)  # 保存最近3帧
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        
        # 统计信息
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        
    def start(self):
        """启动截图线程"""
        if not DXCAM_AVAILABLE:
            print("DXCam不可用，使用备用截图方法")
            return False
            
        try:
            # 创建DXCam实例
            self.camera = dxcam.create()
            if self.camera is None:
                print("无法创建DXCam实例")
                return False
                
            # 启动摄像头
            self.camera.start(target_fps=self.target_fps, region=self.dxcam_region)
            
            # 启动截图线程
            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            print(f"DXCam截图器已启动 - 区域: {self.region} - 目标FPS: {self.target_fps}")
            return True
            
        except Exception as e:
            print(f"启动DXCam失败: {e}")
            return False
    
    def _capture_loop(self):
        """截图线程主循环"""
        while self.is_running:
            try:
                # 获取最新帧（非阻塞）
                frame = self.camera.get_latest_frame()
                
                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.frame_buffer.append({
                            'frame': frame.copy(),  # 深拷贝
                            'timestamp': time.time()
                        })
                        self.frame_count += 1
                    
                    # 计算FPS
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        self.fps = self.frame_count / elapsed
                
                # 微小睡眠避免CPU占用过高
                time.sleep(0.001)
                
            except Exception as e:
                print(f"截图错误: {e}")
                time.sleep(0.01)
    
    def get_frame(self, wait_for_new=True, timeout=0.1):
        """
        获取最新帧
        
        参数：
        wait_for_new: 是否等待新帧
        timeout: 等待超时时间（秒）
        """
        if not self.is_running or self.latest_frame is None:
            return None
            
        if wait_for_new:
            # 记录当前缓冲区大小
            start_size = len(self.frame_buffer)
            start_time = time.time()
            
            # 等待新帧
            while len(self.frame_buffer) <= start_size:
                if time.time() - start_time > timeout:
                    break
                time.sleep(0.001)
        
        with self.frame_lock:
            if self.latest_frame is not None:
                # DXCam返回BGRA格式，转换为BGR
                frame_bgr = cv2.cvtColor(self.latest_frame.copy(), cv2.COLOR_BGRA2BGR)
                return frame_bgr
        return None
    
    def get_fps(self):
        """获取当前截图FPS"""
        return self.fps
    
    def stop(self):
        """停止截图器"""
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
        """析构函数"""
        self.stop()

def validate_movement(dx, dy):
    """验证移动向量是否合理"""
    # 检查移动方向是否异常
    if abs(dy) > abs(dx) * 3:  # 如果垂直移动远大于水平移动
        if DEBUG_MODE and abs(dy) > 50:  # 且垂直移动较大
            print(f"可疑垂直移动: dx={dx:.1f}, dy={dy:.1f}, 比例={abs(dy/dx) if dx != 0 else 'inf'}")
    
    # 限制最大垂直移动
    max_vertical_move = 60  # 最大垂直移动像素
    if abs(dy) > max_vertical_move:
        scale = max_vertical_move / abs(dy)
        dy = dy * scale
        dx = dx * scale
        if DEBUG_MODE:
            print(f"垂直移动限制: 原始dy={dy:.1f}, 缩放后={dy*scale:.1f}")
    
    # 检查是否向上移动过大
    if dy < -40:  # 如果向上移动超过40像素
        if DEBUG_MODE:
            print(f"可疑上拉: dy={dy:.1f}")
        # 进一步限制上拉
        dy = max(dy, -60)  # 限制最大上拉为60像素
    
    return dx, dy

def cleanup_resources():
    """清理所有资源"""
    global mouse_listener, dxcam_capture
    
    print("正在清理资源...")
    
    # 1. 停止鼠标监听器
    if mouse_listener and mouse_listener.is_alive():
        try:
            mouse_listener.stop()
            print("鼠标监听器已停止")
        except Exception as e:
            print(f"停止鼠标监听器时出错: {e}")
    
    # 2. 停止DXCam截图器
    if dxcam_capture:
        try:
            dxcam_capture.stop()
            print("DXCam截图器已停止")
        except Exception as e:
            print(f"停止DXCam截图器时出错: {e}")
    
    # 3. 清理模型资源
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print("GPU缓存已清理")
        except Exception as e:
            print(f"清理GPU缓存时出错: {e}")
    
    # 4. 等待一小段时间让资源释放
    time.sleep(0.1)
    print("资源清理完成")

def on_click(x, y, button, pressed):
    """鼠标点击回调函数 - 左键按住激活瞄准"""
    global aim_active
    if button == mouse.Button.left:  # 处理左键
        aim_active = pressed  # 按下时激活，释放时关闭
        

# 启动鼠标监听器
mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()

def is_in_circle(x, y, center_x, center_y, radius):
    """检查点(x,y)是否在以(center_x, center_y)为圆心、radius为半径的圆内"""
    # 使用平方比较避免开销较大的 sqrt
    dx = x - center_x
    dy = y - center_y
    return dx*dx + dy*dy <= radius*radius

def get_circle_bounding_box(center_x, center_y, radius):
    """获取圆形区域的外接矩形"""
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(screen_width, center_x + radius)
    y2 = min(screen_height, center_y + radius)
    return (x1, y1, x2 - x1, y2 - y1)

def detect_humans_fixed(frame, region, model):
    """修复版人体检测 - 简化版本以提高性能"""
    x, y, width, height = region
    
    if frame is None or frame.size == 0:
        return []
    
    try:
        # 使用YOLO检测，使用固定推理尺寸以提高性能
        with torch.no_grad():
            results = model(frame,
                            verbose=False,
                            classes=[TARGET_CLASS],
                            conf=MIN_CONFIDENCE,
                            imgsz=416,  # 更轻量的推理尺寸
                            device=device)
        
        all_results = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                boxes_np = boxes.xyxy.cpu().numpy()
                confs_np = boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes_np, confs_np):
                    # 转换为整数坐标
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 确保边界框有效
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # 边界检查
                    x1 = max(0, min(x1, frame.shape[1]-1))
                    y1 = max(0, min(y1, frame.shape[0]-1))
                    x2 = max(0, min(x2, frame.shape[1]-1))
                    y2 = max(0, min(y2, frame.shape[0]-1))
                    
                    # 计算上半身中心点
                    body_height = y2 - y1
                    if body_height <= 0:
                        continue
                    
                    # 使用百分比计算胸部位置
                    head_ratio = 0.2 # 头部区域占整个身体高度的比例
                    chest_offset = body_height * head_ratio
                    chest_y = y1 + chest_offset
                    
                    # 身体中心X坐标
                    center_x = (x1 + x2) // 2
                    
                    # 确保胸部位置在边界框内
                    chest_y = max(y1, min(int(chest_y), y2))
                    
                    # 转换为屏幕坐标
                    screen_x = center_x + x
                    screen_y = chest_y + y
                    
                    # 屏幕边界检查
                    screen_x = max(0, min(screen_x, screen_width - 1))
                    screen_y = max(0, min(screen_y, screen_height - 1))
                    
                    # 检查是否在圆形区域内
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
    """直接选择最接近屏幕中心的目标"""
    if not detections:
        return None
    
    # 只考虑高置信度的目标
    detections = [d for d in detections if d['conf'] > MIN_CONFIDENCE]
    
    # 选择最中心的目标
    screen_center_x = SCREEN_CENTER_X
    screen_center_y = SCREEN_CENTER_Y
    min_distance_sq = float('inf')
    best_target = None
    
    for det in detections:
        # 计算与屏幕中心的平方距离（避免 sqrt）
        dx = det['screen_pos'][0] - screen_center_x
        dy = det['screen_pos'][1] - screen_center_y
        distance_sq = dx*dx + dy*dy
        
        # 选择最接近屏幕中心的目标
        if distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            best_target = det
    
    return best_target

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

def adjust_sensitivity():
    """调整灵敏度参数"""
    global SENSITIVITY
    
    # 增加灵敏度
    if keyboard.is_pressed("up"):
        SENSITIVITY = min(1.0, SENSITIVITY + 0.05)
        print(f"灵敏度增加至: {SENSITIVITY:.2f}")
        time.sleep(0.2)  # 防止快速连续调整
    
    # 减少灵敏度
    elif keyboard.is_pressed("down"):
        SENSITIVITY = max(0.1, SENSITIVITY - 0.05)
        print(f"灵敏度减少至: {SENSITIVITY:.2f}")
        time.sleep(0.2)  # 防止快速连续调整

def can_move_now():
    """检查是否可以进行移动操作"""
    global last_move_time
    current_time = time.time()
    return (current_time - last_move_time) >= MOVE_COOLDOWN

def is_already_aimed(target_pos, distance_threshold=CENTER_THRESHOLD):
    """检查目标是否已经在屏幕中心附近"""
    # 计算目标与屏幕中心的距离
    dx = target_pos[0] - SCREEN_CENTER_X
    dy = target_pos[1] - SCREEN_CENTER_Y
    distance_sq = dx*dx + dy*dy
    
    # 如果距离小于阈值，则认为已经瞄准（使用平方比较避免 sqrt）
    return distance_sq <= distance_threshold * distance_threshold

def verify_coordinate_system():
    """验证坐标系是否正确"""
    print("=== 坐标系验证 ===")
    
    # 获取屏幕中心
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    print(f"屏幕中心: ({screen_center_x}, {screen_center_y})")
    
    # 测试几个点
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
    global scan_enabled, aim_active, last_alt_state, SENSITIVITY, last_move_time, dxcam_capture
    
    print("程序启动，按numlock键切换功能开关，按住鼠标R键瞄准，按Alt+C组合键停止")
    print(f"当前灵敏度: {SENSITIVITY:.2f} (使用↑/↓键调整)")
    print(f"中心阈值: {CENTER_THRESHOLD}px (小于此值认为已瞄准)")
    print(f"瞄准稳定时间: {AIM_STABILIZATION_TIME}秒")
    print(f"瞄准部位: 胸部以上 (上半身比例: {UPPER_BODY_RATIO*100}%)")
    print(f"检测区域: 以({CIRCLE_CENTER_X},{CIRCLE_CENTER_Y})为中心, 半径{CIRCLE_RADIUS}px的圆形区域")
    
    # 验证坐标系
    verify_coordinate_system()
    
    # 初始化瞄准稳定器
    aim_stabilizer = AimStabilizer()
    
    # 设置输入参数
    pydirectinput.PAUSE = 0.01
    pydirectinput.FAILSAFE = False
    
    # 获取圆形区域的外接矩形
    game_x, game_y, game_width, game_height = get_circle_bounding_box(CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS)
    print(f"检测区域外接矩形: x={game_x}, y={game_y}, width={game_width}, height={game_height}")
    
    # 初始化DXCam截图器
    if DXCAM_AVAILABLE:
        dxcam_capture = DXCamCapture(
            region=(game_x, game_y, game_width, game_height),
            target_fps=50
        )
        if not dxcam_capture.start():
            print("DXCam启动失败，将使用备用截图方法")
            dxcam_capture = None
    else:
        print("DXCam不可用，将使用pyautogui截图")
    
    # 帧率统计
    frame_count = 0
    start_time = time.time()
    last_fps_update = time.time()
    last_process_time = 0
    
    # 瞄准状态跟踪
    aimed_count = 0  # 连续瞄准计数
    
    # 帧跳过计数器
    frame_skip_counter = 0
    FRAME_SKIP = 2  # 每2帧处理1帧
    
    print("3秒后准备就绪...")
    time.sleep(3)
    
    try:
        while True:
            current_time = time.time()
            
            # 帧跳过逻辑
            frame_skip_counter += 1
            if frame_skip_counter % FRAME_SKIP != 0:
                time.sleep(0.005)
                continue

            # 检测NumLock键按下事件（切换功能开关）
            current_alt_pressed = keyboard.is_pressed('NumLock')
            if current_alt_pressed and not last_alt_state:
                scan_enabled = not scan_enabled
                print(f"扫描功能已{'启用' if scan_enabled else '禁用'}")
            last_alt_state = current_alt_pressed
            
            # 检查灵敏度调整
            if keyboard.is_pressed("up") or keyboard.is_pressed("down"):
                adjust_sensitivity()
            
            # 检查激活状态
            if scan_enabled and aim_active and (current_time - last_process_time >= PROCESSING_INTERVAL):
                last_process_time = current_time
                frame = None
                
                # 使用DXCam截图（如果可用）
                if dxcam_capture and dxcam_capture.is_running:
                    frame = dxcam_capture.get_frame(wait_for_new=False)
                else:
                    # 备用方法：使用pyautogui截图
                    try:
                        screenshot = pyautogui.screenshot(region=(game_x, game_y, game_width, game_height))
                        frame = np.array(screenshot)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"截图失败: {e}")
                        time.sleep(0.1)
                        continue
                
                if frame is not None:
                    # 检测人体 - 使用修复版函数
                    detections = detect_humans_fixed(frame, (game_x, game_y, game_width, game_height), model)
                    
                    # 选择目标
                    selected_target = select_target(detections)
    
                    # 如果找到目标
                    if selected_target:
                        # 检查目标是否已经在中心附近
                        if is_already_aimed(selected_target['screen_pos'], CENTER_THRESHOLD):
                            aimed_count += 1
                            
                            # 当稳定瞄准时，轻微增加冷却时间让瞄准更稳定
                            if aimed_count > 10:  # 连续瞄准超过10帧
                                time.sleep(0.5)  # 轻微延迟，增加稳定性
                            
                            # 每秒更新一次状态
                            # if current_time - last_fps_update >= 1.0:
                            #     # 获取截图FPS
                            #     capture_fps = dxcam_capture.get_fps() if dxcam_capture else 0
                                
                            #     # 计算处理FPS
                            #     fps = frame_count / (current_time - start_time)
                                
                            #     # 显示状态
                            #     status_msg = f"已稳定瞄准 | 稳定帧数: {aimed_count} | 灵敏度: {SENSITIVITY:.2f}"
                            #     if capture_fps > 0:
                            #         status_msg += f" | 截图FPS: {capture_fps:.1f}"
                            #     status_msg += f" | 处理FPS: {fps:.1f}"
                            #     print(status_msg)
                                
                            #     # 重置计数
                            #     frame_count = 0
                            #     start_time = current_time
                            #     last_fps_update = current_time
                            
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
                            current_x, current_y = pyautogui.position()
                            
                            # 计算需要移动的距离（目标与屏幕中心的偏移）
                            dx = target_x - SCREEN_CENTER_X
                            dy = target_y - SCREEN_CENTER_Y
                            
                            # 调试输出
                            if DEBUG_MODE and (abs(dx) > 100 or abs(dy) > 100):
                                print(f"大距离移动: 目标({target_x},{target_y}), 中心({SCREEN_CENTER_X},{SCREEN_CENTER_Y}), "
                                      f"偏移(dx={dx:.1f}, dy={dy:.1f})")
                            
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
                            
                            # 确保移动方向合理
                            if abs(dy) > 0 and target_y < SCREEN_CENTER_Y:
                                # 如果目标是向上的，检查是否合理
                                if DEBUG_MODE:
                                    print(f"向上移动: 目标y={target_y}, 中心y={SCREEN_CENTER_Y}, dy={dy:.1f}")
                            
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
                                    
                                    if DEBUG_MODE and distance > 50:
                                        print(f"移动: 从({current_x},{current_y})到({target_mouse_x:.0f},{target_mouse_y:.0f}), "
                                              f"距离={distance:.1f}px")
                                    
                                    pydirectinput.moveTo(int(target_mouse_x), int(target_mouse_y), duration=move_duration)
                                    time.sleep(move_duration + 0.05)  # 确保移动完成

                                    
                                    # # 帧计数
                                    # frame_count += 1
                                    
                                    # # 每秒更新一次状态
                                    # if current_time - last_fps_update >= 1.0:
                                    #     # 获取截图FPS
                                    #     capture_fps = dxcam_capture.get_fps() if dxcam_capture else 0
                                        
                                    #     # 计算处理FPS
                                    #     fps = frame_count / (current_time - start_time)
                                        
                                    #     # 显示状态
                                    #     status_msg = f"目标锁定 | 距离: {distance:.1f}px | 灵敏度: {SENSITIVITY:.2f}"
                                    #     if capture_fps > 0:
                                    #         status_msg += f" | 截图FPS: {capture_fps:.1f}"
                                    #     status_msg += f" | 处理FPS: {fps:.1f}"
                                    #     print(status_msg)
                                        
                                    #     # 重置计数
                                    #     frame_count = 0
                                    #     start_time = current_time
                                    #     last_fps_update = current_time
                                except Exception as e:
                                    print(f"鼠标移动错误: {str(e)}")
                        
                        # 帧计数
                       # frame_count += 1
                        
                        # 每秒更新一次状态
                    #     if current_time - last_fps_update >= 1.0:
                    #         # 获取截图FPS
                    #         capture_fps = dxcam_capture.get_fps() if dxcam_capture else 0
                            
                    #         # 计算处理FPS
                    #         fps = frame_count / (current_time - start_time)
                            
                    #         # 显示状态
                    #         status_msg = f"扫描中... | 灵敏度: {SENSITIVITY:.2f}"
                    #         if capture_fps > 0:
                    #             status_msg += f" | 截图FPS: {capture_fps:.1f}"
                    #         status_msg += f" | 处理FPS: {fps:.1f}"
                    #         print(status_msg)
                            
                    #         # 重置计数
                    #         frame_count = 0
                    #         start_time = current_time
                    #         last_fps_update = current_time
                    
                    # # 更新最后处理时间
                    # last_process_time = current_time
                    
            else:
                # 没有激活时等待并重置稳定器
                aim_stabilizer.reset()
                aimed_count = 0
                time.sleep(0.01)
                
                # 显示状态信息
                # if current_time - last_fps_update >= 1.0:
                #     status = "等待激活"
                #     if scan_enabled and not aim_active:
                #         status = "扫描已启用，等待鼠标左键"
                #     elif not scan_enabled:
                #         status = "扫描已禁用，按numlock键启用"
                    
                #     # 获取截图FPS
                #     capture_fps = dxcam_capture.get_fps() if dxcam_capture else 0
                    
                #     status_msg = f"状态: {status} | 灵敏度: {SENSITIVITY:.2f}"
                #     if capture_fps > 0:
                #         status_msg += f" | 截图FPS: {capture_fps:.1f}"
                #     print(status_msg)
                #     last_fps_update = current_time
            
            # 退出检测
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
        # 确保清理资源
        cleanup_resources()
        
        print("\n程序已停止")

if __name__ == "__main__":
    main()