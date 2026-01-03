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
import atexit
import warnings

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
model_path = resource_path('yolov8m.pt')  # 使用资源路径函数
model = YOLO(model_path).to(device)       # 加载模型
model.fuse()  

# 获取屏幕尺寸
screen_width, screen_height = pyautogui.size()
print(f"屏幕分辨率: {screen_width}x{screen_height}")

# 目标类别 (person)
TARGET_CLASS = 0

# 程序状态
scan_enabled = False  # Alt键控制扫描开关
aim_active = False    # 鼠标左键控制瞄准激活
last_alt_state = False  # 记录上一次Alt键状态

# 优化参数
MAX_MOVE_DISTANCE = 2000    # 最大有效移动距离(像素)
MIN_CONFIDENCE = 0.7        # 最低置信度阈值
SENSITIVITY = 0.30           # 灵敏度参数 (0.1-1.0)
CENTER_THRESHOLD = 50      # 中心点阈值(像素)，小于此值认为已瞄准

# 上半身比例 (从头部到胸部)
UPPER_BODY_RATIO = 0.3       # 上半身占整个身体高度的比例，越小越锁头

# 圆形检测区域参数
CIRCLE_CENTER_X = 1280       # 圆形区域中心X坐标
CIRCLE_CENTER_Y = 800        # 圆形区域中心Y坐标
CIRCLE_RADIUS = 600          # 圆形区域半径

# 防止连续移动
last_move_time = 0
MOVE_COOLDOWN = 0.20  # 冷却时间

# 性能优化参数
PROCESSING_INTERVAL = 0.02  # 处理间隔(秒)，控制处理频率

class DXCamCapture:
    """
    高性能截图类，使用DXCam进行游戏截图
    """
    def __init__(self, region, target_fps=60):
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

def on_click(x, y, button, pressed):
    """鼠标点击回调函数 - 左键按住激活瞄准"""
    global aim_active
    if button == mouse.Button.left:  # 处理左键
        aim_active = pressed  # 按下时激活，释放时关闭
        if scan_enabled:  # 只在功能启用时显示状态
            status = "激活" if pressed else "关闭"
            print(f"瞄准系统已{status}")

# 启动鼠标监听器
mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()

def is_in_circle(x, y, center_x, center_y, radius):
    """检查点(x,y)是否在以(center_x, center_y)为圆心、radius为半径的圆内"""
    distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
    return distance <= radius

def get_circle_bounding_box(center_x, center_y, radius):
    """获取圆形区域的外接矩形"""
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(screen_width, center_x + radius)
    y2 = min(screen_height, center_y + radius)
    return (x1, y1, x2 - x1, y2 - y1)

def detect_humans(frame, region, model):
    """使用YOLO检测人体并计算上半身中心点"""
    x, y, width, height = region
    
    # 多尺度检测 - 创建不同尺度的图像
    scales = [1.0, 0.5]  # 原始尺寸和缩小尺寸
    all_results = []
    
    for scale in scales:
        # 调整图像尺寸
        if scale != 1.0:
            scaled_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            scaled_frame = frame
            
        # 使用YOLO检测
        results = model(scaled_frame, verbose=False, classes=[TARGET_CLASS])
        
        # 处理检测结果
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(boxes, confs, clss):
                if conf > MIN_CONFIDENCE:
                    # 调整框坐标到原始尺寸
                    if scale != 1.0:
                        box = box / scale
                    
                    # 转换为整数坐标
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, frame.shape[1]-1))
                    y1 = max(0, min(y1, frame.shape[0]-1))
                    x2 = max(0, min(x2, frame.shape[1]-1))
                    y2 = max(0, min(y2, frame.shape[0]-1))
                    
                    # 计算上半身中心点（胸部位置）
                    upper_height = (y2 - y1) * UPPER_BODY_RATIO
                    upper_y = y1 + upper_height / 2
                    cx = (x1 + x2) // 2
                    cy = int(upper_y)
                    
                    # 转换为屏幕坐标
                    screen_x = cx + x
                    screen_y = cy + y
                    
                    # 确保坐标在屏幕范围内
                    screen_x = max(0, min(screen_x, screen_width - 1))
                    screen_y = max(0, min(screen_y, screen_height - 1))
                    
                    # 检查是否在圆形区域内
                    if is_in_circle(screen_x, screen_y, CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS):
                        # 保存结果
                        all_results.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (cx, cy),
                            'screen_pos': (screen_x, screen_y),
                            'conf': conf
                        })
    
    return all_results

def select_target(detections):
    """直接选择最接近屏幕中心的目标"""
    if not detections:
        return None
    
    # 只考虑高置信度的目标
    detections = [d for d in detections if d['conf'] > MIN_CONFIDENCE]
    
    # 选择最中心的目标
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    min_distance = float('inf')
    best_target = None
    
    for det in detections:
        # 计算与屏幕中心的距离
        dx = det['screen_pos'][0] - screen_center_x
        dy = det['screen_pos'][1] - screen_center_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # 选择最接近屏幕中心的目标
        if distance < min_distance:
            min_distance = distance
            best_target = det
    
    return best_target

def apply_sensitivity_adjustment(dx, dy):
    """应用灵敏度调整到移动向量"""
    # 计算移动距离
    distance = math.sqrt(dx**2 + dy**2)
    
    # 应用灵敏度缩放
    scaled_distance = distance * SENSITIVITY
    
    # 如果缩放后距离为0，则返回原始值
    if scaled_distance < 1:
        return dx, dy
    
    # 计算缩放因子
    scale_factor = scaled_distance / distance
    
    # 应用缩放
    return dx * scale_factor, dy * scale_factor

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

def is_already_aimed(target_pos):
    """检查目标是否已经在屏幕中心附近"""
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    
    # 计算目标与屏幕中心的距离
    dx = target_pos[0] - screen_center_x
    dy = target_pos[1] - screen_center_y
    distance = math.sqrt(dx*dx + dy*dy)
    
    # 如果距离小于阈值，则认为已经瞄准
    return distance <= CENTER_THRESHOLD

def cleanup_resources():
    """清理所有资源"""
    global mouse_listener, dxcam_capture, model
    
    print("\n" + "="*50)
    print("正在清理资源...")
    
    # 忽略资源释放时的警告
    warnings.filterwarnings("ignore", category=ResourceWarning)
    
    # 1. 停止鼠标监听器
    if 'mouse_listener' in globals() and mouse_listener is not None:
        try:
            if mouse_listener.is_alive():
                mouse_listener.stop()
                mouse_listener = None
                print("✓ 鼠标监听器已停止")
        except Exception as e:
            print(f"✗ 停止鼠标监听器时出错: {e}")
    
    # 2. 停止DXCam截图器
    if 'dxcam_capture' in globals() and dxcam_capture is not None:
        try:
            dxcam_capture.stop()
            dxcam_capture = None
            print("✓ DXCam截图器已停止")
        except Exception as e:
            print(f"✗ 停止DXCam截图器时出错: {e}")
    
    # 3. 清理模型资源
    if 'model' in globals() and model is not None:
        try:
            # 清除GPU缓存
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            model = None
            print("✓ 模型资源已清理")
        except Exception as e:
            print(f"✗ 清理模型资源时出错: {e}")
    
    # 4. 清理COM对象（如果有）
    try:
        import comtypes
        comtypes.CoUninitialize()
        print("✓ COM对象已清理")
    except:
        pass
    
    # 5. 等待一小段时间让资源释放
    time.sleep(0.1)
    print("资源清理完成")
    print("="*50 + "\n")

def exit_handler():
    """程序退出时的清理函数"""
    print("程序退出，执行清理...")
    cleanup_resources()

def main():
    global scan_enabled, aim_active, last_alt_state, SENSITIVITY, last_move_time
    
    print("程序启动，按Alt键切换功能开关，按住鼠标左键瞄准，按Alt+C组合键停止")
    print(f"当前灵敏度: {SENSITIVITY:.2f} (使用↑/↓键调整)")
    print(f"中心阈值: {CENTER_THRESHOLD}px (小于此值认为已瞄准)")
    print(f"瞄准部位: 胸部以上 (上半身比例: {UPPER_BODY_RATIO*100}%)")
    print(f"检测区域: 以({CIRCLE_CENTER_X},{CIRCLE_CENTER_Y})为中心, 半径{CIRCLE_RADIUS}px的圆形区域")
    
    # 设置输入参数
    pydirectinput.PAUSE = 0.01
    pydirectinput.FAILSAFE = False
    
    # 获取圆形区域的外接矩形
    game_x, game_y, game_width, game_height = get_circle_bounding_box(CIRCLE_CENTER_X, CIRCLE_CENTER_Y, CIRCLE_RADIUS)
    print(f"检测区域外接矩形: x={game_x}, y={game_y}, width={game_width}, height={game_height}")
    
    # 初始化DXCam截图器
    dxcam_capture = None
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
    
    # 帧率统计
    frame_count = 0
    start_time = time.time()
    last_fps_update = time.time()
    last_process_time = 0
    
    # 瞄准状态跟踪
    aimed_count = 0  # 连续瞄准计数
    
    print("3秒后准备就绪...")
    time.sleep(3)
    
    try:
        while True:
            current_time = time.time()
            
            # 检测Alt键按下事件（切换功能开关）
            current_alt_pressed = keyboard.is_pressed('alt')
            if current_alt_pressed and not last_alt_state:
                scan_enabled = not scan_enabled
                print(f"扫描功能已{'启用' if scan_enabled else '禁用'}")
            last_alt_state = current_alt_pressed
            
            # 检查灵敏度调整
            if keyboard.is_pressed("up") or keyboard.is_pressed("down"):
                adjust_sensitivity()
            
            # 检查激活状态
            if scan_enabled and aim_active and (current_time - last_process_time >= PROCESSING_INTERVAL):
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
                    # 检测人体
                    detections = detect_humans(frame, (game_x, game_y, game_width, game_height), model)
                    
                    # 选择目标
                    selected_target = select_target(detections)
                    
                    # 如果找到目标
                    if selected_target:
                        # 检查目标是否已经在中心附近
                        if is_already_aimed(selected_target['screen_pos']):
                            aimed_count += 1
                            time.sleep(0.05)  
                            # 每秒更新一次状态
                            if current_time - last_fps_update >= 1.0:
                                # 获取截图FPS
                                capture_fps = dxcam_capture.get_fps() if dxcam_capture else 0
                                
                                # 计算处理FPS
                                fps = frame_count / (current_time - start_time)
                                
                                # 显示状态
                                status_msg = f"已瞄准 | 连续瞄准帧: {aimed_count} | 灵敏度: {SENSITIVITY:.2f}"
                                if capture_fps > 0:
                                    status_msg += f" | 截图FPS: {capture_fps:.1f}"
                                status_msg += f" | 处理FPS: {fps:.1f}"
                                print(status_msg)
                                
                                # 重置计数
                                frame_count = 0
                                start_time = current_time
                                last_fps_update = current_time
                            
                            # 跳过移动操作
                            continue
                        else:
                            aimed_count = 0  # 重置连续瞄准计数
                        
                        # 检查是否可以移动
                        if can_move_now():
                            # 获取目标屏幕坐标
                            target_x, target_y = selected_target['screen_pos']
                            
                            # 获取当前鼠标位置
                            current_x, current_y = pyautogui.position()
                            
                            # 计算需要移动的距离（目标与屏幕中心的偏移）
                            screen_center_x = screen_width // 2
                            screen_center_y = screen_height // 2
                            
                            # 计算目标与屏幕中心的偏移
                            dx = target_x - screen_center_x
                            dy = target_y - screen_center_y
                            
                            # 应用灵敏度调整
                            dx, dy = apply_sensitivity_adjustment(dx, dy)
                            
                            # 计算目标鼠标位置（当前鼠标位置 + 偏移）
                            target_mouse_x = current_x + dx
                            target_mouse_y = current_y + dy
                            
                            # 计算实际移动距离
                            distance = math.sqrt(dx**2 + dy**2)
                            
                            # 如果距离在有效范围内
                            if distance <= MAX_MOVE_DISTANCE:
                                try:
                                    # 一次性移动到目标位置（带平滑过渡）
                                    pydirectinput.moveTo(int(target_mouse_x), int(target_mouse_y), duration=0.05)
                                    # 更新最后移动时间
                                    last_move_time = current_time
                                    
                                    # 帧计数
                                    frame_count += 1
                                    
                                    # 每秒更新一次状态
                                    if current_time - last_fps_update >= 1.0:
                                        # 获取截图FPS
                                        capture_fps = dxcam_capture.get_fps() if dxcam_capture else 0
                                        
                                        # 计算处理FPS
                                        fps = frame_count / (current_time - start_time)
                                        
                                        # 显示状态
                                        status_msg = f"目标锁定 | 距离: {distance:.1f}px | 灵敏度: {SENSITIVITY:.2f}"
                                        if capture_fps > 0:
                                            status_msg += f" | 截图FPS: {capture_fps:.1f}"
                                        status_msg += f" | 处理FPS: {fps:.1f}"
                                        print(status_msg)
                                        
                                        # 重置计数
                                        frame_count = 0
                                        start_time = current_time
                                        last_fps_update = current_time
                                except Exception as e:
                                    print(f"鼠标移动错误: {str(e)}")
                    else:
                        # 没有找到目标时重置瞄准计数
                        aimed_count = 0
                        
                        # 帧计数
                        frame_count += 1
                        
                        # 每秒更新一次状态
                        if current_time - last_fps_update >= 1.0:
                            # 获取截图FPS
                            capture_fps = dxcam_capture.get_fps() if dxcam_capture else 0
                            
                            # 计算处理FPS
                            fps = frame_count / (current_time - start_time)
                            
                            # 显示状态
                            status_msg = f"扫描中... | 灵敏度: {SENSITIVITY:.2f}"
                            if capture_fps > 0:
                                status_msg += f" | 截图FPS: {capture_fps:.1f}"
                            status_msg += f" | 处理FPS: {fps:.1f}"
                            print(status_msg)
                            
                            # 重置计数
                            frame_count = 0
                            start_time = current_time
                            last_fps_update = current_time
                    
                    # 更新最后处理时间
                    last_process_time = current_time
                    
            else:
                # 没有激活时等待
                time.sleep(0.01)
                
                # 显示状态信息
                if current_time - last_fps_update >= 1.0:
                    status = "等待激活"
                    if scan_enabled and not aim_active:
                        status = "扫描已启用，等待鼠标左键"
                    elif not scan_enabled:
                        status = "扫描已禁用，按Alt键启用"
                    
                    # 获取截图FPS
                    capture_fps = dxcam_capture.get_fps() if dxcam_capture else 0
                    
                    status_msg = f"状态: {status} | 灵敏度: {SENSITIVITY:.2f}"
                    if capture_fps > 0:
                        status_msg += f" | 截图FPS: {capture_fps:.1f}"
                    print(status_msg)
                    last_fps_update = current_time
            
            # 退出检测
            if keyboard.is_pressed('alt') and keyboard.is_pressed('c'):
                print("\n检测到停止快捷键")
                break
                
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
    finally:
         cleanup_resources()
    
    # 确保线程结束
    time.sleep(0.2)
    
    print("程序已停止")
    sys.exit(0)

if __name__ == "__main__":
    main()