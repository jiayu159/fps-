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

# 检查GPU可用性
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载YOLOv8模型 (轻量级版本)
model = YOLO('yolov8n.pt').to(device)
model.fuse()  

# 获取屏幕尺寸
screen_width, screen_height = pyautogui.size()
print(f"屏幕分辨率: {screen_width}x{screen_height}")

# 目标类别 (person)
TARGET_CLASS = 0

# 鼠标状态
aim_active = False  # 点击切换模式

# 优化参数
MAX_MOVE_DISTANCE = 2000    # 最大有效移动距离(像素)
MIN_CONFIDENCE = 0.5        # 最低置信度阈值
SENSITIVITY = 0.49           # 灵敏度参数 (0.1-1.0)
CENTER_THRESHOLD = 50      # 中心点阈值(像素)，小于此值认为已瞄准

# 上半身比例 (从头部到胸部)
UPPER_BODY_RATIO = 0.5       # 上半身占整个身体高度的比例，越小越锁头

# 防止连续移动
last_move_time = 0
MOVE_COOLDOWN = 0.19  # 冷却时间

def on_click(x, y, button, pressed):
    """鼠标点击回调函数 - 左键按住激活"""
    global aim_active
    if button == mouse.Button.left:  # 处理左键
        aim_active = pressed  # 按下时激活，释放时关闭
        status = "激活" if pressed else "关闭"
        print(f"瞄准系统已{status}")

# 启动鼠标监听器
mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()

def find_game_window(window_title="游戏窗口"):  #窗口标题可以根据实际游戏修改
    """查找并激活游戏窗口"""
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if window_title.lower() in title.lower():
                rect = win32gui.GetWindowRect(hwnd)
                windows.append((hwnd, rect))
    
    windows = []
    win32gui.EnumWindows(callback, windows)
    
    if windows:
        hwnd, rect = windows[0]
        x, y, right, bottom = rect
        width = right - x
        height = bottom - y
        print(f"找到游戏窗口: '{win32gui.GetWindowText(hwnd)}' 位置: ({x}, {y}) 尺寸: {width}x{height}")
        
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.5)
        
        return x, y, width, height
    else:
        print(f"未找到标题包含 '{window_title}' 的窗口")
        return 0, 0, screen_width, screen_height

def detect_humans(frame, region, model):
    """检测人体并返回上半身中心点"""
    x, y, width, height = region
    
    # 多尺度检测 - 创建不同尺度的图像
    scales = [0.8, 0.5, 0.3]  # 原始尺寸和缩小尺寸
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
                if conf > MIN_CONFIDENCE:  # 使用提高后的置信度阈值
                    # 调整框坐标到原始尺寸
                    if scale != 1.0:
                        box = box / scale
                    
                    # 转换为整数坐标
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 计算上半身中心点（胸部位置）
                    # 上半身高度 = 整个身体高度 * UPPER_BODY_RATIO
                    upper_height = (y2 - y1) * UPPER_BODY_RATIO
                    # 上半身中心Y坐标 = 头部Y坐标 + 上半身高度的一半
                    upper_y = y1 + upper_height / 2
                    cx = (x1 + x2) // 2
                    cy = int(upper_y)
                    
                    # 转换为屏幕坐标
                    screen_x = cx + x
                    screen_y = cy + y
                    
                    # 确保坐标在屏幕范围内
                    screen_x = max(0, min(screen_x, screen_width - 1))
                    screen_y = max(0, min(screen_y, screen_height - 1))
                    
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
    
    # 过滤低置信度的检测结果
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
        time.sleep(0.2)  
    
    # 减少灵敏度
    elif keyboard.is_pressed("down"):
        SENSITIVITY = max(0.1, SENSITIVITY - 0.05)
        print(f"灵敏度减少至: {SENSITIVITY:.2f}")
        time.sleep(0.2)  

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

def main():
    global aim_active, SENSITIVITY, last_move_time  
    
    print("程序启动，点击鼠标右键切换瞄准状态，按Alt+C组合键停止")
    print(f"当前灵敏度: {SENSITIVITY:.2f} (使用↑/↓键调整)")
    print(f"中心阈值: {CENTER_THRESHOLD}px (小于此值认为已瞄准)")
    print(f"瞄准部位: 胸部以上 (上半身比例: {UPPER_BODY_RATIO*100}%)")
    
    # 设置输入参数
    pydirectinput.PAUSE = 0.01
    pydirectinput.FAILSAFE = False
    
    # 查找并激活游戏窗口
    game_region = find_game_window()
    game_x, game_y, game_width, game_height = game_region
    
    # 帧率统计
    frame_count = 0
    start_time = time.time()
    last_fps_update = time.time()
    
    # 瞄准状态跟踪
    aimed_count = 0  
    
    print("3秒后准备就绪...")
    time.sleep(3)
    
    try:
        while True:
            # 检查灵敏度调整
            adjust_sensitivity()
            
            # 检查激活状态
            if aim_active:
                # 截取游戏区域
                screenshot = pyautogui.screenshot(region=(game_x, game_y, game_width, game_height))
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 检测人体
                detections = detect_humans(frame, game_region, model)
                
                # 选择目标
                selected_target = select_target(detections)
                
                # 如果找到目标
                if selected_target:
                    # 检查目标是否已经在中心附近
                    if is_already_aimed(selected_target['screen_pos']):
                        aimed_count += 1
                        
                        # 每秒更新一次状态
                        if time.time() - last_fps_update >= 1.0:
                            # 计算并显示FPS
                            fps = frame_count / (time.time() - start_time)
                            print(f"已瞄准 | 连续瞄准帧: {aimed_count} | 灵敏度: {SENSITIVITY:.2f} | FPS: {fps:.1f}")
                            frame_count = 0
                            start_time = time.time()
                            last_fps_update = time.time()
                        
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
                        
                        # 计算需要移动的距离
                        dx = target_x - current_x
                        dy = target_y - current_y
                        
                        # 应用灵敏度调整
                        dx, dy = apply_sensitivity_adjustment(dx, dy)
                        
                        # 计算实际移动距离
                        distance = math.sqrt(dx**2 + dy**2)
                        
                        # 如果距离在有效范围内
                        if distance <= MAX_MOVE_DISTANCE:
                            # 计算目标位置
                            target_x = current_x + dx
                            target_y = current_y + dy
                            try:
                                # 一次性移动到目标位置（带平滑过渡）
                                pydirectinput.moveTo(int(target_x), int(target_y), duration=0.05)
                                # pydirectinput.mouseDown(button='left')
                                # time.sleep(0.1) 
                                # pydirectinput.mouseUp(button='left')
                                # 更新最后移动时间
                                last_move_time = time.time()
                                
                                # 帧计数
                                frame_count += 1
                                
                                # 每秒更新一次状态
                                if time.time() - last_fps_update >= 1.0:
                                    # 计算并显示FPS
                                    fps = frame_count / (time.time() - start_time)
                                    print(f"目标锁定 | 距离: {distance:.1f}px | 灵敏度: {SENSITIVITY:.2f} | FPS: {fps:.1f}")
                                    frame_count = 0
                                    start_time = time.time()
                                    last_fps_update = time.time()
                            except Exception as e:
                                print(f"鼠标移动错误: {str(e)}")
                else:
                    # 没有找到目标时重置瞄准计数
                    aimed_count = 0
                    
                    # 帧计数
                    frame_count += 1
                    
                    # 每秒更新一次状态
                    if time.time() - last_fps_update >= 1.0:
                        # 计算并显示FPS
                        fps = frame_count / (time.time() - start_time)
                        print(f"扫描中... | 灵敏度: {SENSITIVITY:.2f} | FPS: {fps:.1f}")
                        frame_count = 0
                        start_time = time.time()
                        last_fps_update = time.time()
            else:
                # 没有激活时等待
                time.sleep(0.01)
            
            # 退出检测
            if keyboard.is_pressed('alt') and keyboard.is_pressed('c'):
                print("\n检测到停止快捷键")
                break
                
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
    finally:
        # 确保停止鼠标监听器
        mouse_listener.stop()
        print("\n程序已停止")

if __name__ == "__main__":
    main()