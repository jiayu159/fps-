# GameAimAssistant (GAA)

基于 YOLO11 + DXCam 的 FPS 游戏辅助瞄准工具，需要GPU 加速（CUDA），目前实测6G显存需将游戏帧率控制在100fps才能有效运作。

## 功能

- **YOLO11s 人体检测** — 实时检测画面中的人形目标（使用 `yolo11s.pt`，支持 FP16 推理加速）
- **DXCam 截图** — 独立线程捕获游戏画面
- **圆形检测区域** — 只处理屏幕中心圆形范围内的目标（半径可调）
- **动态置信度** — 越靠近屏幕中心阈值越低，能稳定识别准星处的近身/半身目标
- **站立目标优先** — 优先锁定站立目标，倒地目标仅在没有站立目标时才会被选中
- **智能目标选择与跟踪** — 锁定最近目标，支持丢失保持（1s 内不换人），显著优于近目标自动切换
- **瞄准稳定器** — 平滑减速跟随，防止抖动
- **距离梯度灵敏度** — 近距离低灵敏度防过冲，远距离全速跟进
- **垂直移动限制** — 抑制异常上拉/下拉，提升瞄准体验
- **快捷键控制** — NumLock 开关，↑/↓ 调灵敏度，Alt+C 退出
- **左/右键触发** — 按住鼠标左键或右键均激活瞄准

## 使用方法

```bash
python run.py
```

### 快捷键

| 按键         | 功能     |
| ---------- | ------ |
| NumLock    | 切换扫描开关 |
| 鼠标左键（按住）  | 激活瞄准   |
| 鼠标右键（按住）  | 激活瞄准   |
| ↑ / ↓      | 调整灵敏度  |
| Alt + C    | 退出程序   |

## 配置

编辑 `aim_assist/config.py` 可调整参数：

- `MIN_CONFIDENCE` — 边缘最低置信度阈值 (默认 0.45)
- `CENTER_CONF_THRESHOLD` — 屏幕中心最低置信度 (默认 0.15)
- `SENSITIVITY` — 鼠标灵敏度 (默认 0.4)
- `CIRCLE_RADIUS` — 检测区域半径 (默认 400px)
- `CENTER_THRESHOLD` — 瞄准判定阈值 (默认 10px)
- `MAX_LOST_FRAMES` — 目标丢失保持帧数 (默认 60，约 1s)
- `PROCESSING_INTERVAL` — 处理间隔 (默认 0.01s)
- `DEBUG_MODE` — 开启调试日志 (默认 False)

## 项目结构

```
GAA/
├── aim_assist/              # 核心包
│   ├── __init__.py          # 统一导出
│   ├── config.py            # 配置参数
│   ├── capture.py           # DXCam 截图
│   ├── detection.py         # YOLO 检测 + 动态置信度
│   ├── stabilizer.py        # 瞄准稳定器
│   ├── movement.py          # 移动计算、校验与距离梯度
│   ├── target_tracker.py    # 目标跟踪与丢失保持
│   ├── input_handler.py     # 鼠标/键盘输入
│   └── app.py               # 主循环
├── run.py                   # 启动入口
├── build.py                 # PyInstaller 构建脚本
├── GameAimAssistant.spec    # PyInstaller spec
├── legacy/                  # 原始单文件备份
├── aim_assist_settings.json # 设置存档
├── yolo11s.pt               # 当前检测模型
├── yolov8n.pt               # YOLOv8 备选模型(小)
├── icon.png                 # 应用图标
└── interception.dll         # 输入拦截库
```

## 构建

```bash
python build.py
```

或

```bash
pyinstaller GameAimAssistant.spec
```

## 依赖

- ultralytics (YOLO11)
- torch (PyTorch, CUDA)
- dxcam (截图调用)
- pydirectinput / pyautogui (鼠标控制)
- opencv-python (图像处理)
- pynput / keyboard (输入监听)
- pywin32 (Windows API)
