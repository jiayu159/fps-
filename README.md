# GameAimAssistant (GAA)

基于 YOLOv8 + DXCam 的 FPS 游戏辅助瞄准工具，支持 GPU 加速。

## 功能

- **YOLOv8 人体检测** — 实时检测画面中的人形目标（使用 `yolov8m.pt`）
- **DXCam 截图** — 独立线程捕获游戏画面
- **圆形检测区域** — 只处理屏幕中心圆形范围内的目标
- **智能目标选择** — 选取最接近屏幕中心的目标
- **瞄准稳定器** — 平滑减速跟随，防止抖动
- **垂直移动限制** — 抑制异常上拉/下拉，提升瞄准体验
- **快捷键控制** — NumLock 开关，↑/↓ 调灵敏度，Alt+C 退出

## 使用方法

```bash
python run.py
```

### 快捷键

| 按键       | 功能     |
| -------- | ------ |
| NumLock  | 切换扫描开关 |
| 鼠标左键（按住） | 激活瞄准   |
| ↑ / ↓    | 调整灵敏度  |
| Alt + C  | 退出程序   |

## 配置

编辑 `aim_assist/config.py` 可调整参数：

- `MIN_CONFIDENCE` — 最低置信度阈值 (默认 0.6)
- `SENSITIVITY` — 鼠标灵敏度 (默认 0.4)
- `CIRCLE_RADIUS` — 检测区域半径 (默认 200px)
- `CENTER_THRESHOLD` — 瞄准判定阈值 (默认 10px)
- `PROCESSING_INTERVAL` — 处理间隔 (默认 0.016s)
- `DEBUG_MODE` — 开启调试日志 (默认 False)

## 项目结构

```
GAA/
├── aim_assist/              # 核心包
│   ├── __init__.py          # 统一导出
│   ├── config.py            # 配置参数
│   ├── capture.py           # DXCam 截图
│   ├── detection.py         # YOLO 检测 + 目标选择
│   ├── stabilizer.py        # 瞄准稳定器
│   ├── movement.py          # 移动计算与校验
│   ├── input_handler.py     # 鼠标/键盘输入
│   └── app.py               # 主循环
├── run.py                   # 启动入口
├── build.py                 # PyInstaller 构建脚本
├── GameAimAssistant.spec    # PyInstaller spec
├── legacy/                  # 原始单文件备份
├── main.py                  # OpenAI 聊天客户端
├── tcp_frame_sender.py      # TCP 屏幕发送端
├── test.py                  # TileLang MoE 测试
├── aim_assist_settings.json # 设置存档
├── yolov8m.pt               # YOLOv8 模型(大)
├── yolov8n.pt               # YOLOv8 模型(小)
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

- ultralytics (YOLOv8)
- torch (PyTorch)
- dxcam (截图调用)
- pydirectinput / pyautogui (鼠标控制)
- opencv-python (图像处理)
- pynput / keyboard (输入监听)
- pywin32 (Windows API)

