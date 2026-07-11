import os
import torch
import pyautogui

os.environ['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'

DXCAM_AVAILABLE = True

DEBUG_MODE = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

screen_width, screen_height = pyautogui.size()
SCREEN_CENTER_X = screen_width // 2
SCREEN_CENTER_Y = screen_height // 2

TARGET_CLASS = 0

scan_enabled = False
aim_active = False
last_alt_state = False

MAX_MOVE_DISTANCE = 2000
MIN_CONFIDENCE = 0.6
SENSITIVITY = 0.4
CENTER_THRESHOLD = 10

AIM_STABILIZATION_TIME = 0.5
STABILIZATION_MOVEMENT_SCALE = 0.2
MIN_STABILIZATION_FRAMES = 10

UPPER_BODY_RATIO = 0.5

CIRCLE_CENTER_X = SCREEN_CENTER_X
CIRCLE_CENTER_Y = SCREEN_CENTER_Y
CIRCLE_RADIUS = 200

PROCESSING_INTERVAL = 0.016
