"""
PyInstaller 构建脚本
用法: python build.py 或 pyinstaller GameAimAssistant.spec
"""
import PyInstaller.__main__

PyInstaller.__main__.run([
    'run.py',
    '--name=GameAimAssistant',
    '--console',
    '--add-data=yolov8m.pt;.',
    '--hidden-import=PyQt5.sip',
    '--hidden-import=pynput.keyboard._win32',
    '--hidden-import=pynput.mouse._win32',
    '--hidden-import=win32timezone',
    '--hidden-import=ultralytics.models',
    '--hidden-import=torch._C',
    '--hidden-import=cv2',
    '--hidden-import=keyboard',
    '--hidden-import=pydirectinput',
    '--hidden-import=numpy',
    '--hidden-import=win32gui',
    '--hidden-import=win32con',
    '--hidden-import=dxcam',
    '--collect-all=aim_assist',
    '--noconfirm',
])
