import time
import keyboard
from pynput import mouse
from .config import aim_active, SENSITIVITY


mouse_listener = None


def on_click(x, y, button, pressed):
    global aim_active
    if button == mouse.Button.left:
        aim_active = pressed


def start_mouse_listener():
    global mouse_listener
    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()


def stop_mouse_listener():
    global mouse_listener
    if mouse_listener and mouse_listener.is_alive():
        try:
            mouse_listener.stop()
        except:
            pass


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
