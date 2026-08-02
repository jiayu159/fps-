import time
import keyboard
from pynput import mouse
from . import config


mouse_listener = None
left_pressed = False
right_pressed = False


def on_click(x, y, button, pressed):
    global left_pressed, right_pressed
    if button == mouse.Button.left:
        left_pressed = pressed
    elif button == mouse.Button.right:
        right_pressed = pressed
    config.aim_active = left_pressed or right_pressed


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
    if keyboard.is_pressed("up"):
        config.SENSITIVITY = min(1.0, config.SENSITIVITY + 0.05)
        print(f"灵敏度增加至: {config.SENSITIVITY:.2f}")
        time.sleep(0.2)
    elif keyboard.is_pressed("down"):
        config.SENSITIVITY = max(0.1, config.SENSITIVITY - 0.05)
        print(f"灵敏度减少至: {config.SENSITIVITY:.2f}")
        time.sleep(0.2)
