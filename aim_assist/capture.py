import cv2
import time
import threading
from collections import deque
import dxcam
from .config import DXCAM_AVAILABLE


class DXCamCapture:
    def __init__(self, region, target_fps=45):
        self.region = region
        self.target_fps = target_fps
        self.dxcam_region = (
            region[0],
            region[1],
            region[0] + region[2],
            region[1] + region[3]
        )
        self.camera = None
        self.is_running = False
        self.frame_buffer = deque(maxlen=3)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0

    def start(self):
        if not DXCAM_AVAILABLE:
            print("DXCam不可用，使用备用截图方法")
            return False
        try:
            self.camera = dxcam.create()
            if self.camera is None:
                print("无法创建DXCam实例")
                return False
            self.camera.start(target_fps=self.target_fps, region=self.dxcam_region)
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
        while self.is_running:
            try:
                frame = self.camera.get_latest_frame()
                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.frame_buffer.append({
                            'frame': frame.copy(),
                            'timestamp': time.time()
                        })
                        self.frame_count += 1
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        self.fps = self.frame_count / elapsed
                time.sleep(0.001)
            except Exception as e:
                print(f"截图错误: {e}")
                time.sleep(0.01)

    def get_frame(self, wait_for_new=True, timeout=0.1):
        if not self.is_running or self.latest_frame is None:
            return None

        if wait_for_new:
            start_size = len(self.frame_buffer)
            start_time = time.time()
            while len(self.frame_buffer) <= start_size:
                if time.time() - start_time > timeout:
                    break
                time.sleep(0.001)

        with self.frame_lock:
            if self.latest_frame is not None:
                frame_bgr = cv2.cvtColor(self.latest_frame.copy(), cv2.COLOR_BGRA2BGR)
                return frame_bgr
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
