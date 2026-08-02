import cv2
import time
import dxcam
from .config import DXCAM_AVAILABLE


class DXCamCapture:
    def __init__(self, region, target_fps=60):
        self.region = region
        self.target_fps = target_fps
        self.camera = None
        self.is_running = False
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
            self.camera.start(target_fps=self.target_fps, video_mode=True)
            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            x, y, w, h = self.region
            print(f"DXCam截图器已启动 - 区域: ({x},{y},{w},{h})")
            return True
        except Exception as e:
            print(f"启动DXCam失败: {e}")
            return False

    def _crop_region(self, frame):
        x, y, w, h = self.region
        return frame[y:y+h, x:x+w]

    def get_frame(self):
        if not self.is_running:
            return None
        frame = self.camera.get_latest_frame()
        if frame is None:
            return None
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        cropped = self._crop_region(frame)
        return cv2.cvtColor(cropped, cv2.COLOR_BGRA2BGR)

    def get_fps(self):
        return self.fps

    def stop(self):
        self.is_running = False
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass
        if self.frame_count > 0:
            print(f"DXCam已停止 - 平均FPS: {self.fps:.1f}")

    def __del__(self):
        self.stop()
