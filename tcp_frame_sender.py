"""
示例：TCP 屏幕发送端（生产者）。
协议：4 字节小端长度 + JPEG 数据
用法：python tcp_frame_sender.py [host] [port] [width] [height]
"""
import sys
import socket
import struct
import time
import cv2
import numpy as np
from mss import mss

HOST = sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
OUT_W = int(sys.argv[3]) if len(sys.argv) > 3 else 640
OUT_H = int(sys.argv[4]) if len(sys.argv) > 4 else 360

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Connecting to {HOST}:{PORT}...")
        s.connect((HOST, PORT))
        print("Connected, start sending frames...")
        with mss() as sct:
            monitor = sct.monitors[1]
            while True:
                img = np.array(sct.grab(monitor))[:, :, :3]
                # resize to OUT_W x OUT_H to reduce bandwidth
                img = cv2.resize(img, (OUT_W, OUT_H))
                ret, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret:
                    time.sleep(0.01)
                    continue
                data = buf.tobytes()
                length = struct.pack('<I', len(data))
                try:
                    s.sendall(length + data)
                except BrokenPipeError:
                    print("Connection closed by receiver")
                    break
                time.sleep(0.01)

if __name__ == '__main__':
    main()
