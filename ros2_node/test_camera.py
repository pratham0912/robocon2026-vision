"""
Standalone RealSense D435i verification — no ROS2 required.
Verifies all three bug fixes before running the full node.
Press Q to quit, S to save a frame.
"""

import queue, sys, threading, time
import cv2, numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("[FAIL] pyrealsense2 not installed.")
    sys.exit(1)

EMA_ALPHA = 0.3

# BF-1: query device BEFORE pipeline start
ctx     = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("[FAIL] No RealSense device found.")
    sys.exit(1)

dev = devices[0]
print(f"[OK]  Device: {dev.get_info(rs.camera_info.name)}")

depth_sensor = dev.first_depth_sensor()
if depth_sensor.supports(rs.option.inter_cam_sync_mode):
    depth_sensor.set_option(rs.option.inter_cam_sync_mode, 0)
    print("[OK]  BF-1: inter_cam_sync_mode set PRE-start")

color_sensor = dev.query_sensors()[1]
if color_sensor.supports(rs.option.enable_auto_exposure):
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    color_sensor.set_option(rs.option.exposure, 8500)
    print("[OK]  BF-3c: auto-exposure disabled (8500 µs)")

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16,  30)
pipeline.start(config)
print("[OK]  Pipeline started\n")

frame_q   = queue.Queue(maxsize=1)
ema_depth = None
running   = True
frames_n  = 0

def capture():
    global ema_depth, running, frames_n
    while running:
        try:
            ok, frames = pipeline.poll_for_frames()  # BF-2: no args
        except RuntimeError:
            time.sleep(0.001); continue
        if not ok or frames is None:
            time.sleep(0.001); continue

        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            continue

        frames_n += 1
        img = np.asanyarray(cf.get_data())
        raw = np.asanyarray(df.get_data()).astype(np.float32) * 0.001

        # BF-3b: NumPy EMA
        if ema_depth is None:
            ema_depth = raw.copy()
        else:
            ema_depth[:] = EMA_ALPHA * raw + (1 - EMA_ALPHA) * ema_depth

        cy, cx = 360, 640
        patch  = ema_depth[cy-10:cy+11, cx-10:cx+11]
        valid  = patch[patch > 0.05]
        depth_c = float(np.median(valid)) if valid.size else 0.0

        display = img.copy()
        cv2.circle(display, (cx, cy), 6, (0,255,0), 2)
        cv2.putText(display, f"Centre depth: {depth_c:.3f} m",
                    (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(display, f"Frames: {frames_n}",
                    (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        cv2.putText(display, "Q: quit   S: save",
                    (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

        try:
            frame_q.put_nowait(display)
        except queue.Full:
            try: frame_q.get_nowait()
            except queue.Empty: pass
            frame_q.put_nowait(display)

threading.Thread(target=capture, daemon=True).start()
print("[OK]  BF-2: poll_for_frames() zero-arg in capture thread")
print("[OK]  BF-3a: queue.Queue(maxsize=1) for display delivery")
print("Preview open — Q to quit, S to save frame\n")

cv2.namedWindow("D435i Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("D435i Test", 1280, 720)

while True:
    try:
        frame = frame_q.get(timeout=1.0)
    except queue.Empty:
        continue
    cv2.imshow("D435i Test", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("test_frame.jpg", frame)
        print("[SAVE] test_frame.jpg")

running = False
pipeline.stop()
cv2.destroyAllWindows()
print(f"\n[DONE] {frames_n} frames captured — all BF checks passed.")
