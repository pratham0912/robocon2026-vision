"""
ROS2 node — Spearhead detection, 3-D localisation, and ID assignment.
Intel RealSense D435i + YOLOv8s. ABU Robocon 2026 "Kung Fu Quest".

Bug fixes applied:
  BF-1: inter_cam_sync_mode set via rs.context() BEFORE pipeline.start()
  BF-2: poll_for_frames() called with zero arguments (Python binding)
  BF-3: Display freeze — threading.Event -> queue.Queue(maxsize=1),
         SDK temporal_filter -> NumPy EMA, colour auto-exposure disabled
"""

import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Int32, String
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

try:
    import pupil_apriltags as apriltag
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False

WEIGHTS_PATH   = Path(__file__).resolve().parents[2] / "spearhead" / "weights" / "best.pt"
CONF_THRESHOLD = 0.70
IOU_THRESHOLD  = 0.70
IMGSZ          = 768
DEPTH_PATCH    = 21
DEPTH_SCALE    = 0.001
EMA_ALPHA      = 0.3
NUM_SPEARHEADS = 6
HALF_SPLIT_X   = 0.0
DIST_GATE      = 0.25
PICKED_HOLDOUT = 5.0
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720
COLOR_WIDTH    = 1280
COLOR_HEIGHT   = 720
DEPTH_WIDTH    = 1280
DEPTH_HEIGHT   = 720
FPS            = 30


class SpearheadDetectorNode(Node):
    def __init__(self):
        super().__init__("spearhead_detector")

        self.declare_parameter("weights",        str(WEIGHTS_PATH))
        self.declare_parameter("conf",           CONF_THRESHOLD)
        self.declare_parameter("iou",            IOU_THRESHOLD)
        self.declare_parameter("imgsz",          IMGSZ)
        self.declare_parameter("display",        True)
        self.declare_parameter("picked_holdout", PICKED_HOLDOUT)
        self.declare_parameter("waypoint_dist",  DIST_GATE)

        self.conf    = self.get_parameter("conf").value
        self.iou     = self.get_parameter("iou").value
        self.imgsz   = self.get_parameter("imgsz").value
        self.show    = self.get_parameter("display").value
        self.holdout = self.get_parameter("picked_holdout").value
        self.wp_dist = self.get_parameter("waypoint_dist").value
        weights_str  = self.get_parameter("weights").value

        self.get_logger().info(f"Loading model: {weights_str}")
        self.model = YOLO(weights_str)

        self.pub_targets = self.create_publisher(String,  "/spearhead/targets",     10)
        self.pub_best    = self.create_publisher(Point,   "/spearhead/best_target", 10)
        self.pub_zone    = self.create_publisher(Int32,   "/spearhead/zone_id",     10)

        self.confirmed_picked: dict[int, float] = {}
        self.tracked_targets:  dict[int, dict]  = {}
        self._ema_depth: Optional[np.ndarray]   = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)  # BF-3

        self._apriltag_detector = None
        if APRILTAG_AVAILABLE:
            self._apriltag_detector = apriltag.Detector(families="tag36h11")
            self.get_logger().info("AprilTag detector ready")
        else:
            self.get_logger().warn("pupil_apriltags not installed — zone detection disabled")

        if not REALSENSE_AVAILABLE:
            self.get_logger().error("pyrealsense2 not found")
            raise RuntimeError("pyrealsense2 unavailable")

        self._pipeline, self._intrinsics = self._start_pipeline()
        self.get_logger().info("RealSense pipeline started")

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        if self.show:
            self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self._display_thread.start()

        self.get_logger().info("SpearheadDetectorNode ready")

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _start_pipeline(self):
        """BF-1: set inter_cam_sync_mode via rs.context() BEFORE pipeline.start()"""
        ctx     = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense device found")
        dev = devices[0]
        self.get_logger().info(f"Device: {dev.get_info(rs.camera_info.name)}")

        # BF-1: set sync mode pre-start
        depth_sensor = dev.first_depth_sensor()
        if depth_sensor.supports(rs.option.inter_cam_sync_mode):
            depth_sensor.set_option(rs.option.inter_cam_sync_mode, 0)

        # BF-3c: disable colour auto-exposure before start
        color_sensor = dev.query_sensors()[1]
        if color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, 8500)

        pipeline = rs.pipeline()
        config   = rs.config()
        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16,  FPS)

        profile    = pipeline.start(config)
        intrinsics = (profile.get_stream(rs.stream.color)
                              .as_video_stream_profile()
                              .get_intrinsics())
        return pipeline, intrinsics

    # ── Capture loop ──────────────────────────────────────────────────────────

    def _capture_loop(self):
        """BF-2: poll_for_frames() with NO arguments"""
        while self._running and rclpy.ok():
            try:
                success, frames = self._pipeline.poll_for_frames()  # BF-2: no args
            except RuntimeError:
                time.sleep(0.001)
                continue

            if not success or frames is None:
                time.sleep(0.001)
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_raw   = np.asanyarray(depth_frame.get_data())

            # BF-3b: NumPy EMA instead of rs.temporal_filter
            depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE
            if self._ema_depth is None:
                self._ema_depth = depth_m.copy()
            else:
                self._ema_depth = EMA_ALPHA * depth_m + (1.0 - EMA_ALPHA) * self._ema_depth

            results    = self.model.predict(source=color_image, conf=self.conf,
                                            iou=self.iou, imgsz=self.imgsz, verbose=False)
            detections = self._parse_detections(results[0])
            world_pts  = [self._backproject(d["cx"], d["cy"]) for d in detections]

            self._update_tracking(detections, world_pts)
            zone_id = self._detect_zone(color_image)
            self._publish(zone_id)

            if self.show:
                annotated = results[0].plot()
                # BF-3a: queue.Queue instead of threading.Event
                try:
                    self._frame_q.put_nowait(annotated)
                except queue.Full:
                    try:
                        self._frame_q.get_nowait()
                    except queue.Empty:
                        pass
                    self._frame_q.put_nowait(annotated)

    # ── Detection parsing ─────────────────────────────────────────────────────

    def _parse_detections(self, result):
        detections = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx    = int((x1 + x2) / 2)
            cy    = int((y1 + y2) / 2)
            cls   = int(box.cls[0])
            label = result.names[cls]
            conf  = float(box.conf[0])
            detections.append({"cx": cx, "cy": cy, "label": label, "conf": conf})
        return detections

    # ── Depth back-projection ─────────────────────────────────────────────────

    def _backproject(self, cx, cy):
        if self._ema_depth is None:
            return [0.0, 0.0, 0.0]
        H, W   = self._ema_depth.shape
        half   = DEPTH_PATCH // 2
        r0, r1 = max(0, cy - half), min(H, cy + half + 1)
        c0, c1 = max(0, cx - half), min(W, cx + half + 1)
        patch  = self._ema_depth[r0:r1, c0:c1]
        valid  = patch[patch > 0.1]
        if valid.size == 0:
            return [0.0, 0.0, 0.0]
        Z   = float(np.median(valid))
        fx  = self._intrinsics.fx
        fy  = self._intrinsics.fy
        ppx = self._intrinsics.ppx
        ppy = self._intrinsics.ppy
        X   = (cx - ppx) * Z / fx
        Y   = (cy - ppy) * Z / fy
        return [X, Y, Z]

    # ── Tracking ──────────────────────────────────────────────────────────────

    def _update_tracking(self, detections, world_pts):
        now = time.time()
        matched_ids = set()

        for det, wpt in zip(detections, world_pts):
            if wpt[2] == 0.0:
                continue
            best_id, best_dist = None, float("inf")
            right_half = wpt[0] >= HALF_SPLIT_X

            for tid, track in self.tracked_targets.items():
                if track["label"] != det["label"]:
                    continue
                if (track["pos"][0] >= HALF_SPLIT_X) != right_half:
                    continue
                d = np.linalg.norm(np.array(wpt) - np.array(track["pos"]))
                if d < best_dist and d < DIST_GATE:
                    best_dist, best_id = d, tid

            if best_id is not None:
                self.tracked_targets[best_id]["pos"]       = wpt
                self.tracked_targets[best_id]["last_seen"] = now
                matched_ids.add(best_id)
            else:
                new_id = self._next_free_id(right_half)
                if new_id is not None:
                    if new_id in self.confirmed_picked:
                        if now - self.confirmed_picked[new_id] < self.holdout:
                            continue
                        del self.confirmed_picked[new_id]
                    self.tracked_targets[new_id] = {
                        "label": det["label"], "pos": wpt, "last_seen": now
                    }
                    matched_ids.add(new_id)

        stale = [tid for tid, t in self.tracked_targets.items()
                 if now - t["last_seen"] > 1.0 and tid not in matched_ids]
        for tid in stale:
            del self.tracked_targets[tid]

    def _next_free_id(self, right_half):
        half_ids = range(4, 7) if right_half else range(1, 4)
        used     = set(self.tracked_targets.keys())
        for i in half_ids:
            if i not in used:
                return i
        return None

    # ── Zone / publish / display ──────────────────────────────────────────────

    def _detect_zone(self, frame_bgr):
        if self._apriltag_detector is None:
            return 0
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        tags = self._apriltag_detector.detect(gray)
        if not tags:
            return 0
        return int(max(tags, key=lambda t: t.decision_margin).tag_id)

    def _best_target(self):
        candidates = [(tid, t) for tid, t in self.tracked_targets.items()
                      if tid not in self.confirmed_picked and t["pos"][2] > 0.1]
        if not candidates:
            return None
        tid, t = min(candidates, key=lambda x: x[1]["pos"][2])
        return {"id": tid, **t}

    def _publish(self, zone_id):
        payload = [{"id": tid, "label": t["label"],
                    "x": round(t["pos"][0], 4),
                    "y": round(t["pos"][1], 4),
                    "z": round(t["pos"][2], 4)}
                   for tid, t in self.tracked_targets.items()]
        self.pub_targets.publish(String(data=json.dumps(payload)))

        best = self._best_target()
        if best is not None:
            pt = Point()
            pt.x, pt.y, pt.z = best["pos"]
            self.pub_best.publish(pt)

        self.pub_zone.publish(Int32(data=zone_id))

    def _display_loop(self):
        cv2.namedWindow("Spearhead Detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Spearhead Detector", DISPLAY_WIDTH, DISPLAY_HEIGHT)
        while self._running and rclpy.ok():
            try:
                frame = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            cv2.imshow("Spearhead Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._running = False
                break
        cv2.destroyAllWindows()

    def mark_picked(self, track_id):
        self.confirmed_picked[track_id] = time.time()
        self.tracked_targets.pop(track_id, None)

    def destroy_node(self):
        self._running = False
        if hasattr(self, "_pipeline"):
            self._pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpearheadDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
