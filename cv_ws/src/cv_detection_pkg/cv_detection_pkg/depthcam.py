import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool, Float32

from ultralytics import YOLO
from pupil_apriltags import Detector
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

import pyrealsense2 as rs
import cv2
import math
import numpy as np
import threading
import time
from collections import deque


# ── Model Path ─────────────────────────────────────────────────────────────
pkg_path   = get_package_share_directory("cv_detection_pkg")
MODEL_PATH = Path(pkg_path) / "models" / "spearhead" / "best.pt"

PRIORITY_ORDER = [5, 6, 4, 1, 2, 3]

VALID_IDS_FOR_LABEL = {
    "spear": {1, 6},
    "fist":  {2, 5},
    "palm":  {3, 4},
}

VALID_TAGS      = [4]
VALID_TAGS_LOOP = [5]

WORLD_DIST_WEIGHT             = 200.0
PIXEL_DIST_WEIGHT             = 1.0
AREA_DIFF_WEIGHT              = 50.0
LABEL_MISMATCH_PENALTY        = 200.0
TRACKING_CONFIDENCE_THRESHOLD = 150.0
PICKUP_ABSENCE_FRAMES         = 45
DEPTH_SAMPLE_HALF             = 10
WAYPOINT_DISTANCE_GATE_M      = 5.0
APRILTAG_EVERY_N_FRAMES       = 3
TIMING_LOG_EVERY_N            = 30

# If FrameGrabber produces no new frame for this many seconds,
# restart the RealSense pipeline automatically.
PIPELINE_RESTART_TIMEOUT_S    = 3.0


# ══════════════════════════════════════════════════════════════════════════
# STAGE TIMER
# ══════════════════════════════════════════════════════════════════════════

class StageTimer:
    def __init__(self):
        self._last   = time.perf_counter()
        self._stages : dict[str, deque] = {}
        self._order  : list[str]        = []
        self._frames = 0

    def tick(self, stage: str):
        now = time.perf_counter()
        ms  = (now - self._last) * 1000
        self._last = now
        if stage not in self._stages:
            self._stages[stage] = deque(maxlen=TIMING_LOG_EVERY_N or 30)
            self._order.append(stage)
        self._stages[stage].append(ms)

    def report(self):
        if not TIMING_LOG_EVERY_N:
            return
        self._frames += 1
        if self._frames % TIMING_LOG_EVERY_N != 0:
            return
        parts = [f"{s}:{sum(self._stages[s])/len(self._stages[s]):.1f}ms"
                 for s in self._order if self._stages[s]]
        total = sum(sum(v)/len(v) for v in self._stages.values())
        print(f"[timing] {' | '.join(parts)} | total:{total:.1f}ms (~{1000/max(total,1):.0f}fps)")


# ══════════════════════════════════════════════════════════════════════════
# REALSENSE SETUP
# ══════════════════════════════════════════════════════════════════════════

def build_realsense_pipeline():
    """
    Build and start a RealSense D435i pipeline.

    Extra settings applied here:
      • enable_auto_exposure  — keeps exposure stable when stationary
      • laser_power max       — improves depth on low-texture/static scenes
      • hole_filling_filter   — fills missing depth pixels (reflections etc.)
      • temporal_filter       — smooths depth over time (helps static scenes)
    """
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)

    profile = pipeline.start(config)
    align   = rs.align(rs.stream.color)

    # ── Depth sensor tuning ─────────────────────────────────────────────
    device       = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    # Ensure auto-exposure is on (can get stuck off after firmware update)
    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

    # Max laser power → better depth on static, low-texture scenes
    if depth_sensor.supports(rs.option.laser_power):
        max_laser = depth_sensor.get_option_range(rs.option.laser_power).max
        depth_sensor.set_option(rs.option.laser_power, max_laser)

    # ── Post-processing filters ──────────────────────────────────────────
    # These run inside the FrameGrabber thread, not in the processing thread.
    hole_filter     = rs.hole_filling_filter()   # fill invalid depth pixels
    temporal_filter = rs.temporal_filter()        # smooth depth over time

    # ── Intrinsics ──────────────────────────────────────────────────────
    color_stream = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intr         = color_stream.get_intrinsics()

    return pipeline, align, depth_scale, intr, hole_filter, temporal_filter


# ══════════════════════════════════════════════════════════════════════════
# FRAME GRABBER
# ══════════════════════════════════════════════════════════════════════════

class FrameGrabber(threading.Thread):
    """
    Runs in a daemon thread.  Continuously drains the RealSense queue and
    stores only the LATEST (color_np, depth_np) pair.

    Key fixes vs. the previous version
    ────────────────────────────────────
    FIX 1 — Don't skip on invalid depth.
        Old code:  if not color_frame or not depth_frame: continue
        Problem:   When the camera is stationary, the D435i depth stream
                   frequently returns invalid frames (reflective surfaces,
                   uniform walls, low texture).  The old code discarded
                   the color frame too, so _new_frame never got set and
                   the display froze even though the camera was running.
        Fix:       Always forward the color frame.  If depth is invalid,
                   store depth_np=None.  Processing code handles None depth
                   by skipping 3-D position computation for that frame.

    FIX 2 — Catch ALL exceptions, not just RuntimeError.
        Old code:  except RuntimeError: pass
        Problem:   wait_for_frames() can raise RuntimeError on timeout AND
                   various SDK errors.  Other exception types (ValueError,
                   OSError) were propagating uncaught and killing the thread
                   silently — meaning _new_frame was never set again.
        Fix:       except Exception: log and sleep briefly, keep looping.

    FIX 3 — Automatic pipeline restart after sustained timeout.
        If no valid frame arrives for PIPELINE_RESTART_TIMEOUT_S seconds
        (default 3 s), stop and restart the pipeline entirely.  This
        recovers from the D435i firmware hang that sometimes occurs on
        static scenes without needing to restart the whole node.

    FIX 4 — Post-processing filters applied in this thread.
        Hole-filling and temporal filtering run here so the processing
        thread gets clean depth data with no extra latency penalty.
    """

    def __init__(self, pipeline, align, depth_scale, intr,
                 hole_filter, temporal_filter):
        super().__init__(daemon=True, name="FrameGrabber")

        self._pipeline       = pipeline
        self._align          = align
        self._depth_scale    = depth_scale
        self._intr           = intr
        self._hole_filter    = hole_filter
        self._temporal_filter= temporal_filter

        self._lock           = threading.Lock()
        self._latest         = None          # (color_np, depth_np | None)
        self._new_frame      = threading.Event()
        self._running        = True

        self._last_frame_ts  = time.time()   # watchdog timestamp
        self._grab_times     = deque(maxlen=30)
        self._frame_count    = 0
        self._error_count    = 0

    # ── Public API ────────────────────────────────────────────────────────

    def get_latest_frame(self, timeout: float = 0.1):
        """
        Returns (color_np, depth_np) when a new frame is ready.
        depth_np may be None if the depth stream returned invalid data.
        Returns (None, None) on timeout.
        """
        if not self._new_frame.wait(timeout):
            return None, None
        self._new_frame.clear()
        with self._lock:
            return self._latest   # (color_np, depth_np | None)

    def stop(self):
        self._running = False

    @property
    def seconds_since_last_frame(self) -> float:
        return time.time() - self._last_frame_ts

    @property
    def depth_scale(self):
        return self._depth_scale

    @property
    def intr(self):
        return self._intr

    def avg_grab_ms(self) -> float:
        if not self._grab_times:
            return 0.0
        return sum(self._grab_times) / len(self._grab_times)

    # ── Thread body ───────────────────────────────────────────────────────

    def run(self):
        while self._running:
            try:
                self._grab_one_frame()

            except Exception as e:
                # FIX 2: catch everything so the thread never dies silently.
                self._error_count += 1
                print(f"[FrameGrabber] error #{self._error_count}: {e}")

                # FIX 3: if we haven't had a good frame for a while, restart.
                if self.seconds_since_last_frame > PIPELINE_RESTART_TIMEOUT_S:
                    print("[FrameGrabber] timeout — restarting pipeline...")
                    self._restart_pipeline()

                time.sleep(0.05)   # brief back-off before retry

    def _grab_one_frame(self):
        t0     = time.perf_counter()
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        self._grab_times.append((time.perf_counter() - t0) * 1000)

        aligned     = self._align.process(frames)
        color_frame = aligned.get_color_frame()

        # FIX 1a: only skip if the color frame itself is missing.
        if not color_frame:
            return

        color_np = np.asanyarray(color_frame.get_data()).copy()

        # FIX 1b: depth is optional — use None when invalid.
        depth_frame = aligned.get_depth_frame()
        if depth_frame:
            # Apply post-processing filters (hole-fill + temporal)
            depth_frame = self._hole_filter.process(depth_frame)
            depth_frame = self._temporal_filter.process(depth_frame)
            depth_np    = np.asanyarray(depth_frame.get_data()).copy()
        else:
            depth_np = None   # processing thread will skip 3-D coords this frame

        # Publish the latest frame and wake the processing thread
        with self._lock:
            self._latest = (color_np, depth_np)
        self._new_frame.set()

        self._last_frame_ts = time.time()
        self._frame_count  += 1

    def _restart_pipeline(self):
        """Stop and restart the RealSense pipeline in-place."""
        try:
            self._pipeline.stop()
        except Exception:
            pass
        time.sleep(1.0)
        try:
            pipeline, align, depth_scale, intr, hf, tf = build_realsense_pipeline()
            self._pipeline        = pipeline
            self._align           = align
            self._depth_scale     = depth_scale
            self._intr            = intr
            self._hole_filter     = hf
            self._temporal_filter = tf
            self._last_frame_ts   = time.time()
            print("[FrameGrabber] pipeline restarted successfully.")
        except Exception as e:
            print(f"[FrameGrabber] restart failed: {e}")


# ══════════════════════════════════════════════════════════════════════════
# DEPTH / GEOMETRY
# ══════════════════════════════════════════════════════════════════════════

def sample_depth_np(depth_np,          # np.ndarray | None
                    depth_scale: float,
                    cx: int, cy: int,
                    half: int = DEPTH_SAMPLE_HALF) -> float | None:
    """
    Fast median depth (metres) via numpy slice.
    Returns None if depth_np is None (invalid frame) or all pixels are zero.
    """
    if depth_np is None:               # FIX 1 downstream: handle None depth
        return None

    h, w = depth_np.shape[:2]
    x1, x2 = max(0, cx-half), min(w, cx+half)
    y1, y2 = max(0, cy-half), min(h, cy+half)

    region = depth_np[y1:y2, x1:x2].astype(np.float32)
    valid  = region[region > 0]

    if valid.size == 0:
        return None

    dm = float(np.median(valid)) * depth_scale
    return dm if dm > 0.05 else None


def world_xyz(px, py, depth_m, intr):
    X = (px - intr.ppx) * depth_m / intr.fx
    Y = (py - intr.ppy) * depth_m / intr.fy
    return (X, Y, depth_m)


def dist3d(p1, p2):
    return math.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))


def bot_to_spearhead(wp):
    return math.sqrt(sum(v**2 for v in wp))


# ══════════════════════════════════════════════════════════════════════════
# ID ASSIGNMENT  (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════

def _enumerate_alignments(detections, confirmed_picked):
    n = len(detections)
    if n == 0: return [()]
    results = []
    def bt(i, last, cur):
        if i == n:
            results.append(tuple(cur)); return
        avail = VALID_IDS_FOR_LABEL.get(detections[i]["label"], set()) - confirmed_picked
        for cid in sorted(avail):
            if cid > last:
                cur.append(cid); bt(i+1, cid, cur); cur.pop()
    bt(0, 0, [])
    return results


def _score_alignment(alignment, detections, tracked_targets):
    total = 0.0
    for det, aid in zip(detections, alignment):
        if aid not in tracked_targets:
            total += 50.0; continue
        tracked = tracked_targets[aid]
        lp = 0.0 if det["label"] == tracked["label"] else LABEL_MISMATCH_PENALTY
        if "world_pos" in det and "world_pos" in tracked:
            total += WORLD_DIST_WEIGHT * dist3d(det["world_pos"], tracked["world_pos"]) + lp
        else:
            cx,cy = det["center"]; tcx,tcy = tracked["center"]
            x1,y1,x2,y2 = det["box"]
            da = max(1,(x2-x1)*(y2-y1)); ta = max(1, tracked["area"])
            total += (PIXEL_DIST_WEIGHT * math.sqrt((cx-tcx)**2+(cy-tcy)**2)
                      + AREA_DIFF_WEIGHT * abs(da-ta)/max(da,ta) + lp)
    return total


def assign_ids(detections, tracked_targets, confirmed_picked):
    if not detections: return []
    alignments = _enumerate_alignments(detections, confirmed_picked)
    if not alignments:
        return _fallback_tracking(detections, tracked_targets, confirmed_picked)
    if len(alignments) == 1:
        best = alignments[0]
    else:
        scored = sorted((_score_alignment(a, detections, tracked_targets), a)
                        for a in alignments)
        best_cost, best = scored[0]
        if best_cost >= 50.0 * len(detections):
            best = min(alignments, key=lambda a: (a[0], a[-1]))
    return [{**d, "id": aid} for d, aid in zip(detections, best)]


def _fallback_tracking(detections, tracked_targets, confirmed_picked):
    result, used = [], set()
    for det in detections:
        avail = VALID_IDS_FOR_LABEL.get(det["label"], set()) - confirmed_picked
        best_id, best_score = None, float("inf")
        for tid, tracked in tracked_targets.items():
            if tid in used or tid not in avail: continue
            lp = 0.0 if det["label"] == tracked["label"] else LABEL_MISMATCH_PENALTY
            if "world_pos" in det and "world_pos" in tracked:
                score = WORLD_DIST_WEIGHT * dist3d(det["world_pos"], tracked["world_pos"]) + lp
            else:
                cx,cy = det["center"]; tcx,tcy = tracked["center"]
                score = PIXEL_DIST_WEIGHT*math.sqrt((cx-tcx)**2+(cy-tcy)**2) + lp
            if score < best_score:
                best_score, best_id = score, tid
        if best_id is not None and best_score < TRACKING_CONFIDENCE_THRESHOLD:
            result.append({**det, "id": best_id}); used.add(best_id)
    return result


# ══════════════════════════════════════════════════════════════════════════
# ROS NODE
# ══════════════════════════════════════════════════════════════════════════

class SpearheadDetectorNode(Node):

    def __init__(self):
        super().__init__("spearhead_detector_node")

        self.publisher_    = self.create_publisher(Int32,   "/spearhead/target_id",      10)
        self.dist_pub_     = self.create_publisher(Float32, "/spearhead/target_depth_m", 10)
        self.apriltag_pub  = self.create_publisher(Bool,    "/apriltag/status",          10)
        self.apriltag_pub2 = self.create_publisher(Bool,    "/apriltag/loop",            10)

        self.model_ = YOLO(MODEL_PATH)
        self.apriltag_detector = Detector(
            families="tag36h11", nthreads=2,
            quad_decimate=2.0, quad_sigma=0.0, refine_edges=1,
        )

        self.tracked_targets  : dict = {}
        self.frames_absent    : dict = {i: 0 for i in range(1, 7)}
        self.confirmed_picked : set  = set()
        self._frame_count     = 0
        self._timer           = StageTimer()

        self.run_detection_loop()

    # ── Tracking helpers ──────────────────────────────────────────────────

    def _update_tracking(self, targets):
        for t in targets:
            x1,y1,x2,y2 = t["box"]
            e = {"center":t["center"],"label":t["label"],
                 "area":(x2-x1)*(y2-y1),"bbox":t["box"]}
            if "world_pos" in t: e["world_pos"] = t["world_pos"]
            if "depth_m"   in t: e["depth_m"]   = t["depth_m"]
            self.tracked_targets[t["id"]] = e

    def _update_pickup_state(self, visible_ids):
        for aid in range(1, 7):
            if aid in self.confirmed_picked: continue
            if aid in visible_ids:
                self.frames_absent[aid] = 0
            else:
                self.frames_absent[aid] += 1
                if self.frames_absent[aid] >= PICKUP_ABSENCE_FRAMES:
                    self.confirmed_picked.add(aid)
                    self.tracked_targets.pop(aid, None)
                    self.get_logger().info(
                        f"[pickup] ID {aid} collected. "
                        f"Remaining: {sorted(set(range(1,7))-self.confirmed_picked)}"
                    )

    def _check_and_readmit(self, detections):
        from collections import Counter
        for label, count in Counter(d["label"] for d in detections).items():
            avail = VALID_IDS_FOR_LABEL.get(label, set()) - self.confirmed_picked
            if count > len(avail):
                for pid in sorted(VALID_IDS_FOR_LABEL.get(label, set())
                                  & self.confirmed_picked):
                    self.confirmed_picked.discard(pid)
                    self.frames_absent[pid] = 0
                    self.get_logger().info(f"[readmit] ID {pid} ({label}) returned.")

    def _compute_distances(self, targets):
        bot_dist, inter_dist = {}, {}
        for t in targets:
            if "world_pos" in t:
                bot_dist[t["id"]] = bot_to_spearhead(t["world_pos"])
        id_map     = {t["id"]: t for t in targets if "world_pos" in t}
        sorted_ids = sorted(id_map.keys())
        for i in range(len(sorted_ids)-1):
            a, b = sorted_ids[i], sorted_ids[i+1]
            inter_dist[(a,b)] = dist3d(id_map[a]["world_pos"], id_map[b]["world_pos"])
        return bot_dist, inter_dist

    def _select_priority_target(self, id_map, bot_dist):
        for pid in PRIORITY_ORDER:
            if pid not in id_map: continue
            if pid in bot_dist and bot_dist[pid] > WAYPOINT_DISTANCE_GATE_M: continue
            return pid, id_map[pid]
        return 0, None

    # ── Main loop ─────────────────────────────────────────────────────────

    def run_detection_loop(self):

        pipeline, align, depth_scale, intr, hf, tf = build_realsense_pipeline()

        grabber = FrameGrabber(pipeline, align, depth_scale, intr, hf, tf)
        grabber.start()

        self.get_logger().info(
            f"D435i ready | fx={intr.fx:.1f} | "
            f"depth_scale={depth_scale} | {intr.width}×{intr.height}"
        )

        last_tags      = []
        last_at_data   = False
        last_atl_data  = False

        try:
            while True:
                self._timer.tick("wait")

                color_np, depth_np = grabber.get_latest_frame(timeout=0.1)

                # depth_np may be None — that is now normal and handled below.
                # Only skip if color is missing.
                if color_np is None:
                    # Show a "waiting" overlay on the last known frame
                    # so the window doesn't appear frozen.
                    cv2.waitKey(1)
                    continue

                frame_height, frame_width = color_np.shape[:2]
                self._timer.tick("grab")

                # ── YOLO ─────────────────────────────────────────────────
                yolo_result = self.model_.predict(
                    color_np, conf=0.7, iou=0.5, imgsz=640, verbose=False
                )[0]
                self._timer.tick("yolo")

                # ── Build detections ──────────────────────────────────────
                raw_detections = []
                if yolo_result.boxes is not None:
                    for box in yolo_result.boxes:
                        cls  = int(box.cls[0])
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        label = self.model_.names[cls].lower()
                        if   "spear" in label: label = "spear"
                        elif "fist"  in label: label = "fist"
                        elif "palm"  in label: label = "palm"
                        else: continue

                        cx=(x1+x2)//2; cy=(y1+y2)//2
                        det = {"label":label,"box":(x1,y1,x2,y2),
                               "center":(cx,cy),"conf":float(box.conf[0])}

                        # sample_depth_np handles depth_np=None gracefully
                        dm = sample_depth_np(depth_np, depth_scale, cx, cy)
                        if dm is not None:
                            det["depth_m"]   = dm
                            det["world_pos"] = world_xyz(cx, cy, dm, intr)
                            det["world_x"]   = det["world_pos"][0]

                        raw_detections.append(det)

                raw_detections.sort(key=lambda d: d.get("world_x", d["center"][0]))
                self._timer.tick("depth")

                # ── ID assignment ─────────────────────────────────────────
                self._check_and_readmit(raw_detections)
                targets = assign_ids(raw_detections, self.tracked_targets,
                                     self.confirmed_picked)
                self._update_tracking(targets)
                self._update_pickup_state({t["id"] for t in targets})
                self._timer.tick("assign")

                # ── AprilTag every Nth frame ──────────────────────────────
                self._frame_count += 1
                if self._frame_count % APRILTAG_EVERY_N_FRAMES == 0:
                    gray = cv2.cvtColor(color_np, cv2.COLOR_BGR2GRAY)
                    last_tags     = self.apriltag_detector.detect(gray)
                    last_at_data  = any(t.tag_id in VALID_TAGS      for t in last_tags)
                    last_atl_data = any(t.tag_id in VALID_TAGS_LOOP  for t in last_tags)
                self._timer.tick("apriltag")

                # ── Publish ───────────────────────────────────────────────
                bot_dist, inter_dist = self._compute_distances(targets)
                id_map = {t["id"]: t for t in targets}
                priority_id, best_target = self._select_priority_target(id_map, bot_dist)

                m = Int32();   m.data = priority_id;                    self.publisher_.publish(m)
                dm = Float32();dm.data = float(bot_dist.get(priority_id,-1.0)); self.dist_pub_.publish(dm)
                a  = Bool();   a.data  = last_at_data;                  self.apriltag_pub.publish(a)
                al = Bool();   al.data = last_atl_data;                 self.apriltag_pub2.publish(al)
                self._timer.tick("publish")

                # ── Draw ──────────────────────────────────────────────────
                frame = color_np.copy()

                for tag in last_tags:
                    corners = tag.corners.astype(int)
                    for i in range(4):
                        cv2.line(frame,tuple(corners[i]),
                                 tuple(corners[(i+1)%4]),(255,0,0),2)
                    cv2.putText(frame,f"TAG {tag.tag_id}",
                                tuple(tag.center.astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

                for t in targets:
                    x1,y1,x2,y2 = t["box"]
                    bd  = bot_dist.get(t["id"])
                    ds  = f" {bd:.2f}m" if bd is not None else " (no depth)"
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                    cv2.putText(frame,f"{t['label']} ID:{t['id']}{ds}",
                                (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)

                if best_target is not None:
                    x1,y1,x2,y2 = best_target["box"]
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                    cv2.putText(frame,f"TARGET ID:{priority_id}",
                                (20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                else:
                    cv2.putText(frame,"NO TARGET",(20,40),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

                # Depth stream health indicator
                depth_ok = depth_np is not None
                dcolor   = (0,200,0) if depth_ok else (0,0,255)
                cv2.putText(frame,
                            f"DEPTH:{'OK' if depth_ok else 'INVALID — no 3D'}",
                            (20,200),cv2.FONT_HERSHEY_SIMPLEX,0.65,dcolor,2)

                # Grabber watchdog indicator
                stall = grabber.seconds_since_last_frame
                wcolor = (0,200,0) if stall < 1.0 else (0,0,255)
                cv2.putText(frame,f"CAM stall:{stall:.1f}s",
                            (20,230),cv2.FONT_HERSHEY_SIMPLEX,0.55,wcolor,1)

                y_off = frame_height - 20
                for (a_id,b_id),d in sorted(inter_dist.items()):
                    cv2.putText(frame,f"ID{a_id}↔ID{b_id}:{d:.2f}m",
                                (10,y_off),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,255),1)
                    y_off -= 20

                picked_str = str(sorted(self.confirmed_picked)) if self.confirmed_picked else "none"
                cv2.putText(frame,f"PICKED:{picked_str}",
                            (20,160),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,165,255),2)
                cv2.putText(frame,f"ASSEMBLY:{last_at_data}",
                            (20,80), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                cv2.putText(frame,f"LOOP:{last_atl_data}",
                            (20,120),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                cv2.imshow("Spearhead Detector", frame)
                self._timer.tick("draw")
                self._timer.report()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            grabber.stop()
            grabber.join(timeout=2.0)
            pipeline.stop()
            cv2.destroyAllWindows()

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpearheadDetectorNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()