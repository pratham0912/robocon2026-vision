import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool

from ultralytics import YOLO
from pupil_apriltags import Detector
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

import cv2
import math


# ── Model Path ─────────────────────────────────────────────────────────────
pkg_path = get_package_share_directory("cv_detection_pkg")
MODEL_PATH = Path(pkg_path) / "models" / "spearhead" / "best.pt"


# ── Priority Order ────────────────────────────────────────────────────────
PRIORITY_ORDER = [1, 2, 3, 5, 6, 4]

# ── Fixed Arena Sequence ──────────────────────────────────────────────────
FULL_SEQUENCE = [
    "spear",   # ID 1
    "fist",    # ID 2
    "palm",    # ID 3
    "palm",    # ID 4
    "fist",    # ID 5
    
    "spear",   # ID 6
]

VALID_IDS = {
    "spear": [1, 6],
    "fist": [2, 5],
    "palm": [3, 4]
}

# ── Valid AprilTag IDs ────────────────────────────────────────────────────
VALID_TAGS = [4]
VALID_TAGS_LOOP = [ 5]

GAP_MULTIPLIER = 1.3

CENTER_DIST_WEIGHT     = 1.0    # weight on Euclidean centre-pixel distance
AREA_DIFF_WEIGHT       = 50.0   # weight on normalised bounding-box area difference
LABEL_MISMATCH_PENALTY = 200.0  # hard penalty when detection label ≠ tracked label

# A combined score below this threshold is treated as a confident track hit.
TRACKING_CONFIDENCE_THRESHOLD = 150.0


class SpearheadDetectorNode(Node):

    def __init__(self):
        
        super().__init__('spearhead_detector_node')

        # ── ROS Publishers ──────────────────────────────────────────────
        self.publisher_ = self.create_publisher(Int32,'/spearhead/target_id',10)    #spearhead priority ID (1–6, or 0 if no target)
        
        self.apriltag_pub = self.create_publisher(Bool,'/apriltag/status',10)   #AprilTag in Bool
        self.apriltag_pub2 = self.create_publisher(Bool,'/apriltag/loop',10)   #AprilTag in Bool for loop
        
        # self.get_logger().info("Loading YOLO model...")

        # ── YOLO Model ───────────────────────────────────────────────────
        self.model_ = YOLO(MODEL_PATH)

        # ── AprilTag Detector ───────────────────────────────────────────
        self.apriltag_detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1
        )

        # ── Tracking Memory ──────────────────────────────────────────────
        # Persists the last confirmed state for each arena ID (1–6).
        # Entries are retained even when an object is temporarily invisible
        # so that re-identification works when it reappears.
        #
        # Schema:
        #   self.tracked_targets = {
        #       id (int): {
        #           "center": (cx, cy),   # pixel centre from last seen frame
        #           "label" : str,        # "spear" | "fist" | "palm"
        #           "area"  : int,        # bounding-box pixel area
        #           "bbox"  : (x1,y1,x2,y2)
        #       },
        #       ...
        #   }
        self.tracked_targets = {}

        # self.get_logger().info("YOLO + AprilTag Ready")

        # ── Start Detection Loop ────────────────────────────────────────
        self.run_detection_loop()

    # ══════════════════════════════════════════════════════════════════════
    # GAP ESTIMATION
    # Compute horizontal spacing between consecutive sorted detections.
    # ══════════════════════════════════════════════════════════════════════
    def _estimate_gap(self, detections):
       
        if len(detections) < 2:
            return [], 0.0

        xs    = [d["center"][0] for d in detections]
        gaps  = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        
        # print("GAPS:", gaps)
        # print("AVG GAP:", avg_gap)

        return gaps, avg_gap

    # ══════════════════════════════════════════════════════════════════════
    # MISSING TARGET INFERENCE
    # Insert None placeholders where a large gap signals missing objects.
    # ══════════════════════════════════════════════════════════════════════
    def _expand_with_virtual_slots(self, detections, gaps, avg_gap):
        if not detections:
            return []

        expanded = [detections[0]]

        for i, gap in enumerate(gaps):
            if avg_gap > 0 and gap > GAP_MULTIPLIER * avg_gap:
                # ── Infer how many spearheads are hidden in this gap ────
                num_missing = max(1, round(gap / avg_gap) - 1)
                for _ in range(num_missing):
                    expanded.append(None)   # virtual missing slot

            expanded.append(detections[i + 1])
            # print("CURRENT GAP:", gap)
            # print("THRESHOLD:", GAP_MULTIPLIER * avg_gap)

        return expanded

    # ══════════════════════════════════════════════════════════════════════
    # ID ASSIGNMENT  –  subsequence alignment + gap + frame-side weighting
    # ══════════════════════════════════════════════════════════════════════
    def _assign_ids(self, expanded):

        if not expanded:
            return []

        targets = []

        for i, item in enumerate(expanded):

            if item is None:
                continue

            assigned_id = i + 1

            if assigned_id > 6:
                continue

            item = dict(item)
            item["id"] = assigned_id

            targets.append(item)

        return targets

    # ══════════════════════════════════════════════════════════════════════
    # TRACKING – match a detection to the closest previous tracked target
    # ══════════════════════════════════════════════════════════════════════
    def _match_to_tracked(self, detection):
       
        if not self.tracked_targets:
            return None, float('inf')

        cx, cy          = detection["center"]
        x1, y1, x2, y2 = detection["box"]
        det_area        = max(1, (x2 - x1) * (y2 - y1))
        label           = detection["label"]

        best_id    = None
        best_score = float('inf')

        for tid, tracked in self.tracked_targets.items():
            tcx, tcy = tracked["center"]

            # ── Centre-pixel Euclidean distance ─────────────────────────
            centre_dist = math.sqrt((cx - tcx) ** 2 + (cy - tcy) ** 2)

            # ── Normalised bounding-box area difference ──────────────────
            t_area         = max(1, tracked["area"])
            max_area       = max(det_area, t_area)
            norm_area_diff = abs(det_area - t_area) / max_area

            # ── Label consistency ────────────────────────────────────────
            label_penalty = 0.0 if label == tracked["label"] else LABEL_MISMATCH_PENALTY

            # ── Combined score ───────────────────────────────────────────
            score = (CENTER_DIST_WEIGHT * centre_dist
                     + AREA_DIFF_WEIGHT * norm_area_diff
                     + label_penalty)

            if score < best_score:
                best_score = score
                best_id    = tid

        return best_id, best_score

    # ══════════════════════════════════════════════════════════════════════
    # TRACKING – persist confirmed assignments into memory
    # ══════════════════════════════════════════════════════════════════════
    def _update_tracking(self, targets):
        
        for t in targets:
            x1, y1, x2, y2 = t["box"]
            area = (x2 - x1) * (y2 - y1)
            self.tracked_targets[t["id"]] = {
                "center": t["center"],
                "label":  t["label"],
                "area":   area,
                "bbox":   t["box"],
            }

    # ══════════════════════════════════════════════════════════════════════
    # MAIN DETECTION LOOP
    # ══════════════════════════════════════════════════════════════════════
    def run_detection_loop(self):

        # ── Camera Source ───────────────────────────────────────────────
        CAMERA_SOURCE = 0
        # Example URL camera:
        # CAMERA_SOURCE = "http://192.168.0.101:4747/video"

        results = self.model_.predict(
            source=CAMERA_SOURCE,
            conf=0.40,
            iou=0.4,
            imgsz=800,
            stream=True,
            verbose=False
        )

        for r in results:

            frame  = r.orig_img
            boxes  = r.boxes
            frame_height, frame_width = frame.shape[:2]

            raw_detections = []

            # ────────────────────────────────────────────────────────────
            # YOLO DETECTION
            # Collect all bounding boxes from this frame.
            # ────────────────────────────────────────────────────────────
            if boxes is not None:

                for box in boxes:

                    cls  = int(box.cls[0])
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    label = self.model_.names[cls].lower()

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    raw_detections.append({
                        "label":  label,
                        "box":    (x1, y1, x2, y2),
                        "center": (cx, cy),
                        "conf":   conf,
                    })

            # ── LEFT → RIGHT SORTING ────────────────────────────────────
            raw_detections.sort(key=lambda d: d["center"][0])

            filtered = []
            for det in raw_detections:
                lbl = det["label"]
                if "spear" in lbl:
                    det["label"] = "spear"
                    filtered.append(det)
                elif "fist" in lbl:
                    det["label"] = "fist"
                    filtered.append(det)
                elif "palm" in lbl:
                    det["label"] = "palm"
                    filtered.append(det)

           
            gaps, avg_gap = self._estimate_gap(filtered)
            expanded = self._expand_with_virtual_slots(filtered, gaps, avg_gap)
            gap_targets = self._assign_ids(expanded)

            targets          = []
            used_tracked_ids = set()   # prevent two detections claiming the same ID

            for det in gap_targets:

                candidate_id = det.get("id")
                best_tracked_id, track_score = self._match_to_tracked(det)

                # Label → Valid ID check
                if best_tracked_id is not None:

                    valid_ids = {
                     "spear": [1, 6],
                     "fist": [2, 5],
                     "palm": [3, 4]
                }

                    if best_tracked_id not in valid_ids.get(det["label"], []):
                        best_tracked_id = None

                if (best_tracked_id is not None
                    and track_score < TRACKING_CONFIDENCE_THRESHOLD
                    and best_tracked_id not in used_tracked_ids):

                    final_id = best_tracked_id
                    used_tracked_ids.add(best_tracked_id)

                elif candidate_id is not None:
                    final_id = candidate_id

                else:
                    continue

                det         = dict(det)   # avoid mutating the original
                det["id"]   = final_id
                targets.append(det)

            # ── Persist confirmed IDs into tracking memory ─────────────
            self._update_tracking(targets)

            # ────────────────────────────────────────────────────────────
            # PRIORITY SELECTION
            # Walk PRIORITY_ORDER and publish the first ID that is visible.
            # ────────────────────────────────────────────────────────────
            id_map      = {t["id"]: t for t in targets}
            priority_id = 0
            best_target = None

            for pid in PRIORITY_ORDER:
                if pid in id_map:
                    priority_id = pid
                    best_target = id_map[pid]
                    break

            # ────────────────────────────────────────────────────────────
            # PUBLISH SPEARHEAD PRIORITY
            # ────────────────────────────────────────────────────────────
            msg      = Int32()
            msg.data = priority_id
            self.publisher_.publish(msg)

            # ────────────────────────────────────────────────────────────
            # APRILTAG DETECTION 
            # ────────────────────────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = self.apriltag_detector.detect(gray)

            apriltag_msg      = Bool()
            apriltag_loop_msg      = Bool()
            
            apriltag_msg.data = False   # default → False
            apriltag_loop_msg.data = False   # default → False

            for tag in tags:
                if tag.tag_id in VALID_TAGS:   # only selected IDs publish True
                    apriltag_msg.data = True
                
                if tag.tag_id in VALID_TAGS_LOOP:
                    apriltag_loop_msg.data = True

            self.apriltag_pub.publish(apriltag_msg)
            self.apriltag_pub2.publish(apriltag_loop_msg)

            # ────────────────────────────────────────────────────────────
            # DRAW APRILTAGS  
            # ────────────────────────────────────────────────────────────
            for tag in tags:

                corners = tag.corners.astype(int)

                for i in range(4):
                    pt1 = tuple(corners[i])
                    pt2 = tuple(corners[(i + 1) % 4])
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

                center = tuple(tag.center.astype(int))
                cv2.putText(frame,f"TAG {tag.tag_id}",center,cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)

            # ────────────────────────────────────────────────────────────
            # DRAW YOLO TARGETS  –  green boxes + assigned ID  (unchanged)
            # ────────────────────────────────────────────────────────────
            for t in targets:
                x1, y1, x2, y2 = t["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame,f"{t['label']} | ID:{t['id']}",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

            # ────────────────────────────────────────────────────────────
            # HIGHLIGHT PRIORITY TARGET  –  red box  (unchanged)
            # ────────────────────────────────────────────────────────────
            if best_target is not None:
                x1, y1, x2, y2 = best_target["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame,f"TARGET ID: {priority_id}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)

            else:
                cv2.putText(frame, "NO TARGET",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)


            cv2.putText(frame,f"ASSEMBLY: {apriltag_msg.data}",(20, 80),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
            cv2.putText(frame,f"LOOP: {apriltag_loop_msg.data}",(20, 120),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
            cv2.imshow("YOLO + AprilTag Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SpearheadDetectorNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()