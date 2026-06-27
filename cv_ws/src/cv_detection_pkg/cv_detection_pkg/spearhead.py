import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool

from ultralytics import YOLO
from pupil_apriltags import Detector

import cv2


# ── Model Path ─────────────────────────────────────────────────────────────
MODEL_PATH = "/home/luffy/PycharmProjects/KFS_detection/runs/detect/SPEARHEAD_MAY/spearhead_may/weights/best.pt"

# ── Priority Order ────────────────────────────────────────────────────────
PRIORITY_ORDER = [5, 6, 2, 1, 3, 4]

# ── Fixed Arena Sequence ──────────────────────────────────────────────────
FULL_SEQUENCE = [
    "spear",
    "fist",
    "palm",
    "palm",
    "fist",
    "spear"
]

# ── Valid AprilTag IDs ────────────────────────────────────────────────────
# ONLY these IDs will publish True
VALID_TAGS = [4]


class SpearheadDetectorNode(Node):

    def __init__(self):

        super().__init__('spearhead_detector_node')

        # ── ROS Publishers ──────────────────────────────────────────────
        self.publisher_ = self.create_publisher(
            Int32,
            '/spearhead/target_id',
            10
        )

        self.apriltag_pub = self.create_publisher(
            Bool,
            '/apriltag/status',
            10
        )

        self.get_logger().info(
            "Loading YOLO model..."
        )

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

        self.get_logger().info(
            "YOLO + AprilTag Ready"
        )

        # ── Start Detection Loop ────────────────────────────────────────
        self.run_detection_loop()

    # ──────────────────────────────────────────────────────────────────────
    def run_detection_loop(self):

        # ── Camera Source ───────────────────────────────────────────────
        CAMERA_SOURCE = 1

        # Example URL Camera:
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

            frame = r.orig_img

            boxes = r.boxes

            raw_detections = []

            # ────────────────────────────────────────────────────────────
            # YOLO DETECTION
            # ────────────────────────────────────────────────────────────
            if boxes is not None:

                for box in boxes:

                    cls = int(box.cls[0])

                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    label = self.model_.names[cls].lower()

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    raw_detections.append({

                        "label": label,

                        "box": (x1, y1, x2, y2),

                        "center": (cx, cy),

                        "conf": conf
                    })

            # ── LEFT → RIGHT SORTING ────────────────────────────────────
            raw_detections.sort(
                key=lambda d: d["center"][0]
            )

            # ────────────────────────────────────────────────────────────
            # BUILD DETECTED SEQUENCE
            # ────────────────────────────────────────────────────────────
            detected_sequence = []

            filtered_detections = []

            for det in raw_detections:

                label = det["label"].lower()

                if "spear" in label:

                    detected_sequence.append("spear")

                    filtered_detections.append(det)

                elif "fist" in label:

                    detected_sequence.append("fist")

                    filtered_detections.append(det)

                elif "palm" in label:

                    detected_sequence.append("palm")

                    filtered_detections.append(det)

# ────────────────────────────────────────────────────────────
            # ────────────────────────────────────────────────────────────
            # ROBUST ID MATCHING (MISSING OBJECT TOLERANCE)
            # ────────────────────────────────────────────────────────────

            targets = []

            full_idx = 0

            for det in filtered_detections:

                label = det["label"]

                matched = False

                while full_idx < len(FULL_SEQUENCE):

                    # Match current visible label
                    if FULL_SEQUENCE[full_idx] == label:

                        det["id"] = full_idx + 1

                        targets.append(det)

                        full_idx += 1

                        matched = True

                        break

                    # Skip missing detections
                    full_idx += 1

                # Reset if sequence breaks
                if not matched:

                    full_idx = 0
            # ────────────────────────────────────────────────────────────
            # PRIORITY SELECTION
            # ────────────────────────────────────────────────────────────
            id_map = {

                t["id"]: t

                for t in targets
            }

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
            msg = Int32()

            msg.data = priority_id

            self.publisher_.publish(msg)

            # ────────────────────────────────────────────────────────────
            # APRILTAG DETECTION
            # ────────────────────────────────────────────────────────────
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            tags = self.apriltag_detector.detect(gray)

            apriltag_msg = Bool()

            # Default → False
            apriltag_msg.data = False

            for tag in tags:

                # ONLY selected IDs
                if tag.tag_id in VALID_TAGS:

                    apriltag_msg.data = True

                    break

            # Publish AprilTag status
            self.apriltag_pub.publish(apriltag_msg)

            # ────────────────────────────────────────────────────────────
            # DRAW APRILTAGS
            # ────────────────────────────────────────────────────────────
            for tag in tags:

                corners = tag.corners.astype(int)

                for i in range(4):

                    pt1 = tuple(corners[i])

                    pt2 = tuple(corners[(i + 1) % 4])

                    cv2.line(
                        frame,
                        pt1,
                        pt2,
                        (255, 0, 0),
                        2
                    )

                center = tuple(tag.center.astype(int))

                cv2.putText(
                    frame,
                    f"TAG {tag.tag_id}",
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

            # ────────────────────────────────────────────────────────────
            # DRAW YOLO TARGETS
            # ────────────────────────────────────────────────────────────
            for t in targets:

                x1, y1, x2, y2 = t["box"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{t['label']} | ID:{t['id']}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # ────────────────────────────────────────────────────────────
            # HIGHLIGHT PRIORITY TARGET
            # ────────────────────────────────────────────────────────────
            if best_target is not None:

                x1, y1, x2, y2 = best_target["box"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    f"TARGET ID: {priority_id}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            else:

                cv2.putText(
                    frame,
                    "NO TARGET",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # ────────────────────────────────────────────────────────────
            # APRILTAG STATUS TEXT
            # ────────────────────────────────────────────────────────────
            cv2.putText(
                frame,
                f"APRILTAG: {apriltag_msg.data}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3
            )

            # ────────────────────────────────────────────────────────────
            # DISPLAY
            # ────────────────────────────────────────────────────────────
            cv2.imshow(
                "YOLO + AprilTag Detection",
                frame
            )

            # ────────────────────────────────────────────────────────────
            # QUIT
            # ────────────────────────────────────────────────────────────
            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

        cv2.destroyAllWindows()

    # ──────────────────────────────────────────────────────────────────────
    def destroy_node(self):

        super().destroy_node()


# ──────────────────────────────────────────────────────────────────────────
def main(args=None):

    rclpy.init(args=args)

    node = SpearheadDetectorNode()

    node.destroy_node()

    rclpy.shutdown()


# ──────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    main()