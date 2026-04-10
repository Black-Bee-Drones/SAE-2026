import time

import cv2
from datetime import datetime
from pathlib import Path

import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.control import MavrosDrone, MoveReference
from nectar.vision import ImageHandler
from nectar.ai.detection import Detector

from hook.core.constants import (
    SPHERE_CONF_THRESHOLD,
    SPHERE_DETECTION_CONFIRMATIONS,
    SPHERE_DETECT_TIMEOUT,
    YAW_SCAN_VELOCITY,
    SAVE_DETECTIONS,
    DETECTION_SAVE_PATH,
)


class DetectSphere(State):
    """Detect the orange sphere from search altitude to identify the correct rope."""

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.save_dir = None
        self.frame_count = 0

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        detector: Detector = blackboard["sphere_detector"]

        if SAVE_DETECTIONS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blackboard["mission_timestamp"] = timestamp
            self.save_dir = Path(DETECTION_SAVE_PATH) / timestamp / "detect_sphere"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        yasmin.YASMIN_LOG_INFO("Searching for orange sphere...")

        detection_count = 0
        latest_detection = None
        start_time = time.time()
        scanning = False

        while time.time() - start_time < SPHERE_DETECT_TIMEOUT:
            frame = camera.take_photo()
            if frame is None:
                time.sleep(0.05)
                continue

            result = detector.detect(frame, conf=SPHERE_CONF_THRESHOLD)

            if len(result) > 0:
                best = max(result.detections, key=lambda d: d.confidence)
                detection_count += 1
                latest_detection = best
                scanning = False
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)

                yasmin.YASMIN_LOG_INFO(
                    f"Sphere detected ({detection_count}/{SPHERE_DETECTION_CONFIRMATIONS}): "
                    f"conf={best.confidence:.2f}, center=({best.center[0]:.0f}, {best.center[1]:.0f})"
                )

                if SAVE_DETECTIONS and self.save_dir:
                    self.frame_count += 1
                    annotated = detector.draw_detections(frame, result)
                    cv2.imwrite(
                        str(self.save_dir / f"sphere_{self.frame_count:04d}.jpg"),
                        annotated,
                    )

                if detection_count >= SPHERE_DETECTION_CONFIRMATIONS:
                    cx, cy = latest_detection.center
                    blackboard["sphere_center"] = (cx, cy)
                    blackboard["sphere_bbox"] = latest_detection.xyxy.tolist()
                    yasmin.YASMIN_LOG_INFO(
                        f"Sphere confirmed at pixel ({cx:.0f}, {cy:.0f})."
                    )
                    return SUCCEED
            else:
                detection_count = 0
                if not scanning:
                    yasmin.YASMIN_LOG_INFO("No sphere found, scanning with yaw rotation...")
                    scanning = True
                drone.move_velocity(
                    vyaw=YAW_SCAN_VELOCITY, reference=MoveReference.BODY
                )

            time.sleep(0.05)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, duration=1.0)
        yasmin.YASMIN_LOG_ERROR("Sphere detection timed out.")
        return ABORT
