"""Final descent: same dual-anchor PID as AlignToHose, with vz < 0 on LIDAR."""

import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import rclpy
import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
from yasmin_ros.yasmin_node import YasminNode

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import Segmentor
from nectar.control import AltitudeSource, MavrosDrone, MoveReference, PIDController
from nectar.vision import ImageHandler

from hook.core.constants import (
    DESCEND_ANCHOR_KP,
    DESCEND_ANCHOR_TOLERANCE_PX,
    DESCEND_ANGLE_KP,
    DESCEND_ANGLE_TOLERANCE_DEG,
    DESCEND_CENTER_KP,
    DESCEND_CENTER_TOLERANCE_PX,
    DESCEND_MAX_LOST_FRAMES,
    DESCEND_MAX_VELOCITY_XY,
    DESCEND_MAX_YAW_VELOCITY,
    DESCEND_RELEASE_CONFIRMATIONS,
    DESCEND_TIMEOUT,
    DESCEND_VELOCITY,
    DETECTION_SAVE_PATH,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    RELEASE_ALTITUDE,
    SAVE_DETECTIONS,
    SPHERE_ANCHOR_DISTANCE_M,
)
from hook.core.perception import (
    best_sphere,
    hook_image_offset,
    hose_pose,
    hose_segments,
    pick_hose_by_dir,
    px_per_meter,
    run_seg,
)


class DescendAndAlign(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.pid_center = None
        self.pid_angle = None
        self.pid_anchor = None
        self.save_dir = None
        self.frame_count = 0

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        segmentor: Segmentor = blackboard["segmentor"]
        class_filter: PerClassConfidenceFilter = blackboard["class_filter"]
        side_unit: Tuple[float, float] = blackboard["hose_side_image_unit"]
        anchor_sign: int = (
            blackboard["anchor_sign"] if "anchor_sign" in blackboard else 1
        )

        if self.pid_center is None:
            self.pid_center = PIDController(
                kp=DESCEND_CENTER_KP,
                setpoint=IMAGE_CENTER_Y,
                output_limits=(-DESCEND_MAX_VELOCITY_XY, DESCEND_MAX_VELOCITY_XY),
            )
        if self.pid_angle is None:
            self.pid_angle = PIDController(
                kp=DESCEND_ANGLE_KP,
                setpoint=0.0,
                output_limits=(-DESCEND_MAX_YAW_VELOCITY, DESCEND_MAX_YAW_VELOCITY),
            )
        if self.pid_anchor is None:
            self.pid_anchor = PIDController(
                kp=DESCEND_ANCHOR_KP,
                setpoint=0.0,
                output_limits=(-DESCEND_MAX_VELOCITY_XY, DESCEND_MAX_VELOCITY_XY),
            )

        if SAVE_DETECTIONS:
            ts = (
                blackboard["mission_timestamp"]
                if "mission_timestamp" in blackboard
                else datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "descend"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        yasmin.YASMIN_LOG_INFO(f"Descending to {RELEASE_ALTITUDE}m on LIDAR...")

        confirmed = 0
        lost = 0
        start = time.time()

        while time.time() - start < DESCEND_TIMEOUT:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)

            altitude = drone.get_altitude(AltitudeSource.LIDAR)
            if altitude is None:
                altitude = drone.get_altitude(AltitudeSource.AUTO)
            if altitude is None:
                time.sleep(0.05)
                continue

            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            chosen = pick_hose_by_dir(sphere, hose_segments(result), side_unit)
            pose = hose_pose(chosen) if chosen is not None else None

            if sphere is None or pose is None:
                lost += 1
                if lost > DESCEND_MAX_LOST_FRAMES:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    yasmin.YASMIN_LOG_WARN("Lost sphere or chosen hose during descent.")
                    return ABORT
                drone.move_velocity(
                    vx=0.0,
                    vy=0.0,
                    vz=-DESCEND_VELOCITY,
                    vyaw=0.0,
                    reference=MoveReference.BODY,
                )
                time.sleep(0.05)
                continue

            lost = 0
            hose_cx, hose_cy, angle, _, _ = pose
            sphere_cx, _ = sphere.center

            ppm = px_per_meter(altitude)
            hook_dx, hook_dy = hook_image_offset(altitude)
            anchor_px = SPHERE_ANCHOR_DISTANCE_M * ppm
            target_hose_cy = IMAGE_CENTER_Y + hook_dy
            target_sphere_cx = IMAGE_CENTER_X + hook_dx + anchor_sign * anchor_px

            self.pid_center.setpoint = target_hose_cy
            err_center = hose_cy - target_hose_cy
            err_anchor = sphere_cx - target_sphere_cx
            err_angle = angle

            centered = abs(err_center) < DESCEND_CENTER_TOLERANCE_PX
            angle_ok = abs(err_angle) < DESCEND_ANGLE_TOLERANCE_DEG
            anchor_ok = abs(err_anchor) < DESCEND_ANCHOR_TOLERANCE_PX

            if altitude <= RELEASE_ALTITUDE and centered and angle_ok and anchor_ok:
                confirmed += 1
                if confirmed >= DESCEND_RELEASE_CONFIRMATIONS:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    yasmin.YASMIN_LOG_INFO(f"Release pose reached: alt={altitude:.2f}m")
                    return SUCCEED
            else:
                confirmed = 0

            vx = self.pid_center.update(hose_cy)
            vy = self.pid_anchor.update(err_anchor)
            vyaw = self.pid_angle.update(angle)

            drone.move_velocity(
                vx=vx,
                vy=vy,
                vz=-DESCEND_VELOCITY,
                vyaw=vyaw,
                reference=MoveReference.BODY,
            )

            self._save_frame(
                frame,
                result,
                segmentor,
                hose_cx,
                target_sphere_cx,
                angle,
                altitude,
            )

            if int(time.time() * 2) % 2 == 0:
                yasmin.YASMIN_LOG_INFO(
                    f"Descend: alt={altitude:.2f}m hose_cy_err={err_center:+.0f} "
                    f"angle={angle:+.1f} anchor_err={err_anchor:+.0f}"
                )

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR("Descent timed out.")
        return ABORT

    def _save_frame(
        self, frame, result, segmentor, hose_cx, target_cx, angle, altitude
    ):
        if not (SAVE_DETECTIONS and self.save_dir and frame is not None and result):
            return
        self.frame_count += 1
        if self.frame_count % 10 != 0:
            return
        annotated = segmentor.draw_segmentations(frame, result)
        cv2.drawMarker(
            annotated,
            (int(hose_cx), IMAGE_CENTER_Y),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            30,
            2,
        )
        cv2.drawMarker(
            annotated,
            (int(target_cx), IMAGE_CENTER_Y),
            (0, 0, 255),
            cv2.MARKER_TRIANGLE_UP,
            30,
            2,
        )
        cv2.putText(
            annotated,
            f"alt={altitude:.2f}m angle={angle:+.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(
            str(self.save_dir / f"descend_{self.frame_count:04d}.jpg"), annotated
        )
