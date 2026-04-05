import cv2
import time
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.ai.detection.models.ultralytics import UltralyticsModel

from .navigation import Navigation

from nectar.vision.camera.config import OpenCVConfig
from nectar.vision.camera.drivers.opencv_cam import OpenCVCam
from ..parameters import SIMULATION

from zaxis.drone import Drone

from sensor_msgs.msg import CompressedImage

import threading

from zaxis.runtime import CommandHandle

PIXELS_PER_METER = 1.0 / 0.0010210406  # k = 0.0010210406 m/px


class GaugeReading(State):
    def __init__(self, model_path, confidence_threshold=0.85, cam=None):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.node = YasminNode.get_instance()
        self.confidence_threshold = confidence_threshold
        self.cam = cam
        self._stop_event = threading.Event()

        # Initialize YOLODetector from Mirela SDK
        try:
            self.detector = UltralyticsModel(model_name=model_path)
            self.detector.load_model()
        except Exception as e:
            self.node.get_logger().error(f"Error loading YOLODetector: {e}")
            self.detector = None

        self.drone : Drone = None

    def expanding_square(self, max_step, step_size):
        """
        Expanding square search in FRD frame.
        Distances: 1, 1, 2, 2, 3, 3, ... × step_size
        """
        for step in range(1, max_step + 1):
            if self._stop_event.is_set():
                return

            dist = ((step + 1) // 2) * step_size

            if step % 4 == 1:    # Forward
                offset = (0, dist, 0)
            elif step % 4 == 2:  # Right
                offset = (dist, 0, 0)
            elif step % 4 == 3:  # Backward
                offset = (0, -dist, 0)
            else:                 # Left (step % 4 == 0)
                offset = (-dist, 0, 0)

            task = self.drone.goto_offset(*offset, ned=False, face_wp=False) 

            if task is not None:
                try:
                    while not task.done():
                        if self._stop_event.is_set():
                            task.stop()
                            return
                        time.sleep(0.1)
                except Exception:
                    pass

    def get_square_crops(self, frame, overlap=0.3):
        """Sliding window square crops covering the full FOV."""
        h, w = frame.shape[:2]
        size = min(h, w)
        step = int(size * (1 - overlap))
        crops = []

        if w >= h:
            x = 0
            while x + size <= w:
                crops.append((frame[0:size, x:x+size], x, 0))
                x += step
            if not crops or crops[-1][1] + size < w:
                crops.append((frame[0:size, w-size:w], w-size, 0))
        else:
            y = 0
            while y + size <= h:
                crops.append((frame[y:y+size, 0:size], 0, y))
                y += step
            if not crops or crops[-1][2] + size < h:
                crops.append((frame[h-size:h, 0:size], 0, h-size))

        return crops

    def _center_on_detection(self, detection, x_off, y_off, frame_shape, frame_location):

        K = 0.0010210406  # meters per pixel

        h, w = frame_shape[:2]
        frame_cx = w / 2.0
        frame_cy = h / 2.0

        x1, y1, x2, y2 = detection.xyxy
        # Reconstruct bbox center in full-frame coordinates
        bbox_cx = (x1 + x2) / 2.0 + x_off
        bbox_cy = (y1 + y2) / 2.0 + y_off

        dx_px = bbox_cx - frame_cx  # positive → bbox is to the right
        dy_px = bbox_cy - frame_cy  # positive → bbox is below center (further back)

        forward_m = -dy_px * K   # image Y inverted → FRD X
        right_m   =  dx_px * K  # image X direct   → FRD Y

        # Maintain current altitude (z=0 relative to origin keeps same height)
        current_z = self.drone.local_position.z

        self.node.get_logger().info(
            f"[GaugeReading] Centering: dx={dx_px:.1f}px → right={right_m:.3f}m, "
            f"dy={dy_px:.1f}px → fwd={forward_m:.3f}m"
        )

        task = self.drone.goto_local(
            x=forward_m,
            y=right_m,
            z=current_z,
            origin=frame_location,
        )
        return task

    def execute(self, blackboard: Blackboard):
        """Execute capture and inference with timeout and consecutive detections"""
        
        if self.detector is None:
            self.node.get_logger().error("YOLODetector was not loaded")
            return ABORT
        
        if "inference_image_publisher" not in blackboard:
            blackboard["inference_image_publisher"] = self.node.create_publisher(CompressedImage, "/gauge/compressed", 10)

        if "drone" in blackboard:
            self.drone = blackboard["drone"]

        self._stop_event.clear()

        if self.cam is None:
            config = OpenCVConfig(
                name="webcam",
                device_index=0,
                fps=30,
                fourcc="MJPG",
                buffer_size=1,
                threaded=True,
            )

            self.cam = OpenCVCam(config)
            self.cam.start()
            
        # Configuration
        max_duration = 100.0  
        consecutive_limit = 20
        
        # System state
        start_time = time.time()
        time_last_det = None
        consecutive_count = 0
        last_class_id = None
        best_detection_overall = None
        best_confidence_overall = 0.0
        best_inference_image_overall = None
        last_inference_image = None
        frame_count = 0
        frame_location = None
        centering_done = False  # one-shot flag

        t = None
        goto_handler : CommandHandle = blackboard["goto_handler"]
        
        try:
            while (time.time() - start_time) < max_duration:
                if time_last_det is not None and best_detection_overall is not None and time.time() - time_last_det > 10.0:
                    self._stop_event.set()
                    break

                if not goto_handler.done():
                    start_time = time.time()

                if best_detection_overall is not None and t is not None and t.is_alive():
                    self.node.get_logger().info("Best detection found, overriding default behavior to only 1 detection.")
                    self._stop_event.set()
                    break

                if best_detection_overall is None and (t is None or not t.is_alive()) and time.time() - start_time > 5.0:
                    self._stop_event.clear()
                    t = threading.Thread(target=self.expanding_square, args=(15, 0.2), daemon=True)
                    t.start()

                frame = self.cam.get_frame() if not SIMULATION else self.cam.frame
                frame_location = self.drone.capture_origin()

                if frame is None:
                    self.node.get_logger().warn(
                        "No frame captured from camera, retrying...",
                        throttle_duration_sec=2.5
                    )
                    continue
                frame_count += 1

                # Process current frame detections across all crops
                best_detection = None
                best_confidence = 0.0
                best_crop_x_off = 0
                best_crop_y_off = 0

                for crop, x_off, y_off in self.get_square_crops(frame):
                    detection_result = self.detector.detect(crop, conf=self.confidence_threshold)
                    detections = detection_result if isinstance(detection_result, list) else detection_result.detections

                    for detection in detections:
                        if detection.confidence > best_confidence:
                            best_detection = detection
                            best_confidence = detection.confidence
                            best_crop_x_off = x_off
                            best_crop_y_off = y_off
                            last_inference_image = self.detector.draw_detections(crop, detection_result)

                # On first detection ever, center the drone over the manometer
                if best_detection is not None and not centering_done:
                    centering_done = True
                    self._stop_event.set()  # halt expanding square
                    self.node.get_logger().info("[GaugeReading] First detection — centering drone.")
                    self._center_on_detection(
                        best_detection,
                        best_crop_x_off,
                        best_crop_y_off,
                        frame.shape,
                        frame_location,
                    )

                # Compress OpenCV image to JPEG
                _, buffer = cv2.imencode('.jpg', (last_inference_image if best_detection is not None else frame), [cv2.IMWRITE_JPEG_QUALITY, 85])  # 85% quality

                if best_detection is not None:
                    time_last_det = time.time()

                # Create compressed message
                compressed_msg = CompressedImage()
                compressed_msg.header.stamp = self.node.get_clock().now().to_msg()
                compressed_msg.format = "jpeg"
                compressed_msg.data = buffer.tobytes()

                image_publisher = blackboard["inference_image_publisher"]
                image_publisher.publish(compressed_msg)

                num_detections = 1 if best_detection is not None else 0
                self.node.get_logger().info(
                    f"Frame {frame_count}: {num_detections} detection(s) found"
                )

                # Update overall best detection
                if best_detection is not None:
                    class_id = best_detection.class_id
                    
                    # Check consecutive detections
                    if class_id == last_class_id:
                        consecutive_count += 1
                    else:
                        consecutive_count = 1
                        last_class_id = class_id
                    
                    # Update best detection if necessary
                    if best_confidence > best_confidence_overall:
                        best_detection_overall = best_detection
                        best_confidence_overall = best_confidence
                        best_inference_image_overall = last_inference_image

                    self.node.get_logger().info(
                        f"Frame {frame_count}: Class {class_id}, Conf {best_confidence:.3f}, "
                        f"Consecutive: {consecutive_count}/{consecutive_limit}"
                    )
                    
                    # Stop condition
                    if consecutive_count >= consecutive_limit:
                        self.node.get_logger().info(
                            f"SUCCESS: {consecutive_limit} consecutive detections of class {class_id}!"
                        )
                        break
        finally:
            try:
                self._stop_event.set()
                self.cam.close()
            except Exception:
                pass
        
        if last_class_id is not None:            
            # Save results to blackboard
            blackboard["gauge_reading"] = last_class_id
            
            if last_inference_image is not None:
                blackboard["inference_image_cv"] = last_inference_image
            
            return SUCCEED
        elif best_detection_overall is not None:
            blackboard["gauge_reading"] = best_detection_overall.class_id

            if best_inference_image_overall is not None:
                blackboard["inference_image_cv"] = best_inference_image_overall
            return SUCCEED
        else:
            self.node.get_logger().warn(
                f"Timeout of {max_duration}s reached without valid detections. "
                f"Continuing mission..."
            )
            # Even without detection, continue mission (SUCCEED)
            blackboard["gauge_reading"] = -1  # Indicates no detection
            blackboard["inference_image_cv"] = None
            
            return SUCCEED