import cv2
import time
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.ai.detection.models.ultralytics import UltralyticsModel

from .navigation import Navigation

# from nectar.vision.camera.config import OpenCVConfig
# from nectar.vision.camera.drivers.opencv_cam import OpenCVCam
from nectar.vision.camera.drivers.oakd_cam import OakdCam
from nectar.vision.camera.config import OakDConfig

from sensor_msgs.msg import CompressedImage


class GaugeReading(State):
    def __init__(self, model_path, confidence_threshold=0.8):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.node = YasminNode.get_instance()
        self.confidence_threshold = confidence_threshold
        
        # Initialize YOLODetector from Mirela SDK
        try:
            self.detector = UltralyticsModel(model_name=model_path)
            self.detector.load_model()
        except Exception as e:
            self.node.get_logger().error(f"Error loading YOLODetector: {e}")
            self.detector = None

    def execute(self, blackboard: Blackboard):
        """Execute capture and inference with timeout and consecutive detections"""
        
        if self.detector is None:
            self.node.get_logger().error("YOLODetector was not loaded")
            return ABORT
        
        if "inference_image_publisher" not in blackboard:
            blackboard["inference_image_publisher"] = self.node.create_publisher(CompressedImage, "/gauge/compressed", 10)

        config = OakDConfig()

        # config = OpenCVConfig(
        #     name="webcam",
        #     device_index=0,
        #     width=640,
        #     height=640,
        #     fps=30,
        #     fourcc="MJPG",
        #     buffer_size=1,
        #     threaded=True,
        # )

        self.camera = OakdCam(config)
        # self.camera = OpenCVCam(config)
        self.camera.start()
            
        # Configuration
        max_duration = 60.0  
        consecutive_limit = 3 
        
        # System state
        start_time = time.time()
        consecutive_count = 0
        last_class_id = None
        best_detection_overall = None
        best_confidence_overall = 0.0
        best_inference_image_overall = None
        last_inference_image = None
        frame_count = 0
        
        try:
            while (time.time() - start_time) < max_duration:
                frame = self.camera.get_frame()

                if frame is None:
                    self.node.get_logger().warn(
                        "No frame captured from camera, retrying...",
                        throttle_duration_sec=2.5
                    )
                    continue
                frame_count += 1

                height, width = frame.shape[:2]
                if height > width:
                    diff = height - width
                    frame = frame[diff // 2:diff // 2 + width, :]
                elif width > height:
                    diff = width - height
                    frame = frame[:, diff // 2:diff // 2 + height]

                # Execute detection
                detection_result = self.detector.detect(frame, conf=self.confidence_threshold)

                # Pass the detection_result to draw_detections
                last_inference_image = self.detector.draw_detections(frame, detection_result)
                # Compress OpenCV image to JPEG
                _, buffer = cv2.imencode('.jpg', last_inference_image, [cv2.IMWRITE_JPEG_QUALITY, 85])  # 85% quality

                # Create compressed message
                compressed_msg = CompressedImage()
                compressed_msg.header.stamp = self.node.get_clock().now().to_msg()
                compressed_msg.format = "jpeg"
                compressed_msg.data = buffer.tobytes()

                image_publisher = blackboard["inference_image_publisher"]
                image_publisher.publish(compressed_msg)
                

                # Log detection status
                detections = detection_result if isinstance(detection_result, list) else detection_result.detections
                num_detections = len(detections)
                self.node.get_logger().info(
                    f"Frame {frame_count}: {num_detections} detection(s) found"
                )

                # Process current frame detections
                best_detection = None
                best_confidence = 0.0
                
                for detection in detections:
                    confidence = detection.confidence
                    class_id = detection.class_id
                    
                    if confidence > best_confidence:
                        best_detection = detection
                        best_confidence = confidence
                
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
                else:
                    # Reset counter if nothing detected
                    consecutive_count = 0
                    last_class_id = None
        finally:
            self.camera.close()
            # Stop the diamond flight immediately
            nav : Navigation = blackboard["navigation_state"]
            if nav is not None:
                nav.stop_diamond()
    
        
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
