import cv2
import time
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.ai.detection.models.ultralytics import UltralyticsModel

from mirela_sdk.vision.camera.config import OpenCVConfig
from mirela_sdk.vision.camera.drivers.opencv_cam import OpenCVCam


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

        config = OpenCVConfig(
            name="webcam",
            device_index=0,
            width=640,
            height=640,
            fps=30,
            fourcc="MJPG",
            buffer_size=1,
            threaded=False,
        )

        self.camera = OpenCVCam(config)
        self.camera.start()
            


        # Configuration
        max_duration = 30.0  
        consecutive_limit = 3 
        
        # System state
        start_time = time.time()
        consecutive_count = 0
        last_class_id = None
        best_detection_overall = None
        best_confidence_overall = 0.0
        last_inference_image = None
        frame_count = 0
        
        try:
            while (time.time() - start_time) < max_duration:
                frame = self.camera.get_frame()

                if frame is None:
                    print("Frame is None")
                    continue

                height, width = frame.shape[:2]
                if height > width:
                    diff = height - width
                    frame = frame[diff // 2:diff // 2 + width, :]
                elif width > height:
                    diff = width - height
                    frame = frame[:, diff // 2:diff // 2 + height]

                    
                frame_count += 1
                
                # Execute detection
                detection_result = self.detector.detect(frame, conf=0.75)
                
                # Log detection status
                detections = detection_result if isinstance(detection_result, list) else detection_result.detections
                num_detections = len(detections)
                self.node.get_logger().info(
                    f"Frame {frame_count}: {num_detections} detection(s) found"
                )
                cv2.imwrite(f"/tmp/gauge_reading_frame_{frame_count}.jpg", frame)


                # Process current frame detections
                best_detection = None
                best_confidence = 0.0
                
                # Iterar sobre as detecções
                for detection in detections:
                    confidence = detection.confidence
                    class_id = detection.class_id
                    
                    self.node.get_logger().info(
                        f"  Detection: Class {class_id}, Confidence {confidence:.3f}"
                    )
                    
                    if confidence > self.confidence_threshold and confidence > best_confidence:
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
                        # Pass the detection_result or list to draw_detections
                        last_inference_image = self.detector.draw_detections(frame, detection_result)
                    
                    self.node.get_logger().info(
                        f"Frame {frame_count}: Class {class_id}, Conf {best_confidence:.3f}, "
                        f"Consecutive: {consecutive_count}/{consecutive_limit}"
                    )
                    
                    # Stop condition: 12 equal consecutive detections
                    if consecutive_count >= consecutive_limit:
                        self.node.get_logger().info(
                            f"SUCCESS: {consecutive_limit} consecutive detections of class {class_id}!"
                        )
                        break
                else:
                    # Reset counter if nothing detected
                    consecutive_count = 0
                    last_class_id = None
                
                # Small delay
                time.sleep(0.1)
                
        finally:
            self.camera.close()
        
        # Final analysis
        elapsed_time = time.time() - start_time
        
        if best_detection_overall is not None:
            final_class_id = best_detection_overall.class_id
            
            self.node.get_logger().info(
                f"=== FINAL RESULT ===\n"
                f"Elapsed time: {elapsed_time:.1f}s\n"
                f"Frames processed: {frame_count}\n"
                f"Detected class: {final_class_id}\n"
                f"Best confidence: {best_confidence_overall:.3f}\n"
                f"Final consecutive detections: {consecutive_count}"
            )
            
            # Save results to blackboard
            blackboard["gauge_reading"] = final_class_id
            
            if last_inference_image is not None:
                blackboard["inference_image_cv"] = last_inference_image
            
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
