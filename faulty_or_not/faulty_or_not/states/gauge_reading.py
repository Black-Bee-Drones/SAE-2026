import cv2
import time
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.ai.detection.models.ultralytics import UltralyticsModel


class GaugeReading(State):
    def __init__(self, model_path, confidence_threshold=0.5):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.node = YasminNode.get_instance()
        self.confidence_threshold = confidence_threshold
        
        # Initialize YOLODetector from Mirela SDK
        try:
            self.detector = UltralyticsModel(model_name=model_path)
            self.detector.load_model()
            # self.detector = YOLODetector(model_source=model_path, 
            #                             confidence_threshold=self.confidence_threshold,
            #                             image_size=1280)
        except Exception as e:
            self.node.get_logger().error(f"Error loading YOLODetector: {e}")
            self.detector = None
        

    def execute(self, blackboard: Blackboard):
        """Execute capture and inference with timeout and consecutive detections"""
        
        if self.detector is None:
            self.node.get_logger().error("YOLODetector was not loaded")
            return ABORT
        
        # Initialize camera
        # TODO: Change the image source to the proper mirela-sdk handler

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.node.get_logger().error("Could not open webcam")
            return ABORT
            
        # Configuration
        max_duration = 15.0  # Maximum 15 seconds
        consecutive_limit = 12  # 12 equal consecutive detections
        
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
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame_count += 1
                
                # Execute detection
                detections = self.detector.detect(frame,conf=0.75)

                # Process current frame detections
                best_detection = None
                best_confidence = 0.0
                
                for detection in detections:
                    confidence = detection.confidence if hasattr(detection, 'confidence') else detection['confidence']
                    
                    if confidence > self.confidence_threshold and confidence > best_confidence:
                        best_detection = detection
                        best_confidence = confidence
                
                # Update overall best detection
                if best_detection is not None:
                    class_id = best_detection.class_id if hasattr(best_detection, 'class_id') else best_detection['class_id']
                    
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
                        # Create DetectionResult with only the best detection
                        best_result = DetectionResult(detections=[best_detection])
                        last_inference_image = self.detector.draw_detections(frame, best_result)
                    
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
            cap.release()
        
        # Final analysis
        elapsed_time = time.time() - start_time
        
        if best_detection_overall is not None:
            final_class_id = best_detection_overall.class_id if hasattr(best_detection_overall, 'class_id') else best_detection_overall['class_id']
            
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
