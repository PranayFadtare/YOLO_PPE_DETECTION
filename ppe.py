import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from datetime import datetime
import time
import math

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        
        # Fixed MediaPipe Pose initialization with correct parameters
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img
        
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
        
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        if len(self.lmList) <= max(p1, p2, p3):
            return 0
            
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
            
        # Draw
        if draw and img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

class RealTimePPEDetector:
    def __init__(self, model_path="best.pt", confidence=0.5):
        """
        Real-time PPE Detection with improved OpenPose
        Using CSS dataset classes
        """
        # Load YOLO model
        self.model = YOLO(model_path)
        self.confidence = confidence
        
        # CSS Dataset classes (from your training notebook)
        self.classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
                       'NO-Safety Vest', 'Person', 'Safety Cone', 
                       'Safety Vest', 'machinery', 'vehicle']
        
        # Initialize improved pose detector
        self.pose_detector = poseDetector(
            mode=False,
            upBody=False,
            smooth=True,
            detectionCon=0.5,
            trackCon=0.5
        )
        
        # Colors for visualization
        self.colors = {
            'Hardhat': (0, 255, 255),        # Yellow
            'Mask': (255, 0, 255),           # Magenta
            'NO-Hardhat': (0, 0, 255),       # Red
            'NO-Mask': (0, 0, 255),          # Red
            'NO-Safety Vest': (0, 0, 255),   # Red
            'Person': (0, 255, 0),           # Green
            'Safety Cone': (255, 255, 0),    # Cyan
            'Safety Vest': (255, 165, 0),    # Orange
            'machinery': (128, 0, 128),      # Purple
            'vehicle': (255, 20, 147)        # Deep pink
        }
        
        # Tracking variables
        self.frame_count = 0
        self.total_detections = 0
        self.violation_count = 0
        self.pTime = 0  # For FPS calculation

    def detect_posture_violations(self, lmList):
        """Detect unsafe postures using pose landmarks"""
        posture_violations = []
        
        if len(lmList) > 25:  # Ensure we have enough landmarks
            try:
                # Store current lmList in pose_detector for angle calculations
                self.pose_detector.lmList = lmList
                
                # Check for unsafe bending (back angle)
                # Using shoulder (11), hip (23), and knee (25) landmarks
                if len(lmList) > 25:
                    shoulder_angle = self.pose_detector.findAngle(None, 11, 23, 25, draw=False)
                    
                    # Check if person is bending too much (unsafe posture)
                    if 45 <= shoulder_angle <= 135:  # Unsafe bending range
                        posture_violations.append("Unsafe Bending Posture")
                
                # Check for raised arms (reaching overhead - potential fall risk)
                if len(lmList) > 16:
                    left_arm_angle = self.pose_detector.findAngle(None, 11, 13, 15, draw=False)
                    right_arm_angle = self.pose_detector.findAngle(None, 12, 14, 16, draw=False)
                    
                    if left_arm_angle < 45 or right_arm_angle < 45:
                        posture_violations.append("Overhead Work - Fall Risk")
                        
            except (IndexError, ValueError, ZeroDivisionError):
                # Handle cases where pose landmarks are not complete or calculations fail
                pass
        
        return posture_violations

    def detect_violations_in_frame(self, detections, lmList):
        """Detect safety violations in current frame including posture"""
        violations = []
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Group detections by class
        frame_detections = {cls: [] for cls in self.classes}
        
        if hasattr(detections, 'boxes') and detections.boxes is not None:
            for box in detections.boxes:
                class_id = int(box.cls[0].cpu().numpy())
                if class_id < len(self.model.names):
                    class_name = self.model.names[class_id]
                    if class_name in self.classes:
                        bbox = box.xyxy[0].cpu().numpy()
                        frame_detections[class_name].append(bbox)
        
        # Check for persons and their PPE compliance
        persons = frame_detections.get('Person', [])
        
        for i, person_box in enumerate(persons):
            person_violations = []
            
            # Check for hardhat
            has_hardhat = self.check_ppe_overlap(person_box, frame_detections.get('Hardhat', []))
            has_no_hardhat = len(frame_detections.get('NO-Hardhat', [])) > 0
            
            if has_no_hardhat or not has_hardhat:
                person_violations.append("No Hardhat")
            
            # Check for safety vest
            has_vest = self.check_ppe_overlap(person_box, frame_detections.get('Safety Vest', []))
            has_no_vest = len(frame_detections.get('NO-Safety Vest', [])) > 0
            
            if has_no_vest or not has_vest:
                person_violations.append("No Safety Vest")
            
            # Check for mask
            has_mask = self.check_ppe_overlap(person_box, frame_detections.get('Mask', []))
            has_no_mask = len(frame_detections.get('NO-Mask', [])) > 0
            
            if has_no_mask or not has_mask:
                person_violations.append("No Mask")
            
            # Check for posture violations
            posture_violations = self.detect_posture_violations(lmList)
            person_violations.extend(posture_violations)
            
            if person_violations:
                violations.append({
                    'person_id': i,
                    'timestamp': current_time,
                    'violations': person_violations,
                    'bbox': person_box
                })
                self.violation_count += 1
        
        return violations

    def check_ppe_overlap(self, person_box, ppe_boxes, threshold=0.1):
        """Check if PPE overlaps with person bounding box"""
        if not ppe_boxes:
            return False
        
        px1, py1, px2, py2 = person_box
        
        for ppe_box in ppe_boxes:
            ex1, ey1, ex2, ey2 = ppe_box
            
            # Calculate intersection
            overlap_x = max(0, min(px2, ex2) - max(px1, ex1))
            overlap_y = max(0, min(py2, ey2) - max(py1, ey1))
            overlap_area = overlap_x * overlap_y
            
            # Calculate PPE area
            ppe_area = (ex2 - ex1) * (ey2 - ey1)
            
            # Check if overlap is significant
            if ppe_area > 0 and overlap_area > threshold * ppe_area:
                return True
        
        return False

    def draw_detections_and_pose(self, frame, results, violations, show_pose=True):
        """Draw detection boxes and pose landmarks with improved visualization"""
        annotated_frame = frame.copy()
        
        # Draw PPE detections
        if results.boxes is not None:
            for box in results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                if class_id < len(self.model.names):
                    class_name = self.model.names[class_id]
                    color = self.colors.get(class_name, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with background
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1-label_size[1]-10), 
                                (x1+label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw improved pose estimation
        if show_pose:
            annotated_frame = self.pose_detector.findPose(annotated_frame, draw=True)
            lmList = self.pose_detector.findPosition(annotated_frame, draw=False)
            
            # Highlight key pose points for safety analysis
            if len(lmList) > 0:
                # Highlight head (for hardhat detection)
                if len(lmList) > 0:
                    cv2.circle(annotated_frame, (lmList[0][1], lmList[0][2]), 10, (0, 255, 255), 3)
                
                # Highlight shoulders and torso (for vest detection)
                if len(lmList) > 11:
                    cv2.circle(annotated_frame, (lmList[11][1], lmList[11][2]), 8, (255, 165, 0), 3)
                if len(lmList) > 12:
                    cv2.circle(annotated_frame, (lmList[12][1], lmList[12][2]), 8, (255, 165, 0), 3)
        
        # Draw violation alerts with improved styling
        if violations:
            # Draw alert background
            alert_height = 30 + len(violations) * 25
            cv2.rectangle(annotated_frame, (5, 5), (500, alert_height), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (5, 5), (500, alert_height), (0, 0, 255), 2)
            
            alert_text = f"WARNING VIOLATIONS: {len(violations)}"
            cv2.putText(annotated_frame, alert_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            y_offset = 50
            for violation in violations:
                violation_text = f"Person {violation['person_id']+1}: {', '.join(violation['violations'])}"
                cv2.putText(annotated_frame, violation_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - self.pTime) if self.pTime > 0 else 0
        self.pTime = cTime
        
        # Draw improved status info
        info_bg_height = 80
        cv2.rectangle(annotated_frame, (10, frame.shape[0] - info_bg_height), 
                     (600, frame.shape[0] - 10), (0, 0, 0), -1)
        
        info_texts = [
            f"FPS: {int(fps)}",
            f"Frame: {self.frame_count} | Detections: {self.total_detections}",
            f"Total Violations: {self.violation_count}"
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(annotated_frame, text, (15, frame.shape[0] - 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame

    def run_real_time_surveillance(self):
        """Run real-time surveillance using webcam with improved pose tracking"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        print("🚀 Starting Enhanced Real-time PPE Surveillance")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save screenshot")
        print("   - Press 'v' to toggle violation alerts")
        print("   - Press 'p' to toggle pose estimation")
        print("   - Press 'r' to reset violation counter")
        print()
        
        show_violations = True
        show_pose = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from webcam")
                break
            
            self.frame_count += 1
            
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence, verbose=False)[0]
            
            # Count total detections
            if results.boxes is not None:
                self.total_detections += len(results.boxes)
            
            # Get pose landmarks for violation detection
            temp_frame = frame.copy()
            self.pose_detector.findPose(temp_frame, draw=False)
            lmList = self.pose_detector.findPosition(temp_frame, draw=False)
            
            # Detect violations (including posture)
            violations = self.detect_violations_in_frame(results, lmList) if show_violations else []
            
            # Print violations to console with timestamps
            if violations:
                for violation in violations:
                    print(f"🚨 [{violation['timestamp']}] Person {violation['person_id']+1}: {', '.join(violation['violations'])}")
            
            # Draw everything on frame
            annotated_frame = self.draw_detections_and_pose(frame, results, violations, show_pose)
            
            # Display frame
            cv2.imshow('Enhanced PPE Safety Surveillance', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'ppe_surveillance_{timestamp}.jpg'
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('v'):
                show_violations = not show_violations
                print(f"🔍 Violation detection: {'✅ ON' if show_violations else '❌ OFF'}")
            elif key == ord('p'):
                show_pose = not show_pose
                print(f"🤸 Pose estimation: {'✅ ON' if show_pose else '❌ OFF'}")
            elif key == ord('r'):
                self.violation_count = 0
                print("🔄 Violation counter reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Surveillance Session Summary:")
        print(f"   📹 Total frames processed: {self.frame_count}")
        print(f"   🎯 Total detections: {self.total_detections}")
        print(f"   ⚠️  Total violations: {self.violation_count}")
        if self.frame_count > 0:
            print(f"   📈 Detection rate: {self.total_detections/self.frame_count:.2f} per frame")

def main():
    """Main function to run the enhanced surveillance system"""
    print("🔧 Initializing Enhanced Real-time PPE Surveillance System...")
    
    try:
        # Initialize detector
        detector = RealTimePPEDetector(
            model_path="best.pt",
            confidence=0.5
        )
        
        print("✅ System initialized successfully!")
        print("🎥 Starting surveillance...")
        
        # Run surveillance
        detector.run_real_time_surveillance()
        
    except FileNotFoundError:
        print("❌ Error: Model file 'best.pt' not found!")
        print("   Please ensure the YOLO model file is in the current directory.")
    except Exception as e:
        print(f"❌ Error initializing system: {str(e)}")

if __name__ == "__main__":
    main()