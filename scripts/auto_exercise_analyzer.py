#!/usr/bin/env python3
"""
Complete Auto Exercise Analyzer - Photo/Video/Realtime Integrated Version (No Emoji)
Step 1: AI automatically detects exercise type
Step 2: Precise angle analysis based on detected exercise
Step 3: Exercise-specific detailed feedback + green/red screen display
Step 4: Photo, video, realtime all supported
"""
import cv2
import numpy as np
import mediapipe as mp
import time
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import json
from datetime import datetime
import tempfile



class CompleteAutoExerciseAnalyzer:
    """Complete automated exercise analyzer - Photo/Video/Realtime integrated"""
    
    def __init__(self):
        # MediaPipe initialization
        self.mp_pose = mp.solutions.pose
        self.pose_static = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_video = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # AI exercise classification model loading
        self.exercise_classifier = None
        self.model_loaded = False
        self.temp_dir = tempfile.mkdtemp()
        self.load_exercise_model()
        
        # Enhanced angle criteria
        self.exercise_thresholds = {
            'squat': {
                'left_knee': {'points': [23, 25, 27], 'range': (55, 140), 'weight': 1.1, 'name_en': 'Left Knee'},
                'right_knee': {'points': [24, 26, 28], 'range': (55, 140), 'weight': 1.1, 'name_en': 'Right Knee'},
                'left_hip': {'points': [11, 23, 25], 'range': (55, 140), 'weight': 0.9, 'name_en': 'Left Hip'},
                'right_hip': {'points': [12, 24, 26], 'range': (55, 140), 'weight': 0.9, 'name_en': 'Right Hip'},
                'back_straight': {'points': [11, 23, 25], 'range': (110, 170), 'weight': 1.1, 'name_en': 'Back Straight'},
                'spine_angle': {'points': [23, 11, 13], 'range': (110, 170), 'weight': 0.9, 'name_en': 'Spine Angle'},
            },
            'push_up': {
                'left_elbow': {'points': [11, 13, 15], 'range': (40, 160), 'weight': 1.0, 'name_en': 'Left Elbow'},
                'right_elbow': {'points': [12, 14, 16], 'range': (40, 160), 'weight': 1.0, 'name_en': 'Right Elbow'},
                'body_line': {'points': [11, 23, 25], 'range': (140, 180), 'weight': 1.2, 'name_en': 'Body Line'},
                'leg_straight': {'points': [23, 25, 27], 'range': (140, 180), 'weight': 0.8, 'name_en': 'Leg Straight'},
                'shoulder_alignment': {'points': [13, 11, 23], 'range': (120, 180), 'weight': 0.6, 'name_en': 'Shoulder Align'},
                'core_stability': {'points': [11, 12, 23], 'range': (140, 180), 'weight': 1.0, 'name_en': 'Core Stability'},
            },
            'deadlift': {
                'left_knee': {'points': [23, 25, 27], 'range': (80, 140), 'weight': 0.6, 'name_en': 'Left Knee'},
                'right_knee': {'points': [24, 26, 28], 'range': (80, 140), 'weight': 0.6, 'name_en': 'Right Knee'},
                'hip_hinge': {'points': [11, 23, 25], 'range': (80, 180), 'weight': 0.7, 'name_en': 'Hip Hinge'},
                'back_straight': {'points': [11, 23, 12], 'range': (120, 180), 'weight': 1.0, 'name_en': 'Back Straight'},
                'chest_up': {'points': [23, 11, 13], 'range': (50, 140), 'weight': 0.5, 'name_en': 'Chest Up'},
                'spine_neutral': {'points': [23, 11, 24], 'range': (120, 180), 'weight': 0.8, 'name_en': 'Spine Neutral'},
            },
            'bench_press': {
                'left_elbow': {'points': [11, 13, 15], 'range': (50, 145), 'weight': 1.1, 'name_en': 'Left Elbow'},
                'right_elbow': {'points': [12, 14, 16], 'range': (50, 145), 'weight': 1.1, 'name_en': 'Right Elbow'},
                'left_shoulder': {'points': [13, 11, 23], 'range': (50, 150), 'weight': 0.9, 'name_en': 'Left Shoulder'},
                'right_shoulder': {'points': [14, 12, 24], 'range': (50, 150), 'weight': 0.9, 'name_en': 'Right Shoulder'},
                'back_arch': {'points': [11, 23, 25], 'range': (90, 170), 'weight': 0.7, 'name_en': 'Back Arch'},
                'wrist_alignment': {'points': [13, 15, 17], 'range': (70, 180), 'weight': 0.6, 'name_en': 'Wrist Align'},
            },
            'lunge': {
                'front_knee': {'points': [23, 25, 27], 'range': (70, 120), 'weight': 1.2, 'name_en': 'Front Knee'},
                'back_knee': {'points': [24, 26, 28], 'range': (120, 180), 'weight': 1.0, 'name_en': 'Back Knee'},
                'front_hip': {'points': [11, 23, 25], 'range': (70, 120), 'weight': 0.8, 'name_en': 'Front Hip'},
                'torso_upright': {'points': [11, 23, 25], 'range': (100, 180), 'weight': 1.2, 'name_en': 'Torso Upright'},
                'front_ankle': {'points': [25, 27, 31], 'range': (80, 110), 'weight': 0.8, 'name_en': 'Front Ankle'},
                'back_hip_extension': {'points': [12, 24, 26], 'range': (150, 180), 'weight': 1.0, 'name_en': 'Back Hip Ext'},
            }
        }
        
        # Exercise-specific detailed feedback messages (English)
        self.detailed_feedback = {
            'squat': {
                'left_knee': {
                    'too_low': 'Raise your left knee more (knee too bent)',
                    'too_high': 'Bend your left knee more (squat deeper)',
                    'good': 'Perfect left knee angle!'
                },
                'right_knee': {
                    'too_low': 'Raise your right knee more (knee too bent)',
                    'too_high': 'Bend your right knee more (squat deeper)',
                    'good': 'Perfect right knee angle!'
                },
                'left_hip': {
                    'too_low': 'Push your left hip back more',
                    'too_high': 'Lower your left hip more',
                    'good': 'Great left hip position!'
                },
                'right_hip': {
                    'too_low': 'Push your right hip back more',
                    'too_high': 'Lower your right hip more',
                    'good': 'Great right hip position!'
                },
                'back_straight': {
                    'too_low': 'Straighten your back (back is curved)',
                    'too_high': 'Lean forward slightly',
                    'good': 'Perfect straight back!'
                },
                'general': 'Keep knees behind toes'
            },
            'push_up': {
                'left_elbow': {
                    'too_low': 'Extend your left arm more',
                    'too_high': 'Bend your left elbow more',
                    'good': 'Perfect left arm angle!'
                },
                'right_elbow': {
                    'too_low': 'Extend your right arm more',
                    'too_high': 'Bend your right elbow more',
                    'good': 'Perfect right arm angle!'
                },
                'body_line': {
                    'too_low': 'Raise your hips (body is sagging)',
                    'too_high': 'Lower your hips (hips too high)',
                    'good': 'Perfect straight body line!'
                },
                'shoulder_alignment': {
                    'too_low': 'Keep shoulders more stable',
                    'too_high': 'Relax your shoulders naturally',
                    'good': 'Perfect shoulder alignment!'
                },
                'general': 'Keep elbows close to body'
            },
            'deadlift': {
                'left_knee': {
                    'too_low': 'Extend your left knee slightly more',
                    'too_high': 'Bend your left knee slightly',
                    'good': 'Perfect left knee!'
                },
                'right_knee': {
                    'too_low': 'Extend your right knee slightly more',
                    'too_high': 'Bend your right knee slightly',
                    'good': 'Perfect right knee!'
                },
                'hip_hinge': {
                    'too_low': 'Push your hips back more (hip hinge)',
                    'too_high': 'Lower your hips more',
                    'good': 'Perfect hip hinge movement!'
                },
                'back_straight': {
                    'too_low': 'Straighten your back - very important!',
                    'too_high': 'Relax your back naturally',
                    'good': 'Perfect straight back!'
                },
                'chest_up': {
                    'too_low': 'Lift your chest and look forward',
                    'too_high': 'Dont over-extend your chest',
                    'good': 'Perfect chest position!'
                },
                'general': 'Keep bar close to body'
            },
            'bench_press': {
                'left_elbow': {
                    'too_low': 'Extend your left arm more',
                    'too_high': 'Bend your left elbow more',
                    'good': 'Perfect left arm!'
                },
                'right_elbow': {
                    'too_low': 'Extend your right arm more',
                    'too_high': 'Bend your right elbow more',
                    'good': 'Perfect right arm!'
                },
                'left_shoulder': {
                    'too_low': 'Keep left shoulder stable',
                    'too_high': 'Relax your left shoulder',
                    'good': 'Perfect left shoulder!'
                },
                'right_shoulder': {
                    'too_low': 'Keep right shoulder stable',
                    'too_high': 'Relax your right shoulder',
                    'good': 'Perfect right shoulder!'
                },
                'back_arch': {
                    'too_low': 'Create natural back arch',
                    'too_high': 'Dont over-arch your back',
                    'good': 'Perfect back arch!'
                },
                'general': 'Control the bar slowly'
            },
            'lunge': {
                'front_knee': {
                    'too_low': 'Adjust front knee to 90 degrees (too bent)',
                    'too_high': 'Bend front knee more (to 90 degrees)',
                    'good': 'Perfect 90-degree front knee!'
                },
                'back_knee': {
                    'too_low': 'Extend your back knee more',
                    'too_high': 'Perfect back knee!',
                    'good': 'Perfect extended back knee!'
                },
                'torso_upright': {
                    'too_low': 'Keep your torso more upright',
                    'too_high': 'Perfect torso!',
                    'good': 'Perfect upright torso!'
                },
                'front_ankle': {
                    'too_low': 'Keep front ankle more stable',
                    'too_high': 'Relax your front ankle',
                    'good': 'Perfect front ankle!'
                },
                'general': 'Maintain balance and move slowly'
            }
        }
        
        # Enhanced classification thresholds
        self.classification_thresholds = {
            'squat': 0.5,
            'push_up': 0.7,
            'deadlift': 0.8,  # relaxed
            'bench_press': 0.5,
            'lunge': 0.6,
        }
        
        # Exercise icons and English names (no emoji)
        self.exercise_info = {
            'squat': {'symbol': '[SQ]', 'name_en': 'SQUAT', 'name_display': 'Squat'},
            'push_up': {'symbol': '[PU]', 'name_en': 'PUSH-UP', 'name_display': 'Push-up'},
            'deadlift': {'symbol': '[DL]', 'name_en': 'DEADLIFT', 'name_display': 'Deadlift'},
            'bench_press': {'symbol': '[BP]', 'name_en': 'BENCH PRESS', 'name_display': 'Bench Press'},
            'lunge': {'symbol': '[LG]', 'name_en': 'LUNGE', 'name_display': 'Lunge'}
        }
        
        # Status management
        self.current_exercise = "detecting..."
        self.current_pose_quality = "unknown"
        self.exercise_confidence = 0.0
        self.pose_confidence = 0.0
        
        # History for stabilization
        self.exercise_history = deque(maxlen=10)
        self.pose_history = deque(maxlen=5)
        
        # Statistics
        self.stats = {'good': 0, 'bad': 0, 'frames': 0}
        
        # Screen status (smooth transition)
        self.screen_color = (128, 128, 128)  # default gray
        self.target_color = (128, 128, 128)
        self.color_transition_speed = 0.15
        
        # Timing
        self.last_classification_time = 0
        self.classification_interval = 2.0  # every 2 seconds
        
        # Feedback message management
        self.current_feedback_messages = []
        self.last_feedback_time = 0
        self.feedback_interval = 1.0  # every 1 second
    
    def load_exercise_model(self):
        """Load AI exercise classification model (considering scripts/ folder)"""
        # Check possible model paths
        possible_paths = [
            "models/exercise_classifier.pkl",           # current folder
            "scripts/models/exercise_classifier.pkl",   # scripts folder
            "../models/exercise_classifier.pkl",        # parent folder
            "./exercise_classifier.pkl"                 # same folder
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        print(f"[INFO] Searching for model in multiple locations...")
        for path in possible_paths:
            exists = "[OK]" if os.path.exists(path) else "[NO]"
            print(f"  {exists} {path}")
        
        if not model_path:
            print("[ERROR] No AI Model Found in any location")
            print(f"[INFO] Current directory: {os.getcwd()}")
            print(f"[INFO] Files in current dir: {os.listdir('.')}")
            if os.path.exists('scripts'):
                print(f"[INFO] Files in scripts/: {os.listdir('scripts')}")
            if os.path.exists('models'):
                print(f"[INFO] Files in models/: {os.listdir('models')}")
            if os.path.exists('scripts/models'):
                print(f"[INFO] Files in scripts/models/: {os.listdir('scripts/models')}")
            self.model_loaded = False
            return
        
        print(f"[OK] Found model at: {model_path}")
        
        try:
            print("[INFO] Model file found, attempting to import...")
            try:
                from exercise_classifier import ExerciseClassificationModel
                print("[OK] Successfully imported ExerciseClassificationModel")
            except ImportError as ie:
                print(f"[ERROR] Import Error: {ie}")
                print("[INFO] Make sure exercise_classifier.py is in the current directory")
                # Try import from scripts folder
                try:
                    import sys
                    if 'scripts' not in sys.path:
                        sys.path.append('scripts')
                    from exercise_classifier import ExerciseClassificationModel
                    print("[OK] Successfully imported from scripts folder")
                except ImportError as ie2:
                    print(f"[ERROR] Import from scripts also failed: {ie2}")
                    self.model_loaded = False
                    return
            
            print("[INFO] Creating model instance...")
            self.exercise_classifier = ExerciseClassificationModel()
            
            print(f"[INFO] Loading model from {model_path}...")
            self.model_loaded = self.exercise_classifier.load_model(model_path)
            
            if self.model_loaded:
                print("[OK] AI Exercise Classification Model Loaded Successfully")
                # Print supported exercises list
                if hasattr(self.exercise_classifier, 'label_encoder'):
                    exercises = list(self.exercise_classifier.label_encoder.keys())
                    print(f"[INFO] Supported exercises: {exercises}")
                
                # Model test
                print("[TEST] Testing model with dummy prediction...")
                # Skip simple test (requires actual image)
                
            else:
                print("[ERROR] Model Load Failed - model.load_model() returned False")
                print("[INFO] Try retraining the model: python main.py --mode train")
                
        except Exception as e:
            print(f"[ERROR] Model Load Error: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """Calculate angle"""
        try:
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float64)
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float64)
            
            v1_mag = np.linalg.norm(v1)
            v2_mag = np.linalg.norm(v2)
            
            if v1_mag < 1e-6 or v2_mag < 1e-6:
                return 180.0
            
            cos_angle = np.dot(v1, v2) / (v1_mag * v2_mag)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return float(angle)
        except:
            return 180.0
    
    def classify_exercise(self, frame: np.ndarray) -> Tuple[str, float]:
        """[AI] Step 1: AI automatic exercise detection"""
        current_time = time.time()
        
        # Classification interval control (every 2 seconds)
        if current_time - self.last_classification_time < self.classification_interval:
            return self.current_exercise, self.exercise_confidence
        
        if not self.model_loaded:
            return "manual_mode", 0.0
        
        try:
            # Save temporary image
            temp_path = os.path.join(self.temp_dir, "temp_frame.jpg")
            cv2.imwrite(temp_path, frame)
            
            # AI exercise classification
            exercise, confidence = self.exercise_classifier.predict(temp_path)
            
            # History stabilization
            self.exercise_history.append((exercise, confidence))
            
            if len(self.exercise_history) >= 3:
                # Consensus from recent 3 results
                recent = list(self.exercise_history)[-3:]
                high_conf_predictions = [(ex, conf) for ex, conf in recent if conf > 0.6]
                
                if high_conf_predictions:
                    from collections import Counter
                    exercises = [ex for ex, conf in high_conf_predictions]
                    most_common = Counter(exercises).most_common(1)[0]
                    
                    if most_common[1] >= 2:  # detected 2+ times
                        new_exercise = most_common[0]
                        if new_exercise != self.current_exercise:
                            self.current_exercise = new_exercise
                            self.exercise_confidence = confidence
                            exercise_info = self.exercise_info.get(new_exercise, {})
                            symbol = exercise_info.get('symbol', '[??]')
                            name_display = exercise_info.get('name_display', new_exercise)
                            print(f"AI Detected: {symbol} {name_display} (Confidence: {confidence:.1%})")
            
            self.last_classification_time = current_time
            
            # Delete temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return self.current_exercise, self.exercise_confidence
            
        except Exception as e:
            print(f"Exercise Classification Error: {e}")
            return self.current_exercise, self.exercise_confidence
    
    def analyze_pose_angles(self, landmarks, exercise: str) -> Dict:
        """[TARGET] Step 2: Precise angle analysis based on detected exercise"""
        if exercise not in self.exercise_thresholds:
            return {'valid': False, 'error': f'Unsupported exercise: {exercise}'}
        
        thresholds = self.exercise_thresholds[exercise]
        angles = {}
        violations = []
        weighted_violation_score = 0.0
        total_weight = 0.0
        
        for joint_name, config in thresholds.items():
            try:
                p1_idx, p2_idx, p3_idx = config['points']
                min_angle, max_angle = config['range']
                weight = config['weight']
                
                # Visibility check
                if (landmarks[p1_idx].visibility < 0.25 or 
                    landmarks[p2_idx].visibility < 0.25 or 
                    landmarks[p3_idx].visibility < 0.25):
                    continue
                
                p1 = (landmarks[p1_idx].x, landmarks[p1_idx].y)
                p2 = (landmarks[p2_idx].x, landmarks[p2_idx].y)
                p3 = (landmarks[p3_idx].x, landmarks[p3_idx].y)
                
                angle = self.calculate_angle(p1, p2, p3)
                angles[joint_name] = {
                    'value': angle,
                    'range': (min_angle, max_angle),
                    'weight': weight,
                    'in_range': min_angle <= angle <= max_angle,
                    'name_en': config.get('name_en', joint_name)
                }
                
                total_weight += weight
                
                if not (min_angle <= angle <= max_angle):
                    violations.append({
                        'joint': joint_name,
                        'angle': angle,
                        'expected_range': (min_angle, max_angle),
                        'weight': weight,
                        'name_en': config.get('name_en', joint_name)
                    })
                    weighted_violation_score += weight
                    
            except Exception as e:
                continue
        
        # Enhanced classification
        violation_ratio = weighted_violation_score / max(total_weight, 1.0)
        classification_threshold = self.classification_thresholds.get(exercise, 0.6)
        is_good = violation_ratio < classification_threshold
        
        return {
            'valid': True,
            'classification': 'good' if is_good else 'bad',
            'confidence': 1.0 - violation_ratio,
            'angles': angles,
            'violations': violations,
            'violation_ratio': violation_ratio,
            'threshold': classification_threshold
        }
    
    def generate_detailed_feedback(self, exercise: str, pose_result: Dict) -> List[str]:
        """[FEEDBACK] Generate exercise-specific detailed feedback (English)"""
        current_time = time.time()
        
        # Feedback interval limit
        if current_time - self.last_feedback_time < self.feedback_interval:
            return self.current_feedback_messages
        
        messages = []
        
        if not pose_result.get('valid', False):
            messages.append("Cannot recognize pose")
            return messages
        
        violations = pose_result.get('violations', [])
        exercise_feedback = self.detailed_feedback.get(exercise, {})
        
        if not violations:
            # All poses are perfect
            exercise_info = self.exercise_info.get(exercise, {})
            name_display = exercise_info.get('name_display', exercise)
            messages.append(f"Perfect {name_display} form! [OK]")
            messages.append("Keep this form!")
        else:
            # Violations exist - sort by weight
            violations_sorted = sorted(violations, key=lambda x: x['weight'], reverse=True)
            
            for i, violation in enumerate(violations_sorted[:3]):  # top 3 only
                joint = violation['joint']
                angle = violation['angle']
                min_angle, max_angle = violation['expected_range']
                name_en = violation.get('name_en', joint)
                
                joint_feedback = exercise_feedback.get(joint, {})
                
                if angle < min_angle:
                    # angle too small
                    message = joint_feedback.get('too_low', f'Increase {name_en} angle')
                elif angle > max_angle:
                    # angle too large
                    message = joint_feedback.get('too_high', f'Decrease {name_en} angle')
                else:
                    message = joint_feedback.get('good', f'{name_en} is good!')
                
                messages.append(f"[!] {message}")
                
                # Add specific angle info
                if i == 0:  # most important issue only
                    messages.append(f"   Current: {angle:.0f}° -> Target: {min_angle:.0f}-{max_angle:.0f}°")
            
            # Add general exercise advice
            general_advice = exercise_feedback.get('general', '')
            if general_advice and len(violations_sorted) <= 2:
                messages.append(f"[TIP] {general_advice}")
        
        self.current_feedback_messages = messages
        self.last_feedback_time = current_time
        return messages
    
    def update_screen_color(self, pose_quality: str):
        """[COLOR] Update green/red screen color"""
        if pose_quality == 'good':
            self.target_color = (0, 255, 0)      # green
        elif pose_quality == 'bad':
            self.target_color = (0, 0, 255)      # red
        elif pose_quality == 'detecting':
            self.target_color = (255, 255, 0)    # yellow
        else:
            self.target_color = (128, 128, 128)  # gray
        
        # Smooth color transition
        for i in range(3):
            current = self.screen_color[i]
            target = self.target_color[i]
            diff = target - current
            self.screen_color = tuple(
                int(current + diff * self.color_transition_speed) if j == i 
                else self.screen_color[j] for j in range(3)
            )
    
    def draw_enhanced_overlay(self, frame: np.ndarray, exercise: str, pose_result: Dict) -> np.ndarray:
        """[DISPLAY] Enhanced analysis result screen overlay (English text)"""
        height, width = frame.shape[:2]
        
        # [COLOR] Full screen color overlay and border
        if pose_result.get('valid', False):
            pose_quality = pose_result['classification']
            self.update_screen_color(pose_quality)
            
            # Transparent color overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), self.screen_color, -1)
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # [BORDER] Thick border
        border_thickness = 30
        cv2.rectangle(frame, (0, 0), (width, height), self.screen_color, border_thickness)
        
        # [TOP-LEFT] Exercise type display
        exercise_info = self.exercise_info.get(exercise, {})
        if exercise != "detecting..." and exercise != "manual_mode":
            symbol = exercise_info.get('symbol', '[??]')
            name_display = exercise_info.get('name_display', exercise)
            name_en = exercise_info.get('name_en', exercise.upper())
            
            # Background box
            cv2.rectangle(frame, (40, 40), (400, 140), (0, 0, 0), -1)
            cv2.rectangle(frame, (40, 40), (400, 140), self.screen_color, 3)
            
            # Exercise name display (reduced font size)
            exercise_text = f"{symbol} {name_display}"
            cv2.putText(frame, exercise_text, (60, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # 1.2 -> 0.8
            
            # English name display (reduced font size)
            cv2.putText(frame, name_en, (60, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)  # 0.8 -> 0.6
            
            # Confidence display (reduced font size)
            if self.model_loaded:
                confidence_text = f"AI: {self.exercise_confidence:.0%}"
            else:
                confidence_text = "Manual Mode"
            cv2.putText(frame, confidence_text, (60, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)  # 0.6 -> 0.5
            
        elif exercise == "detecting...":
            cv2.rectangle(frame, (40, 40), (300, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (40, 40), (300, 100), (255, 255, 0), 3)
            cv2.putText(frame, "[AI] Detecting...", (60, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # reduced font size
        else:
            cv2.rectangle(frame, (40, 40), (320, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (40, 40), (320, 100), (128, 128, 128), 3)
            cv2.putText(frame, "[!] No AI Model", (60, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # reduced font size
        
        # [CENTER] Status message
        if pose_result.get('valid', False):
            pose_quality = pose_result['classification']
            confidence = pose_result['confidence']
            
            if pose_quality == 'good':
                status_text = "Perfect Form! [OK]"
                status_color = (0, 255, 0)
            else:
                status_text = "Form Needs Work [!]"
                status_color = (0, 0, 255)
            
            # Center status display
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            status_x = (width - status_size[0]) // 2
            status_y = height // 2 - 80
            
            cv2.rectangle(frame, (status_x - 30, status_y - 40), 
                         (status_x + status_size[0] + 30, status_y + 20), (0, 0, 0), -1)
            cv2.rectangle(frame, (status_x - 30, status_y - 40), 
                         (status_x + status_size[0] + 30, status_y + 20), status_color, 4)
            
            cv2.putText(frame, status_text, (status_x, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)  # 1.5 -> 1.0
            
            # Confidence score (reduced font size)
            score_text = f"Form Score: {confidence:.0%}"
            score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]  # 0.8 -> 0.6
            score_x = (width - score_size[0]) // 2
            cv2.putText(frame, score_text, (score_x, status_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # 0.8 -> 0.6
        
        # [BOTTOM-LEFT] Detailed feedback messages
        if exercise in self.exercise_thresholds:
            feedback_messages = self.generate_detailed_feedback(exercise, pose_result)
            
            if feedback_messages:
                # Feedback area background
                feedback_height = len(feedback_messages) * 35 + 60
                cv2.rectangle(frame, (40, height - feedback_height - 40), 
                             (width - 40, height - 40), (0, 0, 0), -1)
                cv2.rectangle(frame, (40, height - feedback_height - 40), 
                             (width - 40, height - 40), self.screen_color, 3)
                
                # Feedback title (reduced font size)
                cv2.putText(frame, "[FEEDBACK] Feedback:", (60, height - feedback_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # 0.7 -> 0.6
                
                # Feedback messages (reduced font size)
                for i, message in enumerate(feedback_messages[:5]):  # max 5
                    y_pos = height - feedback_height + 20 + (i * 30)  # 35 -> 30 (reduced line spacing)
                    
                    # Message color determination
                    if "Perfect" in message or "[OK]" in message:
                        msg_color = (0, 255, 0)  # green
                    elif "[!]" in message:
                        msg_color = (0, 100, 255)  # orange
                    elif "[TIP]" in message:
                        msg_color = (255, 255, 0)  # yellow
                    else:
                        msg_color = (255, 255, 255)  # white
                    
                    cv2.putText(frame, message, (60, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, msg_color, 2)  # 0.6 -> 0.5
        
        # [TOP-RIGHT] Statistics info
        if self.stats['frames'] > 0:
            total = self.stats['good'] + self.stats['bad']
            if total > 0:
                good_ratio = self.stats['good'] / total
                
                # Statistics background
                cv2.rectangle(frame, (width - 300, 40), (width - 40, 140), (0, 0, 0), -1)
                cv2.rectangle(frame, (width - 300, 40), (width - 40, 140), (255, 255, 255), 2)
                
                # Statistics text (reduced font size)
                cv2.putText(frame, "[STATS] Stats", (width - 280, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # 0.6 -> 0.5
                
                stats_text = f"Good: {self.stats['good']} | Bad: {self.stats['bad']}"
                cv2.putText(frame, stats_text, (width - 280, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # 0.5 -> 0.4
                
                ratio_text = f"Success: {good_ratio:.1%}"
                cv2.putText(frame, ratio_text, (width - 280, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if good_ratio > 0.7 else (255, 255, 255), 1)  # 0.5 -> 0.4
        
        # [BOTTOM] Control guide (reduced font size)
        guide_text = "Q: Quit  |  R: Reset  |  S: Screenshot  |  C: Change Exercise  |  SPACE: Toggle Mode"
        cv2.putText(frame, guide_text, (50, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)  # 0.5 -> 0.4
        
        return frame
    
    def analyze_single_image(self, image_path: str) -> Dict:
        """[PHOTO] Complete automatic single image analysis"""
        if not os.path.exists(image_path):
            return {'error': f'Image file not found: {image_path}'}
        
        print(f"[PHOTO] Starting automatic image analysis: {os.path.basename(image_path)}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Cannot read image'}
        
        # Pose detection (high precision model for static images)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose_static.process(image_rgb)
        
        if not results.pose_landmarks:
            return {'error': 'Cannot detect pose'}
        
        # [AI] Step 1: AI exercise detection
        exercise, confidence = self.classify_exercise(image)
        exercise_info = self.exercise_info.get(exercise, {})
        symbol = exercise_info.get('symbol', '[??]')
        name_display = exercise_info.get('name_display', exercise)
        
        print(f"[AI] AI Detection: {symbol} {name_display} (Confidence: {confidence:.1%})")
        
        # [TARGET] Step 2: Angle analysis
        if exercise in self.exercise_thresholds:
            pose_result = self.analyze_pose_angles(results.pose_landmarks.landmark, exercise)
            
            # [FEEDBACK] Step 3: Generate detailed feedback
            feedback_messages = self.generate_detailed_feedback(exercise, pose_result)
            
            # [DISPLAY] Step 4: Generate annotated image
            annotated_image = image.copy()
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Draw overlay
            annotated_image = self.draw_enhanced_overlay(annotated_image, exercise, pose_result)
            
            # Combine results
            return {
                'success': True,
                'image_path': image_path,
                'detected_exercise': exercise,
                'exercise_info': exercise_info,
                'exercise_confidence': confidence,
                'pose_analysis': pose_result,
                'feedback_messages': feedback_messages,
                'original_image': image,
                'annotated_image': annotated_image,
                'analysis_timestamp': datetime.now().isoformat()
            }
        else:
            return {'error': f'Unsupported exercise: {exercise}'}
    
    def analyze_video_file(self, video_path: str, output_path: str = None) -> Dict:
        """[VIDEO] Complete automatic video file analysis"""
        if not os.path.exists(video_path):
            return {'error': f'Video file not found: {video_path}'}
        
        print(f"[VIDEO] Starting automatic video analysis: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video file'}
        
        # Video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video Info: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Output video setup
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Analysis results storage
        frame_results = []
        exercise_detections = {}
        stats = {'good': 0, 'bad': 0, 'total': 0}
        
        # Temporarily initialize history
        self.exercise_history.clear()
        current_exercise = "detecting..."
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB conversion
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Pose detection
                results = self.pose_video.process(frame_rgb)
                
                if results.pose_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # [AI] Exercise detection (for video)
                    exercise, confidence = self.classify_exercise(frame)
                    
                    # Exercise detection statistics
                    if exercise != "detecting..." and exercise != "manual_mode":
                        if exercise not in exercise_detections:
                            exercise_detections[exercise] = 0
                        exercise_detections[exercise] += 1
                        current_exercise = exercise
                    
                    # [TARGET] Angle analysis
                    if current_exercise in self.exercise_thresholds:
                        pose_result = self.analyze_pose_angles(results.pose_landmarks.landmark, current_exercise)
                        
                        if pose_result['valid']:
                            pose_quality = pose_result['classification']
                            stats[pose_quality] += 1
                            stats['total'] += 1
                            
                            # [FEEDBACK] Generate feedback
                            feedback_messages = self.generate_detailed_feedback(current_exercise, pose_result)
                            
                            # [DISPLAY] Draw overlay
                            frame = self.draw_enhanced_overlay(frame, current_exercise, pose_result)
                            
                            # Save results
                            frame_results.append({
                                'frame': frame_count,
                                'timestamp': frame_count / fps,
                                'exercise': current_exercise,
                                'classification': pose_quality,
                                'confidence': pose_result['confidence'],
                                'feedback': feedback_messages[:3]  # top 3 only
                            })
                        else:
                            frame = self.draw_enhanced_overlay(frame, current_exercise, {'valid': False})
                    else:
                        frame = self.draw_enhanced_overlay(frame, current_exercise, {'valid': False})
                
                # Progress display
                if frame_count % (fps * 5) == 0:  # every 5 seconds
                    progress = (frame_count / total_frames) * 100
                    print(f"[PROGRESS] Analysis Progress: {progress:.1f}%")
                
                # Write to output video
                if output_path:
                    out.write(frame)
                
                frame_count += 1
                
        except Exception as e:
            print(f"[ERROR] Video analysis error: {e}")
            return {'error': f'Video analysis failed: {str(e)}'}
        finally:
            cap.release()
            if output_path:
                out.release()
        
        # Find most frequently detected exercise
        main_exercise = max(exercise_detections.items(), key=lambda x: x[1])[0] if exercise_detections else "unknown"
        
        # Results summary
        success_rate = (stats['good'] / max(stats['total'], 1)) * 100
        
        print(f"\n[COMPLETE] Video analysis complete!")
        print(f"[RESULT] Main exercise: {self.exercise_info.get(main_exercise, {}).get('name_display', main_exercise)}")
        print(f"[STATS] Analysis results: Good {stats['good']} frames, Bad {stats['bad']} frames")
        print(f"[SCORE] Success rate: {success_rate:.1f}%")
        
        return {
            'success': True,
            'video_path': video_path,
            'output_path': output_path,
            'main_exercise': main_exercise,
            'exercise_detections': exercise_detections,
            'stats': stats,
            'success_rate': success_rate,
            'frame_results': frame_results,
            'total_frames_analyzed': len(frame_results),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def run_realtime_analysis(self, camera_id: int = 0, manual_exercise: str = None):
        """[REALTIME] Complete automatic realtime analysis"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open camera {camera_id}")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        cv2.namedWindow('Exercise Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Exercise Analysis', 1600, 1200)  # larger window size
        
        print("\n" + "="*80)
        print("[SYSTEM] Complete Automated Exercise Analysis System")
        print("="*80)
        print("[FEATURES] Features:")
        print("  [AI] Step 1: AI automatically detects exercise type")
        print("  [TARGET] Step 2: Precise angle analysis based on detected exercise")
        print("  [FEEDBACK] Step 3: Exercise-specific detailed feedback")
        print("  [COLOR] Step 4: Real-time green/red screen + border")
        print("  [STATS] Step 5: Real-time statistics and performance tracking")
        print("\n[LAYOUT] Screen Layout:")
        print("  • Top Left: Detected exercise type")
        print("  • Bottom Left: Detailed feedback messages")
        print("  • Top Right: Exercise statistics")
        print("  • Center: Form status (Good/Bad)")
        print("  • Overall: Green/red border + background")
        print("\n[CONTROLS] Controls:")
        print("  Q: Quit | R: Reset Stats | S: Screenshot")
        print("  C: Manual Exercise Selection | SPACE: Auto/Manual Mode Toggle")
        print("="*80)
        
        # Model status check and default exercise setting
        if not self.model_loaded:
            print("[WARNING] No AI Model Found - Starting with default exercise")
            # Even without AI model, start with default exercise (not manual mode)
            if manual_exercise:
                self.current_exercise = manual_exercise
            else:
                self.current_exercise = 'squat'  # default squat
            
            exercise_info = self.exercise_info.get(self.current_exercise, {})
            print(f"Default Exercise: {exercise_info.get('symbol', '[??]')} {exercise_info.get('name_display', self.current_exercise)}")
            print("[INFO] You can change exercise with 'C' key or train AI model for auto-detection")
        
        # Manual exercise selection
        available_exercises = list(self.exercise_thresholds.keys())
        manual_mode = False  # default auto mode (analyze with current set exercise even without AI)
        current_manual_idx = 0
        
        if self.current_exercise in available_exercises:
            current_manual_idx = available_exercises.index(self.current_exercise)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)  # selfie mode
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose_video.process(frame_rgb)
                
                if results.pose_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # [AI] Step 1: AI exercise detection (only when AI model exists)
                    if self.model_loaded and not manual_mode:
                        exercise, confidence = self.classify_exercise(frame)
                    else:
                        # When no AI model or manual mode, use current set exercise
                        exercise = self.current_exercise
                        confidence = 1.0
                    
                    # [TARGET] Step 2: Angle analysis
                    if exercise in self.exercise_thresholds:
                        pose_result = self.analyze_pose_angles(results.pose_landmarks.landmark, exercise)
                        
                        if pose_result['valid']:
                            # Update statistics
                            self.stats['frames'] += 1
                            pose_quality = pose_result['classification']
                            self.stats[pose_quality] += 1
                            
                            # [DISPLAY] Steps 3-4: Feedback + screen overlay
                            frame = self.draw_enhanced_overlay(frame, exercise, pose_result)
                        else:
                            frame = self.draw_enhanced_overlay(frame, exercise, {'valid': False})
                    else:
                        frame = self.draw_enhanced_overlay(frame, exercise, {'valid': False})
                else:
                    # Pose not detected
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 0), 30)
                    message = "Stand in front of camera (full body visible)"
                    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = frame.shape[0] // 2
                    
                    cv2.rectangle(frame, (text_x - 20, text_y - 40), 
                                 (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)
                    cv2.putText(frame, message, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # Frame size adjustment (display larger)
                display_frame = frame.copy()
                height, width = display_frame.shape[:2]
                
                # Resize to desired display size
                target_width = 1280
                target_height = 960
                
                # Resize maintaining aspect ratio
                scale = min(target_width / width, target_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                # Screen output
                window_title = "[SYSTEM] Complete Automated Exercise Analysis System"
                cv2.imshow(window_title, display_frame)
                
                # Key input handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset statistics
                    self.stats = {'good': 0, 'bad': 0, 'frames': 0}
                    self.exercise_history.clear()
                    self.pose_history.clear()
                    print("[RESET] Statistics Reset")
                elif key == ord('s'):
                    # Screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"complete_analysis_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[SCREENSHOT] Screenshot saved: {filename}")
                elif key == ord('c'):
                    # Manual exercise change
                    current_manual_idx = (current_manual_idx + 1) % len(available_exercises)
                    self.current_exercise = available_exercises[current_manual_idx]
                    exercise_info = self.exercise_info.get(self.current_exercise, {})
                    symbol = exercise_info.get('symbol', '[??]')
                    name_display = exercise_info.get('name_display', self.current_exercise)
                    print(f"[CHANGE] Manual Selection: {symbol} {name_display}")
                elif key == ord(' '):
                    # Auto/manual mode toggle (only when AI model exists)
                    if self.model_loaded:
                        manual_mode = not manual_mode
                        mode = "Manual" if manual_mode else "AI Auto"
                        print(f"[MODE] Changed to {mode} Mode")
                    else:
                        print("[INFO] AI model not available - Use 'C' to change exercise manually")
        
        except KeyboardInterrupt:
            print("\n[STOP] User Interrupted")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            total = self.stats['good'] + self.stats['bad']
            if total > 0:
                success_rate = (self.stats['good'] / total) * 100
                print(f"\n[FINAL] Final Statistics:")
                print(f"  [TOTAL] Total Analysis: {total} frames")
                print(f"  [GOOD] Good: {self.stats['good']} ({success_rate:.1f}%)")
                print(f"  [BAD] Bad: {self.stats['bad']} ({100-success_rate:.1f}%)")
                print(f"  [COMPLETE] Exercise-specific analysis complete!")
            
            return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='[SYSTEM] Complete Automated Exercise Analyzer - Photo/Video/Realtime',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
[FEATURES] Complete Automation Features:
  Step 1: [AI] AI automatically detects exercise type (Squat, Push-up, Deadlift, Bench Press, Lunge)
  Step 2: [TARGET] Precise angle analysis based on detected exercise
  Step 3: [FEEDBACK] Exercise-specific detailed feedback
  Step 4: [COLOR] Real-time green/red screen + border
  Step 5: [STATS] Real-time statistics and performance tracking

[LAYOUT] Screen Layout:
  • Top Left: Detected exercise type + confidence
  • Bottom Left: Detailed feedback messages (angle-specific advice)
  • Top Right: Exercise statistics (Good/Bad ratio)
  • Center: Form status (Perfect Form! / Form Needs Work)
  • Overall: Green(Good)/Red(Bad) border + background

[USAGE] Usage Examples:
  # Real-time complete auto analysis
  python auto_exercise_analyzer.py --mode realtime
  
  # Real-time + manual exercise selection
  python auto_exercise_analyzer.py --mode realtime --manual squat
  
  # Photo complete auto analysis
  python auto_exercise_analyzer.py --mode image --input photo.jpg
  
  # Video complete auto analysis
  python auto_exercise_analyzer.py --mode video --input video.mp4 --output analyzed.mp4

[CONTROLS] Real-time Controls:
  Q: Quit  |  R: Reset Stats  |  S: Screenshot
  C: Change Exercise (Manual)  |  SPACE: Auto/Manual Mode Toggle

[EXERCISES] Supported Exercises & Detailed Feedback:
  [SQ] Squat: Knee/hip angles, keep back straight, knees behind toes
  [PU] Push-up: Elbow angles, straight body line, shoulder stability
  [DL] Deadlift: Hip hinge, straight back, knee angles (relaxed criteria)
  [BP] Bench Press: Elbow/shoulder angles, back arch
  [LG] Lunge: Front knee 90°, extend back knee, upright torso

[MODEL] AI Model Required:
  With models/exercise_classifier.pkl: Complete automation
  Without model: Manual exercise selection mode
        """
    )
    
    parser.add_argument('--mode', type=str, default='realtime',
                       choices=['realtime', 'image', 'video'],
                       help='Analysis mode: realtime, image, or video')
    parser.add_argument('--input', type=str,
                       help='Input file path (for image/video mode)')
    parser.add_argument('--output', type=str,
                       help='Output file path (for video mode)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (for realtime mode)')
    parser.add_argument('--manual', type=str,
                       choices=['squat', 'push_up', 'deadlift', 'bench_press', 'lunge'],
                       help='Manual exercise selection (skip AI detection)')
    
    args = parser.parse_args()
    
    # Complete automation analyzer initialization
    try:
        analyzer = CompleteAutoExerciseAnalyzer()
    except Exception as e:
        print(f"[ERROR] System initialization failed: {e}")
        return 1
    
    print("[SYSTEM] Complete Automated Exercise Analysis System Starting!")
    print("="*80)
    print("[FEATURES] Key Features:")
    print("  [AI] AI automatic exercise detection (5 exercises)")
    print("  [ANGLE] Precise angle analysis")
    print("  [FEEDBACK] Exercise-specific detailed feedback")
    print("  [COLOR] Real-time green/red feedback")
    print("  [STATS] Performance tracking")
    print("  [MULTI] Photo/Video/Real-time support")
    
    try:
        if args.mode == 'realtime':
            print(f"\n[REALTIME] Starting real-time analysis (Camera {args.camera})")
            if args.manual:
                exercise_info = analyzer.exercise_info.get(args.manual, {})
                symbol = exercise_info.get('symbol', '[??]')
                name_display = exercise_info.get('name_display', args.manual)
                print(f"[MANUAL] Manual Mode: {symbol} {name_display}")
            success = analyzer.run_realtime_analysis(args.camera, args.manual)
            return 0 if success else 1
            
        elif args.mode == 'image':
            if not args.input:
                print("[ERROR] --input option required (image file path)")
                return 1
            
            print(f"\n[PHOTO] Starting image analysis: {args.input}")
            result = analyzer.analyze_single_image(args.input)
            
            if result.get('success', False):
                # Output results
                exercise_info = result['exercise_info']
                symbol = exercise_info.get('symbol', '[??]')
                name_display = exercise_info.get('name_display', 'unknown')
                exercise_conf = result['exercise_confidence']
                pose_result = result['pose_analysis']
                
                print(f"\n[COMPLETE] Image analysis complete!")
                print(f"[AI] AI Detection: {symbol} {name_display} (Confidence: {exercise_conf:.1%})")
                
                if pose_result['valid']:
                    pose_quality = pose_result['classification']
                    pose_conf = pose_result['confidence']
                    
                    status_symbol = "[OK]" if pose_quality == 'good' else "[!]"
                    print(f"[RESULT] Form Analysis: {status_symbol} {pose_quality.upper()} (Score: {pose_conf:.1%})")
                    
                    # Output feedback messages
                    feedback_messages = result['feedback_messages']
                    if feedback_messages:
                        print(f"\n[FEEDBACK] Detailed Feedback:")
                        for i, message in enumerate(feedback_messages[:5], 1):
                            print(f"  {i}. {message}")
                    
                    # Output violations
                    violations = pose_result.get('violations', [])
                    if violations:
                        print(f"\n[ANGLES] Angle Analysis:")
                        for violation in violations[:3]:
                            joint_en = violation.get('name_en', violation['joint'])
                            angle = violation['angle']
                            range_min, range_max = violation['expected_range']
                            print(f"  • {joint_en}: {angle:.1f}° -> Target: {range_min:.0f}-{range_max:.0f}°")
                
                # Display annotated image
                annotated_image = result['annotated_image']
                
                # Resize image (fit to screen)
                height, width = annotated_image.shape[:2]
                if width > 1200:
                    scale = 1200 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    annotated_image = cv2.resize(annotated_image, (new_width, new_height))
                
                window_title = f"Complete Auto Analysis Result: {symbol} {name_display}"
                cv2.imshow(window_title, annotated_image)
                
                print(f"\n[DISPLAY] Analysis result image displayed... (Press any key to close)")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            else:
                print(f"[ERROR] Image analysis failed: {result.get('error', 'Unknown error')}")
                return 1
                
        elif args.mode == 'video':
            if not args.input:
                print("[ERROR] --input option required (video file path)")
                return 1
            
            print(f"\n[VIDEO] Starting video analysis: {args.input}")
            if args.output:
                print(f"[OUTPUT] Output path: {args.output}")
            
            result = analyzer.analyze_video_file(args.input, args.output)
            
            if result.get('success', False):
                # Output results
                main_exercise = result['main_exercise']
                exercise_info = analyzer.exercise_info.get(main_exercise, {})
                symbol = exercise_info.get('symbol', '[??]')
                name_display = exercise_info.get('name_display', main_exercise)
                
                stats = result['stats']
                success_rate = result['success_rate']
                total_analyzed = result['total_frames_analyzed']
                
                print(f"\n[COMPLETE] Video analysis complete!")
                print(f"[MAIN] Main exercise: {symbol} {name_display}")
                print(f"[STATS] Analysis results:")
                print(f"  • Total analyzed frames: {total_analyzed}")
                print(f"  • [OK] Good form: {stats['good']} frames")
                print(f"  • [!] Bad form: {stats['bad']} frames")
                print(f"  • [SCORE] Success rate: {success_rate:.1f}%")
                
                # Exercise detection statistics
                exercise_detections = result['exercise_detections']
                if len(exercise_detections) > 1:
                    print(f"\n[DETECTION] Exercise detection statistics:")
                    for exercise, count in exercise_detections.items():
                        info = analyzer.exercise_info.get(exercise, {})
                        symbol = info.get('symbol', '[??]')
                        name_display = info.get('name_display', exercise)
                        percentage = (count / sum(exercise_detections.values())) * 100
                        print(f"  • {symbol} {name_display}: {count} frames ({percentage:.1f}%)")
                
                if args.output:
                    print(f"\n[SAVE] Annotated video saved: {args.output}")
                
            else:
                print(f"[ERROR] Video analysis failed: {result.get('error', 'Unknown error')}")
                return 1
    
    except KeyboardInterrupt:
        print("\n[STOP] User interrupted")
        return 0
    except Exception as e:
        print(f"[ERROR] Execution error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())