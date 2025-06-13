#!/usr/bin/env python3
"""
🤖 완전 자동화 운동 분석기 - 사진/영상/실시간 통합 버전 (영어 출력)
1단계: AI가 운동 종류 자동 감지
2단계: 감지된 운동에 맞춰 상세 각도 분석
3단계: 운동별 맞춤 피드백 + 초록/빨강 화면 표시
4단계: 사진, 영상, 실시간 모두 지원
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
    """완전 자동화 운동 분석기 - 사진/영상/실시간 통합"""
    
    def __init__(self):
        # MediaPipe 초기화
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
        
        # AI 운동 분류 모델 로드
        self.exercise_classifier = None
        self.model_loaded = False
        self.temp_dir = tempfile.mkdtemp()
        self.load_exercise_model()
        
        # Enhanced 각도 기준
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
        
        # 운동별 맞춤 피드백 메시지 (영어로 변경)
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
        
        # Enhanced 분류 임계값
        self.classification_thresholds = {
            'squat': 0.5,
            'push_up': 0.7,
            'deadlift': 0.8,  # 완화
            'bench_press': 0.5,
            'lunge': 0.6,
        }
        
        # 운동 이모지 및 영어명
        self.exercise_info = {
            'squat': {'emoji': '🏋️‍♀️', 'name_en': 'SQUAT', 'name_display': 'Squat'},
            'push_up': {'emoji': '💪', 'name_en': 'PUSH-UP', 'name_display': 'Push-up'},
            'deadlift': {'emoji': '🏋️‍♂️', 'name_en': 'DEADLIFT', 'name_display': 'Deadlift'},
            'bench_press': {'emoji': '🔥', 'name_en': 'BENCH PRESS', 'name_display': 'Bench Press'},
            'lunge': {'emoji': '🚀', 'name_en': 'LUNGE', 'name_display': 'Lunge'}
        }
        
        # 상태 관리
        self.current_exercise = "detecting..."
        self.current_pose_quality = "unknown"
        self.exercise_confidence = 0.0
        self.pose_confidence = 0.0
        
        # 안정화를 위한 히스토리
        self.exercise_history = deque(maxlen=10)
        self.pose_history = deque(maxlen=5)
        
        # 통계
        self.stats = {'good': 0, 'bad': 0, 'frames': 0}
        
        # 화면 상태 (부드러운 전환)
        self.screen_color = (128, 128, 128)  # 기본 회색
        self.target_color = (128, 128, 128)
        self.color_transition_speed = 0.15
        
        # 타이밍
        self.last_classification_time = 0
        self.classification_interval = 2.0  # 2초마다 운동 분류
        
        # 피드백 메시지 관리
        self.current_feedback_messages = []
        self.last_feedback_time = 0
        self.feedback_interval = 1.0  # 1초마다 피드백 업데이트
    
    def load_exercise_model(self):
        """AI 운동 분류 모델 로드 (scripts/ 폴더 고려)"""
        # 가능한 모델 경로들 확인
        possible_paths = [
            "models/exercise_classifier.pkl",           # 현재 폴더
            "scripts/models/exercise_classifier.pkl",   # scripts 폴더 안
            "../models/exercise_classifier.pkl",        # 상위 폴더
            "./exercise_classifier.pkl"                 # 같은 폴더
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        print(f"🔍 Searching for model in multiple locations...")
        for path in possible_paths:
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"  {exists} {path}")
        
        if not model_path:
            print("❌ No AI Model Found in any location")
            print(f"💡 Current directory: {os.getcwd()}")
            print(f"💡 Files in current dir: {os.listdir('.')}")
            if os.path.exists('scripts'):
                print(f"💡 Files in scripts/: {os.listdir('scripts')}")
            if os.path.exists('models'):
                print(f"💡 Files in models/: {os.listdir('models')}")
            if os.path.exists('scripts/models'):
                print(f"💡 Files in scripts/models/: {os.listdir('scripts/models')}")
            self.model_loaded = False
            return
        
        print(f"✅ Found model at: {model_path}")
        
        try:
            print("✅ Model file found, attempting to import...")
            try:
                from exercise_classifier import ExerciseClassificationModel
                print("✅ Successfully imported ExerciseClassificationModel")
            except ImportError as ie:
                print(f"❌ Import Error: {ie}")
                print("💡 Make sure exercise_classifier.py is in the current directory")
                # scripts 폴더에서 import 시도
                try:
                    import sys
                    if 'scripts' not in sys.path:
                        sys.path.append('scripts')
                    from exercise_classifier import ExerciseClassificationModel
                    print("✅ Successfully imported from scripts folder")
                except ImportError as ie2:
                    print(f"❌ Import from scripts also failed: {ie2}")
                    self.model_loaded = False
                    return
            
            print("✅ Creating model instance...")
            self.exercise_classifier = ExerciseClassificationModel()
            
            print(f"✅ Loading model from {model_path}...")
            self.model_loaded = self.exercise_classifier.load_model(model_path)
            
            if self.model_loaded:
                print("✅ AI Exercise Classification Model Loaded Successfully")
                # 지원되는 운동 목록 출력
                if hasattr(self.exercise_classifier, 'label_encoder'):
                    exercises = list(self.exercise_classifier.label_encoder.keys())
                    print(f"🎯 Supported exercises: {exercises}")
                
                # 모델 테스트
                print("🧪 Testing model with dummy prediction...")
                # 간단한 테스트는 생략 (실제 이미지가 필요함)
                
            else:
                print("❌ Model Load Failed - model.load_model() returned False")
                print("💡 Try retraining the model: python main.py --mode train")
                
        except Exception as e:
            print(f"❌ Model Load Error: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """각도 계산"""
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
        """🤖 1단계: AI로 운동 종류 자동 감지"""
        current_time = time.time()
        
        # 분류 주기 제어 (2초마다)
        if current_time - self.last_classification_time < self.classification_interval:
            return self.current_exercise, self.exercise_confidence
        
        if not self.model_loaded:
            return "manual_mode", 0.0
        
        try:
            # 임시 이미지 저장
            temp_path = os.path.join(self.temp_dir, "temp_frame.jpg")
            cv2.imwrite(temp_path, frame)
            
            # AI 운동 분류
            exercise, confidence = self.exercise_classifier.predict(temp_path)
            
            # 히스토리 안정화
            self.exercise_history.append((exercise, confidence))
            
            if len(self.exercise_history) >= 3:
                # 최근 3개 결과의 합의
                recent = list(self.exercise_history)[-3:]
                high_conf_predictions = [(ex, conf) for ex, conf in recent if conf > 0.6]
                
                if high_conf_predictions:
                    from collections import Counter
                    exercises = [ex for ex, conf in high_conf_predictions]
                    most_common = Counter(exercises).most_common(1)[0]
                    
                    if most_common[1] >= 2:  # 2번 이상 감지
                        new_exercise = most_common[0]
                        if new_exercise != self.current_exercise:
                            self.current_exercise = new_exercise
                            self.exercise_confidence = confidence
                            exercise_info = self.exercise_info.get(new_exercise, {})
                            emoji = exercise_info.get('emoji', '🏋️')
                            name_display = exercise_info.get('name_display', new_exercise)
                            print(f"AI Detected: {emoji} {name_display} (Confidence: {confidence:.1%})")
            
            self.last_classification_time = current_time
            
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return self.current_exercise, self.exercise_confidence
            
        except Exception as e:
            print(f"Exercise Classification Error: {e}")
            return self.current_exercise, self.exercise_confidence
    
    def analyze_pose_angles(self, landmarks, exercise: str) -> Dict:
        """🎯 2단계: 감지된 운동에 맞춰 상세 각도 분석"""
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
                
                # 가시성 확인
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
        
        # Enhanced 분류
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
        """🗣️ 운동별 상세 피드백 생성 (영어)"""
        current_time = time.time()
        
        # 피드백 주기 제한
        if current_time - self.last_feedback_time < self.feedback_interval:
            return self.current_feedback_messages
        
        messages = []
        
        if not pose_result.get('valid', False):
            messages.append("Cannot recognize pose")
            return messages
        
        violations = pose_result.get('violations', [])
        exercise_feedback = self.detailed_feedback.get(exercise, {})
        
        if not violations:
            # 모든 자세가 완벽한 경우
            exercise_info = self.exercise_info.get(exercise, {})
            name_display = exercise_info.get('name_display', exercise)
            messages.append(f"Perfect {name_display} form! 👍")
            messages.append("Keep this form!")
        else:
            # 위반사항이 있는 경우 - 가중치 순으로 정렬
            violations_sorted = sorted(violations, key=lambda x: x['weight'], reverse=True)
            
            for i, violation in enumerate(violations_sorted[:3]):  # 상위 3개만
                joint = violation['joint']
                angle = violation['angle']
                min_angle, max_angle = violation['expected_range']
                name_en = violation.get('name_en', joint)
                
                joint_feedback = exercise_feedback.get(joint, {})
                
                if angle < min_angle:
                    # 각도가 너무 작음
                    message = joint_feedback.get('too_low', f'Increase {name_en} angle')
                elif angle > max_angle:
                    # 각도가 너무 큼
                    message = joint_feedback.get('too_high', f'Decrease {name_en} angle')
                else:
                    message = joint_feedback.get('good', f'{name_en} is good!')
                
                messages.append(f"⚠️ {message}")
                
                # 구체적인 각도 정보 추가
                if i == 0:  # 가장 중요한 문제만 각도 표시
                    messages.append(f"   Current: {angle:.0f}° → Target: {min_angle:.0f}-{max_angle:.0f}°")
            
            # 일반적인 운동별 조언 추가
            general_advice = exercise_feedback.get('general', '')
            if general_advice and len(violations_sorted) <= 2:
                messages.append(f"💡 {general_advice}")
        
        self.current_feedback_messages = messages
        self.last_feedback_time = current_time
        return messages
    
    def update_screen_color(self, pose_quality: str):
        """🌈 초록/빨강 화면 색상 업데이트"""
        if pose_quality == 'good':
            self.target_color = (0, 255, 0)      # 초록색
        elif pose_quality == 'bad':
            self.target_color = (0, 0, 255)      # 빨간색
        elif pose_quality == 'detecting':
            self.target_color = (255, 255, 0)    # 노란색
        else:
            self.target_color = (128, 128, 128)  # 회색
        
        # 부드러운 색상 전환
        for i in range(3):
            current = self.screen_color[i]
            target = self.target_color[i]
            diff = target - current
            self.screen_color = tuple(
                int(current + diff * self.color_transition_speed) if j == i 
                else self.screen_color[j] for j in range(3)
            )
    
    def draw_enhanced_overlay(self, frame: np.ndarray, exercise: str, pose_result: Dict) -> np.ndarray:
        """✨ 향상된 분석 결과 화면 오버레이 (영어 텍스트)"""
        height, width = frame.shape[:2]
        
        # 🌈 전체 화면 색상 오버레이 및 테두리
        if pose_result.get('valid', False):
            pose_quality = pose_result['classification']
            self.update_screen_color(pose_quality)
            
            # 투명한 색상 오버레이
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), self.screen_color, -1)
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # 🎯 두꺼운 테두리
        border_thickness = 30
        cv2.rectangle(frame, (0, 0), (width, height), self.screen_color, border_thickness)
        
        # 📍 왼쪽 위: 운동 종류 표시
        exercise_info = self.exercise_info.get(exercise, {})
        if exercise != "detecting..." and exercise != "manual_mode":
            emoji = exercise_info.get('emoji', '🏋️')
            name_display = exercise_info.get('name_display', exercise)
            name_en = exercise_info.get('name_en', exercise.upper())
            
            # 배경 박스
            cv2.rectangle(frame, (40, 40), (400, 140), (0, 0, 0), -1)
            cv2.rectangle(frame, (40, 40), (400, 140), self.screen_color, 3)
            
            # 운동명 표시 (글자 크기 줄임)
            exercise_text = f"{emoji} {name_display}"
            cv2.putText(frame, exercise_text, (60, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # 1.2 -> 0.8
            
            # 영어명 표시 (글자 크기 줄임)
            cv2.putText(frame, name_en, (60, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)  # 0.8 -> 0.6
            
            # 신뢰도 표시 (글자 크기 줄임)
            if self.model_loaded:
                confidence_text = f"AI: {self.exercise_confidence:.0%}"
            else:
                confidence_text = "Manual Mode"
            cv2.putText(frame, confidence_text, (60, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)  # 0.6 -> 0.5
            
        elif exercise == "detecting...":
            cv2.rectangle(frame, (40, 40), (300, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (40, 40), (300, 100), (255, 255, 0), 3)
            cv2.putText(frame, "🤖 Detecting...", (60, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # 글자 크기 줄임
        else:
            cv2.rectangle(frame, (40, 40), (320, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (40, 40), (320, 100), (128, 128, 128), 3)
            cv2.putText(frame, "⚙️ No AI Model", (60, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 글자 크기 줄임
        
        # 🎯 중앙 상태 메시지
        if pose_result.get('valid', False):
            pose_quality = pose_result['classification']
            confidence = pose_result['confidence']
            
            if pose_quality == 'good':
                status_text = "Perfect Form! 👍"
                status_color = (0, 255, 0)
            else:
                status_text = "Form Needs Work ⚠️"
                status_color = (0, 0, 255)
            
            # 중앙 상태 표시
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            status_x = (width - status_size[0]) // 2
            status_y = height // 2 - 80
            
            cv2.rectangle(frame, (status_x - 30, status_y - 40), 
                         (status_x + status_size[0] + 30, status_y + 20), (0, 0, 0), -1)
            cv2.rectangle(frame, (status_x - 30, status_y - 40), 
                         (status_x + status_size[0] + 30, status_y + 20), status_color, 4)
            
            cv2.putText(frame, status_text, (status_x, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)  # 1.5 -> 1.0
            
            # 신뢰도 점수 (글자 크기 줄임)
            score_text = f"Form Score: {confidence:.0%}"
            score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]  # 0.8 -> 0.6
            score_x = (width - score_size[0]) // 2
            cv2.putText(frame, score_text, (score_x, status_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # 0.8 -> 0.6
        
        # 📍 왼쪽 아래: 상세 피드백 메시지
        if exercise in self.exercise_thresholds:
            feedback_messages = self.generate_detailed_feedback(exercise, pose_result)
            
            if feedback_messages:
                # 피드백 영역 배경
                feedback_height = len(feedback_messages) * 35 + 60
                cv2.rectangle(frame, (40, height - feedback_height - 40), 
                             (width - 40, height - 40), (0, 0, 0), -1)
                cv2.rectangle(frame, (40, height - feedback_height - 40), 
                             (width - 40, height - 40), self.screen_color, 3)
                
                # 피드백 제목 (글자 크기 줄임)
                cv2.putText(frame, "💬 Feedback:", (60, height - feedback_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # 0.7 -> 0.6
                
                # 피드백 메시지들 (글자 크기 줄임)
                for i, message in enumerate(feedback_messages[:5]):  # 최대 5개
                    y_pos = height - feedback_height + 20 + (i * 30)  # 35 -> 30 (줄간격 줄임)
                    
                    # 메시지 색상 결정
                    if "Perfect" in message or "👍" in message:
                        msg_color = (0, 255, 0)  # 초록색
                    elif "⚠️" in message:
                        msg_color = (0, 100, 255)  # 주황색
                    elif "💡" in message:
                        msg_color = (255, 255, 0)  # 노란색
                    else:
                        msg_color = (255, 255, 255)  # 흰색
                    
                    cv2.putText(frame, message, (60, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, msg_color, 2)  # 0.6 -> 0.5
        
        # 📊 오른쪽 위: 통계 정보
        if self.stats['frames'] > 0:
            total = self.stats['good'] + self.stats['bad']
            if total > 0:
                good_ratio = self.stats['good'] / total
                
                # 통계 배경
                cv2.rectangle(frame, (width - 300, 40), (width - 40, 140), (0, 0, 0), -1)
                cv2.rectangle(frame, (width - 300, 40), (width - 40, 140), (255, 255, 255), 2)
                
                # 통계 텍스트 (글자 크기 줄임)
                cv2.putText(frame, "📊 Stats", (width - 280, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # 0.6 -> 0.5
                
                stats_text = f"Good: {self.stats['good']} | Bad: {self.stats['bad']}"
                cv2.putText(frame, stats_text, (width - 280, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # 0.5 -> 0.4
                
                ratio_text = f"Success: {good_ratio:.1%}"
                cv2.putText(frame, ratio_text, (width - 280, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if good_ratio > 0.7 else (255, 255, 255), 1)  # 0.5 -> 0.4
        
        # ⌨️ 하단 조작 가이드 (글자 크기 줄임)
        guide_text = "Q: Quit  |  R: Reset  |  S: Screenshot  |  C: Change Exercise  |  SPACE: Toggle Mode"
        cv2.putText(frame, guide_text, (50, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)  # 0.5 -> 0.4
        
        return frame
    
    def analyze_single_image(self, image_path: str) -> Dict:
        """📷 단일 이미지 완전 자동 분석"""
        if not os.path.exists(image_path):
            return {'error': f'Image file not found: {image_path}'}
        
        print(f"📷 Starting automatic image analysis: {os.path.basename(image_path)}")
        
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Cannot read image'}
        
        # 포즈 검출 (정적 이미지용 고정밀 모델)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose_static.process(image_rgb)
        
        if not results.pose_landmarks:
            return {'error': 'Cannot detect pose'}
        
        # 🤖 1단계: AI 운동 감지
        exercise, confidence = self.classify_exercise(image)
        exercise_info = self.exercise_info.get(exercise, {})
        emoji = exercise_info.get('emoji', '🏋️')
        name_display = exercise_info.get('name_display', exercise)
        
        print(f"🎯 AI Detection: {emoji} {name_display} (Confidence: {confidence:.1%})")
        
        # 🎯 2단계: 각도 분석
        if exercise in self.exercise_thresholds:
            pose_result = self.analyze_pose_angles(results.pose_landmarks.landmark, exercise)
            
            # 🗣️ 3단계: 상세 피드백 생성
            feedback_messages = self.generate_detailed_feedback(exercise, pose_result)
            
            # 📸 4단계: 주석 이미지 생성
            annotated_image = image.copy()
            
            # 랜드마크 그리기
            self.mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            
            # 오버레이 그리기
            annotated_image = self.draw_enhanced_overlay(annotated_image, exercise, pose_result)
            
            # 결과 합치기
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
        """🎬 영상 파일 완전 자동 분석"""
        if not os.path.exists(video_path):
            return {'error': f'Video file not found: {video_path}'}
        
        print(f"🎬 Starting automatic video analysis: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video file'}
        
        # 영상 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video Info: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # 출력 영상 설정
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 분석 결과 저장
        frame_results = []
        exercise_detections = {}
        stats = {'good': 0, 'bad': 0, 'total': 0}
        
        # 임시로 히스토리 초기화
        self.exercise_history.clear()
        current_exercise = "detecting..."
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 포즈 검출
                results = self.pose_video.process(frame_rgb)
                
                if results.pose_landmarks:
                    # 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # 🤖 운동 감지 (영상용)
                    exercise, confidence = self.classify_exercise(frame)
                    
                    # 운동 감지 통계
                    if exercise != "detecting..." and exercise != "manual_mode":
                        if exercise not in exercise_detections:
                            exercise_detections[exercise] = 0
                        exercise_detections[exercise] += 1
                        current_exercise = exercise
                    
                    # 🎯 각도 분석
                    if current_exercise in self.exercise_thresholds:
                        pose_result = self.analyze_pose_angles(results.pose_landmarks.landmark, current_exercise)
                        
                        if pose_result['valid']:
                            pose_quality = pose_result['classification']
                            stats[pose_quality] += 1
                            stats['total'] += 1
                            
                            # 🗣️ 피드백 생성
                            feedback_messages = self.generate_detailed_feedback(current_exercise, pose_result)
                            
                            # ✨ 오버레이 그리기
                            frame = self.draw_enhanced_overlay(frame, current_exercise, pose_result)
                            
                            # 결과 저장
                            frame_results.append({
                                'frame': frame_count,
                                'timestamp': frame_count / fps,
                                'exercise': current_exercise,
                                'classification': pose_quality,
                                'confidence': pose_result['confidence'],
                                'feedback': feedback_messages[:3]  # 상위 3개만
                            })
                        else:
                            frame = self.draw_enhanced_overlay(frame, current_exercise, {'valid': False})
                    else:
                        frame = self.draw_enhanced_overlay(frame, current_exercise, {'valid': False})
                
                # 진행률 표시
                if frame_count % (fps * 5) == 0:  # 5초마다
                    progress = (frame_count / total_frames) * 100
                    print(f"📊 Analysis Progress: {progress:.1f}%")
                
                # 출력 영상에 쓰기
                if output_path:
                    out.write(frame)
                
                frame_count += 1
                
        except Exception as e:
            print(f"❌ Video analysis error: {e}")
            return {'error': f'Video analysis failed: {str(e)}'}
        finally:
            cap.release()
            if output_path:
                out.release()
        
        # 가장 많이 감지된 운동 찾기
        main_exercise = max(exercise_detections.items(), key=lambda x: x[1])[0] if exercise_detections else "unknown"
        
        # 결과 요약
        success_rate = (stats['good'] / max(stats['total'], 1)) * 100
        
        print(f"\n🎉 Video analysis complete!")
        print(f"🎯 Main exercise: {self.exercise_info.get(main_exercise, {}).get('name_display', main_exercise)}")
        print(f"📊 Analysis results: Good {stats['good']} frames, Bad {stats['bad']} frames")
        print(f"🎯 Success rate: {success_rate:.1f}%")
        
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
        """🎮 실시간 완전 자동 분석"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Failed to open camera {camera_id}")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        cv2.namedWindow('Exercise Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Exercise Analysis', 1600, 1200)  # 더 큰 창 크기
        
        print("\n" + "="*80)
        print("🤖 Complete Automated Exercise Analysis System")
        print("="*80)
        print("✨ Features:")
        print("  🤖 Step 1: AI automatically detects exercise type")
        print("  🎯 Step 2: Precise angle analysis based on detected exercise")
        print("  🗣️ Step 3: Exercise-specific detailed feedback")
        print("  🌈 Step 4: Real-time green/red screen + border")
        print("  📊 Step 5: Real-time statistics and performance tracking")
        print("\n📍 Screen Layout:")
        print("  • Top Left: Detected exercise type")
        print("  • Bottom Left: Detailed feedback messages")
        print("  • Top Right: Exercise statistics")
        print("  • Center: Form status (Good/Bad)")
        print("  • Overall: Green/red border + background")
        print("\n⌨️ Controls:")
        print("  Q: Quit | R: Reset Stats | S: Screenshot")
        print("  C: Manual Exercise Selection | SPACE: Auto/Manual Mode Toggle")
        print("="*80)
        
        # 모델 상태 확인 및 기본 운동 설정
        if not self.model_loaded:
            print("⚠️ No AI Model Found - Starting with default exercise")
            # AI 모델이 없어도 기본 운동으로 시작 (수동 모드가 아님)
            if manual_exercise:
                self.current_exercise = manual_exercise
            else:
                self.current_exercise = 'squat'  # 기본값으로 스쿼트 설정
            
            exercise_info = self.exercise_info.get(self.current_exercise, {})
            print(f"Default Exercise: {exercise_info.get('emoji', '🏋️')} {exercise_info.get('name_display', self.current_exercise)}")
            print("💡 You can change exercise with 'C' key or train AI model for auto-detection")
        
        # 수동 운동 선택용
        available_exercises = list(self.exercise_thresholds.keys())
        manual_mode = False  # 기본적으로 자동 모드 (AI 없어도 현재 설정된 운동으로 분석)
        current_manual_idx = 0
        
        if self.current_exercise in available_exercises:
            current_manual_idx = available_exercises.index(self.current_exercise)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)  # 셀카 모드
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose_video.process(frame_rgb)
                
                if results.pose_landmarks:
                    # 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # 🤖 1단계: AI 운동 감지 (AI 모델이 있을 때만)
                    if self.model_loaded and not manual_mode:
                        exercise, confidence = self.classify_exercise(frame)
                    else:
                        # AI 모델이 없거나 수동 모드일 때는 현재 설정된 운동 사용
                        exercise = self.current_exercise
                        confidence = 1.0
                    
                    # 🎯 2단계: 각도 분석
                    if exercise in self.exercise_thresholds:
                        pose_result = self.analyze_pose_angles(results.pose_landmarks.landmark, exercise)
                        
                        if pose_result['valid']:
                            # 통계 업데이트
                            self.stats['frames'] += 1
                            pose_quality = pose_result['classification']
                            self.stats[pose_quality] += 1
                            
                            # ✨ 3-4단계: 피드백 + 화면 오버레이
                            frame = self.draw_enhanced_overlay(frame, exercise, pose_result)
                        else:
                            frame = self.draw_enhanced_overlay(frame, exercise, {'valid': False})
                    else:
                        frame = self.draw_enhanced_overlay(frame, exercise, {'valid': False})
                else:
                    # 포즈 미감지
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 0), 30)
                    message = "Stand in front of camera (full body visible)"
                    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = frame.shape[0] // 2
                    
                    cv2.rectangle(frame, (text_x - 20, text_y - 40), 
                                 (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)
                    cv2.putText(frame, message, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # 프레임 크기 조정 (더 크게 표시)
                display_frame = frame.copy()
                height, width = display_frame.shape[:2]
                
                # 원하는 표시 크기로 리사이즈
                target_width = 1280
                target_height = 960
                
                # 비율 유지하면서 리사이즈
                scale = min(target_width / width, target_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                # 화면 출력
                window_title = "🤖 Complete Automated Exercise Analysis System"
                cv2.imshow(window_title, display_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # 통계 리셋
                    self.stats = {'good': 0, 'bad': 0, 'frames': 0}
                    self.exercise_history.clear()
                    self.pose_history.clear()
                    print("📊 Statistics Reset")
                elif key == ord('s'):
                    # 스크린샷
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"complete_analysis_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('c'):
                    # 수동 운동 변경
                    current_manual_idx = (current_manual_idx + 1) % len(available_exercises)
                    self.current_exercise = available_exercises[current_manual_idx]
                    exercise_info = self.exercise_info.get(self.current_exercise, {})
                    emoji = exercise_info.get('emoji', '🏋️')
                    name_display = exercise_info.get('name_display', self.current_exercise)
                    print(f"🔄 Manual Selection: {emoji} {name_display}")
                elif key == ord(' '):
                    # 자동/수동 모드 토글 (AI 모델이 있을 때만)
                    if self.model_loaded:
                        manual_mode = not manual_mode
                        mode = "Manual" if manual_mode else "AI Auto"
                        print(f"🔄 Changed to {mode} Mode")
                    else:
                        print("💡 AI model not available - Use 'C' to change exercise manually")
        
        except KeyboardInterrupt:
            print("\n⏹️ User Interrupted")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 최종 통계
            total = self.stats['good'] + self.stats['bad']
            if total > 0:
                success_rate = (self.stats['good'] / total) * 100
                print(f"\n📊 Final Statistics:")
                print(f"  🎯 Total Analysis: {total} frames")
                print(f"  ✅ Good: {self.stats['good']} ({success_rate:.1f}%)")
                print(f"  ❌ Bad: {self.stats['bad']} ({100-success_rate:.1f}%)")
                print(f"  🎯 Exercise-specific analysis complete!")
            
            return True

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='🤖 Complete Automated Exercise Analyzer - Photo/Video/Realtime',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 Complete Automation Features:
  Step 1: 🤖 AI automatically detects exercise type (Squat, Push-up, Deadlift, Bench Press, Lunge)
  Step 2: 🎯 Precise angle analysis based on detected exercise
  Step 3: 🗣️ Exercise-specific detailed feedback
  Step 4: 🌈 Real-time green/red screen + border
  Step 5: 📊 Real-time statistics and performance tracking

📍 Screen Layout:
  • Top Left: Detected exercise type + confidence
  • Bottom Left: Detailed feedback messages (angle-specific advice)
  • Top Right: Exercise statistics (Good/Bad ratio)
  • Center: Form status (Perfect Form! / Form Needs Work)
  • Overall: Green(Good)/Red(Bad) border + background

🎯 Usage Examples:
  # Real-time complete auto analysis
  python auto_exercise_analyzer.py --mode realtime
  
  # Real-time + manual exercise selection
  python auto_exercise_analyzer.py --mode realtime --manual squat
  
  # Photo complete auto analysis
  python auto_exercise_analyzer.py --mode image --input photo.jpg
  
  # Video complete auto analysis
  python auto_exercise_analyzer.py --mode video --input video.mp4 --output analyzed.mp4

⌨️ Real-time Controls:
  Q: Quit  |  R: Reset Stats  |  S: Screenshot
  C: Change Exercise (Manual)  |  SPACE: Auto/Manual Mode Toggle

🏋️ Supported Exercises & Detailed Feedback:
  🏋️‍♀️ Squat: Knee/hip angles, keep back straight, knees behind toes
  💪 Push-up: Elbow angles, straight body line, shoulder stability
  🏋️‍♂️ Deadlift: Hip hinge, straight back, knee angles (relaxed criteria)
  🔥 Bench Press: Elbow/shoulder angles, back arch
  🚀 Lunge: Front knee 90°, extend back knee, upright torso

💡 AI Model Required:
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
    
    # 완전 자동화 분석기 초기화
    try:
        analyzer = CompleteAutoExerciseAnalyzer()
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return 1
    
    print("🤖 Complete Automated Exercise Analysis System Starting!")
    print("="*80)
    print("🎯 Key Features:")
    print("  🤖 AI automatic exercise detection (5 exercises)")
    print("  📐 Precise angle analysis")
    print("  🗣️ Exercise-specific detailed feedback")
    print("  🌈 Real-time green/red feedback")
    print("  📊 Performance tracking")
    print("  📷 Photo/🎬 Video/🎮 Real-time support")
    
    try:
        if args.mode == 'realtime':
            print(f"\n🎮 Starting real-time analysis (Camera {args.camera})")
            if args.manual:
                exercise_info = analyzer.exercise_info.get(args.manual, {})
                emoji = exercise_info.get('emoji', '🏋️')
                name_display = exercise_info.get('name_display', args.manual)
                print(f"🔧 Manual Mode: {emoji} {name_display}")
            success = analyzer.run_realtime_analysis(args.camera, args.manual)
            return 0 if success else 1
            
        elif args.mode == 'image':
            if not args.input:
                print("❌ --input option required (image file path)")
                return 1
            
            print(f"\n📷 Starting image analysis: {args.input}")
            result = analyzer.analyze_single_image(args.input)
            
            if result.get('success', False):
                # 결과 출력
                exercise_info = result['exercise_info']
                emoji = exercise_info.get('emoji', '🏋️')
                name_display = exercise_info.get('name_display', 'unknown')
                exercise_conf = result['exercise_confidence']
                pose_result = result['pose_analysis']
                
                print(f"\n🎉 Image analysis complete!")
                print(f"🤖 AI Detection: {emoji} {name_display} (Confidence: {exercise_conf:.1%})")
                
                if pose_result['valid']:
                    pose_quality = pose_result['classification']
                    pose_conf = pose_result['confidence']
                    
                    status_emoji = "✅" if pose_quality == 'good' else "⚠️"
                    print(f"🎯 Form Analysis: {status_emoji} {pose_quality.upper()} (Score: {pose_conf:.1%})")
                    
                    # 피드백 메시지 출력
                    feedback_messages = result['feedback_messages']
                    if feedback_messages:
                        print(f"\n💬 Detailed Feedback:")
                        for i, message in enumerate(feedback_messages[:5], 1):
                            print(f"  {i}. {message}")
                    
                    # 위반사항 출력
                    violations = pose_result.get('violations', [])
                    if violations:
                        print(f"\n📐 Angle Analysis:")
                        for violation in violations[:3]:
                            joint_en = violation.get('name_en', violation['joint'])
                            angle = violation['angle']
                            range_min, range_max = violation['expected_range']
                            print(f"  • {joint_en}: {angle:.1f}° → Target: {range_min:.0f}-{range_max:.0f}°")
                
                # 주석 이미지 표시
                annotated_image = result['annotated_image']
                
                # 이미지 크기 조정 (화면에 맞게)
                height, width = annotated_image.shape[:2]
                if width > 1200:
                    scale = 1200 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    annotated_image = cv2.resize(annotated_image, (new_width, new_height))
                
                window_title = f"Complete Auto Analysis Result: {emoji} {name_display}"
                cv2.imshow(window_title, annotated_image)
                
                print(f"\n🖼️ Analysis result image displayed... (Press any key to close)")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            else:
                print(f"❌ Image analysis failed: {result.get('error', 'Unknown error')}")
                return 1
                
        elif args.mode == 'video':
            if not args.input:
                print("❌ --input option required (video file path)")
                return 1
            
            print(f"\n🎬 Starting video analysis: {args.input}")
            if args.output:
                print(f"📁 Output path: {args.output}")
            
            result = analyzer.analyze_video_file(args.input, args.output)
            
            if result.get('success', False):
                # 결과 출력
                main_exercise = result['main_exercise']
                exercise_info = analyzer.exercise_info.get(main_exercise, {})
                emoji = exercise_info.get('emoji', '🏋️')
                name_display = exercise_info.get('name_display', main_exercise)
                
                stats = result['stats']
                success_rate = result['success_rate']
                total_analyzed = result['total_frames_analyzed']
                
                print(f"\n🎉 Video analysis complete!")
                print(f"🎯 Main exercise: {emoji} {name_display}")
                print(f"📊 Analysis results:")
                print(f"  • Total analyzed frames: {total_analyzed}")
                print(f"  • ✅ Good form: {stats['good']} frames")
                print(f"  • ❌ Bad form: {stats['bad']} frames")
                print(f"  • 🎯 Success rate: {success_rate:.1f}%")
                
                # 운동 감지 통계
                exercise_detections = result['exercise_detections']
                if len(exercise_detections) > 1:
                    print(f"\n📈 Exercise detection statistics:")
                    for exercise, count in exercise_detections.items():
                        info = analyzer.exercise_info.get(exercise, {})
                        emoji = info.get('emoji', '🏋️')
                        name_display = info.get('name_display', exercise)
                        percentage = (count / sum(exercise_detections.values())) * 100
                        print(f"  • {emoji} {name_display}: {count} frames ({percentage:.1f}%)")
                
                if args.output:
                    print(f"\n💾 Annotated video saved: {args.output}")
                
            else:
                print(f"❌ Video analysis failed: {result.get('error', 'Unknown error')}")
                return 1
    
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted")
        return 0
    except Exception as e:
        print(f"❌ Execution error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())