"""
50:50 비율 목표로 조정된 enhanced_pose_analysis.py
스쿼트, 푸쉬업, 데드리프트, 벤치프레스, 런지 - 각도 허용범위 조정
"""

import cv2
import numpy as np
import mediapipe as mp
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import math
from collections import deque

@dataclass
class ViewSpecificThreshold:
    """뷰별 각도 임계값"""
    min_angle: float
    max_angle: float
    joint_points: List[int]
    name: str
    weight: float = 0.5
    view_types: List[str] = None

class EnhancedExerciseClassifier:
    """향상된 운동 분류기 - 50:50 비율 목표 조정"""
    
    def __init__(self):
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        except Exception as e:
            print(f"❌ MediaPipe 초기화 실패: {e}")
            raise
        
        # 🎯 50:50 비율 목표로 조정된 각도 기준
        self.exercise_thresholds = {
            'squat': {
                'side_view': [  # 🏋️‍♀️ 스쿼트 (86.4% → 50% 목표)
                    ViewSpecificThreshold(60, 130, [23, 25, 27], 'left_knee', 1.2, ['side']),      # 40-160 → 60-130 엄격
                    ViewSpecificThreshold(60, 130, [24, 26, 28], 'right_knee', 1.2, ['side']),     # 무릎 각도 엄격
                    ViewSpecificThreshold(60, 130, [11, 23, 25], 'left_hip', 1.0, ['side']),       # 힙 각도 엄격
                    ViewSpecificThreshold(60, 130, [12, 24, 26], 'right_hip', 1.0, ['side']),      # 가중치 증가
                    ViewSpecificThreshold(155, 180, [11, 23, 25], 'back_straight', 1.3, ['side']), # 등 각도 더 엄격
                    ViewSpecificThreshold(155, 180, [23, 11, 13], 'spine_angle', 1.1, ['side']),   # 척추 각도 추가
                ],
                'front_view': [
                    ViewSpecificThreshold(140, 180, [11, 12, 23], 'shoulder_level', 0.8, ['front']),
                    ViewSpecificThreshold(60, 120, [23, 24, 25], 'hip_symmetry', 0.9, ['front']),
                    ViewSpecificThreshold(150, 180, [25, 27, 29], 'knee_tracking', 0.8, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(140, 180, [11, 12, 23], 'back_alignment', 0.7, ['back']),
                    ViewSpecificThreshold(150, 180, [23, 25, 27], 'spine_straight', 0.9, ['back']),
                ]
            },
            
            'push_up': {
                'side_view': [  # 💪 푸쉬업 (92.2% → 50% 목표)
                    ViewSpecificThreshold(60, 140, [11, 13, 15], 'left_elbow', 1.2, ['side']),      # 20-170 → 60-140 엄격
                    ViewSpecificThreshold(60, 140, [12, 14, 16], 'right_elbow', 1.2, ['side']),     # 팔꿈치 각도 엄격
                    ViewSpecificThreshold(160, 180, [11, 23, 25], 'body_line', 1.5, ['side']),      # 100-180 → 160-180 매우 엄격
                    ViewSpecificThreshold(160, 180, [23, 25, 27], 'leg_straight', 1.0, ['side']),   # 다리 직선 엄격
                    ViewSpecificThreshold(140, 180, [13, 11, 23], 'shoulder_alignment', 0.8, ['side']), # 어깨 정렬
                    ViewSpecificThreshold(160, 180, [11, 12, 23], 'core_stability', 1.2, ['side']),  # 코어 안정성 추가
                ],
                'front_view': [
                    ViewSpecificThreshold(130, 180, [11, 12, 13], 'shoulder_width', 0.6, ['front']),
                    ViewSpecificThreshold(140, 180, [15, 16, 17], 'hand_position', 0.6, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(130, 180, [11, 12, 23], 'back_straight', 0.8, ['back']),
                    ViewSpecificThreshold(140, 180, [23, 24, 25], 'hip_level', 0.6, ['back']),
                ]
            },
            
            'deadlift': {
                'side_view': [  # 🏋️‍♂️ 데드리프트 (100% → 50% 목표)
                    ViewSpecificThreshold(140, 180, [23, 25, 27], 'left_knee', 0.8, ['side']),      # 100-180 → 140-180 엄격
                    ViewSpecificThreshold(140, 180, [24, 26, 28], 'right_knee', 0.8, ['side']),     # 무릎 더 엄격
                    ViewSpecificThreshold(120, 180, [11, 23, 25], 'hip_hinge', 1.0, ['side']),      # 80-180 → 120-180 엄격
                    ViewSpecificThreshold(160, 180, [11, 23, 12], 'back_straight', 1.5, ['side']),  # 120-180 → 160-180 매우 엄격
                    ViewSpecificThreshold(70, 120, [23, 11, 13], 'chest_up', 0.8, ['side']),        # 가슴 들기 엄격
                    ViewSpecificThreshold(160, 180, [23, 11, 24], 'spine_neutral', 1.2, ['side']),  # 척추 중립 추가
                ],
                'front_view': [
                    ViewSpecificThreshold(140, 180, [11, 12, 23], 'shoulder_level', 0.6, ['front']),
                    ViewSpecificThreshold(130, 180, [23, 24, 25], 'hip_symmetry', 0.8, ['front']),
                    ViewSpecificThreshold(140, 180, [25, 26, 27], 'knee_alignment', 0.8, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(160, 180, [11, 23, 24], 'spine_neutral', 1.0, ['back']),
                    ViewSpecificThreshold(130, 180, [23, 25, 26], 'hip_level', 0.6, ['back']),
                ]
            },
            
            'bench_press': {
                'side_view': [  # 🔥 벤치프레스 (100% → 50% 목표)
                    ViewSpecificThreshold(60, 130, [11, 13, 15], 'left_elbow', 1.2, ['side']),      # 20-180 → 60-130 엄격
                    ViewSpecificThreshold(60, 130, [12, 14, 16], 'right_elbow', 1.2, ['side']),     # 팔꿈치 각도 엄격
                    ViewSpecificThreshold(60, 140, [13, 11, 23], 'left_shoulder', 1.0, ['side']),   # 20-170 → 60-140 엄격
                    ViewSpecificThreshold(60, 140, [14, 12, 24], 'right_shoulder', 1.0, ['side']),  # 어깨 각도 엄격
                    ViewSpecificThreshold(140, 180, [11, 23, 25], 'back_arch', 0.8, ['side']),      # 등 아치 엄격
                    ViewSpecificThreshold(70, 120, [13, 15, 17], 'wrist_alignment', 0.6, ['side']), # 손목 정렬 추가
                ],
                'front_view': [
                    ViewSpecificThreshold(130, 180, [11, 12, 13], 'shoulder_symmetry', 0.6, ['front']),
                    ViewSpecificThreshold(130, 180, [13, 14, 15], 'arm_symmetry', 0.6, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(130, 180, [11, 12, 23], 'upper_back', 0.6, ['back']),
                ]
            },
            
            'lunge': {  # 🚀 런지 (새로 추가 - 50% 목표)
                'side_view': [
                    # 앞다리 (전진한 다리) - 무릎 각도 (핵심)
                    ViewSpecificThreshold(80, 110, [23, 25, 27], 'front_knee', 1.5, ['side']),         # 70-130 → 80-110 엄격
                    # 뒷다리 (뒤에 있는 다리) - 무릎 각도
                    ViewSpecificThreshold(150, 180, [24, 26, 28], 'back_knee', 1.2, ['side']),         # 120-180 → 150-180 엄격
                    # 앞다리 엉덩이 각도
                    ViewSpecificThreshold(80, 110, [11, 23, 25], 'front_hip', 1.0, ['side']),          # 70-130 → 80-110 엄격
                    # 상체 직립도 (매우 중요)
                    ViewSpecificThreshold(170, 180, [11, 23, 25], 'torso_upright', 1.5, ['side']),     # 160-180 → 170-180 매우 엄격
                    # 발목 안정성
                    ViewSpecificThreshold(85, 105, [25, 27, 31], 'front_ankle', 1.0, ['side']),        # 80-120 → 85-105 엄격
                    # 뒷다리 엉덩이 신전
                    ViewSpecificThreshold(160, 180, [12, 24, 26], 'back_hip_extension', 1.2, ['side']), # 140-180 → 160-180 엄격
                    # 무릎-발끝 정렬
                    ViewSpecificThreshold(170, 180, [23, 25, 27], 'knee_over_ankle', 1.3, ['side']),   # 새로 추가
                ],
                'front_view': [
                    # 무릎 추적 (앞다리) - 매우 중요
                    ViewSpecificThreshold(170, 180, [23, 25, 27], 'knee_tracking', 1.2, ['front']),    # 160-180 → 170-180 엄격
                    # 골반 수평 유지
                    ViewSpecificThreshold(175, 180, [23, 24, 11], 'pelvis_level', 1.0, ['front']),     # 170-180 → 175-180 엄격
                    # 어깨 수평
                    ViewSpecificThreshold(175, 180, [11, 12, 23], 'shoulder_level', 0.8, ['front']),   # 어깨 수평 엄격
                    # 발 너비 (스탠스)
                    ViewSpecificThreshold(170, 180, [27, 28, 31], 'foot_stance', 0.6, ['front']),      # 발 위치 엄격
                ],
                'back_view': [
                    # 척추 정렬
                    ViewSpecificThreshold(170, 180, [11, 23, 25], 'spine_alignment', 1.0, ['back']),   # 160-180 → 170-180 엄격
                    # 어깨 안정성
                    ViewSpecificThreshold(175, 180, [11, 12, 23], 'shoulder_stability', 0.8, ['back']), # 어깨 안정 엄격
                    # 골반 안정성
                    ViewSpecificThreshold(175, 180, [23, 24, 25], 'pelvis_stability', 0.9, ['back']),  # 골반 안정 엄격
                ]
            }
        }
    
    def detect_view_type(self, landmarks: List[Dict]) -> str:
        """촬영 각도/뷰 타입 감지"""
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            hip_width = abs(left_hip['x'] - right_hip['x'])
            
            nose = landmarks[0]
            body_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            
            if shoulder_width < 0.2 and hip_width < 0.2:
                return 'side_view'
            elif shoulder_width > 0.2 and hip_width > 0.15:
                if abs(nose['x'] - body_center_x) < 0.15:
                    return 'front_view'
                else:
                    return 'back_view'
            else:
                return 'side_view'
                
        except Exception as e:
            print(f"뷰 타입 감지 오류: {e}")
            return 'side_view'
    
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
            
        except Exception as e:
            return 180.0
    
    def extract_landmarks(self, image_path: str) -> Optional[Dict]:
        """이미지에서 랜드마크 추출"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                return {
                    'landmarks': landmarks,
                    'image_shape': image.shape
                }
            return None
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def analyze_pose(self, landmarks: List[Dict], exercise_type: str) -> Dict:
        """향상된 자세 분석 - 50:50 비율 목표"""
        if exercise_type not in self.exercise_thresholds:
            return {'valid': False, 'error': f'Unknown exercise type: {exercise_type}'}
        
        view_type = self.detect_view_type(landmarks)
        all_thresholds = self.exercise_thresholds[exercise_type]
        current_thresholds = all_thresholds.get(view_type, [])
        
        if not current_thresholds:
            current_thresholds = all_thresholds.get('side_view', [])
        
        angles = {}
        violations = []
        weighted_violation_score = 0.0
        total_weight = 0.0
        
        for threshold in current_thresholds:
            try:
                p1_idx, p2_idx, p3_idx = threshold.joint_points
                
                if any(idx >= len(landmarks) for idx in [p1_idx, p2_idx, p3_idx]):
                    continue
                
                # 가시성 확인 (조금 더 엄격)
                visibility_threshold = 0.3  # 0.15에서 0.3으로 상향
                if (landmarks[p1_idx]['visibility'] < visibility_threshold or 
                    landmarks[p2_idx]['visibility'] < visibility_threshold or 
                    landmarks[p3_idx]['visibility'] < visibility_threshold):
                    continue
                
                p1 = (landmarks[p1_idx]['x'], landmarks[p1_idx]['y'])
                p2 = (landmarks[p2_idx]['x'], landmarks[p2_idx]['y'])
                p3 = (landmarks[p3_idx]['x'], landmarks[p3_idx]['y'])
                
                angle = self.calculate_angle(p1, p2, p3)
                angles[threshold.name] = angle
                
                if not (threshold.min_angle <= angle <= threshold.max_angle):
                    violations.append({
                        'joint': threshold.name,
                        'angle': angle,
                        'expected_min': threshold.min_angle,
                        'expected_max': threshold.max_angle,
                        'weight': threshold.weight,
                        'view_type': view_type
                    })
                    weighted_violation_score += threshold.weight
                
                total_weight += threshold.weight
                    
            except Exception as e:
                print(f"Error calculating angle for {threshold.name}: {e}")
                continue
        
        # 🎯 50:50 목표로 조정된 분류 기준 (더 엄격)
        violation_ratio = weighted_violation_score / max(total_weight, 1.0)
        
        # 운동별 엄격한 분류 기준 (50:50 목표)
        classification_thresholds = {
            'squat': 0.3,        # 0.5 → 0.3으로 엄격
            'push_up': 0.25,     # 0.8 → 0.25로 매우 엄격
            'deadlift': 0.2,     # 0.7 → 0.2로 매우 엄격
            'bench_press': 0.25, # 0.8 → 0.25로 매우 엄격
            'lunge': 0.3,        # 새로운 런지: 적당히 엄격
        }
        
        threshold = classification_thresholds.get(exercise_type, 0.3)
        is_good = violation_ratio < threshold
        
        return {
            'valid': True,
            'classification': 'good' if is_good else 'bad',
            'angles': angles,
            'violations': violations,
            'violation_count': len(violations),
            'weighted_violation_ratio': violation_ratio,
            'view_type': view_type,
            'confidence': 1.0 - violation_ratio,
            'classification_threshold': threshold,
            'target_ratio': '50:50 balanced'
        }

class AdaptivePostProcessor:
    """적응형 후처리 클래스 - 50:50 비율 목표"""
    
    def __init__(self, hysteresis_threshold: float = 0.3, ema_alpha: float = 0.4):
        self.hysteresis_threshold = hysteresis_threshold
        self.ema_alpha = ema_alpha
        self.history = deque(maxlen=20)
        self.ema_value = None
        self.last_state = 'good'
        self.state_counter = {'good': 0, 'bad': 0}
        
        # 🎯 50:50 목표로 조정된 히스테리시스 (더 엄격)
        self.exercise_hysteresis = {
            'squat': 0.3,        # 0.5 → 0.3으로 엄격
            'push_up': 0.25,     # 0.8 → 0.25로 매우 엄격
            'deadlift': 0.2,     # 0.6 → 0.2로 매우 엄격
            'bench_press': 0.25, # 0.7 → 0.25로 매우 엄격
            'lunge': 0.3,        # 새로운 런지: 적당히 엄격
        }
    
    def apply_ema(self, current_value: float) -> float:
        """지수 이동 평균 적용"""
        if self.ema_value is None:
            self.ema_value = current_value
        else:
            self.ema_value = self.ema_alpha * current_value + (1 - self.ema_alpha) * self.ema_value
        return self.ema_value
    
    def apply_hysteresis(self, violation_ratio: float, exercise_type: str = None) -> str:
        """운동별 50:50 목표 히스테리시스 적용"""
        threshold = self.exercise_hysteresis.get(exercise_type, self.hysteresis_threshold)
        
        if self.last_state == 'good':
            # Good에서 Bad로 전환: 더 쉽게 전환 (50:50 위해)
            if violation_ratio > threshold:
                self.last_state = 'bad'
        else:
            # Bad에서 Good으로 복귀: 더 어렵게 복귀 (50:50 위해)
            recovery_thresholds = {
                'squat': threshold * 0.7,        # 복귀 더 어렵게
                'push_up': threshold * 0.6,      # 복귀 매우 어렵게
                'deadlift': threshold * 0.5,     # 복귀 매우 어렵게
                'bench_press': threshold * 0.6,  # 복귀 매우 어렵게
                'lunge': threshold * 0.7,        # 런지: 적당히 어렵게
            }
            
            recovery_threshold = recovery_thresholds.get(exercise_type, threshold * 0.5)
            
            if violation_ratio < recovery_threshold:
                self.last_state = 'good'
        
        return self.last_state
    
    def process(self, analysis_result: Dict, exercise_type: str = None) -> Dict:
        """50:50 목표 후처리 적용"""
        if not analysis_result['valid']:
            return analysis_result
        
        violation_ratio = analysis_result.get('weighted_violation_ratio', 0)
        smoothed_ratio = self.apply_ema(violation_ratio)
        self.history.append(smoothed_ratio)
        
        final_classification = self.apply_hysteresis(smoothed_ratio, exercise_type)
        self.state_counter[final_classification] += 1
        
        view_type = analysis_result.get('view_type', 'side_view')
        confidence_modifier = {
            'side_view': 1.0,
            'front_view': 0.9,
            'back_view': 0.8
        }
        
        adjusted_confidence = (1.0 - smoothed_ratio) * confidence_modifier.get(view_type, 0.8)
        
        return {
            **analysis_result,
            'final_classification': final_classification,
            'violation_ratio': violation_ratio,
            'smoothed_ratio': smoothed_ratio,
            'confidence': adjusted_confidence,
            'processing_info': {
                'exercise_type': exercise_type,
                'view_type': view_type,
                'hysteresis_threshold': self.exercise_hysteresis.get(exercise_type, self.hysteresis_threshold),
                'state_history': list(self.history)[-5:],
                'target_ratio': '50:50 balanced',
                'strictness_level': 'high'
            }
        }

class EnhancedDatasetProcessor:
    """향상된 데이터셋 처리 클래스 - 50:50 비율 목표"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.classifier = EnhancedExerciseClassifier()
        
        # 50:50 목표로 조정된 후처리기
        self.post_processors = {
            'squat': AdaptivePostProcessor(hysteresis_threshold=0.3, ema_alpha=0.4),      # 엄격
            'push_up': AdaptivePostProcessor(hysteresis_threshold=0.25, ema_alpha=0.4),   # 매우 엄격
            'deadlift': AdaptivePostProcessor(hysteresis_threshold=0.2, ema_alpha=0.4),   # 매우 엄격
            'bench_press': AdaptivePostProcessor(hysteresis_threshold=0.25, ema_alpha=0.4), # 매우 엄격
            'lunge': AdaptivePostProcessor(hysteresis_threshold=0.3, ema_alpha=0.4)       # 적당히 엄격
        }
        
        self.output_path = self.base_path / "processed_data"
        self.output_path.mkdir(exist_ok=True)
        
        self.exercises = ['squat', 'push_up', 'deadlift', 'bench_press', 'lunge']
        for exercise in self.exercises:
            for category in ['good', 'bad']:
                (self.output_path / exercise / category).mkdir(parents=True, exist_ok=True)
    
    def process_exercise_images(self, exercise_name: str, image_dir: str, limit: int = 500):
        """50:50 비율 목표 이미지 처리"""
        print(f"\n=== {exercise_name} 처리 (50:50 비율 목표) ===")
        
        image_path = self.base_path / "data" / "training_images" / image_dir
        if not image_path.exists():
            print(f"Directory not found: {image_path}")
            return {'good': 0, 'bad': 0, 'failed': 0}
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(image_path.glob(ext)))
        
        print(f"Found {len(image_files)} images")
        
        if len(image_files) > limit:
            image_files = image_files[:limit]
            print(f"Limited to {limit} images")
        
        results = {'good': 0, 'bad': 0, 'failed': 0}
        processing_log = []
        view_type_count = {'side_view': 0, 'front_view': 0, 'back_view': 0}
        
        post_processor = self.post_processors.get(exercise_name, 
                                                 AdaptivePostProcessor(hysteresis_threshold=0.3))
        
        for i, img_file in enumerate(image_files):
            try:
                landmarks_data = self.classifier.extract_landmarks(str(img_file))
                if landmarks_data is None:
                    results['failed'] += 1
                    continue
                
                analysis = self.classifier.analyze_pose(
                    landmarks_data['landmarks'], 
                    exercise_name
                )
                
                if not analysis['valid']:
                    results['failed'] += 1
                    continue
                
                final_result = post_processor.process(analysis, exercise_name)
                classification = final_result['final_classification']
                view_type = final_result.get('view_type', 'unknown')
                
                if view_type in view_type_count:
                    view_type_count[view_type] += 1
                
                dest_dir = self.output_path / exercise_name / classification
                dest_file = dest_dir / f"{classification}_{exercise_name}_{view_type}_{i:04d}_balanced.jpg"
                
                import shutil
                shutil.copy2(img_file, dest_file)
                
                results[classification] += 1
                
                log_entry = {
                    'original_file': str(img_file),
                    'classification': classification,
                    'view_type': view_type,
                    'angles': final_result['angles'],
                    'violations': final_result['violations'],
                    'confidence': final_result['confidence'],
                    'weighted_violation_ratio': final_result.get('weighted_violation_ratio', 0),
                    'target_ratio': '50:50 balanced'
                }
                processing_log.append(log_entry)
                
                if (i + 1) % 50 == 0:
                    total_processed = results['good'] + results['bad']
                    good_rate = (results['good'] / max(total_processed, 1)) * 100
                    print(f"  📊 진행률: {i + 1}/{len(image_files)} images")
                    print(f"     현재 Good 비율: {good_rate:.1f}% (목표: 50%)")
                    
                    # 실시간 50:50 목표 달성 여부 체크
                    if 45 <= good_rate <= 55:
                        print(f"     ✅ 50:50 목표 범위 달성! (45-55%)")
                    elif good_rate > 55:
                        print(f"     ⚠️ Good 비율이 높음 - 더 엄격하게 조정 중")
                    else:
                        print(f"     ⚠️ Good 비율이 낮음 - 조금 완화 필요할 수 있음")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                results['failed'] += 1
                continue
        
        # 결과 출력
        total_processed = results['good'] + results['bad']
        good_rate = (results['good'] / max(total_processed, 1)) * 100
        
        print(f"\n📈 {exercise_name.upper()} 최종 결과 (50:50 목표):")
        print(f"  🎯 Good: {results['good']}장 ({good_rate:.1f}%)")
        print(f"  ❌ Bad: {results['bad']}장 ({100-good_rate:.1f}%)")
        print(f"  💥 Failed: {results['failed']}장")
        
        # 50:50 목표 달성 여부 확인
        if 45 <= good_rate <= 55:
            print(f"  🎉 50:50 목표 달성! (45-55% 범위) ✅")
            status = "목표 달성"
        elif good_rate > 70:
            print(f"  ⚠️ Good 비율 과다: {good_rate:.1f}% > 70% (더 엄격한 조정 필요)")
            status = "과도하게 관대함"
        elif good_rate > 55:
            print(f"  ⚠️ Good 비율 높음: {good_rate:.1f}% > 55% (조금 더 엄격하게)")
            status = "약간 관대함"
        elif good_rate < 30:
            print(f"  ⚠️ Good 비율 과소: {good_rate:.1f}% < 30% (너무 엄격함)")
            status = "과도하게 엄격함"
        elif good_rate < 45:
            print(f"  ⚠️ Good 비율 낮음: {good_rate:.1f}% < 45% (조금 더 관대하게)")
            status = "약간 엄격함"
        else:
            status = "목표 달성"
        
        # 뷰 분포
        print(f"  📷 뷰 분포:")
        for view, count in view_type_count.items():
            percentage = (count / max(total_processed, 1)) * 100
            print(f"     {view}: {count}장 ({percentage:.1f}%)")
        
        # 조정 제안
        print(f"  🔧 다음 조정 제안:")
        if good_rate > 60:
            print(f"     - 각도 허용범위 더 축소")
            print(f"     - 가중치 증가")
            print(f"     - 히스테리시스 임계값 낮춤")
        elif good_rate < 40:
            print(f"     - 각도 허용범위 조금 확대")
            print(f"     - 가중치 조정")
            print(f"     - 히스테리시스 임계값 상향")
        else:
            print(f"     - 현재 설정 적절함")
        
        # 로그 저장
        log_file = self.output_path / f"{exercise_name}_50_50_balanced_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'exercise': exercise_name,
                'version': '50_50_balanced',
                'summary': results,
                'good_rate': good_rate,
                'target_achievement': {
                    'target_rate': '50%',
                    'tolerance_range': '45-55%',
                    'achieved': 45 <= good_rate <= 55,
                    'status': status
                },
                'view_distribution': view_type_count,
                'detailed_log': processing_log
            }, f, indent=2, ensure_ascii=False)
        
        return results

# 메인 실행부
if __name__ == "__main__":
    print("🎯 50:50 비율 목표 Enhanced Pose Analysis")
    print("=" * 80)
    print("📋 각도 조정 사항 (Good 비율 낮추기):")
    print("  🏋️‍♀️ 스쿼트: 86.4% → 50% 목표 (각도 범위 축소)")
    print("  💪 푸쉬업: 92.2% → 50% 목표 (몸 일직선 매우 엄격)")
    print("  🏋️‍♂️ 데드리프트: 100% → 50% 목표 (등 각도 매우 엄격)")
    print("  🔥 벤치프레스: 100% → 50% 목표 (팔꿈치 각도 엄격)")
    print("  🚀 런지: 새로 추가 (50% 목표로 설정)")
    print("=" * 80)
    
    processor = EnhancedDatasetProcessor(".")
    
    exercises = {
        'squat': 'squat_exercise',
        'push_up': 'push_up_exercise', 
        'deadlift': 'deadlift_exercise',
        'bench_press': 'bench_press_exercise',
        'lunge': 'lunge_exercise'
    }
    
    total_results = {}
    for exercise, directory in exercises.items():
        print(f"\n{'='*20} {exercise.upper()} 50:50 목표 처리 {'='*20}")
        results = processor.process_exercise_images(exercise, directory, limit=500)
        total_results[exercise] = results
    
    # 전체 결과 저장
    summary_file = processor.output_path / "50_50_balanced_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'version': '50_50_balanced',
            'target_description': 'Adjusted all exercises to achieve 50:50 good:bad ratio',
            'angle_adjustments': {
                'squat': {
                    'before': {'knee': '40-160°', 'hip': '40-160°'},
                    'after': {'knee': '60-130°', 'hip': '60-130°'},
                    'change': 'significantly stricter'
                },
                'push_up': {
                    'before': {'elbow': '20-170°', 'body_line': '100-180°'},
                    'after': {'elbow': '60-140°', 'body_line': '160-180°'},
                    'change': 'very strict body alignment'
                },
                'deadlift': {
                    'before': {'back': '120-180°', 'knee': '100-180°'},
                    'after': {'back': '160-180°', 'knee': '140-180°'},
                    'change': 'extremely strict back angle'
                },
                'bench_press': {
                    'before': {'elbow': '20-180°', 'shoulder': '20-170°'},
                    'after': {'elbow': '60-130°', 'shoulder': '60-140°'},
                    'change': 'much stricter arm positioning'
                },
                'lunge': {
                    'new_exercise': True,
                    'front_knee': '80-110°',
                    'torso': '170-180°',
                    'change': 'designed for 50:50 ratio'
                }
            },
            'results': total_results
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("🎯 50:50 비율 목표 처리 완료!")
    print("="*80)
    print("📊 최종 결과 요약:")
    
    for exercise, results in total_results.items():
        total = sum(results.values())
        if total > 0:
            good_rate = (results['good'] / max(results['good'] + results['bad'], 1)) * 100
            
            emoji_map = {
                'squat': '🏋️‍♀️',
                'push_up': '💪',
                'deadlift': '🏋️‍♂️',
                'bench_press': '🔥',
                'lunge': '🚀'
            }
            
            # 50:50 목표 달성 여부
            if 45 <= good_rate <= 55:
                status = "🎯 목표 달성"
                color = "✅"
            elif good_rate > 60:
                status = "⚠️ 너무 관대"
                color = "🔴"
            elif good_rate < 40:
                status = "⚠️ 너무 엄격"
                color = "🔴"
            else:
                status = "📊 근접함"
                color = "🟡"
            
            print(f"\n{emoji_map.get(exercise, '🏋️')} {exercise.upper()}:")
            print(f"  총 처리: {total}장")
            print(f"  Good: {results['good']}장 ({good_rate:.1f}%)")
            print(f"  Bad: {results['bad']}장 ({100-good_rate:.1f}%)")
            print(f"  목표: 50% | 결과: {color} {status}")
        else:
            print(f"\n⚠️ {exercise.upper()}: 데이터 없음 - 해당 폴더에 이미지를 추가하세요")
    
    print(f"\n🔧 주요 조정사항 (50:50 비율 달성용):")
    print(f"  📐 스쿼트: 무릎/힙 40-160° → 60-130° (엄격)")
    print(f"  📐 푸쉬업: 몸라인 100-180° → 160-180° (매우 엄격)")
    print(f"  📐 데드리프트: 등각도 120-180° → 160-180° (극도로 엄격)")
    print(f"  📐 벤치프레스: 팔꿈치 20-180° → 60-130° (엄격)")
    print(f"  📐 런지: 앞무릎 80-110°, 상체 170-180° (균형잡힌 엄격함)")
    print(f"  ⚖️ 히스테리시스: 모든 운동 0.2-0.3으로 엄격 설정")
    
    print(f"\n💡 다음 단계:")
    print(f"  1. 결과 확인 후 필요시 각도 미세조정")
    print(f"  2. 배드 사진 추가로 데이터 불균형 해결")
    print(f"  3. AI 모델 재훈련: python main.py --mode train")
    print(f"  4. 실시간 분석으로 50:50 비율 검증")
    
    print(f"\n💾 처리된 데이터 위치: {processor.output_path}")
    print("✅ 50:50 비율 달성을 위한 엄격한 기준 적용 완료!")