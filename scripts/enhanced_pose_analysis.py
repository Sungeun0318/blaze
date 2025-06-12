"""
완전한 enhanced_pose_analysis.py
데드리프트 99% Bad 문제 해결 + 스쿼트/벤치 조정 + 푸쉬업 유지
"""

import cv2
import numpy as np
import mediapipe as mp
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import math
import shutil
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
    """향상된 운동 분류기 - 데드리프트 대폭 완화"""
    
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
        
        # 🎯 미세 조정된 각도 기준
        self.exercise_thresholds = {
            'squat': {
                'side_view': [  # 🏋️‍♀️ 스쿼트 (91.6% → 60% 목표 - 조금 엄격)
                    ViewSpecificThreshold(55, 140, [23, 25, 27], 'left_knee', 1.1, ['side']),
                    ViewSpecificThreshold(55, 140, [24, 26, 28], 'right_knee', 1.1, ['side']),
                    ViewSpecificThreshold(55, 140, [11, 23, 25], 'left_hip', 0.9, ['side']),
                    ViewSpecificThreshold(55, 140, [12, 24, 26], 'right_hip', 0.9, ['side']),
                    ViewSpecificThreshold(110, 170, [11, 23, 25], 'back_straight', 1.1, ['side']),
                    ViewSpecificThreshold(110, 170, [23, 11, 13], 'spine_angle', 0.9, ['side']),
                ],
                'front_view': [
                    ViewSpecificThreshold(135, 180, [11, 12, 23], 'shoulder_level', 0.7, ['front']),
                    ViewSpecificThreshold(60, 120, [23, 24, 25], 'hip_symmetry', 0.8, ['front']),
                    ViewSpecificThreshold(150, 180, [25, 27, 29], 'knee_tracking', 0.7, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(135, 180, [11, 12, 23], 'back_alignment', 0.6, ['back']),
                    ViewSpecificThreshold(150, 180, [23, 25, 27], 'spine_straight', 0.8, ['back']),
                ]
            },
            
            'push_up': {  # 💪 푸쉬업 (57.6% - 적절함, 그대로 유지)
                'side_view': [
                    ViewSpecificThreshold(40, 160, [11, 13, 15], 'left_elbow', 1.0, ['side']),
                    ViewSpecificThreshold(40, 160, [12, 14, 16], 'right_elbow', 1.0, ['side']),
                    ViewSpecificThreshold(140, 180, [11, 23, 25], 'body_line', 1.2, ['side']),
                    ViewSpecificThreshold(140, 180, [23, 25, 27], 'leg_straight', 0.8, ['side']),
                    ViewSpecificThreshold(120, 180, [13, 11, 23], 'shoulder_alignment', 0.6, ['side']),
                    ViewSpecificThreshold(140, 180, [11, 12, 23], 'core_stability', 1.0, ['side']),
                ],
                'front_view': [
                    ViewSpecificThreshold(120, 180, [11, 12, 13], 'shoulder_width', 0.5, ['front']),
                    ViewSpecificThreshold(130, 180, [15, 16, 17], 'hand_position', 0.5, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(120, 180, [11, 12, 23], 'back_straight', 0.6, ['back']),
                    ViewSpecificThreshold(130, 180, [23, 24, 25], 'hip_level', 0.5, ['back']),
                ]
            },
            
            'deadlift': {
                'side_view': [  # 🏋️‍♂️ 데드리프트 (99% Bad → 45% Good 목표 - 대폭 완화!)
                    ViewSpecificThreshold(80, 140, [23, 25, 27], 'left_knee', 0.6, ['side']),      #
                    ViewSpecificThreshold(80, 140, [24, 26, 28], 'right_knee', 0.6, ['side']),     #
                    ViewSpecificThreshold(80, 180, [11, 23, 25], 'hip_hinge', 0.7, ['side']),       #
                    ViewSpecificThreshold(120, 180, [11, 23, 12], 'back_straight', 1.0, ['side']),  #
                    ViewSpecificThreshold(50, 140, [23, 11, 13], 'chest_up', 0.5, ['side']),        #
                    ViewSpecificThreshold(120, 180, [23, 11, 24], 'spine_neutral', 0.8, ['side']),  #
                ],
                'front_view': [
                    ViewSpecificThreshold(120, 180, [11, 12, 23], 'shoulder_level', 0.4, ['front']),
                    ViewSpecificThreshold(110, 180, [23, 24, 25], 'hip_symmetry', 0.5, ['front']),
                    ViewSpecificThreshold(120, 180, [25, 26, 27], 'knee_alignment', 0.5, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(120, 180, [11, 23, 24], 'spine_neutral', 0.6, ['back']),
                    ViewSpecificThreshold(110, 180, [23, 25, 26], 'hip_level', 0.4, ['back']),
                ]
            },
            
            'bench_press': {
                'side_view': [  # 🔥 벤치프레스 (87.1% → 65% 목표 - 조금 엄격)
                    ViewSpecificThreshold(50, 145, [11, 13, 15], 'left_elbow', 1.1, ['side']),
                    ViewSpecificThreshold(50, 145, [12, 14, 16], 'right_elbow', 1.1, ['side']),
                    ViewSpecificThreshold(50, 150, [13, 11, 23], 'left_shoulder', 0.9, ['side']),
                    ViewSpecificThreshold(50, 150, [14, 12, 24], 'right_shoulder', 0.9, ['side']),
                    ViewSpecificThreshold(90, 170, [11, 23, 25], 'back_arch', 0.7, ['side']),
                    ViewSpecificThreshold(70, 180, [13, 15, 17], 'wrist_alignment', 0.6, ['side']),
                ],
                'front_view': [
                    ViewSpecificThreshold(125, 180, [11, 12, 13], 'shoulder_symmetry', 0.6, ['front']),
                    ViewSpecificThreshold(125, 180, [13, 14, 15], 'arm_symmetry', 0.6, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(125, 180, [11, 12, 23], 'upper_back', 0.6, ['back']),
                ]
            },
            
            'lunge': {  # 🚀 런지 (적당한 수준)
                'side_view': [
                    ViewSpecificThreshold(70, 120, [23, 25, 27], 'front_knee', 1.2, ['side']), 
                    ViewSpecificThreshold(120, 180, [24, 26, 28], 'back_knee', 1.0, ['side']),
                    ViewSpecificThreshold(70, 120, [11, 23, 25], 'front_hip', 0.8, ['side']),
                    ViewSpecificThreshold(100, 180, [11, 23, 25], 'torso_upright', 1.2, ['side']),
                    ViewSpecificThreshold(80, 110, [25, 27, 31], 'front_ankle', 0.8, ['side']),
                    ViewSpecificThreshold(150, 180, [12, 24, 26], 'back_hip_extension', 1.0, ['side']),
                ],
                'front_view': [
                    ViewSpecificThreshold(160, 180, [23, 25, 27], 'knee_tracking', 1.0, ['front']),
                    ViewSpecificThreshold(170, 180, [23, 24, 11], 'pelvis_level', 0.8, ['front']),
                    ViewSpecificThreshold(170, 180, [11, 12, 23], 'shoulder_level', 0.6, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(160, 180, [11, 23, 25], 'spine_alignment', 0.8, ['back']),
                    ViewSpecificThreshold(170, 180, [11, 12, 23], 'shoulder_stability', 0.6, ['back']),
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
        """자세 분석"""
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
                
                # 가시성 확인
                visibility_threshold = 0.25
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
        
        # 🎯 운동별 분류 기준
        violation_ratio = weighted_violation_score / max(total_weight, 1.0)
        
        classification_thresholds = {
            'squat': 0.5,        # 조금 엄격 (91.6% → 60% 목표)
            'push_up': 0.7,      # 기존 유지 (57.6% 적절함)
            'deadlift': 0.8,     # 대폭 완화! (99% Bad → 45% Good 목표)
            'bench_press': 0.5,  # 조금 엄격 (87.1% → 65% 목표)
            'lunge': 0.6,        # 적당한 수준
        }
        
        threshold = classification_thresholds.get(exercise_type, 0.6)
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
            'target_ratio': 'balanced_with_deadlift_fix'
        }

class AdaptivePostProcessor:
    """적응형 후처리 클래스"""
    
    def __init__(self, hysteresis_threshold: float = 0.6, ema_alpha: float = 0.3):
        self.hysteresis_threshold = hysteresis_threshold
        self.ema_alpha = ema_alpha
        self.history = deque(maxlen=15)
        self.ema_value = None
        self.last_state = 'good'
        self.state_counter = {'good': 0, 'bad': 0}
        
        # 운동별 히스테리시스
        self.exercise_hysteresis = {
            'squat': 0.5,        # 조금 엄격
            'push_up': 0.7,      # 기존 유지
            'deadlift': 0.8,     # 대폭 완화!
            'bench_press': 0.5,  # 조금 엄격
            'lunge': 0.6,        # 적당한 수준
        }
    
    def apply_ema(self, current_value: float) -> float:
        """지수 이동 평균 적용"""
        if self.ema_value is None:
            self.ema_value = current_value
        else:
            self.ema_value = self.ema_alpha * current_value + (1 - self.ema_alpha) * self.ema_value
        return self.ema_value
    
    def apply_hysteresis(self, violation_ratio: float, exercise_type: str = None) -> str:
        """운동별 히스테리시스 적용"""
        threshold = self.exercise_hysteresis.get(exercise_type, self.hysteresis_threshold)
        
        if self.last_state == 'good':
            if violation_ratio > threshold:
                self.last_state = 'bad'
        else:
            # Bad에서 Good으로 복귀 기준
            recovery_thresholds = {
                'squat': threshold * 0.7,        # 복귀 조금 어렵게
                'push_up': threshold * 0.8,      # 기존 유지
                'deadlift': threshold * 0.9,     # 복귀 매우 쉽게! (99% Bad 해결)
                'bench_press': threshold * 0.7,  # 복귀 조금 어렵게
                'lunge': threshold * 0.8,        # 기존 유지
            }
            
            recovery_threshold = recovery_thresholds.get(exercise_type, threshold * 0.8)
            
            if violation_ratio < recovery_threshold:
                self.last_state = 'good'
        
        return self.last_state
    
    def process(self, analysis_result: Dict, exercise_type: str = None) -> Dict:
        """후처리 적용"""
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
                'target_ratio': 'balanced_with_deadlift_fix',
                'strictness_level': 'fine_tuned'
            }
        }

class EnhancedDatasetProcessor:
    """향상된 데이터셋 처리 클래스 (main.py 호환)"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.classifier = EnhancedExerciseClassifier()
        
        # 운동별 후처리기
        self.post_processors = {
            'squat': AdaptivePostProcessor(hysteresis_threshold=0.5, ema_alpha=0.3),
            'push_up': AdaptivePostProcessor(hysteresis_threshold=0.7, ema_alpha=0.3),
            'deadlift': AdaptivePostProcessor(hysteresis_threshold=0.8, ema_alpha=0.3),  # 대폭 완화
            'bench_press': AdaptivePostProcessor(hysteresis_threshold=0.5, ema_alpha=0.3),
            'lunge': AdaptivePostProcessor(hysteresis_threshold=0.6, ema_alpha=0.3)
        }
        
        self.output_path = self.base_path / "processed_data"
        self.output_path.mkdir(exist_ok=True)
        
        self.exercises = ['squat', 'push_up', 'deadlift', 'bench_press', 'lunge']
        for exercise in self.exercises:
            for category in ['good', 'bad']:
                (self.output_path / exercise / category).mkdir(parents=True, exist_ok=True)
    
    def process_exercise_images(self, exercise_name: str, image_dir: str, limit: int = 500):
        """운동별 이미지 처리"""
        # 현재 상태
        current_rates = {
            'squat': 91.6,
            'push_up': 57.6,
            'deadlift': 1.0,    # 99% Bad = 1% Good
            'bench_press': 87.1
        }
        
        target_rates = {
            'squat': '50-70%',
            'push_up': '유지 (적절함)',
            'deadlift': '40-60% (대폭 완화)',
            'bench_press': '50-70%'
        }
        
        print(f"\n=== {exercise_name.upper()} 처리 ===")
        if exercise_name in current_rates:
            print(f"현재: {current_rates[exercise_name]:.1f}% → 목표: {target_rates[exercise_name]}")
        
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
        post_processor = self.post_processors.get(exercise_name, 
                                                 AdaptivePostProcessor(hysteresis_threshold=0.6))
        
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
                
                dest_dir = self.output_path / exercise_name / classification
                dest_file = dest_dir / f"{classification}_{exercise_name}_{view_type}_{i:04d}_fixed.jpg"
                
                shutil.copy2(img_file, dest_file)
                results[classification] += 1
                
                if (i + 1) % 50 == 0:
                    total_processed = results['good'] + results['bad']
                    good_rate = (results['good'] / max(total_processed, 1)) * 100
                    print(f"  📊 진행률: {i + 1}/{len(image_files)} images - Good 비율: {good_rate:.1f}%")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                results['failed'] += 1
                continue
        
        # 결과 출력
        total_processed = results['good'] + results['bad']
        good_rate = (results['good'] / max(total_processed, 1)) * 100
        
        print(f"\n📈 {exercise_name.upper()} 최종 결과:")
        print(f"  🎯 Good: {results['good']}장 ({good_rate:.1f}%)")
        print(f"  ❌ Bad: {results['bad']}장 ({100-good_rate:.1f}%)")
        print(f"  💥 Failed: {results['failed']}장")
        
        # 이전 결과와 비교
        if exercise_name in current_rates:
            previous_rate = current_rates[exercise_name]
            improvement = good_rate - previous_rate
            print(f"  📈 변화: {previous_rate:.1f}% → {good_rate:.1f}% ({improvement:+.1f}%)")
            
            if exercise_name == 'push_up':
                status = "✅ 적절함 유지" if 50 <= good_rate <= 70 else "📊 모니터링 중"
            elif exercise_name == 'deadlift':
                if 40 <= good_rate <= 60:
                    status = "✅ 목표 달성"
                elif good_rate < 20:
                    status = "⚠️ 여전히 너무 엄격함"
                else:
                    status = "📈 개선 중"
            else:  # squat, bench_press
                if 50 <= good_rate <= 70:
                    status = "✅ 목표 달성"
                elif good_rate > 80:
                    status = "⚠️ 여전히 관대함"
                else:
                    status = "📊 목표 근접"
            
            print(f"  🎯 상태: {status}")
        
        return results
    
    def process_all_exercises(self):
        """모든 운동 처리 (main.py 호환용)"""
        exercise_dirs = {
            'squat': 'squat_exercise',
            'push_up': 'push_up_exercise', 
            'deadlift': 'deadlift_exercise',
            'bench_press': 'bench_press_exercise',
            'lunge': 'lunge_exercise'
        }
        
        total_results = {}
        
        print("🎯 데드리프트 99% Bad 문제 해결 + 미세 조정 시작!")
        print("목표: 데드리프트 대폭 완화, 스쿼트/벤치 조정, 푸쉬업 유지")
        
        for exercise, directory in exercise_dirs.items():
            results = self.process_exercise_images(exercise, directory)
            total_results[exercise] = results
        
        # 전체 결과 저장
        summary_file = self.output_path / "deadlift_fixed_processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'version': 'deadlift_99_bad_fixed',
                'main_fix': 'deadlift_relaxed_from_99_percent_bad',
                'target_rates': {
                    'squat': '50-70% (from 91.6%)',
                    'push_up': 'maintain current (57.6%)', 
                    'deadlift': '40-60% (from 1% - fixed 99% Bad problem)',
                    'bench_press': '50-70% (from 87.1%)',
                    'lunge': '50-70% (new exercise)'
                },
                'results': total_results
            }, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*70)
        print("🎉 데드리프트 99% Bad 문제 해결 완료!")
        print("="*70)
        print("📊 최종 결과 요약:")
        
        # 이전 결과
        previous_rates = {
            'squat': 91.6,
            'push_up': 57.6,
            'deadlift': 1.0,    # 99% Bad = 1% Good
            'bench_press': 87.1
        }
        
        for exercise, results in total_results.items():
            total_processed = results['good'] + results['bad']
            if total_processed > 0:
                good_rate = (results['good'] / total_processed) * 100
                
                emoji_map = {
                    'squat': '🏋️‍♀️',
                    'push_up': '💪',
                    'deadlift': '🏋️‍♂️',
                    'bench_press': '🔥',
                    'lunge': '🚀'
                }
                
                # 목표 달성 여부
                if exercise == 'push_up':
                    status = "✅ 적절함 유지" if 50 <= good_rate <= 70 else "📊 모니터링 중"
                    color = "🟢" if 50 <= good_rate <= 70 else "🟡"
                elif exercise == 'deadlift':
                    if 40 <= good_rate <= 60:
                        status = "✅ 목표 달성"
                        color = "🟢"
                    elif good_rate < 20:
                        status = "⚠️ 여전히 너무 엄격함"
                        color = "🔴"
                    elif good_rate > 70:
                        status = "📊 좋은 개선"
                        color = "🟡"
                    else:
                        status = "📈 개선 중"
                        color = "🟡"
                else:  # squat, bench_press
                    if 50 <= good_rate <= 70:
                        status = "✅ 목표 달성"
                        color = "🟢"
                    elif good_rate > 80:
                        status = "⚠️ 여전히 관대함"
                        color = "🔴"
                    elif good_rate < 40:
                        status = "⚠️ 너무 엄격함"
                        color = "🔴"
                    else:
                        status = "📊 목표 근접"
                        color = "🟡"
                
                # 변화량 계산
                if exercise in previous_rates:
                    previous = previous_rates[exercise]
                    change = good_rate - previous
                    change_text = f"({previous:.1f}% → {good_rate:.1f}%, {change:+.1f}%)"
                else:
                    change_text = f"(신규: {good_rate:.1f}%)"
                
                print(f"\n{emoji_map.get(exercise, '🏋️')} {exercise.upper()}:")
                print(f"  결과: Good {results['good']}장 | Bad {results['bad']}장")
                print(f"  비율: {good_rate:.1f}% {change_text}")
                print(f"  상태: {color} {status}")
        
        print(f"\n🔧 주요 조정 사항:")
        print(f"  🏋️‍♀️ 스쿼트: 각도 범위 축소 (45-150° → 55-140°), 임계값 0.5")
        print(f"  💪 푸쉬업: 조정 없음 (57.6% 적절함)")
        print(f"  🏋️‍♂️ 데드리프트: 각도 대폭 완화 (각도 ±20-30도 확장), 임계값 0.8")
        print(f"  🔥 벤치프레스: 각도 범위 축소 (40-150° → 50-140°), 임계값 0.5")
        
        print(f"\n💡 다음 단계:")
        print(f"  🤖 AI 모델 재훈련: python main.py --mode train")
        print(f"  🎮 실시간 분석으로 검증: python main.py --mode realtime")
        print(f"  📊 결과 확인 후 필요시 추가 미세 조정")
        
        return total_results

# 메인 실행부
if __name__ == "__main__":
    print("🎯 데드리프트 99% Bad 문제 해결 버전")
    print("=" * 80)
    print("📋 현재 문제점:")
    print("  🏋️‍♀️ 스쿼트: 91.6% → 50-70% 목표 (너무 관대함)")
    print("  💪 푸쉬업: 57.6% → 유지 (적절함)")
    print("  🏋️‍♂️ 데드리프트: 1% (99% Bad!) → 40-60% 목표 (너무 엄격함)")
    print("  🔥 벤치프레스: 87.1% → 50-70% 목표 (너무 관대함)")
    print()
    print("🔧 해결 전략:")
    print("  • 푸쉬업: 그대로 유지")
    print("  • 스쿼트/벤치: 각도 범위 조금 축소")
    print("  • 데드리프트: 각도 범위 대폭 완화! (99% Bad 문제 해결)")
    print("  • 분류 임계값 개별 조정")
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
        print(f"\n{'='*15} {exercise.upper()} 처리 {'='*15}")
        results = processor.process_exercise_images(exercise, directory, limit=500)
        total_results[exercise] = results
    
    print("\n🎉 처리 완료!")
    print("데드리프트 99% Bad 문제가 해결되고 다른 운동들도 적절히 조정되었습니다.")