"""
스쿼트 80% 목표로 조정된 완전한 enhanced_pose_analysis.py
푸시업은 그대로, 스쿼트만 적당히 엄격하게 조정
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
    """향상된 운동 분류기 - 스쿼트 80% 목표 조정 버전"""
    
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
        
        # 🔥 스쿼트 80% 목표로 조정된 각도 기준
        self.exercise_thresholds = {
            'squat': {
                'side_view': [  # 🏋️‍♀️ 스쿼트 80% 목표 (100% → 80% 조정)
                    ViewSpecificThreshold(40, 160, [23, 25, 27], 'left_knee', 0.8, ['side']),      # 15-175 → 40-160 적당히 제한
                    ViewSpecificThreshold(40, 160, [24, 26, 28], 'right_knee', 0.8, ['side']),     # 무릎 각도 조정
                    ViewSpecificThreshold(40, 160, [11, 23, 25], 'left_hip', 0.8, ['side']),       # 힙 각도도 조정
                    ViewSpecificThreshold(40, 160, [12, 24, 26], 'right_hip', 0.8, ['side']),      # 가중치 0.3→0.8로 증가
                    ViewSpecificThreshold(140, 180, [11, 23, 25], 'back_straight', 0.9, ['side']), # 등 각도 더 엄격 (자세 품질)
                ],
                'front_view': [
                    ViewSpecificThreshold(130, 180, [11, 12, 23], 'shoulder_level', 0.5, ['front']),
                    ViewSpecificThreshold(50, 130, [23, 24, 25], 'hip_symmetry', 0.6, ['front']),
                    ViewSpecificThreshold(140, 180, [25, 27, 29], 'knee_tracking', 0.5, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(130, 180, [11, 12, 23], 'back_alignment', 0.5, ['back']),
                    ViewSpecificThreshold(140, 180, [23, 25, 27], 'spine_straight', 0.7, ['back']),
                ]
            },
            
            'push_up': {
                'side_view': [  # 💪 푸시업은 그대로 유지 (잘 되고 있음)
                    ViewSpecificThreshold(20, 170, [11, 13, 15], 'left_elbow', 0.4, ['side']),      
                    ViewSpecificThreshold(20, 170, [12, 14, 16], 'right_elbow', 0.4, ['side']),     
                    ViewSpecificThreshold(100, 180, [11, 23, 25], 'body_line', 0.5, ['side']),      
                    ViewSpecificThreshold(130, 180, [23, 25, 27], 'leg_straight', 0.2, ['side']),   
                    ViewSpecificThreshold(120, 180, [13, 11, 23], 'shoulder_alignment', 0.2, ['side']),
                ],
                'front_view': [
                    ViewSpecificThreshold(120, 180, [11, 12, 13], 'shoulder_width', 0.2, ['front']),
                    ViewSpecificThreshold(130, 180, [15, 16, 17], 'hand_position', 0.2, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(120, 180, [11, 12, 23], 'back_straight', 0.3, ['back']),
                    ViewSpecificThreshold(130, 180, [23, 24, 25], 'hip_level', 0.2, ['back']),
                ]
            },
            
            'deadlift': {
                'side_view': [  # 🏋️‍♂️ 데드리프트 (그대로 유지)
                    ViewSpecificThreshold(120, 180, [23, 25, 27], 'left_knee', 0.3, ['side']),      
                    ViewSpecificThreshold(120, 180, [24, 26, 28], 'right_knee', 0.3, ['side']),     
                    ViewSpecificThreshold(100, 180, [11, 23, 25], 'hip_hinge', 0.4, ['side']),      
                    ViewSpecificThreshold(140, 180, [11, 23, 12], 'back_straight', 0.6, ['side']),  
                    ViewSpecificThreshold(50, 130, [23, 11, 13], 'chest_up', 0.2, ['side']),       
                ],
                'front_view': [
                    ViewSpecificThreshold(130, 180, [11, 12, 23], 'shoulder_level', 0.2, ['front']),
                    ViewSpecificThreshold(120, 180, [23, 24, 25], 'hip_symmetry', 0.3, ['front']),
                    ViewSpecificThreshold(130, 180, [25, 26, 27], 'knee_alignment', 0.3, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(140, 180, [11, 23, 24], 'spine_neutral', 0.4, ['back']),
                    ViewSpecificThreshold(120, 180, [23, 25, 26], 'hip_level', 0.2, ['back']),
                ]
            },
            
            'bench_press': {
                'side_view': [  # 🔥 벤치프레스 (그대로 유지)
                    ViewSpecificThreshold(30, 170, [11, 13, 15], 'left_elbow', 0.4, ['side']),      
                    ViewSpecificThreshold(30, 170, [12, 14, 16], 'right_elbow', 0.4, ['side']),     
                    ViewSpecificThreshold(30, 150, [13, 11, 23], 'left_shoulder', 0.3, ['side']),   
                    ViewSpecificThreshold(30, 150, [14, 12, 24], 'right_shoulder', 0.3, ['side']),  
                    ViewSpecificThreshold(130, 180, [11, 23, 25], 'back_arch', 0.2, ['side']),      
                ],
                'front_view': [
                    ViewSpecificThreshold(120, 180, [11, 12, 13], 'shoulder_symmetry', 0.2, ['front']),
                    ViewSpecificThreshold(120, 180, [13, 14, 15], 'arm_symmetry', 0.2, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(120, 180, [11, 12, 23], 'upper_back', 0.2, ['back']),
                ]
            },
            
            'pull_up': {
                'side_view': [  # 💯 풀업 (그대로 유지)
                    ViewSpecificThreshold(10, 120, [11, 13, 15], 'left_elbow', 0.4, ['side']),      
                    ViewSpecificThreshold(10, 120, [12, 14, 16], 'right_elbow', 0.4, ['side']),     
                    ViewSpecificThreshold(90, 180, [13, 11, 23], 'left_shoulder', 0.4, ['side']),   
                    ViewSpecificThreshold(90, 180, [14, 12, 24], 'right_shoulder', 0.4, ['side']), 
                    ViewSpecificThreshold(130, 180, [11, 23, 25], 'body_straight', 0.2, ['side']),  
                    ViewSpecificThreshold(120, 180, [23, 25, 27], 'leg_position', 0.1, ['side']),   
                ],
                'front_view': [
                    ViewSpecificThreshold(120, 180, [11, 12, 13], 'shoulder_width', 0.2, ['front']),
                    ViewSpecificThreshold(10, 120, [13, 15, 16], 'grip_symmetry', 0.2, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(120, 180, [11, 12, 23], 'lat_engagement', 0.3, ['back']),
                    ViewSpecificThreshold(130, 180, [23, 24, 25], 'core_stability', 0.2, ['back']),
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
        """향상된 자세 분석 - 스쿼트 80% 목표 조정"""
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
                visibility_threshold = 0.15
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
        
        # 🎯 스쿼트 전용 조정된 분류 기준
        violation_ratio = weighted_violation_score / max(total_weight, 1.0)
        
        if exercise_type == 'squat':
            # 스쿼트: 80% 목표 - 30% 위반까지 허용 (기존 80%에서 30%로 조정)
            is_good = violation_ratio < 0.5  # 50% 위반까지 허용
        else:
            # 다른 운동들: 기존 기준 유지
            is_good = violation_ratio < 0.8
        
        return {
            'valid': True,
            'classification': 'good' if is_good else 'bad',
            'angles': angles,
            'violations': violations,
            'violation_count': len(violations),
            'weighted_violation_ratio': violation_ratio,
            'view_type': view_type,
            'confidence': 1.0 - violation_ratio
        }

class AdaptivePostProcessor:
    """적응형 후처리 클래스 - 스쿼트 80% 목표 조정"""
    
    def __init__(self, hysteresis_threshold: float = 0.4, ema_alpha: float = 0.4):
        self.hysteresis_threshold = hysteresis_threshold
        self.ema_alpha = ema_alpha
        self.history = deque(maxlen=20)
        self.ema_value = None
        self.last_state = 'good'
        self.state_counter = {'good': 0, 'bad': 0}
        
        # 🎯 스쿼트 80% 목표로 조정된 히스테리시스
        self.exercise_hysteresis = {
            'squat': 0.5,        # 0.8 → 0.5로 조정 (적당히 엄격)
            'push_up': 0.8,      # 푸시업은 그대로 유지 (잘 되고 있음)
            'deadlift': 0.7,     # 그대로 유지
            'bench_press': 0.8,  # 그대로 유지
            'pull_up': 0.8       # 그대로 유지
        }
    
    def apply_ema(self, current_value: float) -> float:
        """지수 이동 평균 적용"""
        if self.ema_value is None:
            self.ema_value = current_value
        else:
            self.ema_value = self.ema_alpha * current_value + (1 - self.ema_alpha) * self.ema_value
        return self.ema_value
    
    def apply_hysteresis(self, violation_ratio: float, exercise_type: str = None) -> str:
        """운동별 조정된 히스테리시스 적용"""
        threshold = self.exercise_hysteresis.get(exercise_type, self.hysteresis_threshold)
        
        if self.last_state == 'good':
            if violation_ratio > threshold:
                self.last_state = 'bad'
        else:
            # 스쿼트는 복귀도 적당히 엄격하게
            if exercise_type == 'squat':
                recovery_threshold = threshold * 0.6  # 다른 운동 0.3보다 높게
            else:
                recovery_threshold = threshold * 0.3
            
            if violation_ratio < recovery_threshold:
                self.last_state = 'good'
        
        return self.last_state
    
    def process(self, analysis_result: Dict, exercise_type: str = None) -> Dict:
        """조정된 후처리 적용"""
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
                'squat_80_percent_adjusted': exercise_type == 'squat'
            }
        }

class EnhancedDatasetProcessor:
    """향상된 데이터셋 처리 클래스 - 스쿼트 80% 목표"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.classifier = EnhancedExerciseClassifier()
        
        # 스쿼트 80% 목표로 조정된 후처리기
        self.post_processors = {
            'squat': AdaptivePostProcessor(hysteresis_threshold=0.5, ema_alpha=0.4),      # 조정됨
            'push_up': AdaptivePostProcessor(hysteresis_threshold=0.8, ema_alpha=0.4),    # 유지
            'deadlift': AdaptivePostProcessor(hysteresis_threshold=0.7, ema_alpha=0.4),   # 유지
            'bench_press': AdaptivePostProcessor(hysteresis_threshold=0.8, ema_alpha=0.4), # 유지
            'pull_up': AdaptivePostProcessor(hysteresis_threshold=0.8, ema_alpha=0.4)     # 유지
        }
        
        self.output_path = self.base_path / "processed_data"
        self.output_path.mkdir(exist_ok=True)
        
        self.exercises = ['squat', 'push_up', 'deadlift', 'bench_press', 'pull_up']
        for exercise in self.exercises:
            for category in ['good', 'bad']:
                (self.output_path / exercise / category).mkdir(parents=True, exist_ok=True)
    
    def process_exercise_images(self, exercise_name: str, image_dir: str, limit: int = 500):
        """스쿼트 80% 목표로 조정된 이미지 처리"""
        print(f"\n=== {exercise_name} 처리 (스쿼트 80% 목표 조정) ===")
        
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
                                                 AdaptivePostProcessor(hysteresis_threshold=0.5))
        
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
                dest_file = dest_dir / f"{classification}_{exercise_name}_{view_type}_{i:04d}_adjusted.jpg"
                
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
                    'squat_80_adjusted': exercise_name == 'squat'
                }
                processing_log.append(log_entry)
                
                if (i + 1) % 50 == 0:
                    total_processed = results['good'] + results['bad']
                    good_rate = (results['good'] / max(total_processed, 1)) * 100
                    print(f"  📊 진행률: {i + 1}/{len(image_files)} images")
                    print(f"     현재 Good 비율: {good_rate:.1f}%")
                    
                    # 스쿼트 목표 달성 여부 실시간 체크
                    if exercise_name == 'squat':
                        if good_rate > 85:
                            print(f"     ⚠️ 스쿼트 85% 초과 - 더 엄격하게 조정 필요")
                        elif good_rate < 75:
                            print(f"     ⚠️ 스쿼트 75% 미만 - 조금 더 완화 필요")
                        else:
                            print(f"     ✅ 스쿼트 목표 범위 (75-85%)")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                results['failed'] += 1
                continue
        
        # 결과 출력
        total_processed = results['good'] + results['bad']
        good_rate = (results['good'] / max(total_processed, 1)) * 100
        
        print(f"\n📈 {exercise_name.upper()} 최종 결과:")
        print(f"  🎯 Good: {results['good']}장 ({good_rate:.1f}%)")
        print(f"  ❌ Bad: {results['bad']}장")
        print(f"  💥 Failed: {results['failed']}장")
        
        # 목표 달성 여부 확인
        target_rates = {'squat': 80, 'push_up': 80, 'deadlift': 85, 'bench_press': 85, 'pull_up': 85}
        target = target_rates.get(exercise_name, 80)
        
        if exercise_name == 'squat':
            if 75 <= good_rate <= 85:
                print(f"  🎉 스쿼트 목표 달성! (75-85% 범위) ✅")
            elif good_rate > 85:
                print(f"  ⚠️ 너무 관대함: {good_rate:.1f}% > 85% (더 엄격하게 조정 필요)")
            else:
                print(f"  ⚠️ 너무 엄격함: {good_rate:.1f}% < 75% (조금 더 완화 필요)")
        else:
            if good_rate >= target:
                print(f"  ✅ 목표 달성! ({target}% 이상)")
            else:
                print(f"  ⚠️ 목표 미달성: {good_rate:.1f}% < {target}%")
        
        # 뷰 분포
        print(f"  📷 뷰 분포:")
        for view, count in view_type_count.items():
            percentage = (count / max(total_processed, 1)) * 100
            print(f"     {view}: {count}장 ({percentage:.1f}%)")
        
        # 로그 저장
        log_file = self.output_path / f"{exercise_name}_80_adjusted_processing_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'exercise': exercise_name,
                'squat_80_percent_adjusted': exercise_name == 'squat',
                'summary': results,
                'good_rate': good_rate,
                'target_achievement': {
                    'target_rate': target,
                    'achieved': good_rate >= target if exercise_name != 'squat' else 75 <= good_rate <= 85,
                    'status': 'optimal' if exercise_name == 'squat' and 75 <= good_rate <= 85 else 'achieved' if good_rate >= target else 'needs_adjustment'
                },
                'view_distribution': view_type_count,
                'detailed_log': processing_log
            }, f, indent=2, ensure_ascii=False)
        
        return results

# 메인 실행부
if __name__ == "__main__":
    print("🎯 스쿼트 80% 목표 조정된 Enhanced Pose Analysis")
    print("목표: 스쿼트 75-85%, 푸시업 80%+ 유지")
    
    processor = EnhancedDatasetProcessor(".")
    
    exercises = {
        'squat': 'squat_exercise',
        'push_up': 'push_up_exercise', 
        'deadlift': 'deadlift_exercise',
        'bench_press': 'bench_press_exercise',
        'pull_up': 'pull_up_exercise'
    }
    
    total_results = {}
    for exercise, directory in exercises.items():
        results = processor.process_exercise_images(exercise, directory, limit=500)
        total_results[exercise] = results
    
    # 전체 결과 저장
    summary_file = processor.output_path / "squat_80_adjusted_processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'adjustment_version': 'squat_80_percent_target',
            'target_rates': {
                'squat': '75-85% (조정됨)',
                'push_up': '80%+ (유지)', 
                'deadlift': '85%+',
                'bench_press': '85%+',
                'pull_up': '85%+'
            },
            'results': total_results
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("🎯 스쿼트 80% 목표 조정 완료!")
    print("="*80)
    print("📊 최종 결과 요약:")
    
    target_rates = {'squat': 80, 'push_up': 80, 'deadlift': 85, 'bench_press': 85, 'pull_up': 85}
    
    for exercise, results in total_results.items():
        total = sum(results.values())
        if total > 0:
            good_rate = (results['good'] / max(results['good'] + results['bad'], 1)) * 100
            target = target_rates.get(exercise, 80)
            
            if exercise == 'squat':
                if 75 <= good_rate <= 85:
                    status = "🎯 최적 범위"
                elif good_rate > 85:
                    status = "⚠️ 너무 관대"
                else:
                    status = "⚠️ 너무 엄격"
                target_text = "75-85%"
            else:
                status = "✅ 달성" if good_rate >= target else "⚠️ 미달성"
                target_text = f"{target}%+"
            
            print(f"\n🏋️ {exercise.upper()}:")
            print(f"  총 처리: {total}장")
            print(f"  Good: {results['good']}장 ({good_rate:.1f}%)")
            print(f"  Bad: {results['bad']}장")
            print(f"  목표: {target_text} | 결과: {status}")
        else:
            print(f"\n⚠️ {exercise.upper()}: 데이터 없음")
    
    print(f"\n🔧 주요 조정 사항:")
    print(f"  • 스쿼트 무릎/힙: 15-175° → 40-160° (적당히 제한)")
    print(f"  • 스쿼트 등 각도: 120° → 140° (자세 품질 향상)")
    print(f"  • 스쿼트 가중치: 0.3 → 0.8 (영향력 증가)")
    print(f"  • 스쿼트 히스테리시스: 0.8 → 0.5 (적당히 엄격)")
    print(f"  • 푸시업: 기존 설정 유지 (잘 되고 있음)")
    
    print(f"\n💾 처리된 데이터 위치: {processor.output_path}")
    print("✅ 다음 단계: AI 모델 재훈련 (python main.py --mode train)")