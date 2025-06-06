"""
향상된 자세 분석 시스템 - 5종목 완전 지원 (최종 완성본)
스쿼트, 푸시업, 데드리프트, 벤치프레스, 풀업
- 뷰 타입 자동 감지 + 완화된 각도 기준 적용
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
    weight: float = 1.0  # 중요도 가중치
    view_types: List[str] = None  # 적용되는 뷰 타입

class EnhancedExerciseClassifier:
    """향상된 운동 분류기 - 5종목 뷰 감지 기능 포함"""
    
    def __init__(self):
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        except Exception as e:
            print(f"❌ MediaPipe 초기화 실패: {e}")
            raise
        
        # 5종목 뷰별 운동 각도 기준 설정 (완화된 버전)
        self.exercise_thresholds = {
            'squat': {
                'side_view': [  # 완화된 스쿼트 기준
                    ViewSpecificThreshold(60, 140, [23, 25, 27], 'left_knee', 1.5, ['side']),      # 70→60, 120→140
                    ViewSpecificThreshold(60, 140, [24, 26, 28], 'right_knee', 1.5, ['side']),     # 더 넓은 범위
                    ViewSpecificThreshold(60, 140, [11, 23, 25], 'left_hip', 1.2, ['side']),       # 힙 각도도 완화
                    ViewSpecificThreshold(150, 180, [11, 23, 25], 'back_straight', 1.0, ['side']), # 등 각도 완화 (중요도 낮춤)
                    ViewSpecificThreshold(70, 120, [25, 27, 31], 'left_ankle', 0.6, ['side']),     # 발목은 덜 중요
                ],
                'front_view': [
                    ViewSpecificThreshold(150, 180, [11, 12, 23], 'shoulder_level', 0.8, ['front']),
                    ViewSpecificThreshold(60, 120, [23, 24, 25], 'hip_symmetry', 1.0, ['front']),
                    ViewSpecificThreshold(160, 180, [25, 27, 29], 'knee_tracking', 0.9, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(150, 180, [11, 12, 23], 'back_alignment', 0.8, ['back']),
                    ViewSpecificThreshold(160, 180, [23, 25, 27], 'spine_straight', 1.0, ['back']),
                ]
            },
            
            'push_up': {
                'side_view': [  # 완화된 푸시업 기준
                    ViewSpecificThreshold(70, 130, [11, 13, 15], 'left_elbow', 1.2, ['side']),      # 80→70, 120→130
                    ViewSpecificThreshold(70, 130, [12, 14, 16], 'right_elbow', 1.2, ['side']),     # 팔꿈치 범위 확대
                    ViewSpecificThreshold(150, 180, [11, 23, 25], 'body_line', 1.3, ['side']),      # 몸 일직선 기준 완화
                    ViewSpecificThreshold(160, 180, [23, 25, 27], 'leg_straight', 0.8, ['side']),   # 다리 각도 완화
                    ViewSpecificThreshold(150, 180, [13, 11, 23], 'shoulder_alignment', 0.9, ['side']),
                ],
                'front_view': [
                    ViewSpecificThreshold(150, 180, [11, 12, 13], 'shoulder_width', 0.7, ['front']),
                    ViewSpecificThreshold(160, 180, [15, 16, 17], 'hand_position', 0.8, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(150, 180, [11, 12, 23], 'back_straight', 1.0, ['back']),
                    ViewSpecificThreshold(160, 180, [23, 24, 25], 'hip_level', 0.8, ['back']),
                ]
            },
            
            'deadlift': {
                'side_view': [  # 완화된 데드리프트 기준
                    ViewSpecificThreshold(150, 180, [23, 25, 27], 'left_knee', 1.0, ['side']),      # 160→150으로 완화
                    ViewSpecificThreshold(150, 180, [24, 26, 28], 'right_knee', 1.0, ['side']),     # 무릎 각도 완화
                    ViewSpecificThreshold(150, 180, [11, 23, 25], 'hip_hinge', 1.2, ['side']),      # 힙힌지 완화
                    ViewSpecificThreshold(160, 180, [11, 23, 12], 'back_straight', 1.3, ['side']),  # 등 자세는 여전히 중요
                    ViewSpecificThreshold(70, 110, [23, 11, 13], 'chest_up', 0.9, ['side']),       # 가슴 각도 완화
                ],
                'front_view': [
                    ViewSpecificThreshold(160, 180, [11, 12, 23], 'shoulder_level', 0.8, ['front']),
                    ViewSpecificThreshold(150, 180, [23, 24, 25], 'hip_symmetry', 0.9, ['front']),
                    ViewSpecificThreshold(160, 180, [25, 26, 27], 'knee_alignment', 1.0, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(160, 180, [11, 23, 24], 'spine_neutral', 1.1, ['back']),
                    ViewSpecificThreshold(150, 180, [23, 25, 26], 'hip_level', 0.8, ['back']),
                ]
            },
            
            'bench_press': {
                'side_view': [  # 완화된 벤치프레스 기준
                    ViewSpecificThreshold(60, 130, [11, 13, 15], 'left_elbow', 1.2, ['side']),      # 70→60, 120→130
                    ViewSpecificThreshold(60, 130, [12, 14, 16], 'right_elbow', 1.2, ['side']),     # 팔꿈치 범위 확대
                    ViewSpecificThreshold(50, 110, [13, 11, 23], 'left_shoulder', 1.0, ['side']),   # 어깨 각도 완화
                    ViewSpecificThreshold(50, 110, [14, 12, 24], 'right_shoulder', 1.0, ['side']),  # 60→50, 100→110
                    ViewSpecificThreshold(160, 180, [11, 23, 25], 'back_arch', 0.6, ['side']),      # 등 아치는 덜 중요
                ],
                'front_view': [
                    ViewSpecificThreshold(150, 180, [11, 12, 13], 'shoulder_symmetry', 0.9, ['front']),
                    ViewSpecificThreshold(150, 180, [13, 14, 15], 'arm_symmetry', 0.8, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(150, 180, [11, 12, 23], 'upper_back', 0.8, ['back']),
                ]
            },
            
            'pull_up': {
                'side_view': [  # 완화된 풀업 기준
                    ViewSpecificThreshold(20, 100, [11, 13, 15], 'left_elbow', 1.2, ['side']),      # 30→20, 90→100
                    ViewSpecificThreshold(20, 100, [12, 14, 16], 'right_elbow', 1.2, ['side']),     # 팔꿈치 범위 확대
                    ViewSpecificThreshold(110, 180, [13, 11, 23], 'left_shoulder', 1.1, ['side']),  # 어깨 각도 완화
                    ViewSpecificThreshold(110, 180, [14, 12, 24], 'right_shoulder', 1.1, ['side']), # 120→110으로 완화
                    ViewSpecificThreshold(160, 180, [11, 23, 25], 'body_straight', 0.9, ['side']),  # 몸 각도 완화
                    ViewSpecificThreshold(150, 180, [23, 25, 27], 'leg_position', 0.6, ['side']),   # 다리 위치는 덜 중요
                ],
                'front_view': [
                    ViewSpecificThreshold(150, 180, [11, 12, 13], 'shoulder_width', 0.8, ['front']),
                    ViewSpecificThreshold(20, 100, [13, 15, 16], 'grip_symmetry', 0.9, ['front']),
                ],
                'back_view': [
                    ViewSpecificThreshold(150, 180, [11, 12, 23], 'lat_engagement', 1.0, ['back']),
                    ViewSpecificThreshold(160, 180, [23, 24, 25], 'core_stability', 0.8, ['back']),
                ]
            }
        }
    
    def detect_view_type(self, landmarks: List[Dict]) -> str:
        """촬영 각도/뷰 타입 감지"""
        try:
            # 어깨와 엉덩이의 x 좌표 차이로 뷰 타입 판단
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # 어깨 너비와 엉덩이 너비 계산
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            hip_width = abs(left_hip['x'] - right_hip['x'])
            
            # 코의 x 좌표로 방향 판단
            nose = landmarks[0]
            body_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            
            # 뷰 타입 판단 로직
            if shoulder_width < 0.15 and hip_width < 0.15:
                # 어깨와 엉덩이가 겹쳐 보임 -> 측면 뷰
                return 'side_view'
            elif shoulder_width > 0.25 and hip_width > 0.2:
                # 어깨와 엉덩이가 넓게 보임 -> 정면 또는 후면 뷰
                # 코의 위치로 정면/후면 구분
                if abs(nose['x'] - body_center_x) < 0.1:
                    return 'front_view'
                else:
                    return 'back_view'
            else:
                # 중간 각도 -> 기본적으로 측면으로 처리
                return 'side_view'
                
        except Exception as e:
            print(f"뷰 타입 감지 오류: {e}")
            return 'side_view'  # 기본값
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """수정된 각도 계산 - BlazePose 좌표계 고려"""
        try:
            # BlazePose는 정규화된 좌표 (0~1)를 사용
            # Y축이 아래쪽이 큰 값 (이미지 좌표계)
            
            # 벡터 계산
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # 벡터 크기 확인
            v1_mag = np.linalg.norm(v1)
            v2_mag = np.linalg.norm(v2)
            
            if v1_mag < 1e-6 or v2_mag < 1e-6:  # 너무 작은 벡터
                return 180.0  # 기본값 (펴진 상태)
            
            # 코사인 값 계산
            cos_angle = np.dot(v1, v2) / (v1_mag * v2_mag)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # 각도 계산 (0~180도)
            angle = np.degrees(np.arccos(cos_angle))
            
            # 180도에 가까운 각도들은 "펴진" 상태로 간주
            # 90도에 가까운 각도들은 "구부린" 상태로 간주
            
            return angle
            
        except Exception as e:
            print(f"각도 계산 오류: {e}")
            return 180.0  # 안전한 기본값
    
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
        """향상된 자세 분석 - 뷰 타입 고려 + 완화된 기준"""
        if exercise_type not in self.exercise_thresholds:
            return {'valid': False, 'error': f'Unknown exercise type: {exercise_type}'}
        
        # 뷰 타입 감지
        view_type = self.detect_view_type(landmarks)
        
        # 해당 뷰 타입에 맞는 임계값 선택
        all_thresholds = self.exercise_thresholds[exercise_type]
        current_thresholds = all_thresholds.get(view_type, [])
        
        # 뷰 타입별 임계값이 없으면 측면 뷰 사용
        if not current_thresholds:
            current_thresholds = all_thresholds.get('side_view', [])
        
        angles = {}
        violations = []
        weighted_violation_score = 0.0
        total_weight = 0.0
        
        for threshold in current_thresholds:
            try:
                p1_idx, p2_idx, p3_idx = threshold.joint_points
                
                # 인덱스 범위 확인
                if any(idx >= len(landmarks) for idx in [p1_idx, p2_idx, p3_idx]):
                    continue
                
                # 가시성 확인 (더 관대하게)
                visibility_threshold = 0.2 if view_type == 'side_view' else 0.3  # 기존보다 낮춤
                if (landmarks[p1_idx]['visibility'] < visibility_threshold or 
                    landmarks[p2_idx]['visibility'] < visibility_threshold or 
                    landmarks[p3_idx]['visibility'] < visibility_threshold):
                    continue
                
                p1 = (landmarks[p1_idx]['x'], landmarks[p1_idx]['y'])
                p2 = (landmarks[p2_idx]['x'], landmarks[p2_idx]['y'])
                p3 = (landmarks[p3_idx]['x'], landmarks[p3_idx]['y'])
                
                angle = self.calculate_angle(p1, p2, p3)
                angles[threshold.name] = angle
                
                # 허용 범위 확인
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
        
        # 가중치를 고려한 분류 (더 관대하게)
        violation_ratio = weighted_violation_score / max(total_weight, 1.0)
        is_good = violation_ratio < 0.7  # 기존 0.3에서 0.7로 완화 (70% 위반까지 허용)
        
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
    """적응형 후처리 클래스 - 5종목 지원 (완화된 버전)"""
    
    def __init__(self, hysteresis_threshold: float = 0.2, ema_alpha: float = 0.3):
        self.hysteresis_threshold = hysteresis_threshold
        self.ema_alpha = ema_alpha
        self.history = deque(maxlen=15)
        self.ema_value = None
        self.last_state = 'good'
        self.state_counter = {'good': 0, 'bad': 0}
        
        # 더 관대한 히스테리시스 설정 (기존보다 완화)
        self.exercise_hysteresis = {
            'squat': 0.4,        # 0.2 → 0.4 (더 관대)
            'push_up': 0.3,      # 0.15 → 0.3 (훨씬 완화)  
            'deadlift': 0.5,     # 0.25 → 0.5 (가장 관대)
            'bench_press': 0.35, # 0.18 → 0.35 (관대)
            'pull_up': 0.4       # 0.22 → 0.4 (관대)
        }
    
    def apply_ema(self, current_value: float) -> float:
        """지수 이동 평균 적용"""
        if self.ema_value is None:
            self.ema_value = current_value
        else:
            self.ema_value = self.ema_alpha * current_value + (1 - self.ema_alpha) * self.ema_value
        return self.ema_value
    
    def apply_hysteresis(self, violation_ratio: float, exercise_type: str = None) -> str:
        """운동별 적응형 히스테리시스 적용"""
        # 운동별 임계값 사용
        threshold = self.exercise_hysteresis.get(exercise_type, self.hysteresis_threshold)
        
        if self.last_state == 'good':
            if violation_ratio > threshold:
                self.last_state = 'bad'
        else:
            if violation_ratio < threshold * 0.5:  # 복귀 임계값은 더 낮게
                self.last_state = 'good'
        
        return self.last_state
    
    def process(self, analysis_result: Dict, exercise_type: str = None) -> Dict:
        """적응형 후처리 적용"""
        if not analysis_result['valid']:
            return analysis_result
        
        # 가중치가 적용된 위반 비율 사용
        violation_ratio = analysis_result.get('weighted_violation_ratio', 0)
        
        # EMA 적용
        smoothed_ratio = self.apply_ema(violation_ratio)
        
        # 히스토리 추가
        self.history.append(smoothed_ratio)
        
        # 적응형 히스테리시스 적용
        final_classification = self.apply_hysteresis(smoothed_ratio, exercise_type)
        
        # 상태 카운터 업데이트
        self.state_counter[final_classification] += 1
        
        # 신뢰도 조정 (뷰 타입에 따라)
        view_type = analysis_result.get('view_type', 'side_view')
        confidence_modifier = {
            'side_view': 1.0,    # 측면 뷰가 가장 정확
            'front_view': 0.8,   # 정면 뷰는 약간 낮음
            'back_view': 0.7     # 후면 뷰는 더 낮음
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
                'state_history': list(self.history)[-5:]  # 최근 5개 상태
            }
        }

# 기존 DatasetProcessor 클래스를 새로운 분석기로 업데이트
class EnhancedDatasetProcessor:
    """향상된 데이터셋 처리 클래스 - 5종목 지원"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.classifier = EnhancedExerciseClassifier()
        
        # 5종목별 맞춤 후처리기 (완화된 버전)
        self.post_processors = {
            'squat': AdaptivePostProcessor(hysteresis_threshold=0.4, ema_alpha=0.3),
            'push_up': AdaptivePostProcessor(hysteresis_threshold=0.3, ema_alpha=0.25),
            'deadlift': AdaptivePostProcessor(hysteresis_threshold=0.5, ema_alpha=0.35),
            'bench_press': AdaptivePostProcessor(hysteresis_threshold=0.35, ema_alpha=0.28),
            'pull_up': AdaptivePostProcessor(hysteresis_threshold=0.4, ema_alpha=0.32)
        }
        
        # 결과 저장 디렉토리 생성
        self.output_path = self.base_path / "processed_data"
        self.output_path.mkdir(exist_ok=True)
        
        # 5종목 디렉토리 생성
        self.exercises = ['squat', 'push_up', 'deadlift', 'bench_press', 'pull_up']
        for exercise in self.exercises:
            for category in ['good', 'bad']:
                (self.output_path / exercise / category).mkdir(parents=True, exist_ok=True)
    
    def process_exercise_images(self, exercise_name: str, image_dir: str, limit: int = 500):
        """향상된 이미지 처리 및 분류"""
        print(f"\n=== Processing {exercise_name} with Enhanced Analysis ===")
        
        image_path = self.base_path / "data" / "training_images" / image_dir
        if not image_path.exists():
            print(f"Directory not found: {image_path}")
            return {'good': 0, 'bad': 0, 'failed': 0}
        
        # 이미지 파일 목록 가져오기
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(image_path.glob(ext)))
        
        print(f"Found {len(image_files)} images")
        
        # 처리 제한
        if len(image_files) > limit:
            image_files = image_files[:limit]
            print(f"Limited to {limit} images")
        
        results = {'good': 0, 'bad': 0, 'failed': 0}
        processing_log = []
        view_type_count = {'side_view': 0, 'front_view': 0, 'back_view': 0}
        
        # 해당 운동의 후처리기 선택
        post_processor = self.post_processors.get(exercise_name, 
                                                 AdaptivePostProcessor())
        
        for i, img_file in enumerate(image_files):
            try:
                # 랜드마크 추출
                landmarks_data = self.classifier.extract_landmarks(str(img_file))
                if landmarks_data is None:
                    results['failed'] += 1
                    continue
                
                # 향상된 자세 분석
                analysis = self.classifier.analyze_pose(
                    landmarks_data['landmarks'], 
                    exercise_name
                )
                
                if not analysis['valid']:
                    results['failed'] += 1
                    continue
                
                # 적응형 후처리 적용
                final_result = post_processor.process(analysis, exercise_name)
                classification = final_result['final_classification']
                view_type = final_result.get('view_type', 'unknown')
                
                # 뷰 타입 카운트
                if view_type in view_type_count:
                    view_type_count[view_type] += 1
                
                # 파일 복사
                dest_dir = self.output_path / exercise_name / classification
                dest_file = dest_dir / f"{classification}_{exercise_name}_{view_type}_{i:04d}.jpg"
                
                import shutil
                shutil.copy2(img_file, dest_file)
                
                results[classification] += 1
                
                # 상세 로그 저장
                log_entry = {
                    'original_file': str(img_file),
                    'classification': classification,
                    'view_type': view_type,
                    'angles': final_result['angles'],
                    'violations': final_result['violations'],
                    'confidence': final_result['confidence'],
                    'weighted_violation_ratio': final_result.get('weighted_violation_ratio', 0),
                    'processing_info': final_result.get('processing_info', {})
                }
                processing_log.append(log_entry)
                
                if (i + 1) % 50 == 0:
                    print(f"  📊 진행률: {i + 1}/{len(image_files)} images")
                    print(f"     뷰 분포: Side({view_type_count['side_view']}) Front({view_type_count['front_view']}) Back({view_type_count['back_view']})")
                    print(f"     분류 현황: Good({results['good']}) Bad({results['bad']}) Failed({results['failed']})")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                results['failed'] += 1
                continue
        
        # 결과 출력
        total_processed = results['good'] + results['bad']
        good_rate = (results['good'] / max(total_processed, 1)) * 100
        
        print(f"\n📈 {exercise_name.upper()} 처리 완료:")
        print(f"  ✅ Good: {results['good']}장 ({good_rate:.1f}%)")
        print(f"  ❌ Bad: {results['bad']}장")
        print(f"  💥 Failed: {results['failed']}장")
        print(f"  📷 뷰 분포:")
        for view, count in view_type_count.items():
            percentage = (count / max(total_processed, 1)) * 100
            print(f"     {view}: {count}장 ({percentage:.1f}%)")
        
        # 상세 로그 저장
        log_file = self.output_path / f"{exercise_name}_enhanced_processing_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'exercise': exercise_name,
                'summary': results,
                'view_distribution': view_type_count,
                'detailed_log': processing_log
            }, f, indent=2, ensure_ascii=False)
        
        return results

# 메인 실행부
if __name__ == "__main__":
    processor = EnhancedDatasetProcessor("..")
    
    # 5종목 처리 (현재 3종목만 데이터 있음)
    exercises = {
        'squat': 'squat_exercise',
        'push_up': 'push_up_exercise', 
        'deadlift': 'deadlift_exercise',
        'bench_press': 'bench_press_exercise',  # 미래에 추가될 데이터
        'pull_up': 'pull_up_exercise'           # 미래에 추가될 데이터
    }
    
    total_results = {}
    for exercise, directory in exercises.items():
        results = processor.process_exercise_images(exercise, directory, limit=500)
        total_results[exercise] = results
    
    # 전체 결과 저장
    summary_file = processor.output_path / "enhanced_processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(total_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("🎉 5종목 향상된 데이터 처리 완료!")
    print("="*70)
    print("📊 전체 결과 요약:")
    
    for exercise, results in total_results.items():
        total = sum(results.values())
        if total > 0:
            good_rate = (results['good'] / max(results['good'] + results['bad'], 1)) * 100
            print(f"\n🏋️ {exercise.upper()}:")
            print(f"  총 처리: {total}장")
            print(f"  Good: {results['good']}장 ({good_rate:.1f}%)")
            print(f"  Bad: {results['bad']}장")
            print(f"  Failed: {results['failed']}장")
        else:
            print(f"\n⚠️ {exercise.upper()}: 데이터 없음 (나중에 추가 예정)")
    
    print(f"\n💾 처리된 데이터 위치: {processor.output_path}")
    print("✅ 다음 단계: AI 모델 훈련 (python exercise_classifier.py --mode train)")