import cv2
import numpy as np
import mediapipe as mp
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import math
import shutil
from collections import deque

@dataclass
class AngleThreshold:
    """각도 임계값 설정"""
    min_angle: float
    max_angle: float
    joint_points: List[int]  # 관절 포인트 인덱스
    name: str

class ExerciseClassifier:
    """운동 분류 및 각도 분석 클래스"""
    
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
        
        # 운동별 각도 기준 설정
        self.exercise_thresholds = {
            'bench_press': [
                AngleThreshold(70, 120, [11, 13, 15], 'left_elbow'),  # 왼쪽 팔꿈치
                AngleThreshold(70, 120, [12, 14, 16], 'right_elbow'), # 오른쪽 팔꿈치
                AngleThreshold(60, 100, [13, 11, 23], 'left_shoulder'),  # 왼쪽 어깨
                AngleThreshold(60, 100, [14, 12, 24], 'right_shoulder'), # 오른쪽 어깨
            ],
            'deadlift': [
                AngleThreshold(160, 180, [23, 25, 27], 'left_knee'),   # 왼쪽 무릎
                AngleThreshold(160, 180, [24, 26, 28], 'right_knee'),  # 오른쪽 무릎
                AngleThreshold(160, 180, [11, 23, 25], 'left_hip'),    # 왼쪽 엉덩이
                AngleThreshold(160, 180, [12, 24, 26], 'right_hip'),   # 오른쪽 엉덩이
                AngleThreshold(160, 180, [23, 11, 13], 'left_back'),   # 왼쪽 등
                AngleThreshold(160, 180, [24, 12, 14], 'right_back'),  # 오른쪽 등
            ],
            'pull_up': [
                AngleThreshold(30, 90, [11, 13, 15], 'left_elbow'),
                AngleThreshold(30, 90, [12, 14, 16], 'right_elbow'),
                AngleThreshold(120, 180, [13, 11, 23], 'left_shoulder'),
                AngleThreshold(120, 180, [14, 12, 24], 'right_shoulder'),
            ],
            'push_up': [
                AngleThreshold(80, 120, [11, 13, 15], 'left_elbow'),
                AngleThreshold(80, 120, [12, 14, 16], 'right_elbow'),
                AngleThreshold(160, 180, [11, 23, 25], 'left_hip'),
                AngleThreshold(160, 180, [12, 24, 26], 'right_hip'),
                AngleThreshold(170, 180, [23, 25, 27], 'left_knee'),
                AngleThreshold(170, 180, [24, 26, 28], 'right_knee'),
            ],
            'squat': [
                AngleThreshold(70, 120, [23, 25, 27], 'left_knee'),
                AngleThreshold(70, 120, [24, 26, 28], 'right_knee'),
                AngleThreshold(70, 120, [11, 23, 25], 'left_hip'),
                AngleThreshold(70, 120, [12, 24, 26], 'right_hip'),
                AngleThreshold(170, 180, [23, 11, 13], 'left_back'),
                AngleThreshold(170, 180, [24, 12, 14], 'right_back'),
            ]
        }
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """세 점 사이의 각도 계산"""
        try:
            # 벡터 계산
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # 각도 계산
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
        except:
            return 0.0
    
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
        """자세 분석 및 각도 계산"""
        if exercise_type not in self.exercise_thresholds:
            return {'valid': False, 'error': f'Unknown exercise type: {exercise_type}'}
        
        thresholds = self.exercise_thresholds[exercise_type]
        angles = {}
        violations = []
        
        for threshold in thresholds:
            try:
                # 관절 포인트 추출
                p1_idx, p2_idx, p3_idx = threshold.joint_points
                
                # 인덱스 범위 확인
                if any(idx >= len(landmarks) for idx in [p1_idx, p2_idx, p3_idx]):
                    continue
                
                # 가시성 확인
                if (landmarks[p1_idx]['visibility'] < 0.5 or 
                    landmarks[p2_idx]['visibility'] < 0.5 or 
                    landmarks[p3_idx]['visibility'] < 0.5):
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
                        'expected_max': threshold.max_angle
                    })
                    
            except Exception as e:
                print(f"Error calculating angle for {threshold.name}: {e}")
                continue
        
        # 분류 결과
        is_good = len(violations) == 0
        
        return {
            'valid': True,
            'classification': 'good' if is_good else 'bad',
            'angles': angles,
            'violations': violations,
            'violation_count': len(violations)
        }

class PostProcessor:
    """후처리 클래스 (히스테리시스 + EMA)"""
    
    def __init__(self, hysteresis_threshold: float = 0.3, ema_alpha: float = 0.3, window_size: int = 5):
        self.hysteresis_threshold = hysteresis_threshold
        self.ema_alpha = ema_alpha
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.ema_value = None
        self.last_state = None
    
    def apply_ema(self, current_value: float) -> float:
        """지수 이동 평균 적용"""
        if self.ema_value is None:
            self.ema_value = current_value
        else:
            self.ema_value = self.ema_alpha * current_value + (1 - self.ema_alpha) * self.ema_value
        return self.ema_value
    
    def apply_hysteresis(self, violation_ratio: float) -> str:
        """히스테리시스 적용"""
        if self.last_state is None:
            self.last_state = 'good' if violation_ratio == 0 else 'bad'
            return self.last_state
        
        if self.last_state == 'good':
            # good 상태에서 bad로 변경하려면 임계값 이상의 위반이 필요
            if violation_ratio > self.hysteresis_threshold:
                self.last_state = 'bad'
        else:
            # bad 상태에서 good으로 변경하려면 위반이 없어야 함
            if violation_ratio == 0:
                self.last_state = 'good'
        
        return self.last_state
    
    def process(self, analysis_result: Dict) -> Dict:
        """후처리 적용"""
        if not analysis_result['valid']:
            return analysis_result
        
        # 위반 비율 계산
        total_angles = len(analysis_result['angles'])
        violation_count = analysis_result['violation_count']
        violation_ratio = violation_count / total_angles if total_angles > 0 else 0
        
        # EMA 적용
        smoothed_ratio = self.apply_ema(violation_ratio)
        
        # 히스토리 추가
        self.history.append(smoothed_ratio)
        
        # 히스테리시스 적용
        final_classification = self.apply_hysteresis(smoothed_ratio)
        
        return {
            **analysis_result,
            'final_classification': final_classification,
            'violation_ratio': violation_ratio,
            'smoothed_ratio': smoothed_ratio,
            'confidence': 1.0 - smoothed_ratio
        }

class DatasetProcessor:
    """데이터셋 처리 클래스"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.classifier = ExerciseClassifier()
        self.post_processor = PostProcessor()
        
        # 결과 저장 디렉토리 생성
        self.output_path = self.base_path / "processed_data"
        self.output_path.mkdir(exist_ok=True)
        
        # 운동별 디렉토리 생성
        self.exercises = ['bench_press', 'deadlift', 'pull_up', 'push_up', 'squat']
        for exercise in self.exercises:
            for category in ['good', 'bad']:
                (self.output_path / exercise / category).mkdir(parents=True, exist_ok=True)
    
    def process_exercise_images(self, exercise_name: str, image_dir: str, limit: int = 500):
        """특정 운동의 이미지들을 처리하고 분류"""
        print(f"\n=== Processing {exercise_name} ===")
        
        image_path = self.base_path / "data" / "images" / image_dir
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
        
        results = {'good': 0, 'bad': 0, 'failed': 0}
        processing_log = []
        
        for i, img_file in enumerate(image_files):
            try:
                # 랜드마크 추출
                landmarks_data = self.classifier.extract_landmarks(str(img_file))
                if landmarks_data is None:
                    results['failed'] += 1
                    continue
                
                # 자세 분석
                analysis = self.classifier.analyze_pose(
                    landmarks_data['landmarks'], 
                    exercise_name
                )
                
                if not analysis['valid']:
                    results['failed'] += 1
                    continue
                
                # 후처리 적용
                final_result = self.post_processor.process(analysis)
                classification = final_result['final_classification']
                
                # 파일 복사
                dest_dir = self.output_path / exercise_name / classification
                dest_file = dest_dir / f"{classification}_{exercise_name}_{i:04d}.jpg"
                shutil.copy2(img_file, dest_file)
                
                results[classification] += 1
                
                # 로그 저장
                log_entry = {
                    'original_file': str(img_file),
                    'classification': classification,
                    'angles': final_result['angles'],
                    'violations': final_result['violations'],
                    'confidence': final_result['confidence']
                }
                processing_log.append(log_entry)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                results['failed'] += 1
                continue
        
        print(f"Results - Good: {results['good']}, Bad: {results['bad']}, Failed: {results['failed']}")
        
        # 로그 저장
        log_file = self.output_path / f"{exercise_name}_processing_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(processing_log, f, indent=2, ensure_ascii=False)
        
        return results
    
    def process_all_exercises(self):
        """모든 운동 처리"""
        exercise_dirs = {
            'bench_press': 'bench_press_exercise',
            'deadlift': 'deadlift_exercise',
            'pull_up': 'pull_up_exercise',
            'push_up': 'push_up_exercise',
            'squat': 'squat_exercise'
        }
        
        total_results = {}
        
        for exercise, directory in exercise_dirs.items():
            results = self.process_exercise_images(exercise, directory)
            total_results[exercise] = results
        
        # 전체 결과 저장
        summary_file = self.output_path / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(total_results, f, indent=2, ensure_ascii=False)
        
        print("\n=== PROCESSING COMPLETE ===")
        print("Summary:")
        for exercise, results in total_results.items():
            print(f"{exercise}: Good={results['good']}, Bad={results['bad']}, Failed={results['failed']}")
        
        return total_results

def main():
    """메인 실행 함수"""
    # 기본 경로 설정 (현재 디렉토리 기준)
    base_path = "."
    
    try:
        # 데이터셋 프로세서 초기화
        processor = DatasetProcessor(base_path)
        
        # 모든 운동 처리
        processor.process_all_exercises()
        
        print(f"\nProcessed data saved to: {processor.output_path}")
        print("Directory structure:")
        print("processed_data/")
        print("├── bench_press/")
        print("│   ├── good/")
        print("│   └── bad/")
        print("├── deadlift/")
        print("│   ├── good/")
        print("│   └── bad/")
        print("├── pull_up/")
        print("│   ├── good/")
        print("│   └── bad/")
        print("├── push_up/")
        print("│   ├── good/")
        print("│   └── bad/")
        print("└── squat/")
        print("    ├── good/")
        print("    └── bad/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())