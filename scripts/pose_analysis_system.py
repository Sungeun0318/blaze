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
    """각도 임계값 설정 (enhanced_pose_analysis 기준 적용)"""
    min_angle: float
    max_angle: float
    joint_points: List[int]
    name: str

class ExerciseClassifier:
    """운동 분류 및 각도 분석 클래스 (enhanced_pose_analysis 기준 적용)"""
    
    def __init__(self):
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,  # enhanced와 동일하게
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        except Exception as e:
            print(f"❌ MediaPipe 초기화 실패: {e}")
            raise
        
        # 🎯 enhanced_pose_analysis.py와 동일한 각도 기준 적용
        self.exercise_thresholds = {
            'bench_press': [
                AngleThreshold(50, 145, [11, 13, 15], 'left_elbow'),    # enhanced와 동일
                AngleThreshold(50, 145, [12, 14, 16], 'right_elbow'),   # enhanced와 동일
                AngleThreshold(50, 150, [13, 11, 23], 'left_shoulder'), # enhanced와 동일
                AngleThreshold(50, 150, [14, 12, 24], 'right_shoulder'), # enhanced와 동일
                AngleThreshold(90, 170, [11, 23, 25], 'back_arch'),     # enhanced 추가
                AngleThreshold(70, 180, [13, 15, 17], 'wrist_alignment'), # enhanced 추가
            ],
            'deadlift': [
                AngleThreshold(80, 140, [23, 25, 27], 'left_knee'),    # enhanced와 동일 (대폭 완화)
                AngleThreshold(80, 140, [24, 26, 28], 'right_knee'),   # enhanced와 동일 (대폭 완화)
                AngleThreshold(80, 180, [11, 23, 25], 'left_hip'),     # enhanced와 동일 (대폭 완화)
                AngleThreshold(80, 180, [12, 24, 26], 'right_hip'),    # enhanced와 동일 (대폭 완화)
                AngleThreshold(120, 180, [23, 11, 13], 'left_back'),   # enhanced와 동일 (완화)
                AngleThreshold(120, 180, [24, 12, 14], 'right_back'),  # enhanced와 동일 (완화)
                AngleThreshold(50, 140, [23, 11, 13], 'chest_up'),     # enhanced 추가
            ],
            'pull_up': [  # 기존 유지 (런지로 교체 예정이지만 호환성 위해)
                AngleThreshold(10, 120, [11, 13, 15], 'left_elbow'),
                AngleThreshold(10, 120, [12, 14, 16], 'right_elbow'),
                AngleThreshold(90, 180, [13, 11, 23], 'left_shoulder'),
                AngleThreshold(90, 180, [14, 12, 24], 'right_shoulder'),
            ],
            'push_up': [
                AngleThreshold(40, 160, [11, 13, 15], 'left_elbow'),   # enhanced와 동일
                AngleThreshold(40, 160, [12, 14, 16], 'right_elbow'),  # enhanced와 동일
                AngleThreshold(140, 180, [11, 23, 25], 'left_hip'),    # enhanced와 동일 (body_line)
                AngleThreshold(140, 180, [12, 24, 26], 'right_hip'),   # enhanced와 동일
                AngleThreshold(140, 180, [23, 25, 27], 'left_knee'),   # enhanced와 동일 (leg_straight)
                AngleThreshold(140, 180, [24, 26, 28], 'right_knee'),  # enhanced와 동일
                AngleThreshold(120, 180, [13, 11, 23], 'shoulder_alignment'), # enhanced 추가
            ],
            'squat': [
                AngleThreshold(55, 140, [23, 25, 27], 'left_knee'),    # enhanced와 동일 (조정됨)
                AngleThreshold(55, 140, [24, 26, 28], 'right_knee'),   # enhanced와 동일 (조정됨)
                AngleThreshold(55, 140, [11, 23, 25], 'left_hip'),     # enhanced와 동일 (조정됨)
                AngleThreshold(55, 140, [12, 24, 26], 'right_hip'),    # enhanced와 동일 (조정됨)
                AngleThreshold(110, 170, [23, 11, 13], 'left_back'),   # enhanced와 동일 (back_straight)
                AngleThreshold(110, 170, [24, 12, 14], 'right_back'),  # enhanced와 동일
            ],
            'lunge': [  # enhanced와 동일한 런지 기준
                AngleThreshold(70, 120, [23, 25, 27], 'front_knee'),   # enhanced와 동일
                AngleThreshold(120, 180, [24, 26, 28], 'back_knee'),   # enhanced와 동일
                AngleThreshold(70, 120, [11, 23, 25], 'front_hip'),    # enhanced와 동일
                AngleThreshold(100, 180, [11, 23, 25], 'torso_upright'), # enhanced와 동일
                AngleThreshold(80, 110, [25, 27, 31], 'front_ankle'),  # enhanced와 동일
                AngleThreshold(150, 180, [12, 24, 26], 'back_hip_extension'), # enhanced와 동일
            ]
        }
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """세 점 사이의 각도 계산 (enhanced와 동일)"""
        try:
            # 벡터 계산
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
        """자세 분석 (enhanced_pose_analysis와 동일한 분류 기준)"""
        if exercise_type not in self.exercise_thresholds:
            return {'valid': False, 'error': f'Unknown exercise type: {exercise_type}'}
        
        thresholds = self.exercise_thresholds[exercise_type]
        angles = {}
        violations = []
        weighted_violation_score = 0.0
        total_weight = 0.0
        
        for threshold in thresholds:
            try:
                # 관절 포인트 추출
                p1_idx, p2_idx, p3_idx = threshold.joint_points
                
                # 인덱스 범위 확인
                if any(idx >= len(landmarks) for idx in [p1_idx, p2_idx, p3_idx]):
                    continue
                
                # enhanced와 동일한 가시성 확인 (0.25)
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
                
                # enhanced에서 가중치 적용 방식
                # 가중치를 enhanced에서 가져와서 적용
                exercise_weights = {
                    'squat': {'left_knee': 1.1, 'right_knee': 1.1, 'left_hip': 0.9, 'right_hip': 0.9, 'left_back': 1.1, 'right_back': 1.1},
                    'push_up': {'left_elbow': 1.0, 'right_elbow': 1.0, 'left_hip': 1.2, 'right_hip': 1.2, 'left_knee': 0.8, 'right_knee': 0.8},
                    'deadlift': {'left_knee': 0.6, 'right_knee': 0.6, 'left_hip': 0.7, 'right_hip': 0.7, 'left_back': 1.0, 'right_back': 1.0},
                    'bench_press': {'left_elbow': 1.1, 'right_elbow': 1.1, 'left_shoulder': 0.9, 'right_shoulder': 0.9, 'back_arch': 0.7},
                    'lunge': {'front_knee': 1.2, 'back_knee': 1.0, 'front_hip': 0.8, 'torso_upright': 1.2, 'front_ankle': 0.8}
                }
                
                weight = exercise_weights.get(exercise_type, {}).get(threshold.name, 0.5)
                
                # 허용 범위 확인
                if not (threshold.min_angle <= angle <= threshold.max_angle):
                    violations.append({
                        'joint': threshold.name,
                        'angle': angle,
                        'expected_min': threshold.min_angle,
                        'expected_max': threshold.max_angle,
                        'weight': weight
                    })
                    weighted_violation_score += weight
                
                total_weight += weight
                    
            except Exception as e:
                print(f"Error calculating angle for {threshold.name}: {e}")
                continue
        
        # enhanced_pose_analysis와 동일한 분류 기준
        violation_ratio = weighted_violation_score / max(total_weight, 1.0)
        
        classification_thresholds = {
            'squat': 0.5,        # enhanced와 동일
            'push_up': 0.7,      # enhanced와 동일
            'deadlift': 0.8,     # enhanced와 동일 (대폭 완화)
            'bench_press': 0.5,  # enhanced와 동일
            'lunge': 0.6,        # enhanced와 동일
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
            'total_joints': len(angles),
            'classification_threshold': threshold,
            'enhanced_compatible': True
        }

class PostProcessor:
    """후처리 클래스 (enhanced_pose_analysis 기준 적용)"""
    
    def __init__(self, hysteresis_threshold: float = 0.6, ema_alpha: float = 0.3, window_size: int = 15):
        self.hysteresis_threshold = hysteresis_threshold
        self.ema_alpha = ema_alpha  # enhanced와 동일
        self.window_size = window_size  # enhanced와 동일
        self.history = deque(maxlen=window_size)
        self.ema_value = None
        self.last_state = 'good'
        
        # enhanced와 동일한 운동별 히스테리시스
        self.exercise_hysteresis = {
            'squat': 0.5,        # enhanced와 동일
            'push_up': 0.7,      # enhanced와 동일
            'deadlift': 0.8,     # enhanced와 동일 (대폭 완화)
            'bench_press': 0.5,  # enhanced와 동일
            'lunge': 0.6,        # enhanced와 동일
        }
    
    def apply_ema(self, current_value: float) -> float:
        """지수 이동 평균 적용"""
        if self.ema_value is None:
            self.ema_value = current_value
        else:
            self.ema_value = self.ema_alpha * current_value + (1 - self.ema_alpha) * self.ema_value
        return self.ema_value
    
    def apply_hysteresis(self, violation_ratio: float, exercise_type: str = None) -> str:
        """enhanced와 동일한 히스테리시스 적용"""
        threshold = self.exercise_hysteresis.get(exercise_type, self.hysteresis_threshold)
        
        if self.last_state == 'good':
            if violation_ratio > threshold:
                self.last_state = 'bad'
        else:
            # enhanced와 동일한 복귀 기준
            recovery_thresholds = {
                'squat': threshold * 0.7,        # enhanced와 동일
                'push_up': threshold * 0.8,      # enhanced와 동일
                'deadlift': threshold * 0.9,     # enhanced와 동일 (매우 쉬운 복귀)
                'bench_press': threshold * 0.7,  # enhanced와 동일
                'lunge': threshold * 0.8,        # enhanced와 동일
            }
            
            recovery_threshold = recovery_thresholds.get(exercise_type, threshold * 0.8)
            
            if violation_ratio < recovery_threshold:
                self.last_state = 'good'
        
        return self.last_state
    
    def process(self, analysis_result: Dict, exercise_type: str = None) -> Dict:
        """enhanced와 동일한 후처리 적용"""
        if not analysis_result['valid']:
            return analysis_result
        
        # 위반 비율 사용
        violation_ratio = analysis_result.get('weighted_violation_ratio', 0)
        
        # EMA 적용
        smoothed_ratio = self.apply_ema(violation_ratio)
        
        # 히스토리 추가
        self.history.append(smoothed_ratio)
        
        # enhanced와 동일한 히스테리시스 적용
        final_classification = self.apply_hysteresis(smoothed_ratio, exercise_type)
        
        return {
            **analysis_result,
            'final_classification': final_classification,
            'violation_ratio': violation_ratio,
            'smoothed_ratio': smoothed_ratio,
            'confidence': 1.0 - smoothed_ratio,
            'enhanced_compatible': True,
            'processing_info': {
                'exercise_type': exercise_type,
                'hysteresis_threshold': self.exercise_hysteresis.get(exercise_type, self.hysteresis_threshold),
                'state_history': list(self.history)[-5:]
            }
        }

class DatasetProcessor:
    """데이터셋 처리 클래스 (enhanced 기준 적용)"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.classifier = ExerciseClassifier()
        
        # enhanced와 동일한 운동별 후처리기
        self.post_processors = {
            'squat': PostProcessor(hysteresis_threshold=0.5, ema_alpha=0.3),
            'push_up': PostProcessor(hysteresis_threshold=0.7, ema_alpha=0.3),
            'deadlift': PostProcessor(hysteresis_threshold=0.8, ema_alpha=0.3),  # 대폭 완화
            'bench_press': PostProcessor(hysteresis_threshold=0.5, ema_alpha=0.3),
            'lunge': PostProcessor(hysteresis_threshold=0.6, ema_alpha=0.3)
        }
        
        # 결과 저장 디렉토리 생성
        self.output_path = self.base_path / "processed_data"
        self.output_path.mkdir(exist_ok=True)
        
        # 운동별 디렉토리 생성
        self.exercises = ['bench_press', 'deadlift', 'pull_up', 'push_up', 'squat', 'lunge']
        for exercise in self.exercises:
            for category in ['good', 'bad']:
                (self.output_path / exercise / category).mkdir(parents=True, exist_ok=True)
    
    def process_exercise_images(self, exercise_name: str, image_dir: str, limit: int = 500):
        """특정 운동의 이미지들을 enhanced 기준으로 처리하고 분류"""
        print(f"\n=== {exercise_name} 처리 (enhanced 기준 적용) ===")
        
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
        
        # enhanced와 동일한 후처리기 사용
        post_processor = self.post_processors.get(exercise_name, 
                                                 PostProcessor(hysteresis_threshold=0.6))
        
        for i, img_file in enumerate(image_files):
            try:
                # 랜드마크 추출
                landmarks_data = self.classifier.extract_landmarks(str(img_file))
                if landmarks_data is None:
                    results['failed'] += 1
                    continue
                
                # enhanced와 동일한 자세 분석
                analysis = self.classifier.analyze_pose(
                    landmarks_data['landmarks'], 
                    exercise_name
                )
                
                if not analysis['valid']:
                    results['failed'] += 1
                    continue
                
                # enhanced와 동일한 후처리 적용
                final_result = post_processor.process(analysis, exercise_name)
                classification = final_result['final_classification']
                
                # 파일 복사
                dest_dir = self.output_path / exercise_name / classification
                dest_file = dest_dir / f"{classification}_{exercise_name}_{i:04d}_enhanced_compatible.jpg"
                shutil.copy2(img_file, dest_file)
                
                results[classification] += 1
                
                # 로그 저장
                log_entry = {
                    'original_file': str(img_file),
                    'classification': classification,
                    'angles': final_result['angles'],
                    'violations': final_result['violations'],
                    'confidence': final_result['confidence'],
                    'weighted_violation_ratio': final_result.get('weighted_violation_ratio', 0),
                    'enhanced_compatible': True
                }
                processing_log.append(log_entry)
                
                if (i + 1) % 50 == 0:
                    current_good_rate = (results['good'] / max(results['good'] + results['bad'], 1)) * 100
                    print(f"Processed {i + 1}/{len(image_files)} images - Good Rate: {current_good_rate:.1f}%")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                results['failed'] += 1
                continue
        
        # 최종 결과 출력
        total_processed = results['good'] + results['bad']
        good_rate = (results['good'] / max(total_processed, 1)) * 100
        
        print(f"✅ enhanced 기준 적용 결과 - Good: {results['good']}, Bad: {results['bad']}, Failed: {results['failed']}")
        print(f"🎯 Good 비율: {good_rate:.1f}%")
        
        # enhanced 목표와 비교
        target_rates = {
            'squat': '50-70%',
            'push_up': '50-70% (기존 유지)',
            'deadlift': '40-60% (대폭 완화)',
            'bench_press': '50-70%',
            'lunge': '50-70%'
        }
        
        target = target_rates.get(exercise_name, '50-70%')
        print(f"📊 목표: {target}")
        
        # 로그 저장
        log_file = self.output_path / f"{exercise_name}_enhanced_compatible_processing_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'exercise': exercise_name,
                'enhanced_compatible': True,
                'summary': results,
                'good_rate': good_rate,
                'target_rate': target,
                'detailed_log': processing_log
            }, f, indent=2, ensure_ascii=False)
        
        return results
    
    def process_all_exercises(self):
        """모든 운동 enhanced 기준으로 처리"""
        exercise_dirs = {
            'bench_press': 'bench_press_exercise',
            'deadlift': 'deadlift_exercise',
            'pull_up': 'pull_up_exercise',
            'push_up': 'push_up_exercise',
            'squat': 'squat_exercise',
            'lunge': 'lunge_exercise'
        }
        
        total_results = {}
        
        print("🎯 enhanced_pose_analysis.py 기준으로 모든 운동 처리 시작!")
        print("📊 적용된 기준:")
        print("  🏋️‍♀️ 스쿼트: 55-140도 (조정됨)")
        print("  💪 푸쉬업: 40-160도 (기존 유지)")
        print("  🏋️‍♂️ 데드리프트: 80-180도 (대폭 완화)")
        print("  🔥 벤치프레스: 50-145도 (조정됨)")
        print("  🚀 런지: 70-120도 (신규)")
        
        for exercise, directory in exercise_dirs.items():
            results = self.process_exercise_images(exercise, directory)
            total_results[exercise] = results
        
        # target_rates 정의
        target_rates = {
            'squat': '50-70%',
            'push_up': '50-70% (기존 유지)',
            'deadlift': '40-60% (대폭 완화)',
            'bench_press': '50-70%',
            'lunge': '50-70%',
            'pull_up': '50-70%'
        }
        
        # 전체 결과 저장
        summary_file = self.output_path / "enhanced_compatible_processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'enhanced_compatible': True,
                'source': 'enhanced_pose_analysis.py angles applied',
                'target_rates': target_rates,
                'results': total_results
            }, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*70)
        print("🎉 enhanced_pose_analysis.py 기준 적용 완료!")
        print("="*70)
        print("📊 최종 결과:")
        
        for exercise, results in total_results.items():
            total_processed = results['good'] + results['bad']
            if total_processed > 0:
                good_rate = (results['good'] / total_processed) * 100
                emoji_map = {
                    'squat': '🏋️‍♀️',
                    'push_up': '💪',
                    'deadlift': '🏋️‍♂️',
                    'bench_press': '🔥',
                    'lunge': '🚀',
                    'pull_up': '💯'
                }
                
                print(f"{emoji_map.get(exercise, '🏋️')} {exercise}: Good={results['good']}, Bad={results['bad']}, Failed={results['failed']}")
                print(f"         Good 비율: {good_rate:.1f}%")
        
        return total_results

def main():
    """메인 실행 함수 (enhanced 기준 적용)"""
    print("🎯 enhanced_pose_analysis.py 기준 적용된 Pose Analysis System")
    print("📊 모든 각도 기준이 enhanced_pose_analysis.py와 동일하게 설정됨")
    
    # 기본 경로 설정 (현재 디렉토리 기준)
    base_path = "."
    
    try:
        # 데이터셋 프로세서 초기화
        processor = DatasetProcessor(base_path)
        
        # 모든 운동 enhanced 기준으로 처리
        processor.process_all_exercises()
        
        print(f"\n💾 enhanced 호환 데이터 저장 위치: {processor.output_path}")
        print("🎯 enhanced_pose_analysis.py와 동일한 기준 적용 완료!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())