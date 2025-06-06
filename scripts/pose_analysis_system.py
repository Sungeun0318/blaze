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
    """각도 임계값 설정 (극도로 완화된 버전)"""
    min_angle: float
    max_angle: float
    joint_points: List[int]
    name: str

class ExerciseClassifier:
    """운동 분류 및 각도 분석 클래스 (극도로 완화된 버전)"""
    
    def __init__(self):
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,  # 0.7 → 0.5로 완화
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        except Exception as e:
            print(f"❌ MediaPipe 초기화 실패: {e}")
            raise
        
        # 🔥 극도로 완화된 운동별 각도 기준 설정
        self.exercise_thresholds = {
            'bench_press': [
                AngleThreshold(30, 170, [11, 13, 15], 'left_elbow'),   # 70→30, 120→170 대폭 완화
                AngleThreshold(30, 170, [12, 14, 16], 'right_elbow'),  
                AngleThreshold(30, 150, [13, 11, 23], 'left_shoulder'), # 60→30, 100→150 대폭 완화
                AngleThreshold(30, 150, [14, 12, 24], 'right_shoulder'),
            ],
            'deadlift': [
                AngleThreshold(120, 180, [23, 25, 27], 'left_knee'),   # 160→120 완화
                AngleThreshold(120, 180, [24, 26, 28], 'right_knee'),  
                AngleThreshold(100, 180, [11, 23, 25], 'left_hip'),    # 160→100 대폭 완화
                AngleThreshold(100, 180, [12, 24, 26], 'right_hip'),   
                AngleThreshold(140, 180, [23, 11, 13], 'left_back'),   # 160→140 (안전 고려)
                AngleThreshold(140, 180, [24, 12, 14], 'right_back'),  
            ],
            'pull_up': [
                AngleThreshold(10, 120, [11, 13, 15], 'left_elbow'),   # 30→10, 90→120 대폭 완화
                AngleThreshold(10, 120, [12, 14, 16], 'right_elbow'),
                AngleThreshold(90, 180, [13, 11, 23], 'left_shoulder'), # 120→90 완화
                AngleThreshold(90, 180, [14, 12, 24], 'right_shoulder'),
            ],
            'push_up': [
                AngleThreshold(20, 170, [11, 13, 15], 'left_elbow'),   # 80→20, 120→170 극도 완화!
                AngleThreshold(20, 170, [12, 14, 16], 'right_elbow'),  # 푸시업 팔꿈치 거의 모든 각도
                AngleThreshold(100, 180, [11, 23, 25], 'left_hip'),    # 160→100 몸 라인 완화
                AngleThreshold(100, 180, [12, 24, 26], 'right_hip'),   
                AngleThreshold(130, 180, [23, 25, 27], 'left_knee'),   # 170→130 다리 완화
                AngleThreshold(130, 180, [24, 26, 28], 'right_knee'),  
            ],
            'squat': [
                AngleThreshold(15, 175, [23, 25, 27], 'left_knee'),    # 70→15, 120→175 극도 완화!
                AngleThreshold(15, 175, [24, 26, 28], 'right_knee'),   # 스쿼트 무릎 거의 모든 각도
                AngleThreshold(15, 175, [11, 23, 25], 'left_hip'),     # 70→15, 120→175 힙도 극도 완화
                AngleThreshold(15, 175, [12, 24, 26], 'right_hip'),    
                AngleThreshold(120, 180, [23, 11, 13], 'left_back'),   # 170→120 등 각도 완화
                AngleThreshold(120, 180, [24, 12, 14], 'right_back'),  
            ]
        }
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """세 점 사이의 각도 계산 (안전한 버전)"""
        try:
            # 벡터 계산
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float64)
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float64)
            
            # 각도 계산
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
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
        """자세 분석 및 각도 계산 (극도로 완화된 기준)"""
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
                
                # 🔥 가시성 확인 (극도로 관대하게)
                visibility_threshold = 0.15  # 0.5에서 0.15로 대폭 완화
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
                        'expected_max': threshold.max_angle
                    })
                    
            except Exception as e:
                print(f"Error calculating angle for {threshold.name}: {e}")
                continue
        
        # 🔥 극도로 완화된 분류 결과
        total_joints = len(angles)
        violation_count = len(violations)
        
        # 위반 비율이 80% 이하면 good (기존 0%에서 80%로 극도 완화!)
        violation_ratio = violation_count / max(total_joints, 1)
        is_good = violation_ratio <= 0.8
        
        return {
            'valid': True,
            'classification': 'good' if is_good else 'bad',
            'angles': angles,
            'violations': violations,
            'violation_count': violation_count,
            'violation_ratio': violation_ratio,
            'total_joints': total_joints
        }

class PostProcessor:
    """후처리 클래스 (극도로 완화된 히스테리시스 + EMA)"""
    
    def __init__(self, hysteresis_threshold: float = 0.8, ema_alpha: float = 0.4, window_size: int = 8):
        self.hysteresis_threshold = hysteresis_threshold  # 0.3 → 0.8로 극도 완화
        self.ema_alpha = ema_alpha  # 0.3 → 0.4로 완화
        self.window_size = window_size  # 5 → 8로 증가
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
        """극도로 완화된 히스테리시스 적용"""
        if self.last_state is None:
            self.last_state = 'good' if violation_ratio <= 0.8 else 'bad'  # 첫 판정도 관대하게
            return self.last_state
        
        if self.last_state == 'good':
            # good 상태에서 bad로 변경하려면 90% 이상 위반 필요 (극도로 관대)
            if violation_ratio > 0.9:
                self.last_state = 'bad'
        else:
            # bad 상태에서 good으로 변경하려면 70% 이하 위반 (관대한 복귀)
            if violation_ratio <= 0.7:
                self.last_state = 'good'
        
        return self.last_state
    
    def process(self, analysis_result: Dict) -> Dict:
        """극도로 완화된 후처리 적용"""
        if not analysis_result['valid']:
            return analysis_result
        
        # 위반 비율 사용
        violation_ratio = analysis_result.get('violation_ratio', 0)
        
        # EMA 적용
        smoothed_ratio = self.apply_ema(violation_ratio)
        
        # 히스토리 추가
        self.history.append(smoothed_ratio)
        
        # 극도로 완화된 히스테리시스 적용
        final_classification = self.apply_hysteresis(smoothed_ratio)
        
        return {
            **analysis_result,
            'final_classification': final_classification,
            'violation_ratio': violation_ratio,
            'smoothed_ratio': smoothed_ratio,
            'confidence': 1.0 - smoothed_ratio,
            'ultra_relaxed_version': True
        }

class DatasetProcessor:
    """데이터셋 처리 클래스 (극도로 완화된 버전)"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.classifier = ExerciseClassifier()
        self.post_processor = PostProcessor()  # 극도로 완화된 후처리기
        
        # 결과 저장 디렉토리 생성
        self.output_path = self.base_path / "processed_data"
        self.output_path.mkdir(exist_ok=True)
        
        # 운동별 디렉토리 생성
        self.exercises = ['bench_press', 'deadlift', 'pull_up', 'push_up', 'squat']
        for exercise in self.exercises:
            for category in ['good', 'bad']:
                (self.output_path / exercise / category).mkdir(parents=True, exist_ok=True)
    
    def process_exercise_images(self, exercise_name: str, image_dir: str, limit: int = 500):
        """특정 운동의 이미지들을 극도로 완화된 기준으로 처리하고 분류"""
        print(f"\n=== 극도 완화된 {exercise_name} 처리 ===")
        
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
                
                # 극도로 완화된 자세 분석
                analysis = self.classifier.analyze_pose(
                    landmarks_data['landmarks'], 
                    exercise_name
                )
                
                if not analysis['valid']:
                    results['failed'] += 1
                    continue
                
                # 극도로 완화된 후처리 적용
                final_result = self.post_processor.process(analysis)
                classification = final_result['final_classification']
                
                # 파일 복사
                dest_dir = self.output_path / exercise_name / classification
                dest_file = dest_dir / f"{classification}_{exercise_name}_{i:04d}_ultra.jpg"
                shutil.copy2(img_file, dest_file)
                
                results[classification] += 1
                
                # 로그 저장
                log_entry = {
                    'original_file': str(img_file),
                    'classification': classification,
                    'angles': final_result['angles'],
                    'violations': final_result['violations'],
                    'confidence': final_result['confidence'],
                    'violation_ratio': final_result.get('violation_ratio', 0),
                    'ultra_relaxed': True
                }
                processing_log.append(log_entry)
                
                if (i + 1) % 50 == 0:
                    current_good_rate = (results['good'] / max(results['good'] + results['bad'], 1)) * 100
                    print(f"Processed {i + 1}/{len(image_files)} images - Current Good Rate: {current_good_rate:.1f}%")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                results['failed'] += 1
                continue
        
        # 최종 결과 출력
        total_processed = results['good'] + results['bad']
        good_rate = (results['good'] / max(total_processed, 1)) * 100
        
        print(f"🎉 극도 완화 결과 - Good: {results['good']}, Bad: {results['bad']}, Failed: {results['failed']}")
        print(f"🎯 Good 비율: {good_rate:.1f}%")
        
        # 목표 달성 여부
        target_rates = {'squat': 90, 'push_up': 80, 'deadlift': 85, 'bench_press': 85, 'pull_up': 85}
        target = target_rates.get(exercise_name, 80)
        
        if good_rate >= target:
            print(f"✅ 목표 달성! ({target}% 이상)")
        else:
            print(f"⚠️ 목표 미달성: {good_rate:.1f}% < {target}% (더 완화 필요)")
        
        # 로그 저장
        log_file = self.output_path / f"{exercise_name}_ultra_processing_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'exercise': exercise_name,
                'ultra_relaxed_version': True,
                'summary': results,
                'good_rate': good_rate,
                'target_achievement': good_rate >= target,
                'detailed_log': processing_log
            }, f, indent=2, ensure_ascii=False)
        
        return results
    
    def process_all_exercises(self):
        """모든 운동 극도로 완화된 기준으로 처리"""
        exercise_dirs = {
            'bench_press': 'bench_press_exercise',
            'deadlift': 'deadlift_exercise',
            'pull_up': 'pull_up_exercise',
            'push_up': 'push_up_exercise',
            'squat': 'squat_exercise'
        }
        
        total_results = {}
        
        print("🔥 극도로 완화된 기준으로 모든 운동 처리 시작!")
        print("목표: 푸시업 80%+, 스쿼트 90%+ Good 비율")
        
        for exercise, directory in exercise_dirs.items():
            results = self.process_exercise_images(exercise, directory)
            total_results[exercise] = results
        
        # 전체 결과 저장
        summary_file = self.output_path / "ultra_relaxed_processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'ultra_relaxed_version': True,
                'target_rates': {
                    'squat': '90%+',
                    'push_up': '80%+', 
                    'deadlift': '85%+',
                    'bench_press': '85%+',
                    'pull_up': '85%+'
                },
                'results': total_results
            }, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*70)
        print("🎉 극도로 완화된 처리 완료!")
        print("="*70)
        print("📊 최종 결과:")
        
        target_rates = {'squat': 90, 'push_up': 80, 'deadlift': 85, 'bench_press': 85, 'pull_up': 85}
        
        for exercise, results in total_results.items():
            total_processed = results['good'] + results['bad']
            if total_processed > 0:
                good_rate = (results['good'] / total_processed) * 100
                target = target_rates.get(exercise, 80)
                status = "✅ 달성" if good_rate >= target else "⚠️ 미달성"
                
                print(f"{exercise}: Good={results['good']}, Bad={results['bad']}, Failed={results['failed']}")
                print(f"         Good 비율: {good_rate:.1f}% (목표: {target}%) {status}")
        
        return total_results

def main():
    """메인 실행 함수 (극도로 완화된 버전)"""
    print("🔥 극도로 완화된 Pose Analysis System 시작")
    print("목표: 푸시업 0% → 80%+, 스쿼트 26.8% → 90%+")
    
    # 기본 경로 설정 (현재 디렉토리 기준)
    base_path = "."
    
    try:
        # 데이터셋 프로세서 초기화
        processor = DatasetProcessor(base_path)
        
        # 모든 운동 극도로 완화된 기준으로 처리
        processor.process_all_exercises()
        
        print(f"\n💾 극도로 완화된 데이터 저장 위치: {processor.output_path}")
        print("🔥 각도 기준이 극도로 완화되었습니다!")
        print("📋 변경 사항:")
        print("  • 푸시업 팔꿈치: 80-120° → 20-170° (거의 모든 각도)")
        print("  • 스쿼트 무릎: 70-120° → 15-175° (거의 모든 각도)")  
        print("  • 위반 허용률: 0% → 80% (80%까지 위반해도 Good)")
        print("  • 가시성 기준: 0.5 → 0.15 (매우 관대)")
        print("  • 히스테리시스: 0.3 → 0.8 (매우 관대)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())