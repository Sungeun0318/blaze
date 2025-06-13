#!/usr/bin/env python3
"""
🤖 자동 운동 감지 + 각도 분석 통합 시스템
1단계: AI가 운동 종류 자동 감지
2단계: 감지된 운동에 맞춰 각도 분석 + 초록/빨강 화면 표시
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

class AutoExerciseAnalyzer:
    """자동 운동 감지 + 각도 분석 통합 시스템"""
    
    def __init__(self):
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
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
        self.load_exercise_model()
        
        # Enhanced 각도 기준
        self.exercise_thresholds = {
            'squat': {
                'left_knee': {'points': [23, 25, 27], 'range': (55, 140), 'weight': 1.1},
                'right_knee': {'points': [24, 26, 28], 'range': (55, 140), 'weight': 1.1},
                'left_hip': {'points': [11, 23, 25], 'range': (55, 140), 'weight': 0.9},
                'right_hip': {'points': [12, 24, 26], 'range': (55, 140), 'weight': 0.9},
                'back_straight': {'points': [11, 23, 25], 'range': (110, 170), 'weight': 1.1},
            },
            'push_up': {
                'left_elbow': {'points': [11, 13, 15], 'range': (40, 160), 'weight': 1.0},
                'right_elbow': {'points': [12, 14, 16], 'range': (40, 160), 'weight': 1.0},
                'body_line': {'points': [11, 23, 25], 'range': (140, 180), 'weight': 1.2},
                'leg_straight': {'points': [23, 25, 27], 'range': (140, 180), 'weight': 0.8},
            },
            'deadlift': {
                'left_knee': {'points': [23, 25, 27], 'range': (80, 140), 'weight': 0.6},
                'right_knee': {'points': [24, 26, 28], 'range': (80, 140), 'weight': 0.6},
                'hip_hinge': {'points': [11, 23, 25], 'range': (80, 180), 'weight': 0.7},
                'back_straight': {'points': [11, 23, 12], 'range': (120, 180), 'weight': 1.0},
            },
            'bench_press': {
                'left_elbow': {'points': [11, 13, 15], 'range': (50, 145), 'weight': 1.1},
                'right_elbow': {'points': [12, 14, 16], 'range': (50, 145), 'weight': 1.1},
                'left_shoulder': {'points': [13, 11, 23], 'range': (50, 150), 'weight': 0.9},
                'right_shoulder': {'points': [14, 12, 24], 'range': (50, 150), 'weight': 0.9},
            },
            'lunge': {
                'front_knee': {'points': [23, 25, 27], 'range': (70, 120), 'weight': 1.2},
                'back_knee': {'points': [24, 26, 28], 'range': (120, 180), 'weight': 1.0},
                'front_hip': {'points': [11, 23, 25], 'range': (70, 120), 'weight': 0.8},
                'torso_upright': {'points': [11, 23, 25], 'range': (100, 180), 'weight': 1.2},
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
        
        # 운동 이모지
        self.exercise_emojis = {
            'squat': '🏋️‍♀️',
            'push_up': '💪',
            'deadlift': '🏋️‍♂️',
            'bench_press': '🔥',
            'lunge': '🚀'
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
        self.color_transition_speed = 0.1
        
        # 타이밍
        self.last_classification_time = 0
        self.classification_interval = 2.0  # 2초마다 운동 분류
    
    def load_exercise_model(self):
        """AI 운동 분류 모델 로드"""
        model_path = "models/exercise_classifier.pkl"
        try:
            if os.path.exists(model_path):
                from exercise_classifier import ExerciseClassificationModel
                self.exercise_classifier = ExerciseClassificationModel()
                self.model_loaded = self.exercise_classifier.load_model(model_path)
                if self.model_loaded:
                    print("✅ AI 운동 분류 모델 로드 완료")
                else:
                    print("❌ 모델 로드 실패")
            else:
                print("⚠️ AI 모델 없음 - 수동 운동 선택 모드")
        except Exception as e:
            print(f"❌ 모델 로드 오류: {e}")
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
            temp_path = "temp_frame.jpg"
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
                            emoji = self.exercise_emojis.get(new_exercise, '🏋️')
                            print(f"🤖 AI 감지: {emoji} {new_exercise.upper()} (신뢰도: {confidence:.1%})")
            
            self.last_classification_time = current_time
            
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return self.current_exercise, self.exercise_confidence
            
        except Exception as e:
            print(f"운동 분류 오류: {e}")
            return self.current_exercise, self.exercise_confidence
    
    def analyze_pose_angles(self, landmarks, exercise: str) -> Dict:
        """🎯 2단계: 감지된 운동에 맞춰 각도 분석"""
        if exercise not in self.exercise_thresholds:
            return {'valid': False, 'error': f'지원되지 않는 운동: {exercise}'}
        
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
                    'in_range': min_angle <= angle <= max_angle
                }
                
                total_weight += weight
                
                if not (min_angle <= angle <= max_angle):
                    violations.append({
                        'joint': joint_name,
                        'angle': angle,
                        'expected_range': (min_angle, max_angle),
                        'weight': weight
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
    
    def draw_analysis_overlay(self, frame: np.ndarray, exercise: str, pose_result: Dict) -> np.ndarray:
        """분석 결과 화면 오버레이"""
        height, width = frame.shape[:2]
        
        # 🌈 전체 화면 색상 오버레이
        if pose_result.get('valid', False):
            pose_quality = pose_result['classification']
            self.update_screen_color(pose_quality)
            
            # 투명한 색상 오버레이
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), self.screen_color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        # 🎯 두꺼운 테두리
        border_thickness = 25
        cv2.rectangle(frame, (0, 0), (width, height), self.screen_color, border_thickness)
        
        # 📊 상단 정보 패널
        panel_height = 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # 🤖 1단계: AI 감지 결과
        if exercise != "detecting..." and exercise != "manual_mode":
            emoji = self.exercise_emojis.get(exercise, '🏋️')
            exercise_text = f"AI 감지: {emoji} {exercise.upper().replace('_', ' ')}"
            cv2.putText(frame, exercise_text, (30, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            confidence_text = f"분류 신뢰도: {self.exercise_confidence:.0%}"
            cv2.putText(frame, confidence_text, (30, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        elif exercise == "detecting...":
            cv2.putText(frame, "🤖 AI가 운동을 감지하는 중...", (30, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "수동 모드 - C키로 운동 선택", (30, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 🎯 2단계: 자세 분석 결과
        if pose_result.get('valid', False):
            pose_quality = pose_result['classification']
            pose_confidence = pose_result['confidence']
            
            # 중앙 상태 메시지
            if pose_quality == 'good':
                status_text = "완벽한 자세! 👍"
                status_color = (0, 255, 0)
            else:
                status_text = "자세 교정 필요 ⚠️"
                status_color = (0, 0, 255)
            
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0]
            status_x = (width - status_size[0]) // 2
            status_y = height // 2 - 50
            
            # 상태 배경
            cv2.rectangle(frame, (status_x - 30, status_y - 50), 
                         (status_x + status_size[0] + 30, status_y + 20), (0, 0, 0), -1)
            cv2.rectangle(frame, (status_x - 30, status_y - 50), 
                         (status_x + status_size[0] + 30, status_y + 20), status_color, 3)
            
            cv2.putText(frame, status_text, (status_x, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, status_color, 4)
            
            # 자세 신뢰도
            pose_text = f"자세 점수: {pose_confidence:.0%}"
            cv2.putText(frame, pose_text, (30, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # 📊 하단 통계
        if self.stats['frames'] > 0:
            total = self.stats['good'] + self.stats['bad']
            if total > 0:
                good_ratio = self.stats['good'] / total
                stats_text = f"Good: {self.stats['good']} | Bad: {self.stats['bad']} | 성공률: {good_ratio:.1%}"
                
                cv2.rectangle(frame, (0, height - 60), (width, height), (0, 0, 0), -1)
                cv2.putText(frame, stats_text, (30, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # ⌨️ 조작 가이드
        guide_text = "Q: 종료 | R: 리셋 | S: 스크린샷 | C: 운동 변경 | SPACE: 수동 모드"
        cv2.putText(frame, guide_text, (10, height - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return frame
    
    def run_realtime_analysis(self, camera_id: int = 0, manual_exercise: str = None):
        """🎮 실시간 자동 분석 실행"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ 카메라 {camera_id} 열기 실패")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*80)
        print("🤖 자동 운동 감지 + 각도 분석 시스템")
        print("="*80)
        print("✨ 기능:")
        print("  🤖 1단계: AI가 운동 종류 자동 감지")
        print("  🎯 2단계: 감지된 운동에 맞춰 각도 분석")
        print("  🌈 3단계: 실시간 초록/빨강 화면 피드백")
        print("  📊 4단계: 통계 및 성과 추적")
        print("\n⌨️ 조작법:")
        print("  Q: 종료 | R: 통계 리셋 | S: 스크린샷")
        print("  C: 수동 운동 선택 | SPACE: 자동/수동 모드 토글")
        print("="*80)
        
        if not self.model_loaded:
            print("⚠️ AI 모델 없음 - 수동 모드로 시작")
            if manual_exercise:
                self.current_exercise = manual_exercise
                print(f"수동 선택: {manual_exercise}")
        
        # 수동 운동 선택용
        available_exercises = list(self.exercise_thresholds.keys())
        manual_mode = not self.model_loaded
        current_manual_idx = 0
        
        if manual_exercise and manual_exercise in available_exercises:
            current_manual_idx = available_exercises.index(manual_exercise)
            self.current_exercise = manual_exercise
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)  # 셀카 모드
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # 🤖 1단계: AI 운동 감지 (자동 모드일 때만)
                    if not manual_mode and self.model_loaded:
                        exercise, confidence = self.classify_exercise(frame)
                    else:
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
                            
                            # 🌈 3단계: 화면 오버레이
                            frame = self.draw_analysis_overlay(frame, exercise, pose_result)
                        else:
                            frame = self.draw_analysis_overlay(frame, exercise, {'valid': False})
                    else:
                        frame = self.draw_analysis_overlay(frame, exercise, {'valid': False})
                else:
                    # 포즈 미감지
                    cv2.putText(frame, "카메라 앞에 서서 전신이 보이도록 해주세요", 
                               (frame.shape[1]//2 - 300, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # 화면 출력
                window_title = "🤖 Auto Exercise Detection + Pose Analysis"
                cv2.imshow(window_title, frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # 통계 리셋
                    self.stats = {'good': 0, 'bad': 0, 'frames': 0}
                    self.exercise_history.clear()
                    self.pose_history.clear()
                    print("📊 통계 리셋 완료")
                elif key == ord('s'):
                    # 스크린샷
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"auto_analysis_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 스크린샷 저장: {filename}")
                elif key == ord('c'):
                    # 수동 운동 변경
                    current_manual_idx = (current_manual_idx + 1) % len(available_exercises)
                    self.current_exercise = available_exercises[current_manual_idx]
                    emoji = self.exercise_emojis.get(self.current_exercise, '🏋️')
                    print(f"🔄 수동 선택: {emoji} {self.current_exercise}")
                elif key == ord(' '):
                    # 자동/수동 모드 토글
                    manual_mode = not manual_mode
                    mode = "수동" if manual_mode else "자동"
                    print(f"🔄 {mode} 모드로 변경")
        
        except KeyboardInterrupt:
            print("\n⏹️ 사용자 중단")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 최종 통계
            total = self.stats['good'] + self.stats['bad']
            if total > 0:
                print(f"\n📊 최종 통계:")
                print(f"  🎯 총 분석: {total} 프레임")
                print(f"  ✅ Good: {self.stats['good']} ({self.stats['good']/total:.1%})")
                print(f"  ❌ Bad: {self.stats['bad']} ({self.stats['bad']/total:.1%})")
            
            return True
    
    def analyze_single_image(self, image_path: str) -> Dict:
        """단일 이미지 자동 분석"""
        if not os.path.exists(image_path):
            return {'error': f'이미지 파일을 찾을 수 없습니다: {image_path}'}
        
        print(f"🤖 자동 이미지 분석: {image_path}")
        
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            return {'error': '이미지를 읽을 수 없습니다'}
        
        # 포즈 검출
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return {'error': '포즈를 검출할 수 없습니다'}
        
        # 🤖 1단계: AI 운동 감지
        exercise, confidence = self.classify_exercise(image)
        print(f"🎯 AI 감지: {exercise} (신뢰도: {confidence:.1%})")
        
        # 🎯 2단계: 각도 분석
        if exercise in self.exercise_thresholds:
            pose_result = self.analyze_pose_angles(results.pose_landmarks.landmark, exercise)
            
            # 결과 합치기
            return {
                'success': True,
                'detected_exercise': exercise,
                'exercise_confidence': confidence,
                'pose_analysis': pose_result,
                'image_path': image_path
            }
        else:
            return {'error': f'지원되지 않는 운동: {exercise}'}

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='🤖 자동 운동 감지 + 각도 분석 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 사용 예시:
  python auto_exercise_analyzer.py --mode realtime              # 실시간 자동 분석
  python auto_exercise_analyzer.py --mode realtime --camera 1   # 다른 카메라
  python auto_exercise_analyzer.py --mode image --input photo.jpg  # 단일 이미지 분석
  python auto_exercise_analyzer.py --mode realtime --manual squat  # 수동 운동 선택

🤖 시스템 특징:
  1단계: AI가 운동 종류 자동 감지 (스쿼트, 푸쉬업, 데드리프트, 벤치프레스, 런지)
  2단계: 감지된 운동에 맞춰 정확한 각도 분석
  3단계: 실시간 초록(Good)/빨강(Bad) 화면 피드백
  4단계: 통계 및 성과 추적

⌨️ 실시간 조작:
  Q: 종료  |  R: 통계 리셋  |  S: 스크린샷
  C: 수동 운동 변경  |  SPACE: 자동/수동 모드 토글
        """
    )
    
    parser.add_argument('--mode', type=str, default='realtime',
                       choices=['realtime', 'image'],
                       help='분석 모드')
    parser.add_argument('--camera', type=int, default=0,
                       help='카메라 ID')
    parser.add_argument('--input', type=str,
                       help='입력 이미지 파일 (image 모드용)')
    parser.add_argument('--manual', type=str,
                       choices=['squat', 'push_up', 'deadlift', 'bench_press', 'lunge'],
                       help='수동 운동 선택 (AI 감지 건너뛰기)')
    
    args = parser.parse_args()
    
    # 자동 분석기 초기화
    try:
        analyzer = AutoExerciseAnalyzer()
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        return 1
    
    print("🤖 자동 운동 감지 + 각도 분석 시스템 시작!")
    print("="*60)
    print("🎯 기능:")
    print("  🤖 AI 자동 운동 감지")
    print("  📐 Enhanced 각도 분석")
    print("  🌈 실시간 초록/빨강 피드백")
    print("  📊 성과 추적")
    
    try:
        if args.mode == 'realtime':
            print(f"\n🎥 실시간 분석 시작 (카메라 {args.camera})")
            if args.manual:
                print(f"🔧 수동 모드: {args.manual}")
            success = analyzer.run_realtime_analysis(args.camera, args.manual)
            return 0 if success else 1
            
        elif args.mode == 'image':
            if not args.input:
                print("❌ --input 옵션이 필요합니다")
                return 1
            
            print(f"\n🖼️ 이미지 분석 시작: {args.input}")
            result = analyzer.analyze_single_image(args.input)
            
            if result.get('success', False):
                exercise = result['detected_exercise']
                exercise_conf = result['exercise_confidence']
                pose_result = result['pose_analysis']
                
                emoji = analyzer.exercise_emojis.get(exercise, '🏋️')
                print(f"\n🎉 분석 완료!")
                print(f"🤖 AI 감지: {emoji} {exercise.upper()} (신뢰도: {exercise_conf:.1%})")
                
                if pose_result['valid']:
                    pose_quality = pose_result['classification']
                    pose_conf = pose_result['confidence']
                    
                    status_emoji = "✅" if pose_quality == 'good' else "⚠️"
                    print(f"🎯 자세 분석: {status_emoji} {pose_quality.upper()} (점수: {pose_conf:.1%})")
                    
                    if pose_result['violations']:
                        print(f"📐 개선 필요한 부분:")
                        for violation in pose_result['violations'][:3]:
                            joint = violation['joint'].replace('_', ' ').title()
                            angle = violation['angle']
                            range_min, range_max = violation['expected_range']
                            print(f"  • {joint}: {angle:.1f}° → {range_min:.0f}-{range_max:.0f}°")
                else:
                    print("❌ 자세 분석 실패")
            else:
                print(f"❌ 분석 실패: {result.get('error', '알 수 없는 오류')}")
                return 1
    
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
        return 0
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())