"""
향상된 실시간 운동 분석기 - 5종목 완전 지원 (최종 완성본)
스쿼트, 푸시업, 데드리프트, 벤치프레스, 풀업
- 뷰 타입 자동 감지 (측면/정면/후면)
- 운동별 맞춤 각도 기준
- 강화된 시각적 피드백 (초록/빨강 화면)
- 완화된 히스테리시스 기반 안정화
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import json
from typing import Dict, List, Tuple, Optional
import argparse
import tempfile
import os
import sys
from pathlib import Path

def import_modules():
    """필요한 모듈들 import"""
    try:
        from exercise_classifier import ExerciseClassificationModel
        from enhanced_pose_analysis import EnhancedExerciseClassifier, AdaptivePostProcessor
        return ExerciseClassificationModel, EnhancedExerciseClassifier, AdaptivePostProcessor
    except ImportError as e:
        print(f"❌ 모듈 import 오류: {e}")
        print("필요한 파일들이 있는지 확인하세요:")
        print("  - exercise_classifier.py")
        print("  - enhanced_pose_analysis.py")
        return None, None, None

class Enhanced5ExerciseRealTimeAnalyzer:
    """향상된 5종목 실시간 분석기 (완화된 버전)"""
    
    def __init__(self, model_path: str = "models/exercise_classifier.pkl"):
        # 모듈 import
        ExerciseClassificationModel, EnhancedExerciseClassifier, AdaptivePostProcessor = import_modules()
        if not all([ExerciseClassificationModel, EnhancedExerciseClassifier, AdaptivePostProcessor]):
            self.init_success = False
            return
        
        # MediaPipe 초기화
        try:
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
        except Exception as e:
            print(f"❌ MediaPipe 초기화 실패: {e}")
            self.init_success = False
            return
        
        # 운동 분류 모델 로드
        self.exercise_classifier = ExerciseClassificationModel()
        self.model_loaded = False
        
        if os.path.exists(model_path):
            self.model_loaded = self.exercise_classifier.load_model(model_path)
            if self.model_loaded:
                print(f"✅ 5종목 운동 분류 모델 로드 완료: {model_path}")
            else:
                print(f"❌ 모델 로드 실패: {model_path}")
        else:
            print(f"❌ 모델 파일 없음: {model_path}")
            print("먼저 모델을 훈련하세요: python main.py --mode train")
        
        # 향상된 자세 분석기 초기화 (완화된 버전)
        try:
            self.pose_analyzer = EnhancedExerciseClassifier()
            # 5종목별 후처리기 (더 관대한 설정)
            self.post_processors = {
                'squat': AdaptivePostProcessor(hysteresis_threshold=0.4, ema_alpha=0.3),      # 더 관대
                'push_up': AdaptivePostProcessor(hysteresis_threshold=0.3, ema_alpha=0.25),   # 완화됨
                'deadlift': AdaptivePostProcessor(hysteresis_threshold=0.5, ema_alpha=0.35),  # 가장 관대
                'bench_press': AdaptivePostProcessor(hysteresis_threshold=0.35, ema_alpha=0.28),
                'pull_up': AdaptivePostProcessor(hysteresis_threshold=0.4, ema_alpha=0.32)
            }
        except Exception as e:
            print(f"❌ 자세 분석기 초기화 실패: {e}")
            self.init_success = False
            return
        
        # 5종목 지원
        self.available_exercises = ['squat', 'push_up', 'deadlift', 'bench_press', 'pull_up']
        self.current_exercise = "detecting..."
        self.current_pose_quality = "unknown"
        self.current_view_type = "unknown"
        self.classification_confidence = 0.0
        self.pose_confidence = 0.0
        
        # 안정화를 위한 히스토리
        self.exercise_history = deque(maxlen=15)  # 운동 분류 안정화
        self.pose_history = deque(maxlen=8)       # 자세 분석 안정화
        
        # 타이밍 제어
        self.last_classification_time = 0
        self.classification_interval = 3.0  # 3초마다 운동 분류
        self.last_feedback_time = 0
        self.feedback_interval = 1.5  # 1.5초마다 피드백 업데이트
        
        # 5종목 통계
        self.session_stats = {
            'good_count': 0,
            'bad_count': 0,
            'total_frames': 0,
            'view_distribution': {'side_view': 0, 'front_view': 0, 'back_view': 0},
            'exercise_distribution': {ex: 0 for ex in self.available_exercises}
        }
        
        # 피드백 메시지 큐
        self.feedback_messages = deque(maxlen=3)
        
        # 임시 파일 디렉토리
        self.temp_dir = tempfile.mkdtemp()
        
        # 화면 상태 (부드러운 전환을 위함)
        self.screen_state = "neutral"  # neutral, good, bad, detecting
        self.state_transition_frames = 0
        self.transition_duration = 10  # 10프레임 동안 전환
        
        self.init_success = True
    
    def classify_current_exercise(self, frame: np.ndarray) -> Tuple[str, float]:
        """현재 프레임의 5종목 운동 분류"""
        current_time = time.time()
        
        # 분류 주기 제어 (3초마다)
        if current_time - self.last_classification_time < self.classification_interval:
            return self.current_exercise, self.classification_confidence
        
        if not self.model_loaded:
            return "model_not_loaded", 0.0
        
        try:
            # 임시 이미지 파일로 저장
            temp_path = os.path.join(self.temp_dir, "current_frame.jpg")
            cv2.imwrite(temp_path, frame)
            
            # 운동 분류
            exercise, confidence = self.exercise_classifier.predict(temp_path)
            
            # 히스토리에 추가
            self.exercise_history.append((exercise, confidence))
            
            # 안정화: 최근 결과들의 합의
            if len(self.exercise_history) >= 5:
                # 신뢰도 0.4 이상인 예측들만 고려
                recent_predictions = [(ex, conf) for ex, conf in list(self.exercise_history)[-10:] 
                                    if conf > 0.4]
                
                if recent_predictions:
                    # 가중 평균으로 최종 결정
                    exercise_scores = {}
                    for ex, conf in recent_predictions:
                        if ex not in exercise_scores:
                            exercise_scores[ex] = []
                        exercise_scores[ex].append(conf)
                    
                    # 각 운동의 평균 신뢰도 계산
                    avg_scores = {}
                    for ex, scores in exercise_scores.items():
                        avg_scores[ex] = np.mean(scores)
                    
                    # 가장 높은 평균 신뢰도를 가진 운동 선택
                    if avg_scores:
                        best_exercise = max(avg_scores, key=avg_scores.get)
                        if avg_scores[best_exercise] > 0.5 and len(exercise_scores[best_exercise]) >= 3:
                            if best_exercise != self.current_exercise:
                                self.current_exercise = best_exercise
                                self.classification_confidence = avg_scores[best_exercise]
                                print(f"🎯 운동 감지: {best_exercise.upper()} (신뢰도: {avg_scores[best_exercise]:.2f})")
                                
                                # 통계 업데이트
                                self.session_stats['exercise_distribution'][best_exercise] += 1
            
            self.last_classification_time = current_time
            
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return self.current_exercise, self.classification_confidence
            
        except Exception as e:
            print(f"운동 분류 오류: {e}")
            return self.current_exercise, self.classification_confidence
    
    def analyze_pose_quality(self, landmarks) -> Tuple[str, Dict]:
        """향상된 5종목 자세 품질 분석"""
        if self.current_exercise in ["detecting...", "model_not_loaded", "unknown"]:
            return "waiting", {}
        
        try:
            # 랜드마크를 딕셔너리 형태로 변환
            landmarks_dict = []
            for landmark in landmarks:
                landmarks_dict.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            # 향상된 자세 분석
            analysis = self.pose_analyzer.analyze_pose(landmarks_dict, self.current_exercise)
            if not analysis['valid']:
                return "invalid", {}
            
            # 뷰 타입 업데이트
            self.current_view_type = analysis.get('view_type', 'unknown')
            self.session_stats['view_distribution'][self.current_view_type] += 1
            
            # 해당 운동의 후처리기 선택
            post_processor = self.post_processors.get(self.current_exercise, 
                                                     self.post_processors['squat'])
            
            # 후처리 적용
            final_result = post_processor.process(analysis, self.current_exercise)
            pose_quality = final_result['final_classification']
            self.pose_confidence = final_result.get('confidence', 0.0)
            
            # 안정화된 자세 품질 결정 (더 관대하게)
            self.pose_history.append(pose_quality)
            
            if len(self.pose_history) >= 3:
                recent_poses = list(self.pose_history)[-5:]
                good_count = recent_poses.count('good')
                bad_count = recent_poses.count('bad')
                
                # 다수결 원칙 + 히스테리시스 (good에 더 많은 가중치)
                if good_count >= bad_count * 1.2:  # good에 더 많은 가중치 (기존 1.5에서 1.2로 완화)
                    self.current_pose_quality = 'good'
                else:
                    self.current_pose_quality = 'bad'
            else:
                self.current_pose_quality = pose_quality
            
            # 통계 업데이트
            self.session_stats['total_frames'] += 1
            if self.current_pose_quality == 'good':
                self.session_stats['good_count'] += 1
            elif self.current_pose_quality == 'bad':
                self.session_stats['bad_count'] += 1
            
            # 피드백 메시지 생성
            self.generate_feedback_messages(final_result)
            
            return self.current_pose_quality, final_result
            
        except Exception as e:
            print(f"자세 분석 오류: {e}")
            return "error", {}
    
    def generate_feedback_messages(self, analysis_result: Dict):
        """5종목별 맞춤 피드백 메시지 생성"""
        current_time = time.time()
        
        # 피드백 주기 제한 (1.5초마다)
        if current_time - self.last_feedback_time < self.feedback_interval:
            return
        
        if not analysis_result.get('valid', False):
            return
        
        violations = analysis_result.get('violations', [])
        view_type = analysis_result.get('view_type', 'unknown')
        
        # 5종목별 맞춤 메시지
        exercise_messages = {
            'squat': {
                'good': ["완벽한 스쿼트!", "깊이가 훌륭해요!", "무릎 각도 완벽!", "자세 유지 잘하고 있어요!"],
                'bad': {
                    'knee': "무릎이 발끝을 넘지 않게 주의하세요",
                    'hip': "엉덩이를 더 뒤로 빼세요",
                    'back': "등을 곧게 펴세요",
                    'depth': "더 깊이 앉아보세요"
                }
            },
            'push_up': {
                'good': ["완벽한 푸시업!", "몸이 일직선이에요!", "팔꿈치 각도 좋아요!", "코어가 안정적!"],
                'bad': {
                    'elbow': "팔꿈치를 몸에 더 가깝게",
                    'body_line': "몸을 일직선으로 유지하세요",
                    'hip': "엉덩이가 너무 높아요",
                    'depth': "가슴을 바닥에 더 가깝게"
                }
            },
            'deadlift': {
                'good': ["완벽한 데드리프트!", "등이 곧고 좋아요!", "무릎 위치 완벽!", "엉덩이 힌지 훌륭!"],
                'bad': {
                    'back': "등을 곧게 펴세요 - 가장 중요!",
                    'knee': "무릎을 약간 구부리세요",
                    'hip': "엉덩이를 뒤로 더 빼세요",
                    'chest': "가슴을 펴고 어깨를 뒤로"
                }
            },
            'bench_press': {
                'good': ["완벽한 벤치프레스!", "팔꿈치 각도 최적!", "어깨 안정적!", "가동범위 훌륭!"],
                'bad': {
                    'elbow': "팔꿈치 각도를 조정하세요",
                    'shoulder': "어깨를 안정적으로 유지",
                    'arch': "자연스러운 등 아치 유지",
                    'path': "바벨 경로를 일정하게"
                }
            },
            'pull_up': {
                'good': ["완벽한 풀업!", "광배근 활성화 좋음!", "턱이 바 위로!", "몸이 안정적!"],
                'bad': {
                    'elbow': "팔꿈치를 완전히 펴세요",
                    'shoulder': "어깨를 아래로 당기세요",
                    'body': "몸의 흔들림을 줄이세요",
                    'range': "풀 레인지로 동작하세요"
                }
            }
        }
        
        # 메시지 생성
        if len(violations) == 0:
            # 좋은 자세
            good_messages = exercise_messages.get(self.current_exercise, {}).get('good', ["좋은 자세!"])
            import random
            message = random.choice(good_messages)
            self.feedback_messages.append(('good', message, view_type))
        else:
            # 나쁜 자세 - 가장 중요한 위반사항만 선택
            bad_messages = exercise_messages.get(self.current_exercise, {}).get('bad', {})
            
            # 가중치가 높은 위반사항 우선
            violations.sort(key=lambda x: x.get('weight', 1.0), reverse=True)
            
            for violation in violations[:2]:  # 최대 2개만
                joint = violation['joint']
                
                # 관절명을 메시지 키로 매핑
                message_key = self.map_joint_to_message_key(joint)
                message = bad_messages.get(message_key, f"{joint} 자세를 확인하세요")
                
                self.feedback_messages.append(('bad', message, view_type))
        
        self.last_feedback_time = current_time
    
    def map_joint_to_message_key(self, joint: str) -> str:
        """관절명을 메시지 키로 매핑 (5종목 지원)"""
        mapping = {
            'left_knee': 'knee', 'right_knee': 'knee',
            'left_hip': 'hip', 'right_hip': 'hip',
            'left_elbow': 'elbow', 'right_elbow': 'elbow',
            'left_shoulder': 'shoulder', 'right_shoulder': 'shoulder',
            'back_straight': 'back', 'left_back': 'back', 'right_back': 'back',
            'body_line': 'body_line', 'spine_straight': 'back',
            'chest_up': 'chest', 'back_arch': 'arch',
            'grip_symmetry': 'grip', 'lat_engagement': 'lat',
            'core_stability': 'body', 'body_straight': 'body'
        }
        return mapping.get(joint, joint.split('_')[0])
    
    def update_screen_state(self, target_state: str):
        """화면 상태 부드럽게 전환"""
        if target_state != self.screen_state:
            self.screen_state = target_state
            self.state_transition_frames = 0
        
        if self.state_transition_frames < self.transition_duration:
            self.state_transition_frames += 1
    
    def get_border_color(self) -> Tuple[int, int, int]:
        """현재 상태에 따른 테두리 색상 계산"""
        colors = {
            'good': (0, 255, 0),      # 초록색
            'bad': (0, 0, 255),       # 빨간색
            'detecting': (255, 255, 0), # 노란색
            'neutral': (128, 128, 128)  # 회색
        }
        
        base_color = colors.get(self.screen_state, colors['neutral'])
        
        # 전환 효과 적용
        if self.state_transition_frames < self.transition_duration:
            # 부드러운 전환을 위한 알파 블렌딩
            alpha = self.state_transition_frames / self.transition_duration
            neutral_color = colors['neutral']
            
            blended_color = (
                int(neutral_color[0] * (1 - alpha) + base_color[0] * alpha),
                int(neutral_color[1] * (1 - alpha) + base_color[1] * alpha),
                int(neutral_color[2] * (1 - alpha) + base_color[2] * alpha)
            )
            return blended_color
        
        return base_color
    
    def draw_enhanced_feedback(self, image: np.ndarray) -> np.ndarray:
        """5종목 강화된 시각적 피드백"""
        height, width = image.shape[:2]
        
        # 현재 상태 결정
        if self.current_exercise in ["detecting...", "model_not_loaded"]:
            target_state = "detecting"
            main_status = "운동을 감지하는 중..."
        elif self.current_pose_quality == "good":
            target_state = "good"
            main_status = "완벽한 자세! 👍"
        elif self.current_pose_quality == "bad":
            target_state = "bad"
            main_status = "자세를 교정하세요 ⚠️"
        else:
            target_state = "neutral"
            main_status = "분석 중..."
        
        # 화면 상태 업데이트
        self.update_screen_state(target_state)
        border_color = self.get_border_color()
        
        # 🎯 화면 전체에 색상 오버레이 (투명도 적용)
        if target_state in ['good', 'bad']:
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), border_color, -1)
            cv2.addWeighted(overlay, 0.1, image, 0.9, 0, image)
        
        # 🎯 두꺼운 테두리
        border_thickness = 25
        cv2.rectangle(image, (0, 0), (width, height), border_color, border_thickness)
        
        # 🎯 상단 정보 패널 (반투명)
        panel_height = 160
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        # 5종목 운동 종목 및 뷰 타입 표시
        if self.current_exercise not in ["detecting...", "model_not_loaded"]:
            exercise_emoji = {
                'squat': '🏋️‍♀️',
                'push_up': '💪',
                'deadlift': '🏋️‍♂️',
                'bench_press': '🔥',
                'pull_up': '💯'
            }
            
            exercise_text = f"{exercise_emoji.get(self.current_exercise, '🏋️')} {self.current_exercise.upper().replace('_', ' ')}"
            cv2.putText(image, exercise_text, (30, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # 뷰 타입과 신뢰도
            info_text = f"📹 {self.current_view_type.replace('_', ' ').title()} | 신뢰도: {self.classification_confidence:.0%}"
            cv2.putText(image, info_text, (30, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # 자세 신뢰도
            pose_text = f"자세 점수: {self.pose_confidence:.0%}"
            cv2.putText(image, pose_text, (30, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # 지원 종목 표시
            available_text = "지원 종목: 스쿼트 | 푸시업 | 데드리프트 | 벤치프레스 | 풀업"
            cv2.putText(image, available_text, (30, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # 🎯 중앙 상태 메시지 (크고 굵게)
        status_size = cv2.getTextSize(main_status, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0]
        status_x = (width - status_size[0]) // 2
        status_y = height // 2 - 80
        
        # 상태 메시지 배경
        padding = 30
        cv2.rectangle(image, 
                     (status_x - padding, status_y - 60), 
                     (status_x + status_size[0] + padding, status_y + 20), 
                     (0, 0, 0), -1)
        
        # 테두리 추가
        cv2.rectangle(image, 
                     (status_x - padding, status_y - 60), 
                     (status_x + status_size[0] + padding, status_y + 20), 
                     border_color, 3)
        
        cv2.putText(image, main_status, (status_x, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, border_color, 4)
        
        # 🎯 실시간 피드백 메시지들
        if self.feedback_messages:
            feedback_y = height // 2 + 40
            for i, (msg_type, message, view) in enumerate(list(self.feedback_messages)[-2:]):
                msg_color = (0, 255, 255) if msg_type == 'bad' else (0, 255, 0)
                
                # 메시지 배경
                msg_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                msg_x = (width - msg_size[0]) // 2
                
                cv2.rectangle(image, 
                             (msg_x - 20, feedback_y + i*35 - 25), 
                             (msg_x + msg_size[0] + 20, feedback_y + i*35 + 10), 
                             (0, 0, 0), -1)
                
                cv2.putText(image, message, (msg_x, feedback_y + i*35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, msg_color, 2)
        
        # 📊 하단 통계 패널
        if self.current_exercise not in ["detecting...", "model_not_loaded"]:
            stats_y = height - 120
            cv2.rectangle(image, (0, stats_y), (width, height), (0, 0, 0), -1)
            
            # 세션 통계
            total_analyzed = self.session_stats['good_count'] + self.session_stats['bad_count']
            if total_analyzed > 0:
                success_rate = self.session_stats['good_count'] / total_analyzed
                
                stats_text = f"📈 세션 통계: 총 {total_analyzed}프레임 | 성공률 {success_rate:.1%} | Good: {self.session_stats['good_count']} | Bad: {self.session_stats['bad_count']}"
                
                # 통계 텍스트 크기에 맞춰 위치 조정
                stats_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                stats_x = max(10, (width - stats_size[0]) // 2)
                
                cv2.putText(image, stats_text, (stats_x, stats_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 운동별 감지 횟수
                exercise_stats = []
                for ex, count in self.session_stats['exercise_distribution'].items():
                    if count > 0:
                        exercise_stats.append(f"{ex.upper()}: {count}")
                
                if exercise_stats:
                    exercise_text = "🏋️ 감지된 운동: " + " | ".join(exercise_stats[:3])  # 최대 3개만
                    cv2.putText(image, exercise_text, (10, stats_y + 55), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # 성공률 진행 바
                bar_width = width - 60
                bar_height = 15
                bar_x = 30
                bar_y = stats_y + 75
                
                # 배경 바
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                
                # 성공률 바
                success_width = int(bar_width * success_rate)
                bar_color = (0, 255, 0) if success_rate > 0.7 else (0, 255, 255) if success_rate > 0.4 else (0, 0, 255)
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + success_width, bar_y + bar_height), bar_color, -1)
        
        # ⌨️ 조작 가이드 (화면 하단)
        guide_text = "🔴 Q: 종료  |  🔄 R: 리셋  |  📸 S: 스크린샷  |  🎯 C: 운동 변경  |  🆘 H: 도움말"
        guide_size = cv2.getTextSize(guide_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        guide_x = (width - guide_size[0]) // 2
        
        cv2.putText(image, guide_text, (guide_x, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return image
    
    def show_exercise_selection_help(self):
        """운동 선택 도움말 출력"""
        print("\n" + "="*60)
        print("🏋️ BLAZE 5종목 운동 가이드")
        print("="*60)
        print("📋 지원되는 운동:")
        print("  1️⃣ SQUAT (스쿼트)")
        print("     - 최적 뷰: 측면")
        print("     - 핵심: 무릎 각도, 엉덩이 뒤로, 등 곧게")
        print("\n  2️⃣ PUSH_UP (푸시업)")
        print("     - 최적 뷰: 측면")
        print("     - 핵심: 몸 일직선, 팔꿈치 각도")
        print("\n  3️⃣ DEADLIFT (데드리프트)")
        print("     - 최적 뷰: 측면")
        print("     - 핵심: 등 곧게(가장 중요), 힙힌지")
        print("\n  4️⃣ BENCH_PRESS (벤치프레스)")
        print("     - 최적 뷰: 측면")
        print("     - 핵심: 팔꿈치 각도, 어깨 안정성")
        print("\n  5️⃣ PULL_UP (풀업)")
        print("     - 최적 뷰: 측면")
        print("     - 핵심: 풀 레인지, 어깨 안정성")
        print("\n🎯 촬영 팁:")
        print("  - 전신이 화면에 들어오도록")
        print("  - 2-3m 거리에서 촬영")
        print("  - 충분한 조명")
        print("  - 단순한 배경")
        print("="*60)
    
    def cycle_exercise(self):
        """운동 종목 순환 변경"""
        if self.current_exercise in self.available_exercises:
            current_idx = self.available_exercises.index(self.current_exercise)
        else:
            current_idx = -1
        
        next_idx = (current_idx + 1) % len(self.available_exercises)
        self.current_exercise = self.available_exercises[next_idx]
        
        # 해당 운동 히스토리 리셋
        if self.current_exercise in self.post_processors:
            self.post_processors[self.current_exercise].history.clear()
            self.post_processors[self.current_exercise].ema_value = None
            self.post_processors[self.current_exercise].last_state = 'good'
        
        print(f"🔄 운동 변경: {self.current_exercise.upper()}")
        return self.current_exercise
    
    def run_analysis(self, camera_id: int = 0):
        """5종목 실시간 분석 실행"""
        if not self.init_success:
            print("❌ 시스템 초기화 실패")
            return False
        
        # 카메라 초기화
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ 카메라 {camera_id} 열기 실패")
            print("💡 다른 카메라 ID를 시도해보세요: --camera 1")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*80)
        print("🏋️ BLAZE - 5종목 향상된 실시간 운동 자세 분석 시스템 (완화된 버전)")
        print("="*80)
        print("✨ 새로운 기능:")
        print("  • 5종목 완전 지원 (스쿼트, 푸시업, 데드리프트, 벤치프레스, 풀업)")
        print("  • 뷰 타입 자동 감지 (측면/정면/후면)")
        print("  • 운동별 맞춤 각도 기준 및 가중치")
        print("  • 완화된 적응형 히스테리시스 (더 관대한 판정)")
        print("  • 강화된 시각적 피드백 (전체 화면 색상)")
        print("  • 실시간 성과 추적 및 분석")
        print("\n🎯 지원 운동:")
        for i, exercise in enumerate(self.available_exercises, 1):
            emoji = {'squat': '🏋️‍♀️', 'push_up': '💪', 'deadlift': '🏋️‍♂️', 'bench_press': '🔥', 'pull_up': '💯'}
            print(f"  {i}. {emoji.get(exercise, '🏋️')} {exercise.upper().replace('_', ' ')}")
        print("\n⌨️ 조작법:")
        print("  Q: 종료  |  R: 통계 리셋  |  S: 스크린샷  |  C: 운동 변경  |  H: 도움말")
        print("="*80)
        
        if not self.model_loaded:
            print("\n❌ 경고: 운동 분류 모델이 로드되지 않았습니다!")
            print("먼저 모델을 훈련하세요:")
            print("python main.py --mode train")
            print("\n자세 분석은 가능하지만 운동을 수동으로 지정해야 합니다.")
            print("C키를 눌러 운동을 변경할 수 있습니다.")
        
        # FPS 측정용
        fps_counter = deque(maxlen=30)
        frame_count = 0
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("❌ 카메라 읽기 실패")
                    break
                
                # 좌우 반전 (셀카 모드)
                frame = cv2.flip(frame, 1)
                frame_count += 1
                
                # MediaPipe 처리
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # 1단계: 5종목 운동 분류 (모델이 로드된 경우)
                    if self.model_loaded:
                        exercise, class_conf = self.classify_current_exercise(frame)
                    else:
                        exercise, class_conf = self.current_exercise, 0.0
                    
                    # 2단계: 향상된 자세 분석
                    pose_quality, analysis_result = self.analyze_pose_quality(results.pose_landmarks.landmark)
                    
                    # 3단계: 강화된 비주얼 피드백
                    frame = self.draw_enhanced_feedback(frame)
                
                else:
                    # 포즈 미감지
                    self.update_screen_state("neutral")
                    frame = self.draw_enhanced_feedback(frame)
                    
                    # 포즈 미감지 메시지
                    cv2.putText(frame, "카메라 앞에 서서 전신이 보이도록 해주세요", 
                               (frame.shape[1]//2 - 300, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # FPS 계산 및 표시
                end_time = time.time()
                fps = 1 / max(end_time - start_time, 0.001)
                fps_counter.append(fps)
                avg_fps = sum(fps_counter) / len(fps_counter)
                
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 화면 출력
                cv2.imshow('🏋️ BLAZE - 5-Exercise Enhanced Analysis (Relaxed)', frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # 통계 리셋
                    self.session_stats = {
                        'good_count': 0, 'bad_count': 0, 'total_frames': 0,
                        'view_distribution': {'side_view': 0, 'front_view': 0, 'back_view': 0},
                        'exercise_distribution': {ex: 0 for ex in self.available_exercises}
                    }
                    self.exercise_history.clear()
                    self.pose_history.clear()
                    self.feedback_messages.clear()
                    for processor in self.post_processors.values():
                        processor.history.clear()
                        processor.ema_value = None
                        processor.last_state = 'good'
                    print("📊 전체 통계 및 히스토리 리셋 완료")
                    
                elif key == ord('s'):
                    # 스크린샷 저장
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    screenshot_dir = "outputs/screenshots"
                    os.makedirs(screenshot_dir, exist_ok=True)
                    
                    filename = f"{screenshot_dir}/blaze_5exercise_{self.current_exercise}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 스크린샷 저장: {filename}")
                    
                elif key == ord('c'):
                    # 운동 수동 변경
                    self.cycle_exercise()
                    
                elif key == ord('h'):
                    # 도움말 표시
                    self.show_exercise_selection_help()
                
        except KeyboardInterrupt:
            print("\n⏹️ 사용자 중단")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 임시 디렉토리 정리
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            
            # 최종 세션 리포트
            self.print_session_report()
            
            return True
    
    def print_session_report(self):
        """5종목 세션 완료 리포트 출력"""
        total_analyzed = self.session_stats['good_count'] + self.session_stats['bad_count']
        
        print(f"\n" + "="*70)
        print(f"📊 BLAZE 5종목 세션 완료 리포트 (완화된 버전)")
        print(f"="*70)
        
        if total_analyzed > 0:
            success_rate = self.session_stats['good_count'] / total_analyzed
            
            print(f"🎯 총 분석 프레임: {total_analyzed:,}개")
            print(f"✅ 좋은 자세: {self.session_stats['good_count']:,}개 ({success_rate:.1%})")
            print(f"⚠️ 개선 필요: {self.session_stats['bad_count']:,}개 ({1-success_rate:.1%})")
            
            # 뷰 분포
            print(f"\n📹 촬영 각도 분포:")
            total_views = sum(self.session_stats['view_distribution'].values())
            for view, count in self.session_stats['view_distribution'].items():
                if total_views > 0:
                    percentage = count / total_views * 100
                    print(f"  {view.replace('_', ' ').title()}: {count:,}개 ({percentage:.1f}%)")
            
            # 5종목 운동 분포 (모델이 로드된 경우)
            if any(count > 0 for count in self.session_stats['exercise_distribution'].values()):
                print(f"\n🏋️ 감지된 운동 분포:")
                exercise_emoji = {
                    'squat': '🏋️‍♀️', 'push_up': '💪', 'deadlift': '🏋️‍♂️', 
                    'bench_press': '🔥', 'pull_up': '💯'
                }
                for exercise, count in self.session_stats['exercise_distribution'].items():
                    if count > 0:
                        emoji = exercise_emoji.get(exercise, '🏋️')
                        print(f"  {emoji} {exercise.replace('_', ' ').title()}: {count}회 감지")
            
            # 성과 평가 (완화된 기준)
            print(f"\n🏆 성과 평가 (완화된 기준):")
            if success_rate >= 0.6:  # 기존 0.8에서 0.6으로 완화
                print(f"  🥇 훌륭함! 자세가 매우 안정적입니다.")
                print(f"  💡 팁: 다른 운동도 도전해보세요!")
            elif success_rate >= 0.4:  # 기존 0.6에서 0.4로 완화
                print(f"  🥈 좋음! 조금 더 연습하면 완벽해질 거예요.")
                print(f"  💡 팁: 측면 뷰에서 촬영하면 더 정확합니다.")
            elif success_rate >= 0.2:  # 기존 0.4에서 0.2로 완화
                print(f"  🥉 보통! 기본 자세를 더 연습해보세요.")
                print(f"  💡 팁: 천천히 정확한 동작부터 익혀나가세요.")
            else:
                print(f"  💪 화이팅! 천천히 정확한 자세부터 익혀나가세요.")
                print(f"  💡 팁: H키를 눌러 운동별 가이드를 확인하세요.")
        else:
            print(f"분석된 데이터가 없습니다.")
        
        print(f"\n🎓 다음 단계 추천:")
        print(f"  1. 성공률이 낮은 운동은 더 많은 연습이 필요")
        print(f"  2. 측면 뷰에서 촬영하면 가장 정확한 분석 가능")
        print(f"  3. 각 운동의 핵심 포인트를 기억하며 연습")
        print(f"  4. 정기적인 자세 체크로 부상 예방")
        print(f"  5. 완화된 기준으로 더 많은 Good 판정 받기!")
        print(f"="*70)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='🏋️ BLAZE 5종목 향상된 실시간 운동 분석 (완화된 버전)')
    parser.add_argument('--camera', type=int, default=0, help='카메라 ID')
    parser.add_argument('--model', type=str, default='models/exercise_classifier.pkl',
                       help='운동 분류 모델 경로')
    
    args = parser.parse_args()
    
    analyzer = Enhanced5ExerciseRealTimeAnalyzer(args.model)
    analyzer.run_analysis(args.camera)

if __name__ == "__main__":
    main()