#!/usr/bin/env python3
"""
🤖 완전 자동화 운동 분석기 - 사진/영상/실시간 통합 버전
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
                'left_knee': {'points': [23, 25, 27], 'range': (55, 140), 'weight': 1.1, 'name_kr': '왼쪽 무릎'},
                'right_knee': {'points': [24, 26, 28], 'range': (55, 140), 'weight': 1.1, 'name_kr': '오른쪽 무릎'},
                'left_hip': {'points': [11, 23, 25], 'range': (55, 140), 'weight': 0.9, 'name_kr': '왼쪽 엉덩이'},
                'right_hip': {'points': [12, 24, 26], 'range': (55, 140), 'weight': 0.9, 'name_kr': '오른쪽 엉덩이'},
                'back_straight': {'points': [11, 23, 25], 'range': (110, 170), 'weight': 1.1, 'name_kr': '등 곧게'},
                'spine_angle': {'points': [23, 11, 13], 'range': (110, 170), 'weight': 0.9, 'name_kr': '척추 각도'},
            },
            'push_up': {
                'left_elbow': {'points': [11, 13, 15], 'range': (40, 160), 'weight': 1.0, 'name_kr': '왼쪽 팔꿈치'},
                'right_elbow': {'points': [12, 14, 16], 'range': (40, 160), 'weight': 1.0, 'name_kr': '오른쪽 팔꿈치'},
                'body_line': {'points': [11, 23, 25], 'range': (140, 180), 'weight': 1.2, 'name_kr': '몸 일직선'},
                'leg_straight': {'points': [23, 25, 27], 'range': (140, 180), 'weight': 0.8, 'name_kr': '다리 펴기'},
                'shoulder_alignment': {'points': [13, 11, 23], 'range': (120, 180), 'weight': 0.6, 'name_kr': '어깨 정렬'},
                'core_stability': {'points': [11, 12, 23], 'range': (140, 180), 'weight': 1.0, 'name_kr': '코어 안정성'},
            },
            'deadlift': {
                'left_knee': {'points': [23, 25, 27], 'range': (80, 140), 'weight': 0.6, 'name_kr': '왼쪽 무릎'},
                'right_knee': {'points': [24, 26, 28], 'range': (80, 140), 'weight': 0.6, 'name_kr': '오른쪽 무릎'},
                'hip_hinge': {'points': [11, 23, 25], 'range': (80, 180), 'weight': 0.7, 'name_kr': '힙 힌지'},
                'back_straight': {'points': [11, 23, 12], 'range': (120, 180), 'weight': 1.0, 'name_kr': '등 곧게'},
                'chest_up': {'points': [23, 11, 13], 'range': (50, 140), 'weight': 0.5, 'name_kr': '가슴 펴기'},
                'spine_neutral': {'points': [23, 11, 24], 'range': (120, 180), 'weight': 0.8, 'name_kr': '척추 중립'},
            },
            'bench_press': {
                'left_elbow': {'points': [11, 13, 15], 'range': (50, 145), 'weight': 1.1, 'name_kr': '왼쪽 팔꿈치'},
                'right_elbow': {'points': [12, 14, 16], 'range': (50, 145), 'weight': 1.1, 'name_kr': '오른쪽 팔꿈치'},
                'left_shoulder': {'points': [13, 11, 23], 'range': (50, 150), 'weight': 0.9, 'name_kr': '왼쪽 어깨'},
                'right_shoulder': {'points': [14, 12, 24], 'range': (50, 150), 'weight': 0.9, 'name_kr': '오른쪽 어깨'},
                'back_arch': {'points': [11, 23, 25], 'range': (90, 170), 'weight': 0.7, 'name_kr': '등 아치'},
                'wrist_alignment': {'points': [13, 15, 17], 'range': (70, 180), 'weight': 0.6, 'name_kr': '손목 정렬'},
            },
            'lunge': {
                'front_knee': {'points': [23, 25, 27], 'range': (70, 120), 'weight': 1.2, 'name_kr': '앞 무릎'},
                'back_knee': {'points': [24, 26, 28], 'range': (120, 180), 'weight': 1.0, 'name_kr': '뒤 무릎'},
                'front_hip': {'points': [11, 23, 25], 'range': (70, 120), 'weight': 0.8, 'name_kr': '앞 엉덩이'},
                'torso_upright': {'points': [11, 23, 25], 'range': (100, 180), 'weight': 1.2, 'name_kr': '상체 직립'},
                'front_ankle': {'points': [25, 27, 31], 'range': (80, 110), 'weight': 0.8, 'name_kr': '앞 발목'},
                'back_hip_extension': {'points': [12, 24, 26], 'range': (150, 180), 'weight': 1.0, 'name_kr': '뒤 엉덩이 신전'},
            }
        }
        
        # 운동별 맞춤 피드백 메시지
        self.detailed_feedback = {
            'squat': {
                'left_knee': {
                    'too_low': '왼쪽 무릎을 더 올려주세요 (무릎이 너무 구부러져 있어요)',
                    'too_high': '왼쪽 무릎을 더 구부려주세요 (스쿼트 깊이가 부족해요)',
                    'good': '왼쪽 무릎 각도가 완벽해요!'
                },
                'right_knee': {
                    'too_low': '오른쪽 무릎을 더 올려주세요 (무릎이 너무 구부러져 있어요)',
                    'too_high': '오른쪽 무릎을 더 구부려주세요 (스쿼트 깊이가 부족해요)',
                    'good': '오른쪽 무릎 각도가 완벽해요!'
                },
                'left_hip': {
                    'too_low': '왼쪽 엉덩이를 더 뒤로 빼주세요',
                    'too_high': '왼쪽 엉덩이를 더 낮춰주세요',
                    'good': '왼쪽 엉덩이 자세가 좋아요!'
                },
                'right_hip': {
                    'too_low': '오른쪽 엉덩이를 더 뒤로 빼주세요',
                    'too_high': '오른쪽 엉덩이를 더 낮춰주세요',
                    'good': '오른쪽 엉덩이 자세가 좋아요!'
                },
                'back_straight': {
                    'too_low': '등을 더 곧게 펴주세요 (등이 굽어있어요)',
                    'too_high': '상체를 약간 앞으로 기울여주세요',
                    'good': '등이 완벽하게 곧아요!'
                },
                'general': '무릎이 발끝을 넘지 않게 주의하세요'
            },
            'push_up': {
                'left_elbow': {
                    'too_low': '왼쪽 팔을 더 펴주세요',
                    'too_high': '왼쪽 팔꿈치를 더 구부려주세요',
                    'good': '왼쪽 팔 각도가 완벽해요!'
                },
                'right_elbow': {
                    'too_low': '오른쪽 팔을 더 펴주세요',
                    'too_high': '오른쪽 팔꿈치를 더 구부려주세요',
                    'good': '오른쪽 팔 각도가 완벽해요!'
                },
                'body_line': {
                    'too_low': '엉덩이를 올려주세요 (몸이 구부러져 있어요)',
                    'too_high': '엉덩이를 내려주세요 (엉덩이가 너무 높아요)',
                    'good': '몸이 완벽한 일직선이에요!'
                },
                'shoulder_alignment': {
                    'too_low': '어깨를 더 안정적으로 유지하세요',
                    'too_high': '어깨에 힘을 빼고 자연스럽게 하세요',
                    'good': '어깨 정렬이 완벽해요!'
                },
                'general': '팔꿈치를 몸에 가깝게 유지하세요'
            },
            'deadlift': {
                'left_knee': {
                    'too_low': '왼쪽 무릎을 약간 더 펴주세요',
                    'too_high': '왼쪽 무릎을 약간 구부려주세요',
                    'good': '왼쪽 무릎이 완벽해요!'
                },
                'right_knee': {
                    'too_low': '오른쪽 무릎을 약간 더 펴주세요',
                    'too_high': '오른쪽 무릎을 약간 구부려주세요',
                    'good': '오른쪽 무릎이 완벽해요!'
                },
                'hip_hinge': {
                    'too_low': '엉덩이를 더 뒤로 빼주세요 (힙 힌지 동작)',
                    'too_high': '엉덩이를 더 낮춰주세요',
                    'good': '힙 힌지 동작이 완벽해요!'
                },
                'back_straight': {
                    'too_low': '등을 곧게 펴주세요 - 매우 중요해요!',
                    'too_high': '등에 힘을 빼고 자연스럽게 하세요',
                    'good': '등이 완벽하게 곧아요!'
                },
                'chest_up': {
                    'too_low': '가슴을 펴고 시선을 앞으로 향하세요',
                    'too_high': '과도하게 가슴을 펴지 마세요',
                    'good': '가슴 자세가 완벽해요!'
                },
                'general': '바벨을 몸에 가깝게 유지하세요'
            },
            'bench_press': {
                'left_elbow': {
                    'too_low': '왼쪽 팔을 더 펴주세요',
                    'too_high': '왼쪽 팔꿈치를 더 구부려주세요',
                    'good': '왼쪽 팔이 완벽해요!'
                },
                'right_elbow': {
                    'too_low': '오른쪽 팔을 더 펴주세요',
                    'too_high': '오른쪽 팔꿈치를 더 구부려주세요',
                    'good': '오른쪽 팔이 완벽해요!'
                },
                'left_shoulder': {
                    'too_low': '왼쪽 어깨를 안정적으로 유지하세요',
                    'too_high': '왼쪽 어깨에 힘을 빼세요',
                    'good': '왼쪽 어깨가 완벽해요!'
                },
                'right_shoulder': {
                    'too_low': '오른쪽 어깨를 안정적으로 유지하세요',
                    'too_high': '오른쪽 어깨에 힘을 빼세요',
                    'good': '오른쪽 어깨가 완벽해요!'
                },
                'back_arch': {
                    'too_low': '자연스러운 등 아치를 만들어주세요',
                    'too_high': '등 아치를 과도하게 만들지 마세요',
                    'good': '등 아치가 완벽해요!'
                },
                'general': '바벨을 천천히 컨트롤하세요'
            },
            'lunge': {
                'front_knee': {
                    'too_low': '앞 무릎을 90도로 맞춰주세요 (너무 구부러져 있어요)',
                    'too_high': '앞 무릎을 더 구부려주세요 (90도까지)',
                    'good': '앞 무릎이 완벽한 90도에요!'
                },
                'back_knee': {
                    'too_low': '뒤 무릎을 더 펴주세요',
                    'too_high': '뒤 무릎이 완벽해요!',
                    'good': '뒤 무릎이 완벽하게 펴져 있어요!'
                },
                'torso_upright': {
                    'too_low': '상체를 더 곧게 세워주세요',
                    'too_high': '상체가 완벽해요!',
                    'good': '상체가 완벽하게 직립해요!'
                },
                'front_ankle': {
                    'too_low': '앞발목을 더 안정적으로 유지하세요',
                    'too_high': '앞발목에 힘을 빼세요',
                    'good': '앞발목이 완벽해요!'
                },
                'general': '균형을 유지하며 천천히 동작하세요'
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
        
        # 운동 이모지 및 한글명
        self.exercise_info = {
            'squat': {'emoji': '🏋️‍♀️', 'name_kr': '스쿼트', 'name_en': 'SQUAT'},
            'push_up': {'emoji': '💪', 'name_kr': '푸쉬업', 'name_en': 'PUSH-UP'},
            'deadlift': {'emoji': '🏋️‍♂️', 'name_kr': '데드리프트', 'name_en': 'DEADLIFT'},
            'bench_press': {'emoji': '🔥', 'name_kr': '벤치프레스', 'name_en': 'BENCH PRESS'},
            'lunge': {'emoji': '🚀', 'name_kr': '런지', 'name_en': 'LUNGE'}
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
                            name_kr = exercise_info.get('name_kr', new_exercise)
                            print(f"🤖 AI 감지: {emoji} {name_kr} (신뢰도: {confidence:.1%})")
            
            self.last_classification_time = current_time
            
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return self.current_exercise, self.exercise_confidence
            
        except Exception as e:
            print(f"운동 분류 오류: {e}")
            return self.current_exercise, self.exercise_confidence
    
    def analyze_pose_angles(self, landmarks, exercise: str) -> Dict:
        """🎯 2단계: 감지된 운동에 맞춰 상세 각도 분석"""
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
                    'in_range': min_angle <= angle <= max_angle,
                    'name_kr': config.get('name_kr', joint_name)
                }
                
                total_weight += weight
                
                if not (min_angle <= angle <= max_angle):
                    violations.append({
                        'joint': joint_name,
                        'angle': angle,
                        'expected_range': (min_angle, max_angle),
                        'weight': weight,
                        'name_kr': config.get('name_kr', joint_name)
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
        """🗣️ 운동별 상세 피드백 생성"""
        current_time = time.time()
        
        # 피드백 주기 제한
        if current_time - self.last_feedback_time < self.feedback_interval:
            return self.current_feedback_messages
        
        messages = []
        
        if not pose_result.get('valid', False):
            messages.append("포즈를 인식할 수 없습니다")
            return messages
        
        violations = pose_result.get('violations', [])
        exercise_feedback = self.detailed_feedback.get(exercise, {})
        
        if not violations:
            # 모든 자세가 완벽한 경우
            exercise_info = self.exercise_info.get(exercise, {})
            name_kr = exercise_info.get('name_kr', exercise)
            messages.append(f"완벽한 {name_kr} 자세입니다! 👍")
            messages.append("현재 폼을 유지하세요!")
        else:
            # 위반사항이 있는 경우 - 가중치 순으로 정렬
            violations_sorted = sorted(violations, key=lambda x: x['weight'], reverse=True)
            
            for i, violation in enumerate(violations_sorted[:3]):  # 상위 3개만
                joint = violation['joint']
                angle = violation['angle']
                min_angle, max_angle = violation['expected_range']
                name_kr = violation.get('name_kr', joint)
                
                joint_feedback = exercise_feedback.get(joint, {})
                
                if angle < min_angle:
                    # 각도가 너무 작음
                    message = joint_feedback.get('too_low', f'{name_kr} 각도를 높여주세요')
                elif angle > max_angle:
                    # 각도가 너무 큼
                    message = joint_feedback.get('too_high', f'{name_kr} 각도를 낮춰주세요')
                else:
                    message = joint_feedback.get('good', f'{name_kr}가 좋아요!')
                
                messages.append(f"⚠️ {message}")
                
                # 구체적인 각도 정보 추가
                if i == 0:  # 가장 중요한 문제만 각도 표시
                    messages.append(f"   현재: {angle:.0f}° → 목표: {min_angle:.0f}-{max_angle:.0f}°")
            
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
        """✨ 향상된 분석 결과 화면 오버레이"""
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
            name_kr = exercise_info.get('name_kr', exercise)
            name_en = exercise_info.get('name_en', exercise.upper())
            
            # 배경 박스
            cv2.rectangle(frame, (40, 40), (400, 140), (0, 0, 0), -1)
            cv2.rectangle(frame, (40, 40), (400, 140), self.screen_color, 3)
            
            # 운동명 표시
            exercise_text = f"{emoji} {name_kr}"
            cv2.putText(frame, exercise_text, (60, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # 영어명 표시
            cv2.putText(frame, name_en, (60, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # 신뢰도 표시
            confidence_text = f"신뢰도: {self.exercise_confidence:.0%}"
            cv2.putText(frame, confidence_text, (250, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
        elif exercise == "detecting...":
            cv2.rectangle(frame, (40, 40), (300, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (40, 40), (300, 100), (255, 255, 0), 3)
            cv2.putText(frame, "🤖 운동 감지 중...", (60, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.rectangle(frame, (40, 40), (350, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (40, 40), (350, 100), (128, 128, 128), 3)
            cv2.putText(frame, "⚙️ 수동 모드", (60, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 🎯 중앙 상태 메시지
        if pose_result.get('valid', False):
            pose_quality = pose_result['classification']
            confidence = pose_result['confidence']
            
            if pose_quality == 'good':
                status_text = "완벽한 자세! 👍"
                status_color = (0, 255, 0)
            else:
                status_text = "자세 교정 필요 ⚠️"
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
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
            
            # 신뢰도 점수
            score_text = f"자세 점수: {confidence:.0%}"
            score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            score_x = (width - score_size[0]) // 2
            cv2.putText(frame, score_text, (score_x, status_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
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
                
                # 피드백 제목
                cv2.putText(frame, "💬 실시간 피드백:", (60, height - feedback_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 피드백 메시지들
                for i, message in enumerate(feedback_messages[:5]):  # 최대 5개
                    y_pos = height - feedback_height + 20 + (i * 35)
                    
                    # 메시지 색상 결정
                    if "완벽" in message or "👍" in message:
                        msg_color = (0, 255, 0)  # 초록색
                    elif "⚠️" in message:
                        msg_color = (0, 100, 255)  # 주황색
                    elif "💡" in message:
                        msg_color = (255, 255, 0)  # 노란색
                    else:
                        msg_color = (255, 255, 255)  # 흰색
                    
                    cv2.putText(frame, message, (60, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, msg_color, 2)
        
        # 📊 오른쪽 위: 통계 정보
        if self.stats['frames'] > 0:
            total = self.stats['good'] + self.stats['bad']
            if total > 0:
                good_ratio = self.stats['good'] / total
                
                # 통계 배경
                cv2.rectangle(frame, (width - 300, 40), (width - 40, 140), (0, 0, 0), -1)
                cv2.rectangle(frame, (width - 300, 40), (width - 40, 140), (255, 255, 255), 2)
                
                # 통계 텍스트
                cv2.putText(frame, "📊 운동 통계", (width - 280, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                stats_text = f"Good: {self.stats['good']} | Bad: {self.stats['bad']}"
                cv2.putText(frame, stats_text, (width - 280, 95), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                ratio_text = f"성공률: {good_ratio:.1%}"
                cv2.putText(frame, ratio_text, (width - 280, 115), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if good_ratio > 0.7 else (255, 255, 255), 1)
        
        # ⌨️ 하단 조작 가이드
        guide_text = "Q: 종료  |  R: 리셋  |  S: 스크린샷  |  C: 운동 변경  |  SPACE: 모드 변경"
        cv2.putText(frame, guide_text, (50, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return frame
    
    def analyze_single_image(self, image_path: str) -> Dict:
        """📷 단일 이미지 완전 자동 분석"""
        if not os.path.exists(image_path):
            return {'error': f'이미지 파일을 찾을 수 없습니다: {image_path}'}
        
        print(f"📷 이미지 자동 분석 시작: {os.path.basename(image_path)}")
        
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            return {'error': '이미지를 읽을 수 없습니다'}
        
        # 포즈 검출 (정적 이미지용 고정밀 모델)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose_static.process(image_rgb)
        
        if not results.pose_landmarks:
            return {'error': '포즈를 검출할 수 없습니다'}
        
        # 🤖 1단계: AI 운동 감지
        exercise, confidence = self.classify_exercise(image)
        exercise_info = self.exercise_info.get(exercise, {})
        emoji = exercise_info.get('emoji', '🏋️')
        name_kr = exercise_info.get('name_kr', exercise)
        
        print(f"🎯 AI 감지: {emoji} {name_kr} (신뢰도: {confidence:.1%})")
        
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
            return {'error': f'지원되지 않는 운동: {exercise}'}
    
    def analyze_video_file(self, video_path: str, output_path: str = None) -> Dict:
        """🎬 영상 파일 완전 자동 분석"""
        if not os.path.exists(video_path):
            return {'error': f'영상 파일을 찾을 수 없습니다: {video_path}'}
        
        print(f"🎬 영상 자동 분석 시작: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': '영상 파일을 열 수 없습니다'}
        
        # 영상 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 영상 정보: {width}x{height}, {fps}fps, {total_frames}프레임")
        
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
                    print(f"📊 분석 진행률: {progress:.1f}%")
                
                # 출력 영상에 쓰기
                if output_path:
                    out.write(frame)
                
                frame_count += 1
                
        except Exception as e:
            print(f"❌ 영상 분석 중 오류: {e}")
            return {'error': f'영상 분석 실패: {str(e)}'}
        finally:
            cap.release()
            if output_path:
                out.release()
        
        # 가장 많이 감지된 운동 찾기
        main_exercise = max(exercise_detections.items(), key=lambda x: x[1])[0] if exercise_detections else "unknown"
        
        # 결과 요약
        success_rate = (stats['good'] / max(stats['total'], 1)) * 100
        
        print(f"\n🎉 영상 분석 완료!")
        print(f"🎯 주요 운동: {self.exercise_info.get(main_exercise, {}).get('name_kr', main_exercise)}")
        print(f"📊 분석 결과: Good {stats['good']}프레임, Bad {stats['bad']}프레임")
        print(f"🎯 성공률: {success_rate:.1f}%")
        
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
            print(f"❌ 카메라 {camera_id} 열기 실패")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        cv2.namedWindow('Exercise Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Exercise Analysis', 1200, 800)
        
        print("\n" + "="*80)
        print("🤖 완전 자동화 운동 분석 시스템")
        print("="*80)
        print("✨ 기능:")
        print("  🤖 1단계: AI가 운동 종류 자동 감지")
        print("  🎯 2단계: 감지된 운동에 맞춰 정밀 각도 분석")
        print("  🗣️ 3단계: 운동별 맞춤 상세 피드백")
        print("  🌈 4단계: 실시간 초록/빨강 화면 + 테두리")
        print("  📊 5단계: 실시간 통계 및 성과 추적")
        print("\n📍 화면 구성:")
        print("  • 왼쪽 위: 감지된 운동 종류")
        print("  • 왼쪽 아래: 상세 피드백 메시지")
        print("  • 오른쪽 위: 운동 통계")
        print("  • 중앙: 자세 상태 (Good/Bad)")
        print("  • 전체: 초록/빨강 테두리 + 배경")
        print("\n⌨️ 조작법:")
        print("  Q: 종료 | R: 통계 리셋 | S: 스크린샷")
        print("  C: 수동 운동 선택 | SPACE: 자동/수동 모드 토글")
        print("="*80)
        
        # 모델 상태 확인
        if not self.model_loaded:
            print("⚠️ AI 모델 없음 - 수동 모드로 시작")
            if manual_exercise:
                self.current_exercise = manual_exercise
                exercise_info = self.exercise_info.get(manual_exercise, {})
                print(f"수동 선택: {exercise_info.get('emoji', '🏋️')} {exercise_info.get('name_kr', manual_exercise)}")
        
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
                results = self.pose_video.process(frame_rgb)
                
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
                            
                            # ✨ 3-4단계: 피드백 + 화면 오버레이
                            frame = self.draw_enhanced_overlay(frame, exercise, pose_result)
                        else:
                            frame = self.draw_enhanced_overlay(frame, exercise, {'valid': False})
                    else:
                        frame = self.draw_enhanced_overlay(frame, exercise, {'valid': False})
                else:
                    # 포즈 미감지
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 0), 30)
                    message = "전신이 보이도록 카메라 앞에 서주세요"
                    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = frame.shape[0] // 2
                    
                    cv2.rectangle(frame, (text_x - 20, text_y - 40), 
                                 (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)
                    cv2.putText(frame, message, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # 화면 출력
                window_title = "🤖 완전 자동화 운동 분석 시스템"
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
                    filename = f"complete_analysis_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 스크린샷 저장: {filename}")
                elif key == ord('c'):
                    # 수동 운동 변경
                    current_manual_idx = (current_manual_idx + 1) % len(available_exercises)
                    self.current_exercise = available_exercises[current_manual_idx]
                    exercise_info = self.exercise_info.get(self.current_exercise, {})
                    emoji = exercise_info.get('emoji', '🏋️')
                    name_kr = exercise_info.get('name_kr', self.current_exercise)
                    print(f"🔄 수동 선택: {emoji} {name_kr}")
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
                success_rate = (self.stats['good'] / total) * 100
                print(f"\n📊 최종 통계:")
                print(f"  🎯 총 분석: {total} 프레임")
                print(f"  ✅ Good: {self.stats['good']} ({success_rate:.1f}%)")
                print(f"  ❌ Bad: {self.stats['bad']} ({100-success_rate:.1f}%)")
                print(f"  🎯 운동별 분석 완료!")
            
            return True

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='🤖 완전 자동화 운동 분석기 - 사진/영상/실시간 통합',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 완전 자동화 기능:
  1단계: 🤖 AI가 운동 종류 자동 감지 (스쿼트, 푸쉬업, 데드리프트, 벤치프레스, 런지)
  2단계: 🎯 감지된 운동에 맞춰 정밀 각도 분석 
  3단계: 🗣️ 운동별 맞춤 상세 피드백
  4단계: 🌈 실시간 초록/빨강 화면 + 테두리
  5단계: 📊 실시간 통계 및 성과 추적

📍 화면 구성:
  • 왼쪽 위: 감지된 운동 종류 + 신뢰도
  • 왼쪽 아래: 상세 피드백 메시지 (각도별 조언)
  • 오른쪽 위: 운동 통계 (Good/Bad 비율)
  • 중앙: 자세 상태 (완벽한 자세! / 자세 교정 필요)
  • 전체: 초록(Good)/빨강(Bad) 테두리 + 배경


🎯 사용 예시:
  # 실시간 완전 자동 분석
  python complete_auto_analyzer.py --mode realtime
  
  # 실시간 + 수동 운동 지정
  python complete_auto_analyzer.py --mode realtime --manual squat
  
  # 사진 완전 자동 분석 
  python complete_auto_analyzer.py --mode image --input photo.jpg
  
  # 영상 완전 자동 분석
  python complete_auto_analyzer.py --mode video --input video.mp4 --output analyzed.mp4

⌨️ 실시간 조작:
  Q: 종료  |  R: 통계 리셋  |  S: 스크린샷
  C: 수동 운동 변경  |  SPACE: 자동/수동 모드 토글

🏋️ 지원 운동 & 상세 피드백:
  🏋️‍♀️ 스쿼트: 무릎/엉덩이 각도, 등 곧게 펴기, 발끝 넘지 않기
  💪 푸쉬업: 팔꿈치 각도, 몸 일직선, 어깨 안정성
  🏋️‍♂️ 데드리프트: 힙 힌지, 등 곧게, 무릎 각도 (완화 기준)
  🔥 벤치프레스: 팔꿈치/어깨 각도, 등 아치
  🚀 런지: 앞무릎 90도, 뒷무릎 펴기, 상체 직립

💡 AI 모델 필요:
  models/exercise_classifier.pkl 파일이 있으면 완전 자동
  없으면 수동 운동 선택 모드로 동작
        """
    )
    
    parser.add_argument('--mode', type=str, default='realtime',
                       choices=['realtime', 'image', 'video'],
                       help='분석 모드: realtime(실시간), image(사진), video(영상)')
    parser.add_argument('--input', type=str,
                       help='입력 파일 경로 (image/video 모드용)')
    parser.add_argument('--output', type=str,
                       help='출력 파일 경로 (video 모드용)')
    parser.add_argument('--camera', type=int, default=0,
                       help='카메라 ID (realtime 모드용)')
    parser.add_argument('--manual', type=str,
                       choices=['squat', 'push_up', 'deadlift', 'bench_press', 'lunge'],
                       help='수동 운동 선택 (AI 감지 건너뛰기)')
    
    args = parser.parse_args()
    
    # 완전 자동화 분석기 초기화
    try:
        analyzer = CompleteAutoExerciseAnalyzer()
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        return 1
    
    print("🤖 완전 자동화 운동 분석 시스템 시작!")
    print("="*80)
    print("🎯 주요 기능:")
    print("  🤖 AI 자동 운동 감지 (5종목)")
    print("  📐 정밀 각도 분석")
    print("  🗣️ 운동별 맞춤 상세 피드백")
    print("  🌈 실시간 초록/빨강 피드백")
    print("  📊 성과 추적")
    print("  📷 사진/🎬 영상/🎮 실시간 모두 지원")
    
    try:
        if args.mode == 'realtime':
            print(f"\n🎮 실시간 분석 시작 (카메라 {args.camera})")
            if args.manual:
                exercise_info = analyzer.exercise_info.get(args.manual, {})
                emoji = exercise_info.get('emoji', '🏋️')
                name_kr = exercise_info.get('name_kr', args.manual)
                print(f"🔧 수동 모드: {emoji} {name_kr}")
            success = analyzer.run_realtime_analysis(args.camera, args.manual)
            return 0 if success else 1
            
        elif args.mode == 'image':
            if not args.input:
                print("❌ --input 옵션이 필요합니다 (이미지 파일 경로)")
                return 1
            
            print(f"\n📷 이미지 분석 시작: {args.input}")
            result = analyzer.analyze_single_image(args.input)
            
            if result.get('success', False):
                # 결과 출력
                exercise_info = result['exercise_info']
                emoji = exercise_info.get('emoji', '🏋️')
                name_kr = exercise_info.get('name_kr', 'unknown')
                exercise_conf = result['exercise_confidence']
                pose_result = result['pose_analysis']
                
                print(f"\n🎉 이미지 분석 완료!")
                print(f"🤖 AI 감지: {emoji} {name_kr} (신뢰도: {exercise_conf:.1%})")
                
                if pose_result['valid']:
                    pose_quality = pose_result['classification']
                    pose_conf = pose_result['confidence']
                    
                    status_emoji = "✅" if pose_quality == 'good' else "⚠️"
                    print(f"🎯 자세 분석: {status_emoji} {pose_quality.upper()} (점수: {pose_conf:.1%})")
                    
                    # 피드백 메시지 출력
                    feedback_messages = result['feedback_messages']
                    if feedback_messages:
                        print(f"\n💬 상세 피드백:")
                        for i, message in enumerate(feedback_messages[:5], 1):
                            print(f"  {i}. {message}")
                    
                    # 위반사항 출력
                    violations = pose_result.get('violations', [])
                    if violations:
                        print(f"\n📐 각도 분석:")
                        for violation in violations[:3]:
                            joint_kr = violation.get('name_kr', violation['joint'])
                            angle = violation['angle']
                            range_min, range_max = violation['expected_range']
                            print(f"  • {joint_kr}: {angle:.1f}° → 목표: {range_min:.0f}-{range_max:.0f}°")
                
                # 주석 이미지 표시
                annotated_image = result['annotated_image']
                
                # 이미지 크기 조정 (화면에 맞게)
                height, width = annotated_image.shape[:2]
                if width > 1200:
                    scale = 1200 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    annotated_image = cv2.resize(annotated_image, (new_width, new_height))
                
                window_title = f"완전 자동 분석 결과: {emoji} {name_kr}"
                cv2.imshow(window_title, annotated_image)
                
                print(f"\n🖼️ 분석 결과 이미지 표시 중... (아무 키나 눌러서 닫기)")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            else:
                print(f"❌ 이미지 분석 실패: {result.get('error', '알 수 없는 오류')}")
                return 1
                
        elif args.mode == 'video':
            if not args.input:
                print("❌ --input 옵션이 필요합니다 (영상 파일 경로)")
                return 1
            
            print(f"\n🎬 영상 분석 시작: {args.input}")
            if args.output:
                print(f"📁 출력 경로: {args.output}")
            
            result = analyzer.analyze_video_file(args.input, args.output)
            
            if result.get('success', False):
                # 결과 출력
                main_exercise = result['main_exercise']
                exercise_info = analyzer.exercise_info.get(main_exercise, {})
                emoji = exercise_info.get('emoji', '🏋️')
                name_kr = exercise_info.get('name_kr', main_exercise)
                
                stats = result['stats']
                success_rate = result['success_rate']
                total_analyzed = result['total_frames_analyzed']
                
                print(f"\n🎉 영상 분석 완료!")
                print(f"🎯 주요 운동: {emoji} {name_kr}")
                print(f"📊 분석 결과:")
                print(f"  • 총 분석 프레임: {total_analyzed}개")
                print(f"  • ✅ Good 자세: {stats['good']}프레임")
                print(f"  • ❌ Bad 자세: {stats['bad']}프레임")
                print(f"  • 🎯 성공률: {success_rate:.1f}%")
                
                # 운동 감지 통계
                exercise_detections = result['exercise_detections']
                if len(exercise_detections) > 1:
                    print(f"\n📈 운동 감지 통계:")
                    for exercise, count in exercise_detections.items():
                        info = analyzer.exercise_info.get(exercise, {})
                        emoji = info.get('emoji', '🏋️')
                        name_kr = info.get('name_kr', exercise)
                        percentage = (count / sum(exercise_detections.values())) * 100
                        print(f"  • {emoji} {name_kr}: {count}프레임 ({percentage:.1f}%)")
                
                if args.output:
                    print(f"\n💾 주석 영상 저장: {args.output}")
                
            else:
                print(f"❌ 영상 분석 실패: {result.get('error', '알 수 없는 오류')}")
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