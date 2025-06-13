import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import json
from typing import Dict, List, Tuple, Optional
import argparse

class RealtimePoseAnalyzer:
    """실시간 운동 자세 분석기 - enhanced_pose_analysis.py 기준 적용"""
    
    def __init__(self, exercise_type: str = 'squat'):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 실시간 처리를 위해 낮은 복잡도 사용
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.exercise_type = exercise_type
        
        # 🎯 enhanced_pose_analysis.py와 완전히 동일한 각도 기준 적용
        self.exercise_thresholds = {
            'squat': [
                {'name': 'left_knee', 'points': [23, 25, 27], 'range': (55, 140), 'weight': 1.1},      # enhanced와 동일
                {'name': 'right_knee', 'points': [24, 26, 28], 'range': (55, 140), 'weight': 1.1},     # enhanced와 동일
                {'name': 'left_hip', 'points': [11, 23, 25], 'range': (55, 140), 'weight': 0.9},       # enhanced와 동일
                {'name': 'right_hip', 'points': [12, 24, 26], 'range': (55, 140), 'weight': 0.9},      # enhanced와 동일
                {'name': 'back_straight', 'points': [11, 23, 25], 'range': (110, 170), 'weight': 1.1}, # enhanced와 동일
                {'name': 'spine_angle', 'points': [23, 11, 13], 'range': (110, 170), 'weight': 0.9},   # enhanced와 동일
            ],
            'push_up': [
                {'name': 'left_elbow', 'points': [11, 13, 15], 'range': (40, 160), 'weight': 1.0},     # enhanced와 동일
                {'name': 'right_elbow', 'points': [12, 14, 16], 'range': (40, 160), 'weight': 1.0},    # enhanced와 동일
                {'name': 'body_line', 'points': [11, 23, 25], 'range': (140, 180), 'weight': 1.2},     # enhanced와 동일
                {'name': 'leg_straight', 'points': [23, 25, 27], 'range': (140, 180), 'weight': 0.8},  # enhanced와 동일
                {'name': 'shoulder_alignment', 'points': [13, 11, 23], 'range': (120, 180), 'weight': 0.6}, # enhanced와 동일
                {'name': 'core_stability', 'points': [11, 12, 23], 'range': (140, 180), 'weight': 1.0}, # enhanced와 동일
            ],
            'deadlift': [
                # 🏋️‍♂️ 데드리프트: enhanced에서 대폭 완화된 기준 그대로 적용
                {'name': 'left_knee', 'points': [23, 25, 27], 'range': (80, 140), 'weight': 0.6},      # enhanced와 동일 (대폭 완화)
                {'name': 'right_knee', 'points': [24, 26, 28], 'range': (80, 140), 'weight': 0.6},     # enhanced와 동일 (대폭 완화)
                {'name': 'hip_hinge', 'points': [11, 23, 25], 'range': (80, 180), 'weight': 0.7},      # enhanced와 동일 (대폭 완화)
                {'name': 'back_straight', 'points': [11, 23, 12], 'range': (120, 180), 'weight': 1.0}, # enhanced와 동일 (완화)
                {'name': 'chest_up', 'points': [23, 11, 13], 'range': (50, 140), 'weight': 0.5},       # enhanced와 동일
                {'name': 'spine_neutral', 'points': [23, 11, 24], 'range': (120, 180), 'weight': 0.8}, # enhanced와 동일
            ],
            'bench_press': [
                {'name': 'left_elbow', 'points': [11, 13, 15], 'range': (50, 145), 'weight': 1.1},     # enhanced와 동일
                {'name': 'right_elbow', 'points': [12, 14, 16], 'range': (50, 145), 'weight': 1.1},    # enhanced와 동일
                {'name': 'left_shoulder', 'points': [13, 11, 23], 'range': (50, 150), 'weight': 0.9},  # enhanced와 동일
                {'name': 'right_shoulder', 'points': [14, 12, 24], 'range': (50, 150), 'weight': 0.9}, # enhanced와 동일
                {'name': 'back_arch', 'points': [11, 23, 25], 'range': (90, 170), 'weight': 0.7},      # enhanced와 동일
                {'name': 'wrist_alignment', 'points': [13, 15, 17], 'range': (70, 180), 'weight': 0.6}, # enhanced와 동일
            ],
            'lunge': [
                # 🚀 런지: enhanced와 완전히 동일한 기준
                {'name': 'front_knee', 'points': [23, 25, 27], 'range': (70, 120), 'weight': 1.2},     # enhanced와 동일
                {'name': 'back_knee', 'points': [24, 26, 28], 'range': (120, 180), 'weight': 1.0},     # enhanced와 동일
                {'name': 'front_hip', 'points': [11, 23, 25], 'range': (70, 120), 'weight': 0.8},      # enhanced와 동일
                {'name': 'torso_upright', 'points': [11, 23, 25], 'range': (100, 180), 'weight': 1.2}, # enhanced와 동일
                {'name': 'front_ankle', 'points': [25, 27, 31], 'range': (80, 110), 'weight': 0.8},    # enhanced와 동일
                {'name': 'back_hip_extension', 'points': [12, 24, 26], 'range': (150, 180), 'weight': 1.0}, # enhanced와 동일
            ]
        }
        
        # enhanced와 동일한 후처리 설정
        self.ema_alpha = 0.3              # enhanced와 동일
        self.window_size = 15             # enhanced와 동일
        self.visibility_threshold = 0.25  # enhanced와 동일
        
        # enhanced와 동일한 운동별 분류 임계값
        self.classification_thresholds = {
            'squat': 0.5,        # enhanced와 동일
            'push_up': 0.7,      # enhanced와 동일
            'deadlift': 0.8,     # enhanced와 동일 (대폭 완화)
            'bench_press': 0.5,  # enhanced와 동일
            'lunge': 0.6,        # enhanced와 동일
        }
        
        # enhanced와 동일한 히스테리시스 설정
        self.exercise_hysteresis = {
            'squat': 0.5,        # enhanced와 동일
            'push_up': 0.7,      # enhanced와 동일
            'deadlift': 0.8,     # enhanced와 동일 (대폭 완화)
            'bench_press': 0.5,  # enhanced와 동일
            'lunge': 0.6,        # enhanced와 동일
        }
        
        # enhanced와 동일한 복귀 임계값
        self.recovery_thresholds = {
            'squat': 0.35,       # 0.5 * 0.7 (enhanced와 동일)
            'push_up': 0.56,     # 0.7 * 0.8 (enhanced와 동일)
            'deadlift': 0.72,    # 0.8 * 0.9 (enhanced와 동일 - 매우 쉬운 복귀)
            'bench_press': 0.35, # 0.5 * 0.7 (enhanced와 동일)
            'lunge': 0.48,       # 0.6 * 0.8 (enhanced와 동일)
        }
        
        # 상태 추적
        self.history = deque(maxlen=self.window_size)
        self.ema_value = None
        self.last_state = 'good'
        self.state_counter = {'good': 0, 'bad': 0}
        
        # 피드백 시스템
        self.feedback_messages = deque(maxlen=5)
        self.last_feedback_time = 0
        
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """세 점 사이의 각도 계산 (enhanced와 동일)"""
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            
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
    
    def analyze_frame(self, landmarks) -> Dict:
        """프레임 분석 - enhanced_pose_analysis.py와 동일한 방식"""
        if self.exercise_type not in self.exercise_thresholds:
            return {'valid': False, 'error': 'Unknown exercise type'}
        
        thresholds = self.exercise_thresholds[self.exercise_type]
        angles = {}
        violations = []
        weighted_violation_score = 0.0
        total_weight = 0.0
        
        for threshold in thresholds:
            try:
                p1_idx, p2_idx, p3_idx = threshold['points']
                min_angle, max_angle = threshold['range']
                weight = threshold['weight']
                
                # enhanced와 동일한 가시성 확인 (0.25)
                if (landmarks[p1_idx].visibility < self.visibility_threshold or 
                    landmarks[p2_idx].visibility < self.visibility_threshold or 
                    landmarks[p3_idx].visibility < self.visibility_threshold):
                    continue
                
                p1 = np.array([landmarks[p1_idx].x, landmarks[p1_idx].y])
                p2 = np.array([landmarks[p2_idx].x, landmarks[p2_idx].y])
                p3 = np.array([landmarks[p3_idx].x, landmarks[p3_idx].y])
                
                angle = self.calculate_angle(p1, p2, p3)
                angles[threshold['name']] = angle
                
                # 허용 범위 확인
                if not (min_angle <= angle <= max_angle):
                    violations.append({
                        'joint': threshold['name'],
                        'angle': angle,
                        'expected_range': (min_angle, max_angle),
                        'weight': weight
                    })
                    weighted_violation_score += weight
                
                total_weight += weight
                    
            except Exception as e:
                continue
        
        # enhanced와 동일한 분류 기준
        violation_ratio = weighted_violation_score / max(total_weight, 1.0)
        
        return {
            'valid': True,
            'angles': angles,
            'violations': violations,
            'violation_count': len(violations),
            'weighted_violation_ratio': violation_ratio,
            'total_weight': total_weight,
            'enhanced_compatible': True
        }
    
    def apply_post_processing(self, analysis_result: Dict) -> Dict:
        """enhanced와 동일한 후처리 적용"""
        if not analysis_result['valid']:
            return analysis_result
        
        # 위반 비율 계산
        violation_ratio = analysis_result.get('weighted_violation_ratio', 0)
        
        # enhanced와 동일한 EMA 적용
        if self.ema_value is None:
            self.ema_value = violation_ratio
        else:
            self.ema_value = self.ema_alpha * violation_ratio + (1 - self.ema_alpha) * self.ema_value
        
        # 히스토리 추가
        self.history.append(self.ema_value)
        
        # enhanced와 동일한 히스테리시스 적용
        classification_threshold = self.classification_thresholds.get(self.exercise_type, 0.6)
        hysteresis_threshold = self.exercise_hysteresis.get(self.exercise_type, 0.6)
        recovery_threshold = self.recovery_thresholds.get(self.exercise_type, 0.48)
        
        if self.last_state == 'good':
            if self.ema_value > hysteresis_threshold:
                self.last_state = 'bad'
        else:
            if self.ema_value < recovery_threshold:
                self.last_state = 'good'
        
        # 상태 카운터 업데이트
        self.state_counter[self.last_state] += 1
        
        return {
            **analysis_result,
            'final_classification': self.last_state,
            'violation_ratio': violation_ratio,
            'smoothed_ratio': self.ema_value,
            'confidence': 1.0 - self.ema_value,
            'classification_threshold': classification_threshold,
            'hysteresis_threshold': hysteresis_threshold,
            'recovery_threshold': recovery_threshold,
            'enhanced_compatible': True
        }
    
    def generate_feedback(self, analysis_result: Dict) -> str:
        """enhanced 기준에 맞춘 피드백 메시지 생성"""
        current_time = time.time()
        
        # 피드백 주기 제한 (2초마다)
        if current_time - self.last_feedback_time < 2.0:
            return ""
        
        if not analysis_result['valid']:
            return "포즈를 인식할 수 없습니다"
        
        feedback = ""
        violations = analysis_result['violations']
        
        # enhanced 기준 운동별 맞춤 피드백 메시지
        exercise_feedback = {
            'squat': {
                'good': "완벽한 스쿼트 자세입니다! (enhanced 기준)",
                'bad_knee': "무릎 각도를 55-140도로 조정하세요",
                'bad_hip': "엉덩이 각도를 55-140도로 조정하세요",
                'bad_back': "등을 110-170도로 곧게 펴세요"
            },
            'push_up': {
                'good': "훌륭한 푸쉬업 폼입니다! (enhanced 기준)",
                'bad_elbow': "팔꿈치를 40-160도로 조정하세요",
                'bad_body': "몸을 140-180도 일직선으로 유지하세요",
                'bad_shoulder': "어깨 정렬을 120-180도로 유지하세요"
            },
            'deadlift': {
                'good': "완벽한 데드리프트 자세입니다! (enhanced 완화 기준)",
                'bad_knee': "무릎을 80-140도로 조정하세요 (완화됨)",
                'bad_hip': "엉덩이를 80-180도로 뒤로 빼세요 (완화됨)",
                'bad_back': "등을 120-180도로 곧게 펴세요 (완화됨)",
                'bad_chest': "가슴을 50-140도로 펴세요"
            },
            'bench_press': {
                'good': "완벽한 벤치프레스입니다! (enhanced 기준)",
                'bad_elbow': "팔꿈치를 50-145도로 조정하세요",
                'bad_shoulder': "어깨를 50-150도로 조정하세요",
                'bad_arch': "자연스러운 등 아치(90-170도)를 유지하세요"
            },
            'lunge': {
                'good': "완벽한 런지 자세입니다! (enhanced 기준)",
                'bad_front_knee': "앞무릎을 70-120도로 구부리세요",
                'bad_back_knee': "뒷무릎을 120-180도로 펴세요",
                'bad_torso': "상체를 100-180도로 곧게 세우세요",
                'bad_ankle': "앞발목을 80-110도로 안정화하세요"
            }
        }
        
        if len(violations) == 0:
            feedback = exercise_feedback.get(self.exercise_type, {}).get('good', "좋은 자세입니다! (enhanced 기준)")
        else:
            # 운동별 특화 피드백
            for violation in violations[:2]:  # 최대 2개 피드백
                joint = violation['joint']
                angle = violation['angle']
                expected_range = violation['expected_range']
                
                if 'knee' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_knee', 
                        f"{joint}: {angle:.1f}° → {expected_range[0]}-{expected_range[1]}°") + ", "
                elif 'hip' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_hip', 
                        f"{joint}: {angle:.1f}° → {expected_range[0]}-{expected_range[1]}°") + ", "
                elif 'elbow' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_elbow', 
                        f"{joint}: {angle:.1f}° → {expected_range[0]}-{expected_range[1]}°") + ", "
                elif 'shoulder' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_shoulder', 
                        f"{joint}: {angle:.1f}° → {expected_range[0]}-{expected_range[1]}°") + ", "
                elif 'back' in joint or 'spine' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_back', 
                        f"{joint}: {angle:.1f}° → {expected_range[0]}-{expected_range[1]}°") + ", "
                elif 'torso' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_torso', 
                        f"{joint}: {angle:.1f}° → {expected_range[0]}-{expected_range[1]}°") + ", "
                elif 'ankle' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_ankle', 
                        f"{joint}: {angle:.1f}° → {expected_range[0]}-{expected_range[1]}°") + ", "
                else:
                    feedback += f"{joint}: {angle:.1f}° → {expected_range[0]}-{expected_range[1]}°, "
        
        self.last_feedback_time = current_time
        return feedback.rstrip(', ')
    
    def draw_pose_info(self, image: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """enhanced 기준 정보를 이미지에 그리기"""
        height, width = image.shape[:2]
        
        # 운동 이모지
        exercise_emojis = {
            'squat': '🏋️‍♀️',
            'push_up': '💪',
            'deadlift': '🏋️‍♂️',
            'bench_press': '🔥',
            'lunge': '🚀'
        }
        
        # 상태 표시
        state = analysis_result.get('final_classification', 'unknown')
        color = (0, 255, 0) if state == 'good' else (0, 0, 255)
        
        # 운동 종목과 상태 (enhanced 표시)
        exercise_text = f"{exercise_emojis.get(self.exercise_type, '🏋️')} {self.exercise_type.upper()}: {state.upper()} (Enhanced)"
        cv2.putText(image, exercise_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # enhanced 기준 정보 표시
        enhanced_info = f"Enhanced Criteria Applied"
        cv2.putText(image, enhanced_info, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 신뢰도 및 임계값 표시
        confidence = analysis_result.get('confidence', 0)
        threshold = analysis_result.get('classification_threshold', 0.6)
        cv2.putText(image, f"Confidence: {confidence:.2f} | Threshold: {threshold}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 각도 정보 표시 (enhanced 범위와 함께)
        if 'angles' in analysis_result:
            y_offset = 130
            thresholds = self.exercise_thresholds.get(self.exercise_type, [])
            threshold_dict = {t['name']: t['range'] for t in thresholds}
            
            for joint, angle in analysis_result['angles'].items():
                expected_range = threshold_dict.get(joint, (0, 180))
                in_range = expected_range[0] <= angle <= expected_range[1]
                angle_color = (0, 255, 0) if in_range else (0, 0, 255)
                
                angle_text = f"{joint}: {angle:.1f}° ({expected_range[0]}-{expected_range[1]}°)"
                cv2.putText(image, angle_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, angle_color, 1)
                y_offset += 25
        
        # 피드백 메시지 표시
        feedback = self.generate_feedback(analysis_result)
        if feedback:
            self.feedback_messages.append(feedback)
        
        if self.feedback_messages:
            y_offset = height - 150
            for msg in list(self.feedback_messages)[-3:]:  # 최대 3개 메시지
                cv2.putText(image, msg, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
        
        # enhanced 통계 표시
        total_frames = self.state_counter['good'] + self.state_counter['bad']
        if total_frames > 0:
            good_ratio = self.state_counter['good'] / total_frames
            target_range = {
                'squat': '50-70%',
                'push_up': '50-70%',
                'deadlift': '40-60%',  # 완화된 목표
                'bench_press': '50-70%',
                'lunge': '50-70%'
            }.get(self.exercise_type, '50-70%')
            
            stats_text = f"Good: {good_ratio:.1%} | Target: {target_range} (Enhanced)"
            cv2.putText(image, stats_text, (width - 400, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    def run_camera(self, camera_id: int = 0):
        """enhanced 기준으로 카메라 실시간 분석"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"🎯 Enhanced 기준 실시간 분석 시작: {self.exercise_type}")
        print("📊 적용된 기준:")
        thresholds = self.exercise_thresholds.get(self.exercise_type, [])
        for threshold in thresholds[:3]:  # 처음 3개만 표시
            name = threshold['name']
            range_val = threshold['range']
            weight = threshold['weight']
            print(f"  • {name}: {range_val[0]}-{range_val[1]}° (weight: {weight})")
        
        print(f"🎯 분류 임계값: {self.classification_thresholds.get(self.exercise_type, 0.6)}")
        print("Press 'q' to quit, 'r' to reset counters, 's' to save screenshot")
        
        frame_count = 0
        fps_counter = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # 좌우 반전 (셀카 모드)
                frame = cv2.flip(frame, 1)
                
                # RGB 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 포즈 검출
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # enhanced 기준 자세 분석
                    analysis = self.analyze_frame(results.pose_landmarks.landmark)
                    final_result = self.apply_post_processing(analysis)
                    
                    # 정보 표시
                    frame = self.draw_pose_info(frame, final_result)
                else:
                    cv2.putText(frame, "No pose detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # FPS 계산
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                fps_counter.append(fps)
                avg_fps = sum(fps_counter) / len(fps_counter)
                
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 프레임 표시
                cv2.imshow(f'{self.exercise_type.replace("_", " ").title()} Enhanced Analysis', frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.state_counter = {'good': 0, 'bad': 0}
                    self.history.clear()
                    self.ema_value = None
                    self.last_state = 'good'
                    print("✅ Enhanced 기준 카운터 리셋")
                elif key == ord('s'):
                    filename = f"enhanced_screenshot_{self.exercise_type}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Enhanced 스크린샷 저장: {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⏹️ Enhanced 분석 중단")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # enhanced 기준 최종 통계 출력
            total_frames = self.state_counter['good'] + self.state_counter['bad']
            if total_frames > 0:
                good_ratio = self.state_counter['good'] / total_frames
                target_ranges = {
                    'squat': (0.5, 0.7),
                    'push_up': (0.5, 0.7),
                    'deadlift': (0.4, 0.6),  # 완화된 목표
                    'bench_press': (0.5, 0.7),
                    'lunge': (0.5, 0.7)
                }
                target_range = target_ranges.get(self.exercise_type, (0.5, 0.7))
                target_met = target_range[0] <= good_ratio <= target_range[1]
                
                print(f"\n📊 Enhanced 기준 최종 통계 ({self.exercise_type}):")
                print(f"  🎯 총 분석 프레임: {total_frames}")
                print(f"  ✅ Good 자세: {self.state_counter['good']} ({good_ratio:.1%})")
                print(f"  ❌ Bad 자세: {self.state_counter['bad']} ({1-good_ratio:.1%})")
                print(f"  🎯 목표 범위: {target_range[0]:.0%}-{target_range[1]:.0%}")
                print(f"  📈 목표 달성: {'✅ 성공' if target_met else '❌ 미달성'}")
                print(f"  🔧 적용 기준: Enhanced Pose Analysis")
    
    def analyze_video(self, video_path: str, output_path: str = None):
        """enhanced 기준으로 비디오 파일 분석"""
        cap = cv2.VideoCapture(video_path)
        
        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎯 Enhanced 기준 비디오 분석: {video_path}")
        print(f"📹 해상도: {width}x{height}, FPS: {fps}, 총 프레임: {total_frames}")
        print(f"🎯 적용 운동: {self.exercise_type}")
        
        # 출력 비디오 설정
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_results = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 포즈 검출
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # enhanced 기준 자세 분석
                    analysis = self.analyze_frame(results.pose_landmarks.landmark)
                    final_result = self.apply_post_processing(analysis)
                    
                    # 정보 표시
                    frame = self.draw_pose_info(frame, final_result)
                    
                    # 결과 저장
                    frame_results.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'classification': final_result.get('final_classification', 'unknown'),
                        'confidence': final_result.get('confidence', 0),
                        'angles': final_result.get('angles', {}),
                        'violations': final_result.get('violations', []),
                        'weighted_violation_ratio': final_result.get('weighted_violation_ratio', 0),
                        'enhanced_compatible': True,
                        'thresholds_applied': {
                            'classification': final_result.get('classification_threshold', 0.6),
                            'hysteresis': final_result.get('hysteresis_threshold', 0.6),
                            'recovery': final_result.get('recovery_threshold', 0.48)
                        }
                    })
                
                # 진행률 표시
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"📊 Enhanced 분석 진행률: {progress:.1f}%")
                
                # 출력 비디오에 쓰기
                if output_path:
                    out.write(frame)
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⏹️ Enhanced 비디오 분석 중단")
        finally:
            cap.release()
            if output_path:
                out.release()
            
            # enhanced 기준 결과 저장
            if frame_results:
                result_file = video_path.replace('.mp4', f'_{self.exercise_type}_enhanced_analysis.json')
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'video_info': {
                            'source_file': video_path,
                            'exercise_type': self.exercise_type,
                            'enhanced_compatible': True,
                            'total_frames': len(frame_results),
                            'fps': fps,
                            'resolution': f"{width}x{height}"
                        },
                        'enhanced_criteria': {
                            'thresholds_used': self.exercise_thresholds.get(self.exercise_type, []),
                            'classification_threshold': self.classification_thresholds.get(self.exercise_type, 0.6),
                            'hysteresis_threshold': self.exercise_hysteresis.get(self.exercise_type, 0.6),
                            'recovery_threshold': self.recovery_thresholds.get(self.exercise_type, 0.48)
                        },
                        'frame_results': frame_results
                    }, f, indent=2, ensure_ascii=False)
                
                # enhanced 기준 통계 계산
                good_frames = sum(1 for r in frame_results if r['classification'] == 'good')
                bad_frames = len(frame_results) - good_frames
                good_ratio = good_frames / len(frame_results)
                
                target_ranges = {
                    'squat': (0.5, 0.7),
                    'push_up': (0.5, 0.7),
                    'deadlift': (0.4, 0.6),  # 완화된 목표
                    'bench_press': (0.5, 0.7),
                    'lunge': (0.5, 0.7)
                }
                target_range = target_ranges.get(self.exercise_type, (0.5, 0.7))
                target_met = target_range[0] <= good_ratio <= target_range[1]
                
                print(f"\n📊 Enhanced 기준 비디오 분석 완료 ({self.exercise_type}):")
                print(f"  📁 결과 저장: {result_file}")
                print(f"  🎯 총 분석 프레임: {len(frame_results)}")
                print(f"  ✅ Good 자세: {good_frames} ({good_ratio:.1%})")
                print(f"  ❌ Bad 자세: {bad_frames} ({1-good_ratio:.1%})")
                print(f"  🎯 목표 범위: {target_range[0]:.0%}-{target_range[1]:.0%}")
                print(f"  📈 목표 달성: {'✅ 성공' if target_met else '❌ 미달성'}")
                print(f"  🔧 적용 기준: Enhanced Pose Analysis")
                
                if output_path:
                    print(f"  🎬 주석 비디오 저장: {output_path}")

def main():
    """메인 함수 - enhanced_pose_analysis.py 기준 적용"""
    parser = argparse.ArgumentParser(description='Enhanced Real-time Exercise Pose Analysis')
    parser.add_argument('--exercise', type=str, default='squat',
                       choices=['squat', 'push_up', 'deadlift', 'bench_press', 'lunge'],
                       help='Exercise type to analyze (enhanced criteria)')
    parser.add_argument('--mode', type=str, default='camera',
                       choices=['camera', 'video'],
                       help='Analysis mode')
    parser.add_argument('--input', type=str, help='Input video file path (for video mode)')
    parser.add_argument('--output', type=str, help='Output video file path (for video mode)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    
    args = parser.parse_args()
    
    # enhanced 기준 분석기 초기화
    analyzer = RealtimePoseAnalyzer(args.exercise)
    
    print(f"🎯 Enhanced Pose Analysis 실시간 분석기 시작")
    print(f"🏋️ 선택된 운동: {args.exercise}")
    print(f"📊 적용 기준: enhanced_pose_analysis.py와 동일")
    
    # 현재 운동의 enhanced 기준 표시
    thresholds = analyzer.exercise_thresholds.get(args.exercise, [])
    classification_threshold = analyzer.classification_thresholds.get(args.exercise, 0.6)
    
    print(f"\n📐 {args.exercise.upper()} Enhanced 기준:")
    for i, threshold in enumerate(thresholds[:5]):  # 처음 5개만 표시
        name = threshold['name']
        range_val = threshold['range']
        weight = threshold['weight']
        print(f"  {i+1}. {name}: {range_val[0]}-{range_val[1]}° (가중치: {weight})")
    
    if len(thresholds) > 5:
        print(f"  ... 총 {len(thresholds)}개 기준 적용")
    
    print(f"🎯 분류 임계값: {classification_threshold}")
    
    if args.mode == 'camera':
        analyzer.run_camera(args.camera)
    elif args.mode == 'video':
        if not args.input:
            print("❌ Error: Input video file is required for video mode")
            return
        analyzer.analyze_video(args.input, args.output)

if __name__ == "__main__":
    main()