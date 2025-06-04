"""
간소화된 실시간 운동 분석기
- 훈련된 모델로 운동 자동 분류
- 실시간 자세 분석 (good/bad)
- 직관적 비주얼 피드백
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

def import_modules():
    """필요한 모듈들 import"""
    try:
        from exercise_classifier import ExerciseClassificationModel
        from pose_analysis_system import ExerciseClassifier, PostProcessor
        return ExerciseClassificationModel, ExerciseClassifier, PostProcessor
    except ImportError as e:
        print(f"❌ 모듈 import 오류: {e}")
        print("필요한 파일들이 있는지 확인하세요:")
        print("  - exercise_classifier.py")
        print("  - pose_analysis_system.py")
        return None, None, None

class SimplifiedRealTimeAnalyzer:
    """간소화된 실시간 분석기"""
    
    def __init__(self, model_path: str = "models/exercise_classifier.pkl"):
        # 모듈 import
        ExerciseClassificationModel, ExerciseClassifier, PostProcessor = import_modules()
        if not all([ExerciseClassificationModel, ExerciseClassifier, PostProcessor]):
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
                print(f"✅ 운동 분류 모델 로드 완료: {model_path}")
            else:
                print(f"❌ 모델 로드 실패: {model_path}")
        else:
            print(f"❌ 모델 파일 없음: {model_path}")
            print("먼저 모델을 훈련하세요: python main.py --mode train")
        
        # 자세 분석기 초기화
        try:
            self.pose_analyzer = ExerciseClassifier()
            self.post_processor = PostProcessor(hysteresis_threshold=0.2, ema_alpha=0.3)
        except Exception as e:
            print(f"❌ 자세 분석기 초기화 실패: {e}")
            self.init_success = False
            return
        
        # 현재 상태
        self.current_exercise = "detecting..."
        self.current_pose_quality = "unknown"
        self.classification_confidence = 0.0
        self.pose_confidence = 0.0
        
        # 안정화를 위한 히스토리
        self.exercise_history = deque(maxlen=15)  # 운동 분류 안정화
        self.pose_history = deque(maxlen=5)       # 자세 분석 안정화
        
        # 타이밍 제어
        self.last_classification_time = 0
        self.classification_interval = 2.0  # 2초마다 운동 분류
        
        # 통계
        self.good_count = 0
        self.bad_count = 0
        
        # 임시 파일 디렉토리
        self.temp_dir = tempfile.mkdtemp()
        
        self.init_success = True
    
    def classify_current_exercise(self, frame: np.ndarray) -> Tuple[str, float]:
        """현재 프레임의 운동 분류"""
        current_time = time.time()
        
        # 분류 주기 제어 (2초마다)
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
                    # 가장 자주 나온 운동 선택
                    from collections import Counter
                    exercises = [ex for ex, conf in recent_predictions]
                    most_common = Counter(exercises).most_common(1)[0]
                    
                    if most_common[1] >= 3:  # 최소 3번 이상 나온 경우만
                        new_exercise = most_common[0]
                        
                        # 운동이 바뀐 경우에만 업데이트
                        if new_exercise != self.current_exercise:
                            self.current_exercise = new_exercise
                            self.classification_confidence = confidence
                            print(f"🎯 운동 감지: {new_exercise.upper()} (신뢰도: {confidence:.2f})")
            
            self.last_classification_time = current_time
            
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return self.current_exercise, self.classification_confidence
            
        except Exception as e:
            print(f"운동 분류 오류: {e}")
            return self.current_exercise, self.classification_confidence
    
    def analyze_pose_quality(self, landmarks) -> Tuple[str, Dict]:
        """자세 품질 분석"""
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
            
            # 자세 분석
            analysis = self.pose_analyzer.analyze_pose(landmarks_dict, self.current_exercise)
            if not analysis['valid']:
                return "invalid", {}
            
            # 후처리 적용
            final_result = self.post_processor.process(analysis)
            pose_quality = final_result['final_classification']
            self.pose_confidence = final_result.get('confidence', 0.0)
            
            # 안정화
            self.pose_history.append(pose_quality)
            
            if len(self.pose_history) >= 3:
                recent_poses = list(self.pose_history)[-3:]
                good_count = recent_poses.count('good')
                self.current_pose_quality = 'good' if good_count >= 2 else 'bad'
            else:
                self.current_pose_quality = pose_quality
            
            return self.current_pose_quality, final_result
            
        except Exception as e:
            print(f"자세 분석 오류: {e}")
            return "error", {}
    
    def draw_enhanced_feedback(self, image: np.ndarray, exercise: str, pose_quality: str, 
                             classification_conf: float, pose_conf: float, 
                             analysis_result: Dict) -> np.ndarray:
        """향상된 비주얼 피드백"""
        height, width = image.shape[:2]
        
        # 🎯 메인 상태에 따른 테두리 색상
        if exercise in ["detecting...", "model_not_loaded"]:
            border_color = (255, 255, 0)  # 노란색: 대기중
            status_text = "DETECTING EXERCISE..."
        elif pose_quality == "good":
            border_color = (0, 255, 0)    # 초록색: 좋은 자세
            status_text = "EXCELLENT FORM!"
        elif pose_quality == "bad":
            border_color = (0, 0, 255)    # 빨간색: 나쁜 자세
            status_text = "IMPROVE FORM!"
        else:
            border_color = (128, 128, 128) # 회색: 분석중
            status_text = "ANALYZING..."
        
        # 두꺼운 테두리 그리기
        border_thickness = 20
        cv2.rectangle(image, (0, 0), (width, height), border_color, border_thickness)
        
        # 🎯 상단 정보 패널
        panel_height = 120
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 운동 종목 표시
        if exercise not in ["detecting...", "model_not_loaded"]:
            exercise_text = f"EXERCISE: {exercise.upper().replace('_', ' ')}"
            cv2.putText(image, exercise_text, (30, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # 분류 신뢰도
            conf_text = f"Confidence: {classification_conf:.1%}"
            cv2.putText(image, conf_text, (30, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 🎯 중앙 상태 표시
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        status_x = (width - status_size[0]) // 2
        status_y = height // 2 - 100
        
        # 상태 텍스트 배경
        cv2.rectangle(image, (status_x - 20, status_y - 40), 
                     (status_x + status_size[0] + 20, status_y + 10), (0, 0, 0), -1)
        
        cv2.putText(image, status_text, (status_x, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, border_color, 3)
        
        # 📊 하단 통계 패널
        if exercise not in ["detecting...", "model_not_loaded"]:
            total_frames = self.good_count + self.bad_count
            if total_frames > 0:
                good_ratio = self.good_count / total_frames
                
                stats_y = height - 80
                cv2.rectangle(image, (0, stats_y), (width, height), (0, 0, 0), -1)
                
                stats_text = f"GOOD: {self.good_count}  |  BAD: {self.bad_count}  |  SUCCESS RATE: {good_ratio:.1%}"
                stats_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                stats_x = (width - stats_size[0]) // 2
                
                cv2.putText(image, stats_text, (stats_x, stats_y + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 🔍 실시간 피드백 (운동별 구체적 조언)
        if analysis_result.get('violations') and pose_quality == "bad":
            feedback_y = height - 200
            cv2.putText(image, "ADJUSTMENTS NEEDED:", (30, feedback_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            for i, violation in enumerate(analysis_result['violations'][:2]):
                joint = violation['joint'].replace('_', ' ').title()
                feedback_text = f"• {joint} Position"
                cv2.putText(image, feedback_text, (30, feedback_y + 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # ⌨️ 조작 가이드
        guide_text = "Q: Quit  |  R: Reset Stats  |  S: Screenshot"
        cv2.putText(image, guide_text, (30, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return image
    
    def run_analysis(self, camera_id: int = 0):
        """실시간 분석 실행"""
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
        
        print("\n" + "="*60)
        print("🏋️ BLAZE - 실시간 운동 자세 분석 시스템")
        print("="*60)
        print("✨ 기능:")
        print("  • 자동 운동 분류 (스쿼트, 푸시업, 벤치프레스, 데드리프트, 풀업)")
        print("  • 실시간 자세 분석 (Good/Bad)")
        print("  • 스마트 피드백 시스템")
        print("  • 성과 추적")
        print("\n⌨️ 조작법:")
        print("  Q: 종료  |  R: 통계 리셋  |  S: 스크린샷 저장")
        print("="*60)
        
        if not self.model_loaded:
            print("\n❌ 경고: 운동 분류 모델이 로드되지 않았습니다!")
            print("먼저 모델을 훈련하세요:")
            print("python main.py --mode train")
            print("\n그래도 자세 분석은 가능하지만 운동을 수동으로 지정해야 합니다.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 카메라 읽기 실패")
                    break
                
                # 좌우 반전 (셀카 모드)
                frame = cv2.flip(frame, 1)
                
                # MediaPipe 처리
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # 1단계: 운동 분류
                    exercise, class_conf = self.classify_current_exercise(frame)
                    
                    # 2단계: 자세 분석
                    pose_quality, analysis_result = self.analyze_pose_quality(results.pose_landmarks.landmark)
                    
                    # 통계 업데이트
                    if pose_quality == 'good':
                        self.good_count += 1
                    elif pose_quality == 'bad':
                        self.bad_count += 1
                    
                    # 3단계: 비주얼 피드백
                    frame = self.draw_enhanced_feedback(
                        frame, exercise, pose_quality, class_conf, self.pose_confidence, analysis_result
                    )
                
                else:
                    # 포즈 미감지
                    cv2.putText(frame, "STAND IN FRONT OF CAMERA", 
                               (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # 화면 출력
                cv2.imshow('🏋️ BLAZE - Exercise Analysis', frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # 통계 리셋
                    self.good_count = 0
                    self.bad_count = 0
                    self.exercise_history.clear()
                    self.pose_history.clear()
                    self.current_exercise = "detecting..."
                    print("📊 통계 리셋 완료")
                elif key == ord('s'):
                    # 스크린샷 저장
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # 스크린샷 폴더 확인/생성
                    screenshot_dir = "outputs/screenshots"
                    os.makedirs(screenshot_dir, exist_ok=True)
                    
                    filename = f"{screenshot_dir}/blaze_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 스크린샷 저장: {filename}")
                
        except KeyboardInterrupt:
            print("\n⏹️ 사용자 중단")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 임시 디렉토리 정리
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            
            # 최종 리포트
            total = self.good_count + self.bad_count
            if total > 0:
                print(f"\n📊 세션 완료!")
                print(f"  총 분석: {total} 프레임")
                print(f"  성공: {self.good_count} ({self.good_count/total:.1%})")
                print(f"  개선 필요: {self.bad_count} ({self.bad_count/total:.1%})")
                print("🎯 계속 연습해서 폼을 완성하세요!")
            
            return True

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='🏋️ BLAZE 실시간 운동 분석')
    parser.add_argument('--camera', type=int, default=0, help='카메라 ID')
    parser.add_argument('--model', type=str, default='models/exercise_classifier.pkl',
                       help='운동 분류 모델 경로')
    
    args = parser.parse_args()
    
    analyzer = SimplifiedRealTimeAnalyzer(args.model)
    analyzer.run_analysis(args.camera)

if __name__ == "__main__":
    main()