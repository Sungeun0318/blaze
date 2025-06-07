import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import json
from typing import Dict, List, Tuple, Optional
import argparse

class RealtimePoseAnalyzer:
    """실시간 운동 자세 분석기 - 풀업→런지 교체"""
    
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
        
        # 🚀 풀업→런지 교체 운동별 각도 기준 (완화됨)
        self.exercise_thresholds = {
            'bench_press': [
                {'name': 'left_elbow', 'points': [11, 13, 15], 'range': (20, 180)},    # 완화됨
                {'name': 'right_elbow', 'points': [12, 14, 16], 'range': (20, 180)},   # 완화됨
                {'name': 'left_shoulder', 'points': [13, 11, 23], 'range': (20, 170)}, # 완화됨
                {'name': 'right_shoulder', 'points': [14, 12, 24], 'range': (20, 170)}, # 완화됨
            ],
            'deadlift': [
                {'name': 'left_knee', 'points': [23, 25, 27], 'range': (100, 180)},   # 완화됨
                {'name': 'right_knee', 'points': [24, 26, 28], 'range': (100, 180)},  # 완화됨
                {'name': 'left_hip', 'points': [11, 23, 25], 'range': (80, 180)},     # 완화됨
                {'name': 'right_hip', 'points': [12, 24, 26], 'range': (80, 180)},    # 완화됨
            ],
            'lunge': [  # 🚀 새로 추가된 런지
                {'name': 'front_knee', 'points': [23, 25, 27], 'range': (70, 130)},   # 앞무릎 (런지 핵심)
                {'name': 'back_knee', 'points': [24, 26, 28], 'range': (120, 180)},   # 뒷무릎 (거의 펴짐)
                {'name': 'front_hip', 'points': [11, 23, 25], 'range': (70, 130)},    # 앞 엉덩이
                {'name': 'torso_upright', 'points': [11, 23, 25], 'range': (160, 180)}, # 상체 직립
                {'name': 'front_ankle', 'points': [25, 27, 31], 'range': (80, 120)},  # 앞발목 안정성
                {'name': 'back_hip_extension', 'points': [12, 24, 26], 'range': (140, 180)}, # 뒷엉덩이 신전
            ],
            'push_up': [
                {'name': 'left_elbow', 'points': [11, 13, 15], 'range': (20, 170)},   # 기존 유지
                {'name': 'right_elbow', 'points': [12, 14, 16], 'range': (20, 170)},  # 기존 유지
                {'name': 'left_hip', 'points': [11, 23, 25], 'range': (100, 180)},    # 기존 유지
                {'name': 'right_hip', 'points': [12, 24, 26], 'range': (100, 180)},   # 기존 유지
            ],
            'squat': [
                {'name': 'left_knee', 'points': [23, 25, 27], 'range': (40, 160)},    # 기존 유지
                {'name': 'right_knee', 'points': [24, 26, 28], 'range': (40, 160)},   # 기존 유지
                {'name': 'left_hip', 'points': [11, 23, 25], 'range': (40, 160)},     # 기존 유지
                {'name': 'right_hip', 'points': [12, 24, 26], 'range': (40, 160)},    # 기존 유지
            ]
        }
        
        # 후처리 설정 (완화됨)
        self.hysteresis_threshold = 0.4  # 기본값 완화
        self.ema_alpha = 0.2
        self.window_size = 10
        
        # 상태 추적
        self.history = deque(maxlen=self.window_size)
        self.ema_value = None
        self.last_state = 'good'
        self.state_counter = {'good': 0, 'bad': 0}
        
        # 피드백 시스템
        self.feedback_messages = deque(maxlen=5)
        self.last_feedback_time = 0
        
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """세 점 사이의 각도 계산"""
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
        except:
            return 0.0
    
    def analyze_frame(self, landmarks) -> Dict:
        """프레임 분석 - 풀업→런지 교체"""
        if self.exercise_type not in self.exercise_thresholds:
            return {'valid': False, 'error': 'Unknown exercise type'}
        
        thresholds = self.exercise_thresholds[self.exercise_type]
        angles = {}
        violations = []
        
        for threshold in thresholds:
            try:
                p1_idx, p2_idx, p3_idx = threshold['points']
                min_angle, max_angle = threshold['range']
                
                # 가시성 확인 (완화됨)
                if (landmarks[p1_idx].visibility < 0.4 or 
                    landmarks[p2_idx].visibility < 0.4 or 
                    landmarks[p3_idx].visibility < 0.4):
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
                        'expected_range': (min_angle, max_angle)
                    })
                    
            except Exception as e:
                continue
        
        return {
            'valid': True,
            'angles': angles,
            'violations': violations,
            'violation_count': len(violations)
        }
    
    def apply_post_processing(self, analysis_result: Dict) -> Dict:
        """후처리 적용 (완화됨)"""
        if not analysis_result['valid']:
            return analysis_result
        
        # 위반 비율 계산
        total_angles = len(analysis_result['angles'])
        violation_count = analysis_result['violation_count']
        violation_ratio = violation_count / total_angles if total_angles > 0 else 0
        
        # EMA 적용
        if self.ema_value is None:
            self.ema_value = violation_ratio
        else:
            self.ema_value = self.ema_alpha * violation_ratio + (1 - self.ema_alpha) * self.ema_value
        
        # 히스토리 추가
        self.history.append(self.ema_value)
        
        # 완화된 히스테리시스 적용
        exercise_thresholds = {
            'squat': 0.5,        # 기존 유지
            'push_up': 0.8,      # 기존 유지
            'deadlift': 0.6,     # 완화됨
            'bench_press': 0.7,  # 완화됨
            'lunge': 0.5,        # 새로운 런지
        }
        
        threshold = exercise_thresholds.get(self.exercise_type, 0.4)
        
        if self.last_state == 'good':
            if self.ema_value > threshold:
                self.last_state = 'bad'
        else:
            if self.ema_value < threshold * 0.4:  # 복귀 기준도 완화
                self.last_state = 'good'
        
        # 상태 카운터 업데이트
        self.state_counter[self.last_state] += 1
        
        return {
            **analysis_result,
            'final_classification': self.last_state,
            'violation_ratio': violation_ratio,
            'smoothed_ratio': self.ema_value,
            'confidence': 1.0 - self.ema_value
        }
    
    def generate_feedback(self, analysis_result: Dict) -> str:
        """피드백 메시지 생성 - 풀업→런지 교체"""
        current_time = time.time()
        
        # 피드백 주기 제한 (2초마다)
        if current_time - self.last_feedback_time < 2.0:
            return ""
        
        if not analysis_result['valid']:
            return "포즈를 인식할 수 없습니다"
        
        feedback = ""
        violations = analysis_result['violations']
        
        # 🚀 운동별 맞춤 피드백 메시지
        exercise_feedback = {
            'squat': {
                'good': "완벽한 스쿼트 자세입니다!",
                'bad_knee': "무릎 각도를 조정하세요",
                'bad_hip': "엉덩이를 더 뒤로 빼세요"
            },
            'push_up': {
                'good': "훌륭한 푸쉬업 폼입니다!",
                'bad_elbow': "팔꿈치 각도를 확인하세요",
                'bad_hip': "몸을 일직선으로 유지하세요"
            },
            'deadlift': {
                'good': "완벽한 데드리프트 자세입니다!",
                'bad_knee': "무릎을 약간 구부리세요",
                'bad_hip': "엉덩이를 뒤로 더 빼세요"
            },
            'bench_press': {
                'good': "완벽한 벤치프레스입니다!",
                'bad_elbow': "팔꿈치 각도를 조정하세요",
                'bad_shoulder': "어깨 위치를 확인하세요"
            },
            'lunge': {  # 🚀 새로 추가된 런지 피드백
                'good': "완벽한 런지 자세입니다!",
                'bad_front_knee': "앞무릎을 90도로 구부리세요",
                'bad_back_knee': "뒷무릎을 더 펴세요",
                'bad_torso': "상체를 곧게 세우세요",
                'bad_front_ankle': "앞발목 안정성을 유지하세요"
            }
        }
        
        if len(violations) == 0:
            feedback = exercise_feedback.get(self.exercise_type, {}).get('good', "좋은 자세입니다!")
        else:
            # 운동별 특화 피드백
            for violation in violations[:2]:  # 최대 2개 피드백
                joint = violation['joint']
                
                if 'knee' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_knee', f"{joint} 각도를 확인하세요") + ", "
                elif 'hip' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_hip', f"{joint} 각도를 확인하세요") + ", "
                elif 'elbow' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_elbow', f"{joint} 각도를 확인하세요") + ", "
                elif 'shoulder' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_shoulder', f"{joint} 각도를 확인하세요") + ", "
                elif 'torso' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_torso', f"{joint} 자세를 확인하세요") + ", "
                elif 'ankle' in joint:
                    feedback += exercise_feedback.get(self.exercise_type, {}).get('bad_ankle', f"{joint} 안정성을 확인하세요") + ", "
        
        self.last_feedback_time = current_time
        return feedback.rstrip(', ')
    
    def draw_pose_info(self, image: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """이미지에 포즈 정보 그리기 - 풀업→런지 교체"""
        height, width = image.shape[:2]
        
        # 운동 이모지
        exercise_emojis = {
            'squat': '🏋️‍♀️',
            'push_up': '💪',
            'deadlift': '🏋️‍♂️',
            'bench_press': '🔥',
            'lunge': '🚀'  # 새로 추가된 런지
        }
        
        # 상태 표시
        state = analysis_result.get('final_classification', 'unknown')
        color = (0, 255, 0) if state == 'good' else (0, 0, 255)
        
        # 운동 종목과 상태
        exercise_text = f"{exercise_emojis.get(self.exercise_type, '🏋️')} {self.exercise_type.upper()}: {state.upper()}"
        cv2.putText(image, exercise_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 신뢰도 표시
        confidence = analysis_result.get('confidence', 0)
        cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 각도 정보 표시
        if 'angles' in analysis_result:
            y_offset = 110
            for joint, angle in analysis_result['angles'].items():
                cv2.putText(image, f"{joint}: {angle:.1f}°", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        # 피드백 메시지 표시
        feedback = self.generate_feedback(analysis_result)
        if feedback:
            self.feedback_messages.append(feedback)
        
        if self.feedback_messages:
            y_offset = height - 100
            for msg in list(self.feedback_messages)[-3:]:  # 최대 3개 메시지
                cv2.putText(image, msg, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
        
        # 운동 카운터 표시
        total_frames = self.state_counter['good'] + self.state_counter['bad']
        if total_frames > 0:
            good_ratio = self.state_counter['good'] / total_frames
            cv2.putText(image, f"Good Ratio: {good_ratio:.2f}", (width - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    def run_camera(self, camera_id: int = 0):
        """카메라를 사용한 실시간 분석 - 풀업→런지 교체"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"Starting real-time pose analysis for {self.exercise_type} (풀업→런지 교체)")
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
                    
                    # 자세 분석
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
                cv2.imshow(f'{self.exercise_type.replace("_", " ").title()} Pose Analysis', frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.state_counter = {'good': 0, 'bad': 0}
                    self.history.clear()
                    self.ema_value = None
                    self.last_state = 'good'
                    print("Counters reset")
                elif key == ord('s'):
                    filename = f"screenshot_{self.exercise_type}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 최종 통계 출력
            total_frames = self.state_counter['good'] + self.state_counter['bad']
            if total_frames > 0:
                print(f"\n=== Final Statistics ({self.exercise_type}) ===")
                print(f"Total frames analyzed: {total_frames}")
                print(f"Good poses: {self.state_counter['good']} ({self.state_counter['good']/total_frames:.2%})")
                print(f"Bad poses: {self.state_counter['bad']} ({self.state_counter['bad']/total_frames:.2%})")
    
    def analyze_video(self, video_path: str, output_path: str = None):
        """비디오 파일 분석 - 풀업→런지 교체"""
        cap = cv2.VideoCapture(video_path)
        
        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Analyzing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
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
                    
                    # 자세 분석
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
                        'violations': final_result.get('violations', [])
                    })
                
                # 진행률 표시
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}%")
                
                # 출력 비디오에 쓰기
                if output_path:
                    out.write(frame)
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cap.release()
            if output_path:
                out.release()
            
            # 결과 저장
            if frame_results:
                result_file = video_path.replace('.mp4', f'_{self.exercise_type}_analysis.json')
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(frame_results, f, indent=2, ensure_ascii=False)
                
                # 통계 계산
                good_frames = sum(1 for r in frame_results if r['classification'] == 'good')
                bad_frames = len(frame_results) - good_frames
                
                print(f"\n=== Video Analysis Complete ({self.exercise_type}) ===")
                print(f"Results saved to: {result_file}")
                print(f"Total frames analyzed: {len(frame_results)}")
                print(f"Good poses: {good_frames} ({good_frames/len(frame_results):.2%})")
                print(f"Bad poses: {bad_frames} ({bad_frames/len(frame_results):.2%})")
                
                if output_path:
                    print(f"Annotated video saved to: {output_path}")

def main():
    """메인 함수 - 풀업→런지 교체"""
    parser = argparse.ArgumentParser(description='Real-time Exercise Pose Analysis (풀업→런지 교체)')
    parser.add_argument('--exercise', type=str, default='squat',
                       choices=['bench_press', 'deadlift', 'lunge', 'push_up', 'squat'],  # pull_up → lunge
                       help='Exercise type to analyze')
    parser.add_argument('--mode', type=str, default='camera',
                       choices=['camera', 'video'],
                       help='Analysis mode')
    parser.add_argument('--input', type=str, help='Input video file path (for video mode)')
    parser.add_argument('--output', type=str, help='Output video file path (for video mode)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    
    args = parser.parse_args()
    
    # 분석기 초기화
    analyzer = RealtimePoseAnalyzer(args.exercise)
    
    print(f"🚀 풀업→런지 교체 실시간 분석기 시작")
    print(f"지원 운동: 스쿼트, 푸쉬업, 데드리프트, 벤치프레스, 런지")
    
    if args.mode == 'camera':
        analyzer.run_camera(args.camera)
    elif args.mode == 'video':
        if not args.input:
            print("Error: Input video file is required for video mode")
            return
        analyzer.analyze_video(args.input, args.output)

if __name__ == "__main__":
    main()