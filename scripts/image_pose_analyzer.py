#!/usr/bin/env python3
"""
🖼️ 단일 이미지 운동 자세 분석기
사진 하나 넣으면 바로 분석 결과 보여주는 도구
enhanced_pose_analysis.py 기준 적용
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

class ImagePoseAnalyzer:
    """단일 이미지 운동 자세 분석기"""
    
    def __init__(self):
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # 이미지 모드
            model_complexity=2,      # 높은 정확도
            enable_segmentation=False,
            min_detection_confidence=0.5,  # enhanced와 동일
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 🎯 enhanced_pose_analysis.py와 동일한 각도 기준
        self.exercise_thresholds = {
            'squat': {
                'left_knee': {'points': [23, 25, 27], 'range': (55, 140), 'weight': 1.1},
                'right_knee': {'points': [24, 26, 28], 'range': (55, 140), 'weight': 1.1},
                'left_hip': {'points': [11, 23, 25], 'range': (55, 140), 'weight': 0.9},
                'right_hip': {'points': [12, 24, 26], 'range': (55, 140), 'weight': 0.9},
                'back_straight': {'points': [11, 23, 25], 'range': (110, 170), 'weight': 1.1},
                'spine_angle': {'points': [23, 11, 13], 'range': (110, 170), 'weight': 0.9},
            },
            'push_up': {
                'left_elbow': {'points': [11, 13, 15], 'range': (40, 160), 'weight': 1.0},
                'right_elbow': {'points': [12, 14, 16], 'range': (40, 160), 'weight': 1.0},
                'body_line': {'points': [11, 23, 25], 'range': (140, 180), 'weight': 1.2},
                'leg_straight': {'points': [23, 25, 27], 'range': (140, 180), 'weight': 0.8},
                'shoulder_alignment': {'points': [13, 11, 23], 'range': (120, 180), 'weight': 0.6},
                'core_stability': {'points': [11, 12, 23], 'range': (140, 180), 'weight': 1.0},
            },
            'deadlift': {
                # 🏋️‍♂️ 데드리프트: enhanced 대폭 완화 기준
                'left_knee': {'points': [23, 25, 27], 'range': (80, 140), 'weight': 0.6},
                'right_knee': {'points': [24, 26, 28], 'range': (80, 140), 'weight': 0.6},
                'hip_hinge': {'points': [11, 23, 25], 'range': (80, 180), 'weight': 0.7},
                'back_straight': {'points': [11, 23, 12], 'range': (120, 180), 'weight': 1.0},
                'chest_up': {'points': [23, 11, 13], 'range': (50, 140), 'weight': 0.5},
                'spine_neutral': {'points': [23, 11, 24], 'range': (120, 180), 'weight': 0.8},
            },
            'bench_press': {
                'left_elbow': {'points': [11, 13, 15], 'range': (50, 145), 'weight': 1.1},
                'right_elbow': {'points': [12, 14, 16], 'range': (50, 145), 'weight': 1.1},
                'left_shoulder': {'points': [13, 11, 23], 'range': (50, 150), 'weight': 0.9},
                'right_shoulder': {'points': [14, 12, 24], 'range': (50, 150), 'weight': 0.9},
                'back_arch': {'points': [11, 23, 25], 'range': (90, 170), 'weight': 0.7},
                'wrist_alignment': {'points': [13, 15, 17], 'range': (70, 180), 'weight': 0.6},
            },
            'lunge': {
                'front_knee': {'points': [23, 25, 27], 'range': (70, 120), 'weight': 1.2},
                'back_knee': {'points': [24, 26, 28], 'range': (120, 180), 'weight': 1.0},
                'front_hip': {'points': [11, 23, 25], 'range': (70, 120), 'weight': 0.8},
                'torso_upright': {'points': [11, 23, 25], 'range': (100, 180), 'weight': 1.2},
                'front_ankle': {'points': [25, 27, 31], 'range': (80, 110), 'weight': 0.8},
                'back_hip_extension': {'points': [12, 24, 26], 'range': (150, 180), 'weight': 1.0},
            }
        }
        
        # enhanced와 동일한 분류 임계값
        self.classification_thresholds = {
            'squat': 0.5,
            'push_up': 0.7,
            'deadlift': 0.8,  # 대폭 완화
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
        
        # 가시성 임계값 (enhanced와 동일)
        self.visibility_threshold = 0.25
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """세 점 사이의 각도 계산 (enhanced와 동일)"""
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
    
    def detect_view_type(self, landmarks: List[Dict]) -> str:
        """촬영 각도/뷰 타입 감지"""
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            hip_width = abs(left_hip['x'] - right_hip['x'])
            
            nose = landmarks[0]
            body_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            
            if shoulder_width < 0.2 and hip_width < 0.2:
                return 'side_view'
            elif shoulder_width > 0.2 and hip_width > 0.15:
                if abs(nose['x'] - body_center_x) < 0.15:
                    return 'front_view'
                else:
                    return 'back_view'
            else:
                return 'side_view'
        except:
            return 'side_view'
    
    def analyze_image(self, image_path: str, exercise_type: str) -> Dict:
        """단일 이미지 분석 (enhanced 기준)"""
        if not os.path.exists(image_path):
            return {'error': f'이미지 파일을 찾을 수 없습니다: {image_path}'}
        
        if exercise_type not in self.exercise_thresholds:
            return {'error': f'지원되지 않는 운동: {exercise_type}'}
        
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            return {'error': '이미지를 읽을 수 없습니다'}
        
        original_image = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 포즈 검출
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return {'error': '포즈를 검출할 수 없습니다'}
        
        # 랜드마크 변환
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        # 뷰 타입 감지
        view_type = self.detect_view_type(landmarks)
        
        # 각도 분석
        thresholds = self.exercise_thresholds[exercise_type]
        angles = {}
        violations = []
        weighted_violation_score = 0.0
        total_weight = 0.0
        
        for joint_name, config in thresholds.items():
            try:
                p1_idx, p2_idx, p3_idx = config['points']
                min_angle, max_angle = config['range']
                weight = config['weight']
                
                # 인덱스 범위 확인
                if any(idx >= len(landmarks) for idx in [p1_idx, p2_idx, p3_idx]):
                    continue
                
                # 가시성 확인 (enhanced와 동일)
                if (landmarks[p1_idx]['visibility'] < self.visibility_threshold or 
                    landmarks[p2_idx]['visibility'] < self.visibility_threshold or 
                    landmarks[p3_idx]['visibility'] < self.visibility_threshold):
                    continue
                
                p1 = (landmarks[p1_idx]['x'], landmarks[p1_idx]['y'])
                p2 = (landmarks[p2_idx]['x'], landmarks[p2_idx]['y'])
                p3 = (landmarks[p3_idx]['x'], landmarks[p3_idx]['y'])
                
                angle = self.calculate_angle(p1, p2, p3)
                angles[joint_name] = {
                    'value': angle,
                    'range': (min_angle, max_angle),
                    'weight': weight,
                    'in_range': min_angle <= angle <= max_angle,
                    'deviation': min(abs(angle - min_angle), abs(angle - max_angle)) if not (min_angle <= angle <= max_angle) else 0
                }
                
                total_weight += weight
                
                # 허용 범위 확인
                if not (min_angle <= angle <= max_angle):
                    violations.append({
                        'joint': joint_name,
                        'angle': angle,
                        'expected_range': (min_angle, max_angle),
                        'weight': weight,
                        'deviation': min(abs(angle - min_angle), abs(angle - max_angle))
                    })
                    weighted_violation_score += weight
                    
            except Exception as e:
                print(f"Error calculating angle for {joint_name}: {e}")
                continue
        
        # enhanced와 동일한 분류
        violation_ratio = weighted_violation_score / max(total_weight, 1.0)
        classification_threshold = self.classification_thresholds.get(exercise_type, 0.6)
        is_good = violation_ratio < classification_threshold
        
        # 랜드마크가 있는 이미지 생성
        annotated_image = original_image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        return {
            'success': True,
            'image_path': image_path,
            'exercise_type': exercise_type,
            'view_type': view_type,
            'classification': 'good' if is_good else 'bad',
            'confidence': 1.0 - violation_ratio,
            'violation_ratio': violation_ratio,
            'classification_threshold': classification_threshold,
            'angles': angles,
            'violations': violations,
            'total_joints_analyzed': len(angles),
            'violation_count': len(violations),
            'weighted_violation_score': weighted_violation_score,
            'total_weight': total_weight,
            'enhanced_compatible': True,
            'annotated_image': annotated_image,
            'original_image': original_image,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def generate_detailed_report(self, analysis_result: Dict) -> str:
        """상세 분석 리포트 생성"""
        if not analysis_result.get('success', False):
            return f"❌ 분석 실패: {analysis_result.get('error', '알 수 없는 오류')}"
        
        exercise = analysis_result['exercise_type']
        emoji = self.exercise_emojis.get(exercise, '🏋️')
        classification = analysis_result['classification']
        confidence = analysis_result['confidence']
        view_type = analysis_result['view_type']
        
        # 헤더
        report = f"\n{'='*80}\n"
        report += f"{emoji} {exercise.upper()} 자세 분석 리포트 (Enhanced 기준)\n"
        report += f"{'='*80}\n"
        
        # 기본 정보
        report += f"📁 이미지: {Path(analysis_result['image_path']).name}\n"
        report += f"📷 촬영 각도: {view_type.replace('_', ' ').title()}\n"
        report += f"🎯 분석 시간: {analysis_result['analysis_timestamp']}\n\n"
        
        # 전체 결과
        status_color = "✅" if classification == 'good' else "❌"
        status_text = "완벽한 자세!" if classification == 'good' else "개선이 필요한 자세"
        
        report += f"📊 종합 결과\n"
        report += f"  {status_color} 상태: {classification.upper()} ({status_text})\n"
        report += f"  🎯 신뢰도: {confidence:.1%}\n"
        report += f"  📐 위반 비율: {analysis_result['violation_ratio']:.1%}\n"
        report += f"  🔧 분류 기준: {analysis_result['classification_threshold']:.1%} (Enhanced)\n"
        report += f"  🔍 분석된 관절: {analysis_result['total_joints_analyzed']}개\n"
        report += f"  ⚠️ 위반 관절: {analysis_result['violation_count']}개\n\n"
        
        # 각도 상세 분석
        report += f"📐 관절별 각도 분석\n"
        report += f"{'관절명':<20} {'현재각도':<10} {'기준범위':<15} {'상태':<8} {'가중치':<8} {'편차':<10}\n"
        report += f"{'-'*80}\n"
        
        angles = analysis_result['angles']
        for joint_name, angle_data in angles.items():
            current_angle = angle_data['value']
            range_min, range_max = angle_data['range']
            weight = angle_data['weight']
            in_range = angle_data['in_range']
            deviation = angle_data['deviation']
            
            status_icon = "✅" if in_range else "❌"
            joint_display = joint_name.replace('_', ' ').title()
            
            report += f"{joint_display:<20} {current_angle:>7.1f}°   {range_min:>3.0f}-{range_max:<3.0f}°     {status_icon:<8} {weight:<8.1f} {deviation:>7.1f}°\n"
        
        # 위반사항 상세
        if analysis_result['violations']:
            report += f"\n⚠️ 개선이 필요한 부분\n"
            report += f"{'-'*50}\n"
            
            # 가중치 순으로 정렬
            violations = sorted(analysis_result['violations'], key=lambda x: x['weight'], reverse=True)
            
            for i, violation in enumerate(violations[:5], 1):  # 상위 5개만
                joint = violation['joint'].replace('_', ' ').title()
                angle = violation['angle']
                min_angle, max_angle = violation['expected_range']
                weight = violation['weight']
                deviation = violation['deviation']
                
                report += f"{i}. {joint}\n"
                report += f"   현재: {angle:.1f}° → 목표: {min_angle:.0f}-{max_angle:.0f}° (편차: {deviation:.1f}°)\n"
                report += f"   중요도: {weight:.1f} | "
                
                # 구체적인 조언
                if 'knee' in violation['joint']:
                    if exercise == 'squat':
                        report += "무릎이 발끝을 넘지 않게 주의하세요\n"
                    elif exercise == 'deadlift':
                        report += "무릎을 약간만 구부리세요 (Enhanced 완화 기준)\n"
                    elif exercise == 'lunge':
                        if 'front' in violation['joint']:
                            report += "앞무릎을 90도로 구부리세요\n"
                        else:
                            report += "뒷무릎을 더 펴세요\n"
                elif 'hip' in violation['joint']:
                    if exercise == 'squat':
                        report += "엉덩이를 더 뒤로 빼세요\n"
                    elif exercise == 'deadlift':
                        report += "힙 힌지 동작을 더 크게 하세요 (Enhanced 완화 기준)\n"
                elif 'elbow' in violation['joint']:
                    if exercise == 'push_up':
                        report += "팔꿈치를 몸에 더 가깝게 하세요\n"
                    elif exercise == 'bench_press':
                        report += "팔꿈치 각도를 조정하세요 (Enhanced 기준)\n"
                elif 'back' in violation['joint'] or 'spine' in violation['joint']:
                    report += "등을 곧게 펴세요\n"
                elif 'torso' in violation['joint']:
                    report += "상체를 곧게 세우세요\n"
                else:
                    report += "자세를 교정하세요\n"
                
                report += "\n"
        
        # Enhanced 기준 정보
        report += f"🔧 Enhanced 기준 정보\n"
        report += f"{'-'*30}\n"
        report += f"📊 적용된 기준: enhanced_pose_analysis.py와 동일\n"
        
        target_rates = {
            'squat': '50-70%',
            'push_up': '50-70%',
            'deadlift': '40-60% (대폭 완화)',
            'bench_press': '50-70%',
            'lunge': '50-70%'
        }
        
        report += f"🎯 목표 성공률: {target_rates.get(exercise, '50-70%')}\n"
        
        if exercise == 'deadlift':
            report += f"💡 특별 조치: 데드리프트는 99% Bad 문제 해결을 위해 대폭 완화됨\n"
        
        # 권장사항
        report += f"\n💡 종합 권장사항\n"
        report += f"{'-'*20}\n"
        
        if classification == 'good':
            report += f"✅ 훌륭한 자세입니다! 현재 폼을 유지하세요.\n"
            report += f"💪 이 자세로 운동을 계속하시면 안전하고 효과적입니다.\n"
        else:
            report += f"⚠️ 위에 나열된 개선사항들을 차례대로 교정해보세요.\n"
            report += f"🎯 가중치가 높은 관절부터 우선적으로 개선하세요.\n"
            report += f"📷 측면에서 촬영하면 더 정확한 분석이 가능합니다.\n"
            report += f"⏰ 천천히 정확한 동작부터 연습하세요.\n"
        
        report += f"\n{'='*80}\n"
        
        return report
    
    def save_analysis_results(self, analysis_result: Dict, output_dir: str = "analysis_results"):
        """분석 결과 저장 (이미지 + JSON + 리포트)"""
        if not analysis_result.get('success', False):
            print(f"❌ 저장 실패: {analysis_result.get('error', '분석 결과 없음')}")
            return None
        
        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exercise = analysis_result['exercise_type']
        classification = analysis_result['classification']
        base_filename = f"{exercise}_{classification}_{timestamp}"
        
        saved_files = {}
        
        try:
            # 1. 주석이 달린 이미지 저장
            annotated_path = output_path / f"{base_filename}_annotated.jpg"
            cv2.imwrite(str(annotated_path), analysis_result['annotated_image'])
            saved_files['annotated_image'] = str(annotated_path)
            
            # 2. 원본 이미지 복사
            original_path = output_path / f"{base_filename}_original.jpg"
            cv2.imwrite(str(original_path), analysis_result['original_image'])
            saved_files['original_image'] = str(original_path)
            
            # 3. JSON 결과 저장
            json_path = output_path / f"{base_filename}_analysis.json"
            
            # 이미지 데이터 제외하고 저장
            json_data = {k: v for k, v in analysis_result.items() 
                        if k not in ['annotated_image', 'original_image']}
            json_data['saved_files'] = saved_files
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            saved_files['json_analysis'] = str(json_path)
            
            # 4. 텍스트 리포트 저장
            report_path = output_path / f"{base_filename}_report.txt"
            report = self.generate_detailed_report(analysis_result)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            saved_files['text_report'] = str(report_path)
            
            print(f"💾 분석 결과 저장 완료:")
            print(f"  📁 저장 위치: {output_path}")
            print(f"  🖼️ 주석 이미지: {annotated_path.name}")
            print(f"  📄 분석 리포트: {report_path.name}")
            print(f"  📊 JSON 데이터: {json_path.name}")
            
            return saved_files
            
        except Exception as e:
            print(f"❌ 파일 저장 중 오류: {e}")
            return None
    
    def analyze_single_image(self, image_path: str, exercise_type: str, 
                           save_results: bool = True, show_image: bool = True) -> Dict:
        """단일 이미지 완전 분석 (메인 함수)"""
        print(f"🔍 이미지 분석 시작...")
        print(f"📁 파일: {image_path}")
        print(f"🏋️ 운동: {exercise_type}")
        print(f"🔧 기준: Enhanced Pose Analysis")
        
        # 분석 실행
        start_time = time.time()
        result = self.analyze_image(image_path, exercise_type)
        analysis_time = time.time() - start_time
        
        if not result.get('success', False):
            print(f"❌ 분석 실패: {result.get('error', '알 수 없는 오류')}")
            return result
        
        # 결과 출력
        emoji = self.exercise_emojis.get(exercise_type, '🏋️')
        classification = result['classification']
        confidence = result['confidence']
        
        print(f"\n🎉 분석 완료! (소요시간: {analysis_time:.2f}초)")
        print(f"{emoji} {exercise_type.upper()}: {classification.upper()} ({confidence:.1%} 신뢰도)")
        
        # 상세 리포트 출력
        report = self.generate_detailed_report(result)
        print(report)
        
        # 결과 저장
        if save_results:
            saved_files = self.save_analysis_results(result)
            result['saved_files'] = saved_files
        
        # 이미지 표시
        if show_image:
            try:
                # 원본과 주석 이미지 나란히 표시
                original = result['original_image']
                annotated = result['annotated_image']
                
                # 이미지 크기 조정 (화면에 맞게)
                height, width = original.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    original = cv2.resize(original, (new_width, new_height))
                    annotated = cv2.resize(annotated, (new_width, new_height))
                
                # 나란히 배치
                combined = np.hstack([original, annotated])
                
                # 결과 텍스트 추가
                result_text = f"{emoji} {exercise_type.upper()}: {classification.upper()} ({confidence:.1%})"
                cv2.putText(combined, result_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if classification == 'good' else (0, 0, 255), 2)
                
                # 창 제목
                window_title = f"Enhanced Analysis: Original (Left) vs Annotated (Right)"
                cv2.imshow(window_title, combined)
                
                print(f"\n🖼️ 이미지 표시 중... (아무 키나 눌러서 닫기)")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"⚠️ 이미지 표시 오류: {e}")
        
        return result

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='🖼️ Enhanced 단일 이미지 운동 자세 분석기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 사용 예시:
  python image_pose_analyzer.py --image squat_photo.jpg --exercise squat
  python image_pose_analyzer.py --image push_up.png --exercise push_up --no-save
  python image_pose_analyzer.py --image deadlift.jpg --exercise deadlift --no-show

🏋️ 지원 운동:
  • squat (스쿼트)
  • push_up (푸쉬업)  
  • deadlift (데드리프트)
  • bench_press (벤치프레스)
  • lunge (런지)

📊 Enhanced 기준:
  모든 분석이 enhanced_pose_analysis.py와 동일한 기준으로 수행됩니다.
  데드리프트는 99% Bad 문제 해결을 위해 대폭 완화된 기준을 사용합니다.
        """
    )
    
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='분석할 이미지 파일 경로')
    parser.add_argument('--exercise', '-e', type=str, required=True,
                       choices=['squat', 'push_up', 'deadlift', 'bench_press', 'lunge'],
                       help='운동 종류')
    parser.add_argument('--output', '-o', type=str, default='analysis_results',
                       help='결과 저장 디렉토리 (기본값: analysis_results)')
    parser.add_argument('--no-save', action='store_true',
                       help='결과 파일 저장 안함')
    parser.add_argument('--no-show', action='store_true',
                       help='이미지 표시 안함')
    parser.add_argument('--json-only', action='store_true',
                       help='JSON 결과만 출력 (리포트 생략)')
    
    args = parser.parse_args()
    
    # 이미지 파일 존재 확인
    if not os.path.exists(args.image):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {args.image}")
        return 1
    
    # 분석기 초기화
    try:
        analyzer = ImagePoseAnalyzer()
    except Exception as e:
        print(f"❌ 분석기 초기화 실패: {e}")
        return 1
    
    print(f"🎯 Enhanced 이미지 자세 분석기")
    print(f"📁 이미지: {args.image}")
    print(f"🏋️ 운동: {args.exercise}")
    print(f"🔧 기준: enhanced_pose_analysis.py 동일")
    
    # 분석 실행
    try:
        result = analyzer.analyze_single_image(
            args.image,
            args.exercise,
            save_results=not args.no_save,
            show_image=not args.no_show
        )
        
        # JSON 전용 출력
        if args.json_only and result.get('success', False):
            # 이미지 데이터 제외하고 JSON 출력
            json_result = {k: v for k, v in result.items() 
                          if k not in ['annotated_image', 'original_image']}
            print(json.dumps(json_result, indent=2, ensure_ascii=False))
        
        return 0 if result.get('success', False) else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
        return 0
    except Exception as e:
        print(f"❌ 분석 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1

# 배치 분석 함수 추가
def batch_analyze_images(image_directory: str, exercise_type: str, output_dir: str = "batch_analysis_results"):
    """여러 이미지 일괄 분석"""
    print(f"📁 배치 분석 시작: {image_directory}")
    
    analyzer = ImagePoseAnalyzer()
    image_dir = Path(image_directory)
    
    if not image_dir.exists():
        print(f"❌ 디렉토리를 찾을 수 없습니다: {image_directory}")
        return
    
    # 이미지 파일 찾기
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
        image_files.extend(list(image_dir.glob(ext)))
        image_files.extend(list(image_dir.glob(ext.upper())))
    
    if not image_files:
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_directory}")
        return
    
    print(f"🔍 발견된 이미지: {len(image_files)}개")
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 배치 분석
    results = []
    good_count = 0
    bad_count = 0
    failed_count = 0
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n📊 진행률: {i}/{len(image_files)} - {img_file.name}")
        
        try:
            result = analyzer.analyze_image(str(img_file), exercise_type)
            
            if result.get('success', False):
                classification = result['classification']
                confidence = result['confidence']
                
                if classification == 'good':
                    good_count += 1
                else:
                    bad_count += 1
                
                # 간단한 결과 저장
                simple_result = {
                    'filename': img_file.name,
                    'classification': classification,
                    'confidence': confidence,
                    'violation_ratio': result['violation_ratio'],
                    'angles': {k: v['value'] for k, v in result['angles'].items()},
                    'violations_count': result['violation_count']
                }
                results.append(simple_result)
                
                print(f"  ✅ {classification.upper()} ({confidence:.1%})")
            else:
                failed_count += 1
                print(f"  ❌ 실패: {result.get('error', '알 수 없는 오류')}")
                
        except Exception as e:
            failed_count += 1
            print(f"  ❌ 오류: {e}")
    
    # 배치 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_result = {
        'batch_info': {
            'source_directory': str(image_directory),
            'exercise_type': exercise_type,
            'enhanced_compatible': True,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'successful_analyses': len(results),
            'failed_analyses': failed_count
        },
        'summary': {
            'good_count': good_count,
            'bad_count': bad_count,
            'failed_count': failed_count,
            'success_rate': good_count / max(good_count + bad_count, 1),
            'analysis_success_rate': len(results) / len(image_files)
        },
        'detailed_results': results
    }
    
    # JSON 저장
    batch_json_path = output_path / f"batch_analysis_{exercise_type}_{timestamp}.json"
    with open(batch_json_path, 'w', encoding='utf-8') as f:
        json.dump(batch_result, f, indent=2, ensure_ascii=False)
    
    # 요약 리포트 생성
    summary_report = f"""
📊 Enhanced 배치 분석 완료 리포트
{'='*60}
📁 소스: {image_directory}
🏋️ 운동: {exercise_type}
🔧 기준: Enhanced Pose Analysis

📈 분석 결과:
  총 이미지: {len(image_files)}개
  성공 분석: {len(results)}개 ({len(results)/len(image_files):.1%})
  실패 분석: {failed_count}개

🎯 자세 평가:
  ✅ Good: {good_count}개 ({good_count/(good_count+bad_count):.1%})
  ❌ Bad: {bad_count}개 ({bad_count/(good_count+bad_count):.1%})

💾 결과 저장: {batch_json_path}
{'='*60}
    """
    
    print(summary_report)
    
    # 텍스트 리포트 저장
    report_path = output_path / f"batch_summary_{exercise_type}_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    return batch_result

if __name__ == "__main__":
    main()