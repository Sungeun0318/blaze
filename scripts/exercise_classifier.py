"""
풀업 → 런지 교체된 5종목 운동 동작 자동 분류 모델
BlazePose 랜드마크를 기반으로 5가지 운동 종목을 자동으로 분류합니다.
스쿼트, 푸쉬업, 데드리프트, 벤치프레스, 런지 (풀업 대체)
processed_data 구조 지원 추가
"""

import cv2
import numpy as np
import mediapipe as mp
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class ExerciseFeatureExtractor:
    """운동 특징 추출기 - 풀업→런지 교체 5종목 지원"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, image_path: str) -> Optional[np.ndarray]:
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
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                return np.array(landmarks)
            return None
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def calculate_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """주요 관절 각도 계산 - 풀업→런지 교체 5종목 특징 포함"""
        def angle_3points(p1, p2, p3):
            """세 점 사이의 각도 계산"""
            try:
                v1 = p1 - p2
                v2 = p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                return np.degrees(np.arccos(cos_angle))
            except:
                return 0.0
        
        # 랜드마크를 (x, y, z) 형태로 재구성
        points = landmarks.reshape(-1, 3)
        
        angles = []
        
        # 5종목 구분을 위한 주요 각도들 계산 (풀업→런지 교체)
        angle_configs = [
            # 팔꿈치 각도 (벤치프레스, 푸쉬업 구분)
            ([11, 13, 15], 'left_elbow'),    # 왼쪽 팔꿈치
            ([12, 14, 16], 'right_elbow'),   # 오른쪽 팔꿈치
            
            # 어깨 각도 (벤치프레스, 푸쉬업 구분)
            ([13, 11, 23], 'left_shoulder'), # 왼쪽 어깨
            ([14, 12, 24], 'right_shoulder'), # 오른쪽 어깨
            
            # 엉덩이 각도 (스쿼트, 데드리프트, 런지 구분 - 중요!)
            ([11, 23, 25], 'left_hip'),      # 왼쪽 엉덩이
            ([12, 24, 26], 'right_hip'),     # 오른쪽 엉덩이
            
            # 무릎 각도 (스쿼트, 데드리프트, 런지 구분 - 매우 중요!)
            ([23, 25, 27], 'left_knee'),     # 왼쪽 무릎 (런지 앞다리)
            ([24, 26, 28], 'right_knee'),    # 오른쪽 무릎 (런지 뒷다리)
            
            # 발목 각도 (런지 구분용 - 새로 추가)
            ([25, 27, 31], 'left_ankle'),    # 왼쪽 발목
            ([26, 28, 32], 'right_ankle'),   # 오른쪽 발목
            
            # 척추 각도 (데드리프트, 스쿼트, 런지 구분)
            ([11, 23, 25], 'spine_upper'),   # 상체 척추
            ([23, 25, 27], 'spine_lower'),   # 하체 척추
            
            # 손목 각도 (벤치프레스, 푸쉬업 구분)
            ([13, 15, 17], 'left_wrist'),    # 왼쪽 손목
            ([14, 16, 18], 'right_wrist'),   # 오른쪽 손목
            
            # 몸통 각도 (푸쉬업, 런지 구분)
            ([11, 12, 23], 'torso_angle'),   # 몸통 각도
            
            # 🚀 런지 구분을 위한 추가 각도들
            ([11, 23, 24], 'hip_level'),     # 골반 수평도
            ([23, 11, 13], 'torso_lean'),    # 상체 기울기
            ([27, 28, 0], 'foot_separation'), # 발 벌림 정도 (런지 스탠스)
        ]
        
        for indices, name in angle_configs:
            try:
                if all(i < len(points) for i in indices):
                    p1, p2, p3 = points[indices[0]][:2], points[indices[1]][:2], points[indices[2]][:2]
                    angle = angle_3points(p1, p2, p3)
                    angles.append(angle)
                else:
                    angles.append(0.0)
            except:
                angles.append(0.0)
        
        return np.array(angles)
    
    def calculate_distances(self, landmarks: np.ndarray) -> np.ndarray:
        """주요 신체 부위 간 거리 계산 - 풀업→런지 교체 5종목 특징"""
        points = landmarks.reshape(-1, 3)
        
        distances = []
        
        # 5종목 구분을 위한 주요 거리들 (풀업→런지 교체)
        distance_configs = [
            ([11, 12], 'shoulder_width'),    # 어깨 너비
            ([23, 24], 'hip_width'),         # 엉덩이 너비
            ([27, 28], 'ankle_width'),       # 발목 너비 (런지 스탠스 중요!)
            ([11, 23], 'left_torso'),        # 왼쪽 몸통 길이
            ([12, 24], 'right_torso'),       # 오른쪽 몸통 길이
            ([23, 27], 'left_leg'),          # 왼쪽 다리 길이
            ([24, 28], 'right_leg'),         # 오른쪽 다리 길이
            ([15, 16], 'hand_distance'),     # 양손 거리 (벤치프레스, 푸쉬업)
            ([0, 23], 'head_to_hip'),        # 머리에서 엉덩이까지
            ([15, 27], 'hand_to_foot'),      # 손에서 발까지 (푸쉬업)
            ([11, 15], 'shoulder_to_hand'),  # 어깨에서 손까지
            ([23, 31], 'hip_to_toe'),        # 엉덩이에서 발끝까지
            
            # 🚀 런지 구분을 위한 추가 거리들
            ([27, 31], 'front_foot_length'), # 앞발 길이 (런지)
            ([28, 32], 'back_foot_length'),  # 뒷발 길이 (런지)
            ([25, 26], 'knee_separation'),   # 무릎 간 거리 (런지 스탠스)
            ([23, 28], 'hip_to_back_ankle'), # 엉덩이에서 뒷발목까지 (런지 깊이)
        ]
        
        for indices, name in distance_configs:
            try:
                if all(i < len(points) for i in indices):
                    p1, p2 = points[indices[0]][:2], points[indices[1]][:2]
                    dist = np.linalg.norm(p1 - p2)
                    distances.append(dist)
                else:
                    distances.append(0.0)
            except:
                distances.append(0.0)
        
        return np.array(distances)
    
    def calculate_pose_ratios(self, landmarks: np.ndarray) -> np.ndarray:
        """신체 비율 계산 - 풀업→런지 교체 5종목 구분 특징"""
        points = landmarks.reshape(-1, 3)
        
        ratios = []
        
        try:
            # 주요 비율들
            shoulder_width = np.linalg.norm(points[11][:2] - points[12][:2])
            hip_width = np.linalg.norm(points[23][:2] - points[24][:2])
            torso_height = np.linalg.norm(points[11][:2] - points[23][:2])
            leg_length = np.linalg.norm(points[23][:2] - points[27][:2])
            arm_length = np.linalg.norm(points[11][:2] - points[15][:2])
            
            # 🚀 런지를 위한 추가 측정값들
            ankle_separation = np.linalg.norm(points[27][:2] - points[28][:2])  # 발목 간 거리
            front_leg_bend = np.linalg.norm(points[23][:2] - points[27][:2])   # 앞다리 굽힘
            back_leg_stretch = np.linalg.norm(points[24][:2] - points[28][:2]) # 뒷다리 뻗음
            
            # 비율 계산 (풀업→런지 교체 5종목 구분용)
            ratios.extend([
                shoulder_width / max(hip_width, 0.001),         # 어깨/엉덩이 비율
                torso_height / max(leg_length, 0.001),          # 몸통/다리 비율
                hip_width / max(torso_height, 0.001),           # 엉덩이/몸통 비율
                arm_length / max(torso_height, 0.001),          # 팔/몸통 비율
                
                # 높이 비율 (운동 자세 구분)
                points[11][1] / max(points[27][1], 0.001),      # 어깨/발목 높이 비율
                points[23][1] / max(points[27][1], 0.001),      # 엉덩이/발목 높이 비율
                points[15][1] / max(points[27][1], 0.001),      # 손/발 높이 비율 (푸쉬업)
                
                # 가로 비율 (운동 방향성)
                abs(points[15][0] - points[16][0]) / max(shoulder_width, 0.001),  # 손 벌림 정도
                
                # 🚀 런지 특화 비율들
                ankle_separation / max(shoulder_width, 0.001),   # 발목간거리/어깨너비 (런지 스탠스)
                front_leg_bend / max(torso_height, 0.001),       # 앞다리굽힘/몸통높이 (런지 깊이)
                back_leg_stretch / max(torso_height, 0.001),     # 뒷다리뻗음/몸통높이 (런지 확장)
                abs(points[27][1] - points[28][1]) / max(torso_height, 0.001), # 발높이차이/몸통높이
            ])
            
        except:
            ratios = [1.0] * 12  # 런지 추가로 12개로 증가
        
        return np.array(ratios)
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """5종목 구분을 위한 전체 특징 추출"""
        landmarks = self.extract_landmarks(image_path)
        if landmarks is None:
            return None
        
        # 다양한 특징들 추출
        angles = self.calculate_angles(landmarks)
        distances = self.calculate_distances(landmarks)
        ratios = self.calculate_pose_ratios(landmarks)
        
        # 모든 특징을 하나로 합치기
        features = np.concatenate([angles, distances, ratios])
        
        return features

class ExerciseClassificationModel:
    """풀업→런지 교체 5종목 운동 분류 모델 (processed_data 지원)"""
    
    def __init__(self):
        self.feature_extractor = ExerciseFeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=300,  # 5종목이므로 더 많은 트리
            max_depth=20,      # 더 깊은 트리
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'  # 클래스 불균형 해결
        )
        
        # 🚀 풀업 → 런지로 변경된 5종목 라벨 인코딩
        self.label_encoder = {
            'squat': 0,
            'push_up': 1, 
            'deadlift': 2,
            'bench_press': 3,
            'lunge': 4  # pull_up → lunge로 변경
        }
        self.reverse_encoder = {v: k for k, v in self.label_encoder.items()}
        self.is_trained = False
    
    def prepare_training_data_from_processed(self, processed_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """processed_data 구조에서 훈련 데이터 준비"""
        data_dir = Path(processed_data_path)
        
        features_list = []
        labels_list = []
        
        print("🔍 processed_data 구조에서 훈련 데이터 수집 중...")
        
        total_processed = 0
        for exercise_name in self.label_encoder.keys():
            exercise_path = data_dir / exercise_name
            if not exercise_path.exists():
                print(f"⚠️ Warning: {exercise_path} not found - {exercise_name} 데이터 없음")
                continue
            
            print(f"📂 Processing {exercise_name}...")
            
            count = 0
            # good과 bad 폴더 모두 처리
            for category in ['good', 'bad']:
                category_path = exercise_path / category
                if not category_path.exists():
                    continue
                
                # 이미지 파일들 가져오기
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(list(category_path.glob(ext)))
                
                print(f"  📸 {category}: {len(image_files)} images found")
                
                for img_file in image_files:
                    features = self.feature_extractor.extract_features(str(img_file))
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(self.label_encoder[exercise_name])
                        count += 1
                        
                        if count % 100 == 0:
                            print(f"    Processing: {count} images")
                        
                        # 메모리 절약을 위해 제한
                        if count >= 1000:
                            break
                
                if count >= 1000:
                    break
            
            total_processed += count
            print(f"  ✅ {exercise_name}: Total {count} images processed")
        
        if not features_list:
            raise ValueError("❌ No valid training data found in processed_data structure!")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"📊 processed_data 훈련 데이터 준비 완료: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 운동별 데이터 분포:")
        for exercise_name, label in self.label_encoder.items():
            count = np.sum(y == label)
            percentage = (count / len(y)) * 100 if len(y) > 0 else 0
            emoji = {
                'squat': '🏋️‍♀️', 
                'push_up': '💪', 
                'deadlift': '🏋️‍♂️', 
                'bench_press': '🔥', 
                'lunge': '🚀'
            }
            print(f"  {emoji.get(exercise_name, '🏋️')} {exercise_name}: {count}개 ({percentage:.1f}%)")
        
        return X, y
    
    def prepare_training_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """훈련 데이터 준비 (원본 구조 + processed_data 구조 모두 지원)"""
        data_dir = Path(data_path)
        
        # processed_data 구조인지 확인 (squat/good, squat/bad 등)
        squat_path = data_dir / 'squat'
        if squat_path.exists() and (squat_path / 'good').exists():
            print("📁 processed_data 구조 감지됨")
            return self.prepare_training_data_from_processed(data_path)
        
        # 원본 구조 처리 (기존 방식)
        features_list = []
        labels_list = []
        
        # 🚀 풀업 → 런지로 변경된 5종목 디렉토리 매핑
        exercise_dirs = {
            'squat_exercise': 'squat',
            'push_up_exercise': 'push_up',
            'deadlift_exercise': 'deadlift',
            'bench_press_exercise': 'bench_press',
            'lunge_exercise': 'lunge'  # pull_up_exercise → lunge_exercise로 변경
        }
        
        print("🔍 원본 구조에서 훈련 데이터 수집 중...")
        
        total_processed = 0
        for dir_name, exercise_name in exercise_dirs.items():
            exercise_path = data_dir / dir_name
            if not exercise_path.exists():
                print(f"⚠️ Warning: {exercise_path} not found - 해당 운동 데이터 없음")
                if exercise_name == 'lunge':
                    print(f"   💡 런지 데이터를 {exercise_path} 폴더에 추가하세요!")
                continue
            
            print(f"📂 Processing {exercise_name}...")
            
            # 이미지 파일들 가져오기
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(list(exercise_path.glob(ext)))
            
            if not image_files:
                print(f"  ⚠️ {exercise_name}: 이미지 파일 없음")
                continue
            
            count = 0
            for img_file in image_files:
                features = self.feature_extractor.extract_features(str(img_file))
                if features is not None:
                    features_list.append(features)
                    labels_list.append(self.label_encoder[exercise_name])
                    count += 1
                    
                    if count % 50 == 0:
                        print(f"  📸 {exercise_name}: {count} images processed")
                    
                    # 메모리 절약을 위해 제한
                    if count >= 500:
                        break
            
            total_processed += count
            print(f"  ✅ {exercise_name}: Total {count} images")
        
        if not features_list:
            raise ValueError("❌ No valid training data found!")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"📊 원본 구조 훈련 데이터 준비 완료: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 운동별 데이터 분포:")
        for exercise_name, label in self.label_encoder.items():
            count = np.sum(y == label)
            percentage = (count / len(y)) * 100 if len(y) > 0 else 0
            emoji = {
                'squat': '🏋️‍♀️', 
                'push_up': '💪', 
                'deadlift': '🏋️‍♂️', 
                'bench_press': '🔥', 
                'lunge': '🚀'
            }
            print(f"  {emoji.get(exercise_name, '🏋️')} {exercise_name}: {count}개 ({percentage:.1f}%)")
        
        return X, y
    
    def train(self, data_path: str, test_size: float = 0.2):
        """풀업→런지 교체 5종목 모델 훈련 (processed_data 지원)"""
        print("🧠 === processed_data 지원 5종목 운동 분류 모델 훈련 시작 ===")
        
        # 훈련 데이터 준비
        X, y = self.prepare_training_data(data_path)
        
        # 최소 데이터 요구사항 확인
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError("❌ 최소 2종목 이상의 데이터가 필요합니다!")
        
        print(f"📚 사용 가능한 운동: {len(unique_labels)}종목")
        
        # 훈련/테스트 분할 (계층 샘플링)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📚 Training set: {X_train.shape[0]} samples")
        print(f"🔬 Test set: {X_test.shape[0]} samples")
        
        # 모델 훈련
        print("⚙️ 5종목 모델 훈련 중...")
        self.model.fit(X_train, y_train)
        
        # 성능 평가
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ === processed_data 지원 5종목 훈련 완료 ===")
        print(f"🎯 정확도: {accuracy:.3f}")
        print("\n📈 상세 성능 리포트:")
        
        # 실제 존재하는 운동만 표시
        available_exercises = [self.reverse_encoder[label] for label in unique_labels]
        print(classification_report(y_test, y_pred, 
                                  target_names=available_exercises))
        
        # 특징 중요도 (상위 10개)
        feature_importance = self.model.feature_importances_
        print(f"\n🔍 특징 중요도 (평균): {np.mean(feature_importance):.3f}")
        
        # 운동별 예측 신뢰도
        print(f"\n📊 운동별 예측 성능:")
        for exercise in available_exercises:
            label = self.label_encoder[exercise]
            mask = y_test == label
            if np.sum(mask) > 0:
                exercise_accuracy = accuracy_score(y_test[mask], y_pred[mask])
                emoji = {
                    'squat': '🏋️‍♀️', 
                    'push_up': '💪', 
                    'deadlift': '🏋️‍♂️', 
                    'bench_press': '🔥', 
                    'lunge': '🚀'
                }
                print(f"  {emoji.get(exercise, '🏋️')} {exercise}: {exercise_accuracy:.3f}")
        
        self.is_trained = True
        return accuracy
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """단일 이미지 5종목 예측"""
        if not self.is_trained:
            raise ValueError("❌ Model is not trained yet!")
        
        features = self.feature_extractor.extract_features(image_path)
        if features is None:
            return "unknown", 0.0
        
        # 예측 및 확률
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        
        exercise_name = self.reverse_encoder[prediction]
        confidence = probabilities[prediction]
        
        return exercise_name, confidence
    
    def save_model(self, model_path: str):
        """processed_data 지원 5종목 모델 저장"""
        if not self.is_trained:
            raise ValueError("❌ Model is not trained yet!")
        
        # 모델 디렉토리 확인/생성
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'reverse_encoder': self.reverse_encoder,
            'is_trained': self.is_trained,
            'supported_exercises': list(self.label_encoder.keys()),
            'feature_count': len(self.label_encoder),
            'version': 'processed_data_support_v1.0',
            'data_structures_supported': ['original_images', 'processed_data'],
            'changelog': {
                'added': 'processed_data structure support',
                'replaced': 'pull_up → lunge',
                'maintained': ['squat', 'push_up', 'deadlift', 'bench_press']
            }
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 processed_data 지원 5종목 모델 저장 완료: {model_path}")
        print(f"🎯 지원 운동: {', '.join(self.label_encoder.keys())}")
        print(f"📁 지원 구조: 원본 이미지 + processed_data")
    
    def load_model(self, model_path: str):
        """processed_data 지원 5종목 모델 로드"""
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.reverse_encoder = model_data['reverse_encoder']
            self.is_trained = model_data['is_trained']
            
            # 버전 정보 확인
            version = model_data.get('version', 'unknown')
            supported_exercises = model_data.get('supported_exercises', list(self.label_encoder.keys()))
            
            print(f"📥 processed_data 지원 모델 로드 완료: {model_path}")
            print(f"🎯 지원 운동 ({len(supported_exercises)}종목): {', '.join(supported_exercises)}")
            print(f"📋 모델 버전: {version}")
            
            data_structures = model_data.get('data_structures_supported', ['original_images'])
            if 'processed_data' in data_structures:
                print(f"📁 processed_data 구조 지원 ✅")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """메인 실행 함수 (processed_data 지원)"""
    import argparse
    
    parser = argparse.ArgumentParser(description='processed_data 지원 5종목 Exercise Classification Model')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'predict', 'evaluate'],
                       help='실행 모드')
    parser.add_argument('--data_path', type=str, default='./processed_data',
                       help='훈련 데이터 경로 (processed_data 또는 원본 구조)')
    parser.add_argument('--model_path', type=str, default='models/exercise_classifier.pkl',
                       help='모델 파일 경로')
    parser.add_argument('--image', type=str, help='단일 이미지 예측용')
    parser.add_argument('--test_data', type=str, help='별도 테스트 데이터 경로')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # processed_data 지원 5종목 모델 훈련
        model = ExerciseClassificationModel()
        try:
            print("🚀 processed_data 지원 5종목 운동 분류 모델 훈련 시작...")
            print("지원 운동: 스쿼트, 푸쉬업, 데드리프트, 벤치프레스, 런지")
            print("📁 지원 구조: 원본 이미지 폴더 + processed_data 폴더")
            
            accuracy = model.train(args.data_path)
            model.save_model(args.model_path)
            
            print(f"\n🎉 processed_data 지원 5종목 훈련 완료! 정확도: {accuracy:.3f}")
            print("💡 실시간 분석을 시작하려면:")
            print("   python main.py --mode realtime")
            
        except Exception as e:
            print(f"❌ 훈련 실패: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
    elif args.mode == 'predict':
        # 단일 이미지 예측
        if not args.image:
            print("❌ --image 옵션이 필요합니다")
            return 1
        
        model = ExerciseClassificationModel()
        if model.load_model(args.model_path):
            try:
                exercise, confidence = model.predict(args.image)
                
                # 결과 출력
                emoji = {
                    'squat': '🏋️‍♀️', 
                    'push_up': '💪', 
                    'deadlift': '🏋️‍♂️', 
                    'bench_press': '🔥', 
                    'lunge': '🚀'
                }
                print(f"🎯 예측 결과:")
                print(f"  {emoji.get(exercise, '🏋️')} 운동: {exercise.upper()}")
                print(f"  📊 신뢰도: {confidence:.3f} ({confidence*100:.1f}%)")
                
                if confidence > 0.8:
                    print("  ✅ 높은 신뢰도 - 정확한 예측")
                elif confidence > 0.6:
                    print("  ⚠️ 보통 신뢰도 - 추가 확인 권장")
                else:
                    print("  ❌ 낮은 신뢰도 - 불확실한 예측")
                    
            except Exception as e:
                print(f"❌ 예측 실패: {e}")
                return 1
        else:
            return 1
    
    elif args.mode == 'evaluate':
        # 모델 평가
        model = ExerciseClassificationModel()
        if model.load_model(args.model_path):
            print("💡 모델 정보:")
            print(f"  지원 운동: {len(model.label_encoder)}종목")
            print(f"  훈련 상태: {'✅ 완료' if model.is_trained else '❌ 미완료'}")
            print(f"  운동 목록: {', '.join(model.label_encoder.keys())}")
            print(f"  📁 processed_data 구조 지원")
        else:
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())