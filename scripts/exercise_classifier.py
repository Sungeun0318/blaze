"""
운동 동작 자동 분류 모델
BlazePose 랜드마크를 기반으로 운동 종목을 자동으로 분류합니다.
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
    """운동 특징 추출기"""
    
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
        """주요 관절 각도 계산"""
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
        
        # 주요 각도들 계산
        angle_configs = [
            # 팔꿈치 각도
            ([11, 13, 15], 'left_elbow'),    # 왼쪽 팔꿈치
            ([12, 14, 16], 'right_elbow'),   # 오른쪽 팔꿈치
            
            # 어깨 각도  
            ([13, 11, 23], 'left_shoulder'), # 왼쪽 어깨
            ([14, 12, 24], 'right_shoulder'), # 오른쪽 어깨
            
            # 엉덩이 각도
            ([11, 23, 25], 'left_hip'),      # 왼쪽 엉덩이
            ([12, 24, 26], 'right_hip'),     # 오른쪽 엉덩이
            
            # 무릎 각도
            ([23, 25, 27], 'left_knee'),     # 왼쪽 무릎
            ([24, 26, 28], 'right_knee'),    # 오른쪽 무릎
            
            # 발목 각도
            ([25, 27, 31], 'left_ankle'),    # 왼쪽 발목
            ([26, 28, 32], 'right_ankle'),   # 오른쪽 발목
            
            # 척추 각도
            ([11, 23, 25], 'spine_upper'),   # 상체 척추
            ([23, 25, 27], 'spine_lower'),   # 하체 척추
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
        """주요 신체 부위 간 거리 계산"""
        points = landmarks.reshape(-1, 3)
        
        distances = []
        
        # 주요 거리들
        distance_configs = [
            ([11, 12], 'shoulder_width'),    # 어깨 너비
            ([23, 24], 'hip_width'),         # 엉덩이 너비
            ([27, 28], 'ankle_width'),       # 발목 너비
            ([11, 23], 'left_torso'),        # 왼쪽 몸통 길이
            ([12, 24], 'right_torso'),       # 오른쪽 몸통 길이
            ([23, 27], 'left_leg'),          # 왼쪽 다리 길이
            ([24, 28], 'right_leg'),         # 오른쪽 다리 길이
            ([15, 16], 'hand_distance'),     # 양손 거리
            ([0, 23], 'head_to_hip'),        # 머리에서 엉덩이까지
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
        """신체 비율 계산"""
        points = landmarks.reshape(-1, 3)
        
        ratios = []
        
        try:
            # 주요 비율들
            shoulder_width = np.linalg.norm(points[11][:2] - points[12][:2])
            hip_width = np.linalg.norm(points[23][:2] - points[24][:2])
            torso_height = np.linalg.norm(points[11][:2] - points[23][:2])
            leg_length = np.linalg.norm(points[23][:2] - points[27][:2])
            
            # 비율 계산
            ratios.extend([
                shoulder_width / max(hip_width, 0.001),      # 어깨/엉덩이 비율
                torso_height / max(leg_length, 0.001),       # 몸통/다리 비율
                hip_width / max(torso_height, 0.001),        # 엉덩이/몸통 비율
                
                # 높이 비율
                points[11][1] / max(points[27][1], 0.001),   # 어깨/발목 높이 비율
                points[23][1] / max(points[27][1], 0.001),   # 엉덩이/발목 높이 비율
            ])
            
        except:
            ratios = [1.0] * 5
        
        return np.array(ratios)
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """이미지에서 전체 특징 추출"""
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
    """운동 분류 모델"""
    
    def __init__(self):
        self.feature_extractor = ExerciseFeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.label_encoder = {
            'squat': 0,
            'push_up': 1, 
            'bench_press': 2,
            'deadlift': 3,
            'pull_up': 4
        }
        self.reverse_encoder = {v: k for k, v in self.label_encoder.items()}
        self.is_trained = False
    
    def prepare_training_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """훈련 데이터 준비"""
        data_dir = Path(data_path)
        
        features_list = []
        labels_list = []
        
        exercise_dirs = {
            'squat_exercise': 'squat',
            'push_up_exercise': 'push_up',
            'bench_press_exercise': 'bench_press', 
            'deadlift_exercise': 'deadlift',
            'pull_up_exercise': 'pull_up'
        }
        
        print("🔍 훈련 데이터 수집 중...")
        
        for dir_name, exercise_name in exercise_dirs.items():
            exercise_path = data_dir / dir_name
            if not exercise_path.exists():
                print(f"⚠️ Warning: {exercise_path} not found")
                continue
            
            print(f"📂 Processing {exercise_name}...")
            
            # 이미지 파일들 가져오기
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(list(exercise_path.glob(ext)))
            
            count = 0
            for img_file in image_files:
                features = self.feature_extractor.extract_features(str(img_file))
                if features is not None:
                    features_list.append(features)
                    labels_list.append(self.label_encoder[exercise_name])
                    count += 1
                    
                    if count % 50 == 0:
                        print(f"  📸 {exercise_name}: {count} images processed")
            
            print(f"  ✅ {exercise_name}: Total {count} images")
        
        if not features_list:
            raise ValueError("❌ No valid training data found!")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"📊 Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train(self, data_path: str, test_size: float = 0.2):
        """모델 훈련"""
        print("🧠 === 운동 분류 모델 훈련 시작 ===")
        
        # 훈련 데이터 준비
        X, y = self.prepare_training_data(data_path)
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📚 Training set: {X_train.shape[0]} samples")
        print(f"🔬 Test set: {X_test.shape[0]} samples")
        
        # 모델 훈련
        print("⚙️ 모델 훈련 중...")
        self.model.fit(X_train, y_train)
        
        # 성능 평가
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ === 훈련 완료 ===")
        print(f"🎯 정확도: {accuracy:.3f}")
        print("\n📈 상세 성능 리포트:")
        print(classification_report(y_test, y_pred, 
                                  target_names=list(self.label_encoder.keys())))
        
        # 특징 중요도
        feature_importance = self.model.feature_importances_
        print(f"\n🔍 특징 중요도 (평균): {np.mean(feature_importance):.3f}")
        
        self.is_trained = True
        return accuracy
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """단일 이미지 예측"""
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
    
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[str, float]]:
        """여러 이미지 일괄 예측"""
        results = []
        
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)
        
        return results
    
    def save_model(self, model_path: str):
        """모델 저장"""
        if not self.is_trained:
            raise ValueError("❌ Model is not trained yet!")
        
        # 모델 디렉토리 확인/생성
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'reverse_encoder': self.reverse_encoder,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Model saved to: {model_path}")
    
    def load_model(self, model_path: str):
        """모델 로드"""
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.reverse_encoder = model_data['reverse_encoder']
            self.is_trained = model_data['is_trained']
            
            print(f"📥 Model loaded from: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Exercise Classification Model')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'predict'],
                       help='실행 모드')
    parser.add_argument('--data_path', type=str, default='./data/training_images',
                       help='훈련 데이터 경로')
    parser.add_argument('--model_path', type=str, default='models/exercise_classifier.pkl',
                       help='모델 파일 경로')
    parser.add_argument('--image', type=str, help='단일 이미지 예측용')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 모델 훈련
        model = ExerciseClassificationModel()
        try:
            accuracy = model.train(args.data_path)
            model.save_model(args.model_path)
            print(f"🎉 훈련 완료! 정확도: {accuracy:.3f}")
        except Exception as e:
            print(f"❌ 훈련 실패: {e}")
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
                print(f"🎯 예측 결과: {exercise} (신뢰도: {confidence:.3f})")
            except Exception as e:
                print(f"❌ 예측 실패: {e}")
                return 1
        else:
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())