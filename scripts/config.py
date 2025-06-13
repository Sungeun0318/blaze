"""
운동 자세 분석 시스템 설정 - enhanced_pose_analysis.py 기준 적용
"""

import json
from pathlib import Path
from datetime import datetime

class Config:
    """시스템 설정 클래스 - enhanced_pose_analysis.py 기준 적용"""
    
    def __init__(self):
        self.load_default_config()
    
    def load_default_config(self):
        """기본 설정 로드 - enhanced_pose_analysis.py 기준 적용"""
        
        # 경로 설정
        self.BASE_PATH = Path(".")
        self.DATA_PATH = self.BASE_PATH / "data" / "training_images"
        self.OUTPUT_PATH = self.BASE_PATH / "processed_data"
        self.MODEL_PATH = self.BASE_PATH / "models"
        self.LOGS_PATH = self.BASE_PATH / "logs"
        self.OUTPUTS_PATH = self.BASE_PATH / "outputs"
        
        # 처리 설정
        self.MAX_IMAGES_PER_EXERCISE = 500
        self.IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # MediaPipe 설정 (enhanced와 동일)
        self.MEDIAPIPE_CONFIG = {
            'static_image_mode': True,
            'model_complexity': 2,
            'enable_segmentation': False,
            'min_detection_confidence': 0.5,  # enhanced와 동일
            'min_tracking_confidence': 0.5    # enhanced와 동일
        }
        
        # 실시간 분석 설정
        self.REALTIME_CONFIG = {
            'model_complexity': 1,  # 실시간 처리용 낮은 복잡도
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.5,
            'camera_width': 1280,
            'camera_height': 720,
            'fps_target': 30
        }
        
        # 후처리 설정 (enhanced와 동일)
        self.POST_PROCESSING = {
            'hysteresis_threshold': 0.6,        # enhanced 기본값
            'ema_alpha': 0.3,                   # enhanced와 동일
            'window_size': 15,                  # enhanced와 동일
            'feedback_interval': 2.0,
            'classification_interval': 2.0,
            'pose_history_size': 5,
            'exercise_history_size': 15,
            'visibility_threshold': 0.25        # enhanced와 동일
        }
        
        # AI 모델 설정
        self.MODEL_CONFIG = {
            'random_forest': {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 3,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'test_size': 0.2,
            'validation_split': 0.1
        }
        
        # 🎯 enhanced_pose_analysis.py와 완전히 동일한 운동별 각도 임계값
        self.EXERCISE_THRESHOLDS = {
            'squat': {
                'left_knee': {'points': [23, 25, 27], 'range': (55, 140), 'weight': 1.1},      # enhanced와 동일
                'right_knee': {'points': [24, 26, 28], 'range': (55, 140), 'weight': 1.1},     # enhanced와 동일
                'left_hip': {'points': [11, 23, 25], 'range': (55, 140), 'weight': 0.9},       # enhanced와 동일
                'right_hip': {'points': [12, 24, 26], 'range': (55, 140), 'weight': 0.9},      # enhanced와 동일
                'back_straight': {'points': [11, 23, 25], 'range': (110, 170), 'weight': 1.1}, # enhanced와 동일
                'spine_angle': {'points': [23, 11, 13], 'range': (110, 170), 'weight': 0.9},   # enhanced와 동일
            },
            'push_up': {
                'left_elbow': {'points': [11, 13, 15], 'range': (40, 160), 'weight': 1.0},     # enhanced와 동일
                'right_elbow': {'points': [12, 14, 16], 'range': (40, 160), 'weight': 1.0},    # enhanced와 동일
                'body_line': {'points': [11, 23, 25], 'range': (140, 180), 'weight': 1.2},     # enhanced와 동일
                'leg_straight': {'points': [23, 25, 27], 'range': (140, 180), 'weight': 0.8},  # enhanced와 동일
                'shoulder_alignment': {'points': [13, 11, 23], 'range': (120, 180), 'weight': 0.6}, # enhanced와 동일
                'core_stability': {'points': [11, 12, 23], 'range': (140, 180), 'weight': 1.0}, # enhanced와 동일
            },
            'deadlift': {
                # 🏋️‍♂️ 데드리프트: enhanced에서 대폭 완화된 기준 적용
                'left_knee': {'points': [23, 25, 27], 'range': (80, 140), 'weight': 0.6},      # enhanced와 동일 (대폭 완화)
                'right_knee': {'points': [24, 26, 28], 'range': (80, 140), 'weight': 0.6},     # enhanced와 동일 (대폭 완화)
                'hip_hinge': {'points': [11, 23, 25], 'range': (80, 180), 'weight': 0.7},      # enhanced와 동일 (대폭 완화)
                'back_straight': {'points': [11, 23, 12], 'range': (120, 180), 'weight': 1.0}, # enhanced와 동일 (완화)
                'chest_up': {'points': [23, 11, 13], 'range': (50, 140), 'weight': 0.5},       # enhanced와 동일
                'spine_neutral': {'points': [23, 11, 24], 'range': (120, 180), 'weight': 0.8}, # enhanced와 동일
            },
            'bench_press': {
                'left_elbow': {'points': [11, 13, 15], 'range': (50, 145), 'weight': 1.1},     # enhanced와 동일
                'right_elbow': {'points': [12, 14, 16], 'range': (50, 145), 'weight': 1.1},    # enhanced와 동일
                'left_shoulder': {'points': [13, 11, 23], 'range': (50, 150), 'weight': 0.9},  # enhanced와 동일
                'right_shoulder': {'points': [14, 12, 24], 'range': (50, 150), 'weight': 0.9}, # enhanced와 동일
                'back_arch': {'points': [11, 23, 25], 'range': (90, 170), 'weight': 0.7},      # enhanced와 동일
                'wrist_alignment': {'points': [13, 15, 17], 'range': (70, 180), 'weight': 0.6}, # enhanced와 동일
            },
            'lunge': {
                # 🚀 런지: enhanced와 동일한 기준
                'front_knee': {'points': [23, 25, 27], 'range': (70, 120), 'weight': 1.2},     # enhanced와 동일
                'back_knee': {'points': [24, 26, 28], 'range': (120, 180), 'weight': 1.0},     # enhanced와 동일
                'front_hip': {'points': [11, 23, 25], 'range': (70, 120), 'weight': 0.8},      # enhanced와 동일
                'torso_upright': {'points': [11, 23, 25], 'range': (100, 180), 'weight': 1.2}, # enhanced와 동일
                'front_ankle': {'points': [25, 27, 31], 'range': (80, 110), 'weight': 0.8},    # enhanced와 동일
                'back_hip_extension': {'points': [12, 24, 26], 'range': (150, 180), 'weight': 1.0}, # enhanced와 동일
            }
        }
        
        # 🎯 enhanced와 동일한 분류 임계값
        self.CLASSIFICATION_THRESHOLDS = {
            'squat': 0.5,        # enhanced와 동일
            'push_up': 0.7,      # enhanced와 동일
            'deadlift': 0.8,     # enhanced와 동일 (대폭 완화)
            'bench_press': 0.5,  # enhanced와 동일
            'lunge': 0.6,        # enhanced와 동일
        }
        
        # enhanced와 동일한 운동별 히스테리시스
        self.EXERCISE_HYSTERESIS = {
            'squat': 0.5,        # enhanced와 동일
            'push_up': 0.7,      # enhanced와 동일
            'deadlift': 0.8,     # enhanced와 동일 (대폭 완화)
            'bench_press': 0.5,  # enhanced와 동일
            'lunge': 0.6,        # enhanced와 동일
        }
        
        # enhanced와 동일한 복귀 임계값
        self.RECOVERY_THRESHOLDS = {
            'squat': 0.35,       # 0.5 * 0.7 (enhanced와 동일)
            'push_up': 0.56,     # 0.7 * 0.8 (enhanced와 동일)
            'deadlift': 0.72,    # 0.8 * 0.9 (enhanced와 동일 - 매우 쉬운 복귀)
            'bench_press': 0.35, # 0.5 * 0.7 (enhanced와 동일)
            'lunge': 0.48,       # 0.6 * 0.8 (enhanced와 동일)
        }
        
        # 피드백 메시지
        self.FEEDBACK_MESSAGES = {
            'good': [
                "완벽한 자세입니다!",
                "훌륭한 폼을 유지하고 있습니다!",
                "좋은 자세를 계속 유지하세요!",
                "우수한 운동 폼입니다!",
                "완벽하게 수행하고 있습니다!"
            ],
            'bad': {
                'left_elbow': "왼쪽 팔꿈치 각도를 조정하세요",
                'right_elbow': "오른쪽 팔꿈치 각도를 조정하세요",
                'left_knee': "왼쪽 무릎 각도를 확인하세요",
                'right_knee': "오른쪽 무릎 각도를 확인하세요",
                'left_hip': "왼쪽 엉덩이 자세를 교정하세요",
                'right_hip': "오른쪽 엉덩이 자세를 교정하세요",
                'left_shoulder': "왼쪽 어깨 위치를 조정하세요",
                'right_shoulder': "오른쪽 어깨 위치를 조정하세요",
                'back_straight': "등을 곧게 펴세요",
                'spine_angle': "척추 각도를 바르게 하세요",
                'body_line': "몸을 일직선으로 유지하세요",
                'leg_straight': "다리를 곧게 펴세요",
                'hip_hinge': "엉덩이를 뒤로 빼세요",
                'chest_up': "가슴을 펴세요",
                'spine_neutral': "척추를 중립으로 유지하세요",
                'back_arch': "자연스러운 등 아치를 유지하세요",
                'wrist_alignment': "손목을 정렬하세요",
                # 런지 전용 피드백
                'front_knee': "앞무릎을 90도로 구부리세요",
                'back_knee': "뒷무릎을 더 펴세요",
                'front_hip': "앞 엉덩이 각도를 조정하세요",
                'torso_upright': "상체를 곧게 세우세요",
                'front_ankle': "앞발목 안정성을 유지하세요",
                'back_hip_extension': "뒷엉덩이를 더 신전시키세요",
            },
            'exercise_specific': {
                'squat': {
                    'good': "완벽한 스쿼트 자세입니다!",
                    'bad': "무릎과 엉덩이 각도를 확인하세요"
                },
                'push_up': {
                    'good': "훌륭한 푸쉬업 폼입니다!",
                    'bad': "몸을 일직선으로 유지하세요"
                },
                'bench_press': {
                    'good': "완벽한 벤치프레스입니다!",
                    'bad': "팔꿈치와 어깨 각도를 조정하세요"
                },
                'deadlift': {
                    'good': "우수한 데드리프트 자세입니다!",
                    'bad': "등을 곧게 펴고 무릎을 확인하세요"
                },
                'lunge': {
                    'good': "완벽한 런지 동작입니다!",
                    'bad': "앞무릎 90도, 뒷무릎 펴기, 상체 직립을 확인하세요"
                }
            }
        }
        
        # 시각화 설정
        self.VISUALIZATION = {
            'border_thickness': 20,
            'font_scale': 1.0,
            'font_thickness': 2,
            'colors': {
                'good': (0, 255, 0),      # 초록색
                'bad': (0, 0, 255),       # 빨간색
                'detecting': (255, 255, 0), # 노란색
                'error': (128, 128, 128),  # 회색
                'text': (255, 255, 255),   # 흰색
                'feedback': (0, 255, 255)  # 시안색
            },
            'panel_opacity': 0.7,
            'show_landmarks': True,
            'show_angles': True,
            'show_statistics': True
        }
        
        # 로깅 설정
        self.LOGGING = {
            'level': 'INFO',
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'file_rotation': True,
            'max_log_files': 10,
            'log_to_console': True,
            'log_to_file': True
        }
        
        # 성능 설정
        self.PERFORMANCE = {
            'max_processing_time': 5.0,
            'memory_limit_mb': 1024,
            'cpu_optimization': True,
            'gpu_acceleration': False,
            'parallel_processing': False,
            'enhanced_compatible': True  # enhanced 호환 표시
        }
    
    def get_exercise_threshold(self, exercise: str, joint: str) -> dict:
        """특정 운동의 특정 관절 임계값 가져오기"""
        return self.EXERCISE_THRESHOLDS.get(exercise, {}).get(joint, None)
    
    def get_classification_threshold(self, exercise: str) -> float:
        """enhanced와 동일한 분류 임계값 가져오기"""
        return self.CLASSIFICATION_THRESHOLDS.get(exercise, 0.6)
    
    def get_hysteresis_threshold(self, exercise: str) -> float:
        """enhanced와 동일한 히스테리시스 임계값 가져오기"""
        return self.EXERCISE_HYSTERESIS.get(exercise, 0.6)
    
    def get_recovery_threshold(self, exercise: str) -> float:
        """enhanced와 동일한 복귀 임계값 가져오기"""
        return self.RECOVERY_THRESHOLDS.get(exercise, 0.48)
    
    def get_feedback_message(self, exercise: str, quality: str) -> str:
        """운동별 피드백 메시지 가져오기"""
        if quality == 'good':
            import random
            return random.choice(self.FEEDBACK_MESSAGES['good'])
        else:
            return self.FEEDBACK_MESSAGES['exercise_specific'].get(exercise, {}).get('bad', 
                "자세를 교정해주세요")
    
    def get_supported_exercises(self) -> list:
        """지원되는 운동 목록 반환"""
        return list(self.EXERCISE_THRESHOLDS.keys())
    
    def get_exercise_emoji(self, exercise: str) -> str:
        """운동별 이모지 반환"""
        emojis = {
            'squat': '🏋️‍♀️',
            'push_up': '💪',
            'deadlift': '🏋️‍♂️',
            'bench_press': '🔥',
            'lunge': '🚀'
        }
        return emojis.get(exercise, '🏋️')
    
    def get_target_rate(self, exercise: str) -> str:
        """enhanced 기준 목표 Good 비율"""
        target_rates = {
            'squat': '50-70%',
            'push_up': '50-70%',
            'deadlift': '40-60%',  # 대폭 완화
            'bench_press': '50-70%',
            'lunge': '50-70%'
        }
        return target_rates.get(exercise, '50-70%')
    
    def validate_enhanced_compatibility(self) -> bool:
        """enhanced_pose_analysis.py와의 호환성 검증"""
        # 주요 설정들이 enhanced와 일치하는지 확인
        checks = [
            self.MEDIAPIPE_CONFIG['min_detection_confidence'] == 0.5,
            self.POST_PROCESSING['ema_alpha'] == 0.3,
            self.POST_PROCESSING['window_size'] == 15,
            self.POST_PROCESSING['visibility_threshold'] == 0.25,
            self.CLASSIFICATION_THRESHOLDS['deadlift'] == 0.8,  # 데드리프트 완화 확인
            'enhanced_compatible' in self.PERFORMANCE
        ]
        
        return all(checks)
    
    def save_config(self, filepath: str):
        """설정을 JSON 파일로 저장"""
        config_dict = {
            'version': 'enhanced_pose_analysis_compatible_v1.0',
            'source': 'based_on_enhanced_pose_analysis.py',
            'compatibility_check': self.validate_enhanced_compatibility(),
            'major_changes': {
                'deadlift_relaxed': 'angles 80-180, threshold 0.8',
                'squat_adjusted': 'angles 55-140, threshold 0.5',
                'bench_adjusted': 'angles 50-145, threshold 0.5',
                'push_up_maintained': 'angles 40-160, threshold 0.7'
            },
            'MAX_IMAGES_PER_EXERCISE': self.MAX_IMAGES_PER_EXERCISE,
            'IMAGE_EXTENSIONS': self.IMAGE_EXTENSIONS,
            'MEDIAPIPE_CONFIG': self.MEDIAPIPE_CONFIG,
            'REALTIME_CONFIG': self.REALTIME_CONFIG,
            'POST_PROCESSING': self.POST_PROCESSING,
            'MODEL_CONFIG': self.MODEL_CONFIG,
            'EXERCISE_THRESHOLDS': self.EXERCISE_THRESHOLDS,
            'CLASSIFICATION_THRESHOLDS': self.CLASSIFICATION_THRESHOLDS,
            'EXERCISE_HYSTERESIS': self.EXERCISE_HYSTERESIS,
            'RECOVERY_THRESHOLDS': self.RECOVERY_THRESHOLDS,
            'FEEDBACK_MESSAGES': self.FEEDBACK_MESSAGES,
            'VISUALIZATION': self.VISUALIZATION,
            'LOGGING': self.LOGGING,
            'PERFORMANCE': self.PERFORMANCE,
            'created_at': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            print(f"✅ enhanced 호환 설정 저장 완료: {filepath}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def load_config(self, filepath: str):
        """JSON 파일에서 설정 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 설정 업데이트
            for key, value in config_dict.items():
                if hasattr(self, key) and key not in ['created_at', 'version', 'source', 'compatibility_check', 'major_changes']:
                    setattr(self, key, value)
            
            version = config_dict.get('version', 'unknown')
            source = config_dict.get('source', 'unknown')
            
            print(f"✅ enhanced 호환 설정 로드 완료: {filepath}")
            print(f"📋 버전: {version}")
            print(f"🎯 소스: {source}")
            return True
            
        except FileNotFoundError:
            print(f"⚠️ Config file {filepath} not found. Using default enhanced-compatible configuration.")
            return False
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {filepath}. Using default enhanced-compatible configuration.")
            return False
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return False
    
    def get_summary(self):
        """enhanced 호환 설정 요약 정보"""
        summary = {
            'version': 'enhanced_pose_analysis_compatible_v1.0',
            'enhanced_compatible': self.validate_enhanced_compatibility(),
            'exercises_supported': len(self.EXERCISE_THRESHOLDS),
            'total_joint_thresholds': sum(len(thresholds) for thresholds in self.EXERCISE_THRESHOLDS.values()),
            'max_images_per_exercise': self.MAX_IMAGES_PER_EXERCISE,
            'supported_exercises': self.get_supported_exercises(),
            'target_rates': {exercise: self.get_target_rate(exercise) for exercise in self.get_supported_exercises()},
            'classification_thresholds': self.CLASSIFICATION_THRESHOLDS,
            'hysteresis_thresholds': self.EXERCISE_HYSTERESIS,
            'recovery_thresholds': self.RECOVERY_THRESHOLDS,
            'major_relaxations': {
                'deadlift': 'threshold 0.8, recovery 0.72 (99% Bad 문제 해결)',
                'visibility': '0.25 (enhanced와 동일)',
                'ema_alpha': '0.3 (enhanced와 동일)'
            }
        }
        return summary

# 기본 설정 인스턴스 (enhanced 호환)
default_config = Config()

def get_default_config():
    """enhanced 호환 기본 설정 인스턴스 반환"""
    return default_config

def create_enhanced_compatible_config():
    """enhanced 호환 설정 파일 생성"""
    config = Config()
    config.save_config("enhanced_compatible_config.json")
    print("📄 enhanced_pose_analysis.py 호환 설정 파일 생성: enhanced_compatible_config.json")

if __name__ == "__main__":
    # enhanced 호환 설정 테스트
    print("🔧 enhanced_pose_analysis.py 호환 설정 시스템 테스트...")
    
    config = Config()
    
    # enhanced 호환성 검증
    is_compatible = config.validate_enhanced_compatibility()
    print(f"🎯 enhanced 호환성: {'✅ 통과' if is_compatible else '❌ 실패'}")
    
    # 요약 정보 출력
    summary = config.get_summary()
    print("\n📊 enhanced 호환 설정 요약:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 설정 파일 생성
    config.save_config("enhanced_compatible_default_config.json")
    
    # 특정 설정 테스트
    print(f"\n🦵 스쿼트 무릎 임계값: {config.get_exercise_threshold('squat', 'left_knee')}")
    print(f"🏋️‍♂️ 데드리프트 분류 임계값: {config.get_classification_threshold('deadlift')} (대폭 완화)")
    print(f"🚀 런지 목표 비율: {config.get_target_rate('lunge')}")
    
    print(f"\n🎯 지원 운동 목록: {config.get_supported_exercises()}")
    print(f"✅ enhanced_pose_analysis.py 호환 설정 시스템 테스트 완료!")