"""
운동 자세 분석 시스템 설정
"""

import json
from pathlib import Path
from datetime import datetime

class Config:
    """시스템 설정 클래스"""
    
    def __init__(self):
        self.load_default_config()
    
    def load_default_config(self):
        """기본 설정 로드"""
        
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
        
        # MediaPipe 설정
        self.MEDIAPIPE_CONFIG = {
            'static_image_mode': True,
            'model_complexity': 2,
            'enable_segmentation': False,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.5
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
        
        # 후처리 설정
        self.POST_PROCESSING = {
            'hysteresis_threshold': 0.3,
            'ema_alpha': 0.2,
            'window_size': 10,
            'feedback_interval': 2.0,  # 초
            'classification_interval': 2.0,  # 운동 분류 간격
            'pose_history_size': 5,
            'exercise_history_size': 15
        }
        
        # AI 모델 설정
        self.MODEL_CONFIG = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'test_size': 0.2,
            'validation_split': 0.1
        }
        
        # 운동별 각도 임계값
        self.EXERCISE_THRESHOLDS = {
            'bench_press': {
                'left_elbow': {'points': [11, 13, 15], 'range': (70, 120), 'weight': 1.0},
                'right_elbow': {'points': [12, 14, 16], 'range': (70, 120), 'weight': 1.0},
                'left_shoulder': {'points': [13, 11, 23], 'range': (60, 100), 'weight': 0.8},
                'right_shoulder': {'points': [14, 12, 24], 'range': (60, 100), 'weight': 0.8},
            },
            'deadlift': {
                'left_knee': {'points': [23, 25, 27], 'range': (160, 180), 'weight': 1.0},
                'right_knee': {'points': [24, 26, 28], 'range': (160, 180), 'weight': 1.0},
                'left_hip': {'points': [11, 23, 25], 'range': (160, 180), 'weight': 1.2},
                'right_hip': {'points': [12, 24, 26], 'range': (160, 180), 'weight': 1.2},
                'left_back': {'points': [23, 11, 13], 'range': (160, 180), 'weight': 1.5},
                'right_back': {'points': [24, 12, 14], 'range': (160, 180), 'weight': 1.5},
            },
            'pull_up': {
                'left_elbow': {'points': [11, 13, 15], 'range': (30, 90), 'weight': 1.0},
                'right_elbow': {'points': [12, 14, 16], 'range': (30, 90), 'weight': 1.0},
                'left_shoulder': {'points': [13, 11, 23], 'range': (120, 180), 'weight': 1.2},
                'right_shoulder': {'points': [14, 12, 24], 'range': (120, 180), 'weight': 1.2},
            },
            'push_up': {
                'left_elbow': {'points': [11, 13, 15], 'range': (80, 120), 'weight': 1.0},
                'right_elbow': {'points': [12, 14, 16], 'range': (80, 120), 'weight': 1.0},
                'left_hip': {'points': [11, 23, 25], 'range': (160, 180), 'weight': 1.5},
                'right_hip': {'points': [12, 24, 26], 'range': (160, 180), 'weight': 1.5},
                'left_knee': {'points': [23, 25, 27], 'range': (170, 180), 'weight': 1.0},
                'right_knee': {'points': [24, 26, 28], 'range': (170, 180), 'weight': 1.0},
            },
            'squat': {
                'left_knee': {'points': [23, 25, 27], 'range': (70, 120), 'weight': 1.5},
                'right_knee': {'points': [24, 26, 28], 'range': (70, 120), 'weight': 1.5},
                'left_hip': {'points': [11, 23, 25], 'range': (70, 120), 'weight': 1.2},
                'right_hip': {'points': [12, 24, 26], 'range': (70, 120), 'weight': 1.2},
                'left_back': {'points': [23, 11, 13], 'range': (170, 180), 'weight': 1.0},
                'right_back': {'points': [24, 12, 14], 'range': (170, 180), 'weight': 1.0},
            }
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
                'left_back': "등을 곧게 펴세요",
                'right_back': "등 자세를 바르게 하세요",
                'left_ankle': "왼쪽 발목 위치를 확인하세요",
                'right_ankle': "오른쪽 발목 위치를 확인하세요"
            },
            'exercise_specific': {
                'squat': {
                    'good': "완벽한 스쿼트 자세입니다!",
                    'bad': "무릎과 엉덩이 각도를 확인하세요"
                },
                'push_up': {
                    'good': "훌륭한 푸시업 폼입니다!",
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
                'pull_up': {
                    'good': "완벽한 풀업 동작입니다!",
                    'bad': "팔꿈치와 어깨 각도를 확인하세요"
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
            'max_processing_time': 5.0,  # 초
            'memory_limit_mb': 1024,
            'cpu_optimization': True,
            'gpu_acceleration': False,
            'parallel_processing': False
        }
    
    def get_exercise_threshold(self, exercise: str, joint: str) -> dict:
        """특정 운동의 특정 관절 임계값 가져오기"""
        return self.EXERCISE_THRESHOLDS.get(exercise, {}).get(joint, None)
    
    def get_feedback_message(self, exercise: str, quality: str) -> str:
        """운동별 피드백 메시지 가져오기"""
        if quality == 'good':
            import random
            return random.choice(self.FEEDBACK_MESSAGES['good'])
        else:
            return self.FEEDBACK_MESSAGES['exercise_specific'].get(exercise, {}).get('bad', 
                "자세를 교정해주세요")
    
    def save_config(self, filepath: str):
        """설정을 JSON 파일로 저장"""
        config_dict = {
            'MAX_IMAGES_PER_EXERCISE': self.MAX_IMAGES_PER_EXERCISE,
            'IMAGE_EXTENSIONS': self.IMAGE_EXTENSIONS,
            'MEDIAPIPE_CONFIG': self.MEDIAPIPE_CONFIG,
            'REALTIME_CONFIG': self.REALTIME_CONFIG,
            'POST_PROCESSING': self.POST_PROCESSING,
            'MODEL_CONFIG': self.MODEL_CONFIG,
            'EXERCISE_THRESHOLDS': self.EXERCISE_THRESHOLDS,
            'FEEDBACK_MESSAGES': self.FEEDBACK_MESSAGES,
            'VISUALIZATION': self.VISUALIZATION,
            'LOGGING': self.LOGGING,
            'PERFORMANCE': self.PERFORMANCE,
            'created_at': datetime.now().isoformat()
        }
        
        # 경로 객체를 문자열로 변환
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            print(f"✅ Configuration saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def load_config(self, filepath: str):
        """JSON 파일에서 설정 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 설정 업데이트
            for key, value in config_dict.items():
                if hasattr(self, key) and key != 'created_at':
                    setattr(self, key, value)
            
            print(f"✅ Configuration loaded from: {filepath}")
            return True
            
        except FileNotFoundError:
            print(f"⚠️ Config file {filepath} not found. Using default configuration.")
            return False
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {filepath}. Using default configuration.")
            return False
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return False
    
    def validate_config(self):
        """설정 유효성 검사"""
        issues = []
        
        # 경로 존재 확인
        required_paths = [self.BASE_PATH, self.DATA_PATH]
        for path in required_paths:
            if not path.exists():
                issues.append(f"Path does not exist: {path}")
        
        # 임계값 범위 확인
        for exercise, thresholds in self.EXERCISE_THRESHOLDS.items():
            for joint, config in thresholds.items():
                range_min, range_max = config['range']
                if range_min >= range_max:
                    issues.append(f"Invalid range for {exercise}.{joint}: {config['range']}")
                
                if not (0 <= range_min <= 180 and 0 <= range_max <= 180):
                    issues.append(f"Angle out of range for {exercise}.{joint}: {config['range']}")
        
        # 후처리 설정 확인
        if not (0 < self.POST_PROCESSING['ema_alpha'] <= 1):
            issues.append("EMA alpha must be between 0 and 1")
        
        if self.POST_PROCESSING['window_size'] < 1:
            issues.append("Window size must be positive")
        
        if issues:
            print("⚠️ Configuration validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("✅ Configuration validation passed")
        return True
    
    def get_summary(self):
        """설정 요약 정보"""
        summary = {
            'exercises_supported': len(self.EXERCISE_THRESHOLDS),
            'total_joint_thresholds': sum(len(thresholds) for thresholds in self.EXERCISE_THRESHOLDS.values()),
            'max_images_per_exercise': self.MAX_IMAGES_PER_EXERCISE,
            'supported_image_formats': len(self.IMAGE_EXTENSIONS),
            'post_processing_enabled': True,
            'visualization_enabled': True,
            'logging_enabled': self.LOGGING['log_to_file']
        }
        return summary

# 기본 설정 인스턴스
default_config = Config()

def get_default_config():
    """기본 설정 인스턴스 반환"""
    return default_config

def create_sample_config():
    """샘플 설정 파일 생성"""
    config = Config()
    config.save_config("sample_config.json")
    print("📄 Sample configuration created: sample_config.json")

if __name__ == "__main__":
    # 설정 테스트
    print("🔧 Testing configuration system...")
    
    config = Config()
    
    # 설정 유효성 검사
    config.validate_config()
    
    # 요약 정보 출력
    summary = config.get_summary()
    print("\n📊 Configuration Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 샘플 설정 파일 생성
    config.save_config("default_config.json")
    
    # 특정 설정 테스트
    squat_knee = config.get_exercise_threshold('squat', 'left_knee')
    print(f"\n🦵 Squat left knee threshold: {squat_knee}")
    
    good_message = config.get_feedback_message('squat', 'good')
    print(f"💬 Squat good feedback: {good_message}")
    
    print("\n✅ Configuration system test completed!")