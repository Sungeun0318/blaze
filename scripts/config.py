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
        self.DATA_PATH = self.BASE_PATH / "data" / "images"
        self.OUTPUT_PATH = self.BASE_PATH / "processed_data"
        self.MODEL_PATH = self.BASE_PATH / "models"
        
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
            'min_tracking_confidence': 0.5
        }
        
        # 후처리 설정
        self.POST_PROCESSING = {
            'hysteresis_threshold': 0.3,
            'ema_alpha': 0.2,
            'window_size': 10,
            'feedback_interval': 2.0  # 초
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
                'left_back': "등 자세를 곧게 펴세요",
                'right_back': "등 자세를 곧게 펴세요",
            }
        }
    
    def save_config(self, filepath: str):
        """설정을 JSON 파일로 저장"""
        config_dict = {
            'MAX_IMAGES_PER_EXERCISE': self.MAX_IMAGES_PER_EXERCISE,
            'MEDIAPIPE_CONFIG': self.MEDIAPIPE_CONFIG,
            'POST_PROCESSING': self.POST_PROCESSING,
            'EXERCISE_THRESHOLDS': self.EXERCISE_THRESHOLDS,
            'FEEDBACK_MESSAGES': self.FEEDBACK_MESSAGES
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def load_config(self, filepath: str):
        """JSON 파일에서 설정 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except FileNotFoundError:
            print(f"Config file {filepath} not found. Using default configuration.")
        except json.JSONDecodeError:
            print(f"Invalid JSON in {filepath}. Using default configuration.")

# 기본 설정 인스턴스
default_config = Config()

if __name__ == "__main__":
    # 설정 테스트
    config = Config()
    config.save_config("default_config.json")
    print("Configuration file created: default_config.json")