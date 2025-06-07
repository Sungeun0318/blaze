"""
운동 자세 분석 시스템 설정 - 풀업→런지 교체 버전
"""

import json
from pathlib import Path
from datetime import datetime

class Config:
    """시스템 설정 클래스 - 풀업→런지 교체"""
    
    def __init__(self):
        self.load_default_config()
    
    def load_default_config(self):
        """기본 설정 로드 - 풀업→런지 교체"""
        
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
        
        # 후처리 설정 (완화됨)
        self.POST_PROCESSING = {
            'hysteresis_threshold': 0.4,    # 기본값 완화
            'ema_alpha': 0.3,
            'window_size': 10,
            'feedback_interval': 2.0,       # 초
            'classification_interval': 2.0, # 운동 분류 간격
            'pose_history_size': 5,
            'exercise_history_size': 15
        }
        
        # AI 모델 설정
        self.MODEL_CONFIG = {
            'random_forest': {
                'n_estimators': 300,  # 5종목이므로 더 많은 트리
                'max_depth': 20,      # 더 깊은 트리
                'min_samples_split': 3,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'test_size': 0.2,
            'validation_split': 0.1
        }
        
        # 🚀 풀업→런지 교체 및 완화된 운동별 각도 임계값
        self.EXERCISE_THRESHOLDS = {
            'bench_press': {
                'left_elbow': {'points': [11, 13, 15], 'range': (20, 180), 'weight': 0.3},    # 완화됨
                'right_elbow': {'points': [12, 14, 16], 'range': (20, 180), 'weight': 0.3},   # 완화됨
                'left_shoulder': {'points': [13, 11, 23], 'range': (20, 170), 'weight': 0.2}, # 완화됨
                'right_shoulder': {'points': [14, 12, 24], 'range': (20, 170), 'weight': 0.2}, # 완화됨
            },
            'deadlift': {
                'left_knee': {'points': [23, 25, 27], 'range': (100, 180), 'weight': 0.2},   # 완화됨
                'right_knee': {'points': [24, 26, 28], 'range': (100, 180), 'weight': 0.2},  # 완화됨
                'left_hip': {'points': [11, 23, 25], 'range': (80, 180), 'weight': 0.3},     # 완화됨
                'right_hip': {'points': [12, 24, 26], 'range': (80, 180), 'weight': 0.3},    # 완화됨
                'left_back': {'points': [23, 11, 13], 'range': (120, 180), 'weight': 0.4},   # 완화됨
                'right_back': {'points': [24, 12, 14], 'range': (120, 180), 'weight': 0.4},  # 완화됨
            },
            'lunge': {  # 🚀 새로 추가된 런지
                'front_knee': {'points': [23, 25, 27], 'range': (70, 130), 'weight': 0.9},      # 런지의 핵심
                'back_knee': {'points': [24, 26, 28], 'range': (120, 180), 'weight': 0.8},      # 뒷무릎 펴짐
                'front_hip': {'points': [11, 23, 25], 'range': (70, 130), 'weight': 0.8},       # 앞 엉덩이
                'torso_upright': {'points': [11, 23, 25], 'range': (160, 180), 'weight': 0.7},  # 상체 직립
                'front_ankle': {'points': [25, 27, 31], 'range': (80, 120), 'weight': 0.5},     # 앞발목 안정성
                'back_hip_extension': {'points': [12, 24, 26], 'range': (140, 180), 'weight': 0.6}, # 뒷엉덩이 신전
                'pelvis_level': {'points': [23, 24, 11], 'range': (170, 180), 'weight': 0.7},   # 골반 수평
                'knee_tracking': {'points': [23, 25, 27], 'range': (160, 180), 'weight': 0.8},  # 무릎 추적
            },
            'push_up': {
                'left_elbow': {'points': [11, 13, 15], 'range': (20, 170), 'weight': 0.4},     # 기존 유지
                'right_elbow': {'points': [12, 14, 16], 'range': (20, 170), 'weight': 0.4},    # 기존 유지
                'left_hip': {'points': [11, 23, 25], 'range': (100, 180), 'weight': 0.5},      # 기존 유지
                'right_hip': {'points': [12, 24, 26], 'range': (100, 180), 'weight': 0.5},     # 기존 유지
                'left_knee': {'points': [23, 25, 27], 'range': (130, 180), 'weight': 0.2},     # 기존 유지
                'right_knee': {'points': [24, 26, 28], 'range': (130, 180), 'weight': 0.2},    # 기존 유지
            },
            'squat': {
                'left_knee': {'points': [23, 25, 27], 'range': (40, 160), 'weight': 0.8},      # 기존 유지
                'right_knee': {'points': [24, 26, 28], 'range': (40, 160), 'weight': 0.8},     # 기존 유지
                'left_hip': {'points': [11, 23, 25], 'range': (40, 160), 'weight': 0.8},       # 기존 유지
                'right_hip': {'points': [12, 24, 26], 'range': (40, 160), 'weight': 0.8},      # 기존 유지
                'left_back': {'points': [23, 11, 13], 'range': (140, 180), 'weight': 0.9},     # 기존 유지
                'right_back': {'points': [24, 12, 14], 'range': (140, 180), 'weight': 0.9},    # 기존 유지
            }
        }
        
        # 🚀 풀업→런지 교체 피드백 메시지
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
                'right_ankle': "오른쪽 발목 위치를 확인하세요",
                # 🚀 런지 전용 피드백 추가
                'front_knee': "앞무릎을 90도로 구부리세요",
                'back_knee': "뒷무릎을 더 펴세요",
                'front_hip': "앞 엉덩이 각도를 조정하세요",
                'torso_upright': "상체를 곧게 세우세요",
                'front_ankle': "앞발목 안정성을 유지하세요",
                'back_hip_extension': "뒷엉덩이를 더 신전시키세요",
                'pelvis_level': "골반을 수평으로 유지하세요",
                'knee_tracking': "무릎이 발끝 방향을 향하게 하세요"
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
                'lunge': {  # 🚀 새로 추가된 런지 피드백
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
        
        # 성능 설정 (완화됨)
        self.PERFORMANCE = {
            'max_processing_time': 5.0,  # 초
            'memory_limit_mb': 1024,
            'cpu_optimization': True,
            'gpu_acceleration': False,
            'parallel_processing': False,
            'relaxed_criteria': True      # 완화된 기준 사용
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
    
    def get_supported_exercises(self) -> list:
        """지원되는 운동 목록 반환 (풀업→런지 교체)"""
        return list(self.EXERCISE_THRESHOLDS.keys())
    
    def is_lunge_exercise(self, exercise: str) -> bool:
        """런지 운동인지 확인"""
        return exercise == 'lunge'
    
    def get_exercise_emoji(self, exercise: str) -> str:
        """운동별 이모지 반환"""
        emojis = {
            'squat': '🏋️‍♀️',
            'push_up': '💪',
            'deadlift': '🏋️‍♂️',
            'bench_press': '🔥',
            'lunge': '🚀'  # 새로 추가된 런지
        }
        return emojis.get(exercise, '🏋️')
    
    def save_config(self, filepath: str):
        """설정을 JSON 파일로 저장"""
        config_dict = {
            'version': 'pullup_to_lunge_relaxed_v1.0',
            'changelog': {
                'replaced_exercise': 'pull_up → lunge',
                'relaxed_exercises': ['deadlift', 'bench_press'],
                'maintained_exercises': ['squat', 'push_up']
            },
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
            print(f"✅ 풀업→런지 교체 설정 저장 완료: {filepath}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def load_config(self, filepath: str):
        """JSON 파일에서 설정 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 설정 업데이트
            for key, value in config_dict.items():
                if hasattr(self, key) and key not in ['created_at', 'version', 'changelog']:
                    setattr(self, key, value)
            
            version = config_dict.get('version', 'unknown')
            changelog = config_dict.get('changelog', {})
            
            print(f"✅ 풀업→런지 교체 설정 로드 완료: {filepath}")
            print(f"📋 버전: {version}")
            if changelog:
                print(f"🔄 변경사항: {changelog.get('replaced_exercise', 'N/A')}")
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
        """설정 유효성 검사 - 풀업→런지 교체"""
        issues = []
        
        # 경로 존재 확인
        required_paths = [self.BASE_PATH, self.DATA_PATH]
        for path in required_paths:
            if not path.exists():
                issues.append(f"Path does not exist: {path}")
        
        # 5종목 임계값 범위 확인 (풀업→런지 교체)
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
        
        # 런지 특화 검증
        if 'lunge' in self.EXERCISE_THRESHOLDS:
            lunge_config = self.EXERCISE_THRESHOLDS['lunge']
            required_lunge_joints = ['front_knee', 'back_knee', 'front_hip', 'torso_upright']
            for joint in required_lunge_joints:
                if joint not in lunge_config:
                    issues.append(f"Missing required lunge joint: {joint}")
        
        if issues:
            print("⚠️ Configuration validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("✅ 풀업→런지 교체 설정 검증 통과")
        return True
    
    def get_summary(self):
        """설정 요약 정보 - 풀업→런지 교체"""
        summary = {
            'version': 'pullup_to_lunge_relaxed_v1.0',
            'exercises_supported': len(self.EXERCISE_THRESHOLDS),
            'total_joint_thresholds': sum(len(thresholds) for thresholds in self.EXERCISE_THRESHOLDS.values()),
            'max_images_per_exercise': self.MAX_IMAGES_PER_EXERCISE,
            'supported_image_formats': len(self.IMAGE_EXTENSIONS),
            'post_processing_enabled': True,
            'visualization_enabled': True,
            'logging_enabled': self.LOGGING['log_to_file'],
            'relaxed_criteria': self.PERFORMANCE.get('relaxed_criteria', False),
            'supported_exercises': self.get_supported_exercises(),
            'new_exercise': 'lunge (replaced pull_up)',
            'relaxed_exercises': ['deadlift', 'bench_press'],
            'maintained_exercises': ['squat', 'push_up']
        }
        return summary

# 기본 설정 인스턴스
default_config = Config()

def get_default_config():
    """기본 설정 인스턴스 반환"""
    return default_config

def create_sample_config():
    """샘플 설정 파일 생성 - 풀업→런지 교체"""
    config = Config()
    config.save_config("pullup_to_lunge_sample_config.json")
    print("📄 풀업→런지 교체 샘플 설정 파일 생성: pullup_to_lunge_sample_config.json")

if __name__ == "__main__":
    # 설정 테스트
    print("🔧 풀업→런지 교체 설정 시스템 테스트...")
    
    config = Config()
    
    # 설정 유효성 검사
    config.validate_config()
    
    # 요약 정보 출력
    summary = config.get_summary()
    print("\n📊 풀업→런지 교체 설정 요약:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 샘플 설정 파일 생성
    config.save_config("pullup_to_lunge_default_config.json")
    
    # 특정 설정 테스트
    squat_knee = config.get_exercise_threshold('squat', 'left_knee')
    print(f"\n🦵 스쿼트 무릎 임계값: {squat_knee}")
    
    lunge_front_knee = config.get_exercise_threshold('lunge', 'front_knee')
    print(f"🚀 런지 앞무릎 임계값: {lunge_front_knee}")
    
    good_message = config.get_feedback_message('lunge', 'good')
    print(f"💬 런지 Good 피드백: {good_message}")
    
    print(f"\n🎯 지원 운동 목록: {config.get_supported_exercises()}")
    print(f"🚀 런지 운동 여부: {config.is_lunge_exercise('lunge')}")
    
    print("\n✅ 풀업→런지 교체 설정 시스템 테스트 완료!")