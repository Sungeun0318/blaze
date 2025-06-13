"""
BLAZE 시스템 유틸리티 함수들
enhanced_pose_analysis.py 기준 호환
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

def ensure_directory(path: str):
    """디렉토리 생성 확인"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_timestamp():
    """현재 시간 타임스탬프"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(data: Dict, filepath: str):
    """JSON 파일 저장"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"JSON 저장 실패: {e}")
        return False

def load_json(filepath: str) -> Optional[Dict]:
    """JSON 파일 로드"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON 로드 실패: {e}")
        return None

def resize_image(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """이미지 크기 조정"""
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image

def calculate_angle_safe(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """안전한 각도 계산 (enhanced 호환)"""
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

def get_exercise_emoji(exercise: str) -> str:
    """운동별 이모지 반환"""
    emojis = {
        'squat': '🏋️‍♀️',
        'push_up': '💪',
        'deadlift': '🏋️‍♂️',
        'bench_press': '🔥',
        'lunge': '🚀'
    }
    return emojis.get(exercise, '🏋️')

def format_exercise_name(exercise: str) -> str:
    """운동명 포맷팅"""
    return exercise.replace('_', ' ').title()

def calculate_success_rate(good_count: int, bad_count: int) -> float:
    """성공률 계산"""
    total = good_count + bad_count
    return (good_count / total) * 100 if total > 0 else 0.0

def is_target_met(success_rate: float, exercise: str) -> bool:
    """enhanced 기준 목표 달성 여부"""
    target_ranges = {
        'squat': (50, 70),
        'push_up': (50, 70),
        'deadlift': (40, 60),  # 완화된 목표
        'bench_press': (50, 70),
        'lunge': (50, 70)
    }
    
    min_target, max_target = target_ranges.get(exercise, (50, 70))
    return min_target <= success_rate <= max_target

def get_target_range(exercise: str) -> str:
    """운동별 목표 범위 반환"""
    target_ranges = {
        'squat': '50-70%',
        'push_up': '50-70%',
        'deadlift': '40-60% (완화됨)',
        'bench_press': '50-70%',
        'lunge': '50-70%'
    }
    return target_ranges.get(exercise, '50-70%')

def print_progress(current: int, total: int, prefix: str = "진행률"):
    """진행률 출력"""
    percentage = (current / total) * 100
    print(f"  {prefix}: {current}/{total} ({percentage:.1f}%)")

def validate_image_file(filepath: str) -> bool:
    """이미지 파일 유효성 검사"""
    if not os.path.exists(filepath):
        return False
    
    try:
        image = cv2.imread(filepath)
        return image is not None
    except:
        return False

def get_image_files(directory: str) -> List[str]:
    """디렉토리에서 이미지 파일들 찾기"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return sorted(image_files)

def create_summary_report(results: Dict, title: str = "분석 결과") -> str:
    """요약 리포트 생성"""
    report = f"\n{'='*60}\n"
    report += f"{title}\n"
    report += f"{'='*60}\n"
    
    for exercise, result in results.items():
        if isinstance(result, dict) and 'good' in result:
            emoji = get_exercise_emoji(exercise)
            total = result['good'] + result['bad']
            success_rate = calculate_success_rate(result['good'], result['bad'])
            target_range = get_target_range(exercise)
            target_met = is_target_met(success_rate, exercise)
            
            status = "✅ 목표 달성" if target_met else "📊 목표 근접"
            
            report += f"\n{emoji} {format_exercise_name(exercise)}:\n"
            report += f"  Good: {result['good']}장, Bad: {result['bad']}장, 실패: {result.get('failed', 0)}장\n"
            report += f"  성공률: {success_rate:.1f}% (목표: {target_range}) {status}\n"
    
    report += f"\n{'='*60}\n"
    return report

def log_analysis_start(exercise: str, image_count: int):
    """분석 시작 로그"""
    emoji = get_exercise_emoji(exercise)
    print(f"\n{emoji} {format_exercise_name(exercise)} 분석 시작")
    print(f"  📸 이미지 수: {image_count}장")
    print(f"  🔧 기준: Enhanced Pose Analysis")
    print(f"  🎯 목표: {get_target_range(exercise)}")

def log_analysis_result(exercise: str, good: int, bad: int, failed: int):
    """분석 결과 로그"""
    emoji = get_exercise_emoji(exercise)
    total = good + bad
    success_rate = calculate_success_rate(good, bad)
    target_met = is_target_met(success_rate, exercise)
    
    print(f"\n{emoji} {format_exercise_name(exercise)} 분석 완료:")
    print(f"  ✅ Good: {good}장 ({success_rate:.1f}%)")
    print(f"  ❌ Bad: {bad}장")
    print(f"  💥 실패: {failed}장")
    print(f"  🎯 목표 달성: {'✅ 성공' if target_met else '📊 진행 중'}")

def enhanced_compatibility_check() -> Dict:
    """enhanced_pose_analysis.py 호환성 확인"""
    return {
        'enhanced_compatible': True,
        'version': 'enhanced_pose_analysis_v1.0',
        'features': [
            'angle_calculation',
            'view_detection', 
            'weighted_scoring',
            'adaptive_thresholds',
            'deadlift_relaxation'
        ],
        'target_rates': {
            'squat': '50-70%',
            'push_up': '50-70%', 
            'deadlift': '40-60% (relaxed)',
            'bench_press': '50-70%',
            'lunge': '50-70%'
        }
    }

def format_angle_info(angle: float, range_min: float, range_max: float) -> str:
    """각도 정보 포맷팅"""
    in_range = range_min <= angle <= range_max
    status = "✅" if in_range else "❌"
    return f"{angle:.1f}° ({range_min:.0f}-{range_max:.0f}°) {status}"

def calculate_deviation(angle: float, range_min: float, range_max: float) -> float:
    """각도 편차 계산"""
    if range_min <= angle <= range_max:
        return 0.0
    return min(abs(angle - range_min), abs(angle - range_max))

def get_exercise_advice(exercise: str, joint: str) -> str:
    """운동별 관절 조언"""
    advice_map = {
        'squat': {
            'knee': '무릎이 발끝을 넘지 않게 주의하세요',
            'hip': '엉덩이를 더 뒤로 빼세요',
            'back': '등을 곧게 펴세요'
        },
        'push_up': {
            'elbow': '팔꿈치를 몸에 더 가깝게 하세요',
            'body': '몸을 일직선으로 유지하세요',
            'shoulder': '어깨를 안정적으로 유지하세요'
        },
        'deadlift': {
            'knee': '무릎을 약간만 구부리세요 (완화 기준)',
            'hip': '힙 힌지 동작을 크게 하세요',
            'back': '등을 곧게 펴세요 - 가장 중요!'
        },
        'bench_press': {
            'elbow': '팔꿈치 각도를 조정하세요',
            'shoulder': '어깨를 안정적으로 유지하세요',
            'arch': '자연스러운 등 아치를 유지하세요'
        },
        'lunge': {
            'front_knee': '앞무릎을 90도로 구부리세요',
            'back_knee': '뒷무릎을 더 펴세요', 
            'torso': '상체를 곧게 세우세요',
            'ankle': '앞발목 안정성을 유지하세요'
        }
    }
    
    exercise_advice = advice_map.get(exercise, {})
    for key in exercise_advice:
        if key in joint.lower():
            return exercise_advice[key]
    
    return "자세를 교정해보세요"

class ProgressTracker:
    """진행률 추적기"""
    
    def __init__(self, total: int, name: str = "작업"):
        self.total = total
        self.current = 0
        self.name = name
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """진행률 업데이트"""
        self.current += increment
        self.print_progress()
    
    def print_progress(self):
        """진행률 출력"""
        if self.current % 50 == 0 or self.current == self.total:
            percentage = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            eta = (elapsed / max(self.current, 1)) * (self.total - self.current)
            
            print(f"  📊 {self.name}: {self.current}/{self.total} ({percentage:.1f}%) - ETA: {eta:.0f}초")

def main():
    """유틸리티 테스트"""
    print("🔧 BLAZE 유틸리티 시스템 테스트")
    
    # Enhanced 호환성 확인
    compatibility = enhanced_compatibility_check()
    print(f"✅ Enhanced 호환성: {compatibility['enhanced_compatible']}")
    
    # 각도 계산 테스트
    test_angle = calculate_angle_safe((0, 0), (1, 0), (1, 1))
    print(f"📐 테스트 각도: {test_angle:.1f}°")
    
    # 운동별 이모지 테스트
    for exercise in ['squat', 'push_up', 'deadlift', 'bench_press', 'lunge']:
        emoji = get_exercise_emoji(exercise)
        target = get_target_range(exercise)
        print(f"{emoji} {format_exercise_name(exercise)}: {target}")
    
    print("✅ 유틸리티 시스템 정상 작동")

if __name__ == "__main__":
    main()