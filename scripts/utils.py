"""
유틸리티 함수 모음
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from datetime import datetime

def create_directory_structure(base_path: str):
    """디렉토리 구조 생성"""
    base = Path(base_path)
    
    # 필요한 디렉토리들
    directories = [
        "data/images/bench_press_exercise",
        "data/images/deadlift_exercise", 
        "data/images/pull_up_exercise",
        "data/images/push_up_exercise",
        "data/images/squat_exercise",
        "processed_data",
        "models",
        "logs",
        "results"
    ]
    
    for dir_path in directories:
        (base / dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Directory structure created at: {base}")

def validate_image_dataset(data_path: str) -> Dict:
    """이미지 데이터셋 검증"""
    data_dir = Path(data_path)
    validation_report = {}
    
    exercises = ['bench_press_exercise', 'deadlift_exercise', 'pull_up_exercise', 
                'push_up_exercise', 'squat_exercise']
    
    for exercise in exercises:
        exercise_path = data_dir / exercise
        if not exercise_path.exists():
            validation_report[exercise] = {'status': 'missing', 'count': 0}
            continue
        
        # 이미지 파일 카운트
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(exercise_path.glob(ext)))
        
        # 이미지 유효성 검사
        valid_images = 0
        invalid_images = []
        
        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file))
                if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                    valid_images += 1
                else:
                    invalid_images.append(str(img_file))
            except:
                invalid_images.append(str(img_file))
        
        validation_report[exercise] = {
            'status': 'ok',
            'total_files': len(image_files),
            'valid_images': valid_images,
            'invalid_images': invalid_images
        }
    
    return validation_report

def generate_analysis_report(results_path: str, output_path: str = None):
    """분석 결과 리포트 생성"""
    results_dir = Path(results_path)
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_path}")
        return
    
    # 전체 통계 로드
    summary_file = results_dir / "processing_summary.json"
    if not summary_file.exists():
        print("Processing summary not found")
        return
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    # 리포트 생성
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_summary': summary,
        'detailed_analysis': {}
    }
    
    # 운동별 상세 분석
    for exercise in summary.keys():
        log_file = results_dir / f"{exercise}_processing_log.json"
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # 각도 통계 계산
            angle_stats = {}
            for entry in log_data:
                if 'good' in entry['classification']:
                    for joint, angle in entry.get('angles', {}).items():
                        if joint not in angle_stats:
                            angle_stats[joint] = []
                        angle_stats[joint].append(angle)
            
            # 평균 각도 계산
            avg_angles = {}
            for joint, angles in angle_stats.items():
                if angles:
                    avg_angles[joint] = {
                        'mean': np.mean(angles),
                        'std': np.std(angles),
                        'min': np.min(angles),
                        'max': np.max(angles)
                    }
            
            report['detailed_analysis'][exercise] = {
                'processed_count': len(log_data),
                'average_angles': avg_angles,
                'common_violations': get_common_violations(log_data)
            }
    
    # 리포트 저장
    if output_path is None:
        output_path = results_dir / "analysis_report.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis report saved to: {output_path}")
    
    # 시각화 생성
    create_visualization(summary, results_dir / "summary_chart.png")

def get_common_violations(log_data: List[Dict]) -> Dict:
    """일반적인 위반 사항 분석"""
    violation_counts = {}
    
    for entry in log_data:
        for violation in entry.get('violations', []):
            joint = violation['joint']
            if joint not in violation_counts:
                violation_counts[joint] = 0
            violation_counts[joint] += 1
    
    # 빈도순 정렬
    sorted_violations = sorted(violation_counts.items(), 
                             key=lambda x: x[1], reverse=True)
    
    return dict(sorted_violations[:5])  # 상위 5개

def create_visualization(summary: Dict, output_path: str):
    """분석 결과 시각화"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 데이터 준비
        exercises = list(summary.keys())
        good_counts = [summary[ex]['good'] for ex in exercises]
        bad_counts = [summary[ex]['bad'] for ex in exercises]
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 막대 그래프
        x = np.arange(len(exercises))
        width = 0.35
        
        ax1.bar(x - width/2, good_counts, width, label='Good', color='green', alpha=0.7)
        ax1.bar(x + width/2, bad_counts, width, label='Bad', color='red', alpha=0.7)
        
        ax1.set_xlabel('Exercise')
        ax1.set_ylabel('Count')
        ax1.set_title('Good vs Bad Poses by Exercise')
        ax1.set_xticks(x)
        ax1.set_xticklabels([ex.replace('_', ' ').title() for ex in exercises], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 비율 파이 차트
        total_good = sum(good_counts)
        total_bad = sum(bad_counts)
        
        ax2.pie([total_good, total_bad], labels=['Good', 'Bad'], 
                colors=['green', 'red'], autopct='%1.1f%%', alpha=0.7)
        ax2.set_title('Overall Pose Quality Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")

def setup_logging(log_dir: str = "logs"):
    """로깅 설정"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 로그 파일명에 타임스탬프 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"pose_analysis_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def check_camera_availability():
    """카메라 사용 가능성 확인"""
    available_cameras = []
    
    for i in range(5):  # 0~4번 카메라 확인
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
            cap.release()
    
    return available_cameras

def resize_image(image: np.ndarray, target_width: int = 640) -> np.ndarray:
    """이미지 크기 조정 (비율 유지)"""
    height, width = image.shape[:2]
    ratio = target_width / width
    target_height = int(height * ratio)
    
    return cv2.resize(image, (target_width, target_height))

def draw_angle_info(image: np.ndarray, landmarks, angle_info: Dict) -> np.ndarray:
    """이미지에 각도 정보 시각화"""
    height, width = image.shape[:2]
    
    # 각도 정보를 이미지에 표시
    for joint_name, angle in angle_info.items():
        # 관절 위치에 각도 표시
        if 'left' in joint_name and 'elbow' in joint_name:
            # 왼쪽 팔꿈치 위치
            point = landmarks[13]  # 왼쪽 팔꿈치 랜드마크
            x, y = int(point.x * width), int(point.y * height)
            cv2.putText(image, f"{angle:.1f}°", (x-20, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        elif 'right' in joint_name and 'elbow' in joint_name:
            # 오른쪽 팔꿈치 위치
            point = landmarks[14]
            x, y = int(point.x * width), int(point.y * height)
            cv2.putText(image, f"{angle:.1f}°", (x+10, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        elif 'knee' in joint_name:
            # 무릎 위치
            point_idx = 25 if 'left' in joint_name else 26
            point = landmarks[point_idx]
            x, y = int(point.x * width), int(point.y * height)
            cv2.putText(image, f"{angle:.1f}°", (x-20, y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return image

def calculate_pose_similarity(angles1: Dict, angles2: Dict) -> float:
    """두 자세 간의 유사도 계산 (0~1)"""
    if not angles1 or not angles2:
        return 0.0
    
    common_joints = set(angles1.keys()) & set(angles2.keys())
    if not common_joints:
        return 0.0
    
    differences = []
    for joint in common_joints:
        diff = abs(angles1[joint] - angles2[joint])
        # 각도 차이를 0~1 범위로 정규화 (180도 차이를 최대로)
        normalized_diff = diff / 180.0
        differences.append(1.0 - normalized_diff)
    
    return np.mean(differences)

def filter_poses_by_confidence(poses: List[Dict], min_confidence: float = 0.7) -> List[Dict]:
    """신뢰도 기준으로 자세 필터링"""
    filtered_poses = []
    
    for pose in poses:
        if pose.get('confidence', 0) >= min_confidence:
            filtered_poses.append(pose)
    
    return filtered_poses

def export_results_to_csv(results: Dict, output_path: str):
    """결과를 CSV 파일로 내보내기"""
    try:
        import pandas as pd
        
        # 데이터 변환
        rows = []
        for exercise, stats in results.items():
            rows.append({
                'Exercise': exercise,
                'Good_Count': stats.get('good', 0),
                'Bad_Count': stats.get('bad', 0),
                'Failed_Count': stats.get('failed', 0),
                'Total_Count': stats.get('good', 0) + stats.get('bad', 0) + stats.get('failed', 0),
                'Good_Ratio': stats.get('good', 0) / max(1, stats.get('good', 0) + stats.get('bad', 0))
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Results exported to CSV: {output_path}")
        
    except ImportError:
        print("Pandas not available. Cannot export to CSV.")

def load_exercise_config(config_path: str) -> Dict:
    """운동 설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in config file: {config_path}")
        return {}

def save_exercise_config(config: Dict, config_path: str):
    """운동 설정 파일 저장"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Config saved to: {config_path}")
    except Exception as e:
        print(f"Error saving config: {e}")

def create_demo_data():
    """데모용 데이터 생성 (테스트용)"""
    demo_results = {
        'squat': {'good': 150, 'bad': 50, 'failed': 10},
        'push_up': {'good': 120, 'bad': 80, 'failed': 15},
        'bench_press': {'good': 100, 'bad': 60, 'failed': 8},
        'deadlift': {'good': 140, 'bad': 45, 'failed': 12},
        'pull_up': {'good': 90, 'bad': 70, 'failed': 20}
    }
    
    return demo_results

# 메인 실행 부분
if __name__ == "__main__":
    # 유틸리티 함수 테스트
    print("Testing utility functions...")
    
    # 디렉토리 구조 생성 테스트
    create_directory_structure("./test_project")
    
    # 카메라 확인
    cameras = check_camera_availability()
    print(f"Available cameras: {cameras}")
    
    # 데모 데이터로 시각화 테스트
    demo_data = create_demo_data()
    create_visualization(demo_data, "demo_chart.png")
    
    print("Utility functions test completed!")