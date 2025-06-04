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
        "data/training_images/bench_press_exercise",
        "data/training_images/deadlift_exercise", 
        "data/training_images/pull_up_exercise",
        "data/training_images/push_up_exercise",
        "data/training_images/squat_exercise",
        "data/processed_data",
        "models",
        "logs",
        "outputs/screenshots",
        "outputs/reports"
    ]
    
    for dir_path in directories:
        (base / dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Directory structure created at: {base}")

def validate_image_dataset(data_path: str) -> Dict:
    """이미지 데이터셋 검증"""
    data_dir = Path(data_path)
    validation_report = {}
    
    exercises = ['squat_exercise', 'push_up_exercise', 'bench_press_exercise', 
                'deadlift_exercise', 'pull_up_exercise']
    
    total_valid = 0
    total_invalid = 0
    
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
        
        total_valid += valid_images
        total_invalid += len(invalid_images)
        
        validation_report[exercise] = {
            'status': 'ok',
            'total_files': len(image_files),
            'valid_images': valid_images,
            'invalid_images': invalid_images[:5]  # 최대 5개만 표시
        }
    
    validation_report['summary'] = {
        'total_valid': total_valid,
        'total_invalid': total_invalid,
        'total_exercises': len([ex for ex in validation_report.keys() 
                              if ex != 'summary' and validation_report[ex]['status'] == 'ok'])
    }
    
    return validation_report

def generate_analysis_report(results_path: str, output_path: str = None):
    """분석 결과 리포트 생성"""
    results_dir = Path(results_path)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_path}")
        return
    
    # 전체 통계 로드
    summary_file = results_dir / "processing_summary.json"
    if not summary_file.exists():
        print("❌ Processing summary not found")
        return
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    except Exception as e:
        print(f"❌ Error reading summary file: {e}")
        return
    
    # 리포트 생성
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_summary': summary,
        'detailed_analysis': {},
        'statistics': {}
    }
    
    # 전체 통계 계산
    total_good = sum(data.get('good', 0) for data in summary.values())
    total_bad = sum(data.get('bad', 0) for data in summary.values())
    total_failed = sum(data.get('failed', 0) for data in summary.values())
    total_processed = total_good + total_bad + total_failed
    
    report['statistics'] = {
        'total_processed': total_processed,
        'total_good': total_good,
        'total_bad': total_bad,
        'total_failed': total_failed,
        'success_rate': total_good / max(total_good + total_bad, 1),
        'processing_rate': (total_good + total_bad) / max(total_processed, 1)
    }
    
    # 운동별 상세 분석
    for exercise in summary.keys():
        log_file = results_dir / f"{exercise}_processing_log.json"
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                # 각도 통계 계산
                angle_stats = {}
                good_angles = {}
                
                for entry in log_data:
                    if entry.get('classification') == 'good':
                        for joint, angle in entry.get('angles', {}).items():
                            if joint not in good_angles:
                                good_angles[joint] = []
                            good_angles[joint].append(angle)
                
                # 평균 각도 계산
                for joint, angles in good_angles.items():
                    if angles:
                        angle_stats[joint] = {
                            'mean': float(np.mean(angles)),
                            'std': float(np.std(angles)),
                            'min': float(np.min(angles)),
                            'max': float(np.max(angles)),
                            'count': len(angles)
                        }
                
                report['detailed_analysis'][exercise] = {
                    'processed_count': len(log_data),
                    'average_angles': angle_stats,
                    'common_violations': get_common_violations(log_data)
                }
                
            except Exception as e:
                print(f"⚠️ Error processing log for {exercise}: {e}")
    
    # 리포트 저장
    if output_path is None:
        output_path = results_dir / "analysis_report.json"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Analysis report saved to: {output_path}")
        
        # 요약 출력
        stats = report['statistics']
        print(f"📊 분석 요약:")
        print(f"  총 처리: {stats['total_processed']}개")
        print(f"  성공: {stats['total_good']}개 ({stats['success_rate']:.1%})")
        print(f"  개선 필요: {stats['total_bad']}개")
        print(f"  실패: {stats['total_failed']}개")
        
    except Exception as e:
        print(f"❌ Error saving report: {e}")
    
    # 시각화 생성
    create_visualization(summary, results_dir / "summary_chart.png")

def get_common_violations(log_data: List[Dict]) -> Dict:
    """일반적인 위반 사항 분석"""
    violation_counts = {}
    
    for entry in log_data:
        for violation in entry.get('violations', []):
            joint = violation.get('joint', 'unknown')
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
        import matplotlib
        matplotlib.use('Agg')  # GUI 없는 환경에서 사용
        
        # 데이터 준비
        exercises = list(summary.keys())
        good_counts = [summary[ex].get('good', 0) for ex in exercises]
        bad_counts = [summary[ex].get('bad', 0) for ex in exercises]
        
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
        
        if total_good + total_bad > 0:
            ax2.pie([total_good, total_bad], labels=['Good', 'Bad'], 
                    colors=['green', 'red'], autopct='%1.1f%%', alpha=0.7)
            ax2.set_title('Overall Pose Quality Distribution')
        else:
            ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('No Data Available')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to: {output_path}")
        
    except ImportError:
        print("⚠️ Matplotlib not available. Skipping visualization.")
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

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

def check_system_requirements():
    """시스템 요구사항 확인"""
    requirements = {
        'opencv': False,
        'mediapipe': False,
        'numpy': False,
        'sklearn': False,
        'joblib': False
    }
    
    try:
        import cv2
        requirements['opencv'] = True
    except ImportError:
        pass
    
    try:
        import mediapipe
        requirements['mediapipe'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        requirements['numpy'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        requirements['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import joblib
        requirements['joblib'] = True
    except ImportError:
        pass
    
    return requirements

def resize_image(image: np.ndarray, target_width: int = 640) -> np.ndarray:
    """이미지 크기 조정 (비율 유지)"""
    height, width = image.shape[:2]
    ratio = target_width / width
    target_height = int(height * ratio)
    
    return cv2.resize(image, (target_width, target_height))

def export_results_to_csv(results: Dict, output_path: str):
    """결과를 CSV 파일로 내보내기"""
    try:
        import pandas as pd
        
        # 데이터 변환
        rows = []
        for exercise, stats in results.items():
            if isinstance(stats, dict):
                rows.append({
                    'Exercise': exercise,
                    'Good_Count': stats.get('good', 0),
                    'Bad_Count': stats.get('bad', 0),
                    'Failed_Count': stats.get('failed', 0),
                    'Total_Count': stats.get('good', 0) + stats.get('bad', 0) + stats.get('failed', 0),
                    'Good_Ratio': stats.get('good', 0) / max(1, stats.get('good', 0) + stats.get('bad', 0))
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"📊 Results exported to CSV: {output_path}")
        else:
            print("⚠️ No data to export")
        
    except ImportError:
        print("⚠️ Pandas not available. Cannot export to CSV.")
    except Exception as e:
        print(f"❌ Error exporting to CSV: {e}")

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
    print("🔧 Testing utility functions...")
    
    # 시스템 요구사항 확인
    requirements = check_system_requirements()
    print("📋 System requirements:")
    for package, available in requirements.items():
        status = "✅" if available else "❌"
        print(f"  {status} {package}")
    
    # 디렉토리 구조 생성 테스트
    create_directory_structure("./test_project")
    
    # 카메라 확인
    cameras = check_camera_availability()
    print(f"📹 Available cameras: {cameras}")
    
    # 데모 데이터로 시각화 테스트
    demo_data = create_demo_data()
    create_visualization(demo_data, "demo_chart.png")
    
    print("✅ Utility functions test completed!")