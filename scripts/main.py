#!/usr/bin/env python3
"""
🏋️ BLAZE - 운동 자세 분석 시스템 (풀업→런지 교체 완화 버전)
스쿼트, 푸쉬업, 데드리프트, 벤치프레스, 런지 전용 시스템
"""

import sys
import os
import argparse
from pathlib import Path
import logging
from datetime import datetime
import subprocess
import json

def setup_logging():
    """로깅 시스템 설정"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"blaze_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('Blaze')

class Enhanced5ExerciseBlazeManager:
    """향상된 풀업→런지 교체 5종목 BLAZE 시스템 관리자"""
    
    def __init__(self):
        self.logger = setup_logging()
        # 🚀 풀업 → 런지로 변경
        self.available_exercises = ['squat', 'push_up', 'deadlift', 'bench_press', 'lunge']
        self.exercise_emojis = {
            'squat': '🏋️‍♀️',
            'push_up': '💪', 
            'deadlift': '🏋️‍♂️',
            'bench_press': '🔥',
            'lunge': '🚀'  # 풀업 → 런지로 변경
        }
        self.ensure_directories()
    
    def ensure_directories(self):
        """필요한 디렉토리 생성 (풀업→런지 교체 5종목)"""
        directories = [
            "data/training_images/squat_exercise",
            "data/training_images/push_up_exercise", 
            "data/training_images/deadlift_exercise",
            "data/training_images/bench_press_exercise",
            "data/training_images/lunge_exercise",  # pull_up_exercise → lunge_exercise로 변경
            "data/processed_data",
            "models",
            "outputs/screenshots",
            "outputs/reports",
            "logs"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ 풀업→런지 교체 5종목 디렉토리 구조 생성 완료")
    
    def check_dependencies(self):
        """의존성 패키지 확인"""
        required_packages = {
            'cv2': 'opencv-python',
            'mediapipe': 'mediapipe', 
            'numpy': 'numpy',
            'sklearn': 'scikit-learn',
            'joblib': 'joblib'
        }
        
        missing_packages = []
        
        for module, package in required_packages.items():
            try:
                __import__(module)
                self.logger.info(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                self.logger.error(f"❌ {package} 누락")
        
        if missing_packages:
            self.logger.error(f"❌ 설치 필요: {missing_packages}")
            self.logger.info("설치 명령어:")
            self.logger.info(f"pip install {' '.join(missing_packages)}")
            return False
        
        self.logger.info("✅ 모든 의존성 패키지 확인 완료")
        return True
    
    def check_training_data(self):
        """풀업→런지 교체 5종목 훈련 데이터 확인"""
        training_dir = Path("data/training_images")
        # 🚀 풀업 → 런지로 변경된 운동 디렉토리
        exercise_dirs = {
            'squat': 'squat_exercise',
            'push_up': 'push_up_exercise', 
            'deadlift': 'deadlift_exercise',
            'bench_press': 'bench_press_exercise',
            'lunge': 'lunge_exercise'  # pull_up_exercise → lunge_exercise로 변경
        }
        
        total_images = 0
        exercise_counts = {}
        available_exercises = []
        
        self.logger.info("📊 풀업→런지 교체 5종목 훈련 데이터 현황:")
        
        for exercise, dir_name in exercise_dirs.items():
            exercise_path = training_dir / dir_name
            emoji = self.exercise_emojis.get(exercise, '🏋️')
            
            if not exercise_path.exists():
                exercise_counts[exercise] = 0
                status = "폴더 없음 (새로 추가 필요)" if exercise == 'lunge' else "폴더 없음"
                self.logger.info(f"  ❌ {emoji} {exercise}: {status}")
                continue
            
            # 이미지 파일 개수 확인
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(list(exercise_path.glob(ext)))
            
            count = len(image_files)
            exercise_counts[exercise] = count
            total_images += count
            
            if count >= 100:
                status = "(새로 추가됨)" if exercise == 'lunge' else ""
                self.logger.info(f"  ✅ {emoji} {exercise}: {count}개 {status}")
                available_exercises.append(exercise)
            elif count > 0:
                status = "(새로 추가됨, 부족)" if exercise == 'lunge' else "(부족)"
                self.logger.info(f"  ⚠️ {emoji} {exercise}: {count}개 {status}")
                available_exercises.append(exercise)
            else:
                status = "(새로 추가 필요)" if exercise == 'lunge' else ""
                self.logger.info(f"  ❌ {emoji} {exercise}: {count}개 {status}")
        
        self.logger.info(f"📸 총 이미지: {total_images}개")
        self.logger.info(f"🎯 사용 가능한 운동: {len(available_exercises)}종목")
        
        if total_images == 0:
            self.logger.error("❌ 훈련 데이터가 없습니다!")
            self.logger.info("📁 다음 폴더들에 이미지를 넣어주세요:")
            for exercise, dir_name in exercise_dirs.items():
                emoji = self.exercise_emojis.get(exercise, '🏋️')
                status = " (새로 추가됨)" if exercise == 'lunge' else ""
                self.logger.info(f"   {emoji} data/training_images/{dir_name}/{status}")
            return False
        
        if len(available_exercises) < 2:
            self.logger.warning("⚠️ 최소 2종목 이상의 데이터가 필요합니다")
            if 'lunge' not in available_exercises:
                self.logger.info("💡 런지 데이터를 data/training_images/lunge_exercise/ 폴더에 추가하세요!")
        
        return True
    
    def process_data(self):
        """풀업→런지 교체 5종목 데이터 전처리 실행"""
        try:
            self.logger.info("🔄 풀업→런지 교체 5종목 데이터 전처리 시작... (완화된 기준 적용)")
            
            # 향상된 분석 시스템 사용
            from enhanced_pose_analysis import EnhancedDatasetProcessor
            
            processor = EnhancedDatasetProcessor(".")
            
            # 🚀 풀업 → 런지로 변경된 5종목 처리
            exercises = {
                'squat': 'squat_exercise',
                'push_up': 'push_up_exercise', 
                'deadlift': 'deadlift_exercise',
                'bench_press': 'bench_press_exercise',
                'lunge': 'lunge_exercise'  # pull_up_exercise → lunge_exercise로 변경
            }
            
            total_results = {}
            processed_exercises = []
            
            for exercise, directory in exercises.items():
                self.logger.info(f"📂 {self.exercise_emojis.get(exercise, '🏋️')} {exercise} 처리 중...")
                
                # 데이터 폴더 존재 확인
                data_path = Path("data/training_images") / directory
                if data_path.exists() and any(data_path.glob("*.jpg")):
                    results = processor.process_exercise_images(exercise, directory, limit=500)
                    total_results[exercise] = results
                    processed_exercises.append(exercise)
                else:
                    status = " (새로 추가 필요)" if exercise == 'lunge' else " - 건너뜀"
                    self.logger.info(f"  ⚠️ {exercise} 데이터 없음{status}")
                    total_results[exercise] = {'good': 0, 'bad': 0, 'failed': 0}
            
            # 결과 요약 출력
            self.logger.info("✅ 풀업→런지 교체 5종목 데이터 전처리 완료! (완화된 기준)")
            self.logger.info(f"🎯 처리된 운동: {len(processed_exercises)}종목")
            
            for exercise in processed_exercises:
                results = total_results[exercise]
                total_processed = results['good'] + results['bad']
                if total_processed > 0:
                    good_rate = (results['good'] / total_processed) * 100
                    emoji = self.exercise_emojis.get(exercise, '🏋️')
                    status = " (새로 추가됨)" if exercise == 'lunge' else ""
                    self.logger.info(f"  {emoji} {exercise}: Good {results['good']}장 ({good_rate:.1f}%), Bad {results['bad']}장{status}")
            
            return len(processed_exercises) > 0
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 전처리 실패: {e}")
            return False
    
    def train_model(self):
        """풀업→런지 교체 5종목 AI 모델 훈련"""
        try:
            self.logger.info("🧠 풀업→런지 교체 5종목 AI 모델 훈련 시작...")
            
            # exercise_classifier.py가 있는지 확인
            if not Path("exercise_classifier.py").exists():
                self.logger.error("❌ exercise_classifier.py 파일이 없습니다")
                return False
            
            # 모델 훈련 실행
            result = subprocess.run([
                sys.executable, "exercise_classifier.py", 
                "--mode", "train", 
                "--data_path", "./data/training_images"
            ], capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                self.logger.info("✅ 풀업→런지 교체 5종목 모델 훈련 완료!")
                
                # 모델 파일 확인
                model_path = Path("models/exercise_classifier.pkl") 
                if model_path.exists():
                    self.logger.info(f"📁 모델 저장: {model_path}")
                    self.logger.info("🚀 변경사항: 풀업 → 런지 교체")
                    return True
                else:
                    self.logger.error("❌ 모델 파일 생성 실패")
                    return False
            else:
                self.logger.error("❌ 모델 훈련 실패")
                if result.stderr:
                    self.logger.error(f"오류: {result.stderr}")
                if result.stdout:
                    self.logger.info(f"출력: {result.stdout}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 모델 훈련 중 오류: {e}")
            return False
    
    def run_realtime_analysis(self, camera_id=0):
        """풀업→런지 교체 5종목 실시간 분석 실행"""
        model_path = Path("models/exercise_classifier.pkl")
        
        if not model_path.exists():
            self.logger.error("❌ 훈련된 모델이 없습니다!")
            self.logger.info("💡 먼저 모델을 훈련하세요: python main.py --mode train")
            self.logger.info("💡 또는 C키로 운동을 수동 선택할 수 있습니다")
            # 모델 없어도 실행 가능하도록 수정
        
        try:
            self.logger.info("🎮 풀업→런지 교체 5종목 실시간 분석 시작... (완화된 버전)")
            
            # enhanced_realtime_analyzer.py 실행
            if Path("enhanced_realtime_analyzer.py").exists():
                subprocess.run([
                    sys.executable, "enhanced_realtime_analyzer.py", 
                    "--camera", str(camera_id)
                ])
            else:
                self.logger.error("❌ enhanced_realtime_analyzer.py 파일이 없습니다")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 실시간 분석 오류: {e}")
            return False
    
    def show_status(self):
        """풀업→런지 교체 5종목 시스템 상태 표시"""
        print("\n" + "="*80)
        print("🏋️  BLAZE - 풀업→런지 교체 5종목 운동 자세 분석 시스템 상태")
        print("="*80)
        
        # 의존성 확인
        deps_ok = self.check_dependencies()
        
        # 훈련 데이터 확인
        data_ok = self.check_training_data()
        
        # 전처리된 데이터 확인
        processed_data_path = Path("data/processed_data")
        processed_exists = processed_data_path.exists()
        
        print(f"\n🎯 지원 운동 (풀업→런지 교체 5종목):")
        for exercise in self.available_exercises:
            emoji = self.exercise_emojis.get(exercise, '🏋️')
            
            # 원본 데이터 확인
            raw_data_path = Path("data/training_images") / f"{exercise}_exercise"
            raw_count = 0
            if raw_data_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    raw_count += len(list(raw_data_path.glob(ext)))
            
            # 전처리된 데이터 확인
            processed_count = 0
            if processed_exists:
                good_dir = processed_data_path / exercise / "good"
                bad_dir = processed_data_path / exercise / "bad"
                if good_dir.exists():
                    processed_count += len(list(good_dir.glob("*.jpg")))
                if bad_dir.exists():
                    processed_count += len(list(bad_dir.glob("*.jpg")))
            
            status = "✅" if raw_count > 0 else "❌"
            processed_status = "✅" if processed_count > 0 else "❌"
            
            exercise_name = exercise.replace('_', ' ').title()
            note = " (새로 추가됨)" if exercise == 'lunge' else ""
            
            print(f"  {emoji} {exercise_name}: {status} 원본 {raw_count}장 | {processed_status} 처리됨 {processed_count}장{note}")
        
        # 모델 상태 확인
        model_path = Path("models/exercise_classifier.pkl")
        model_exists = model_path.exists()
        print(f"\n🧠 AI 모델: {'✅ 훈련됨 (풀업→런지 교체)' if model_exists else '❌ 없음'}")
        
        # 파일 확인
        required_files = [
            "exercise_classifier.py",
            "enhanced_realtime_analyzer.py", 
            "enhanced_pose_analysis.py",
            "pose_analysis_system.py",
            "utils.py",
            "config.py"
        ]
        
        print(f"\n📄 필수 파일들:")
        all_files_exist = True
        for file_name in required_files:
            exists = Path(file_name).exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {file_name}")
            if not exists:
                all_files_exist = False
        
        print("="*80)
        
        # 완화된 기준 정보
        print(f"\n🎨 풀업→런지 교체 및 완화된 기준 적용:")
        print(f"  🚀 운동 교체: 풀업 → 런지")
        print(f"  📐 데드리프트 완화: 무릎(120°→100°), 힙(100°→80°), 등(140°→120°)")
        print(f"  📐 벤치프레스 완화: 팔꿈치(30-170°→20-180°), 어깨(30-150°→20-170°)")
        print(f"  📐 런지 새 기준: 앞무릎(70-130°), 뒷무릎(120-180°), 상체직립(160-180°)")
        print(f"  ⚖️ 히스테리시스 완화: 데드리프트(0.7→0.6), 벤치프레스(0.8→0.7)")
        print(f"  🏋️‍♀️ 스쿼트, 💪 푸쉬업: 기존 설정 유지 (잘 작동함)")
        
        # 다음 단계 안내
        if not deps_ok:
            print("📋 다음 단계: pip install opencv-python mediapipe numpy scikit-learn joblib")
        elif not all_files_exist:
            print("📋 다음 단계: 누락된 파일들을 추가해주세요")
        elif not data_ok:
            print("📋 다음 단계: data/training_images/ 폴더에 운동별 이미지 넣기")
            print("🚀 특히 런지 데이터를 data/training_images/lunge_exercise/ 폴더에 추가!")
        elif not processed_exists:
            print("📋 다음 단계: python main.py --mode process")
        elif not model_exists:
            print("📋 다음 단계: python main.py --mode train")
        else:
            print("📋 다음 단계: python main.py --mode realtime")
            print("💡 C키로 운동 종목 변경, H키로 도움말 확인")
        
        print(f"\n🎮 실시간 분석 특징 (풀업→런지 교체 완화 버전):")
        print(f"  🚀 5종목 자동 감지: 스쿼트, 푸쉬업, 데드리프트, 벤치프레스, 런지")
        print(f"  📷 뷰 타입 자동 감지 (측면/정면/후면)")
        print(f"  🎯 운동별 맞춤 피드백")
        print(f"  📐 완화된 판정 기준 (더 많은 Good 결과)")
        print(f"  🌈 전체 화면 색상 피드백 (초록=좋음, 빨강=교정필요)")
        print(f"  📊 실시간 통계 및 성과 추적")
        
        print(f"\n📁 배드 사진 추가 방법:")
        print(f"  • processed_data/{exercise}/bad/ 폴더에 직접 배드 사진 추가 가능")
        print(f"  • 학습할 때 Good/Bad 비율이 자동으로 고려됨")
        print(f"  • 더 많은 배드 사진 = 더 정확한 모델")
        print()
    
    def run_full_pipeline(self):
        """풀업→런지 교체 5종목 전체 파이프라인 실행"""
        self.logger.info("🚀 BLAZE 풀업→런지 교체 5종목 전체 파이프라인 시작!")
        
        # 1. 의존성 확인
        if not self.check_dependencies():
            return False
        
        # 2. 훈련 데이터 확인
        if not self.check_training_data():
            return False
        
        # 3. 데이터 전처리
        if not self.process_data():
            return False
        
        # 4. 모델 훈련
        if not self.train_model():
            self.logger.warning("⚠️ 모델 훈련 실패했지만 실시간 분석은 가능합니다")
            self.logger.info("💡 C키로 운동을 수동 선택할 수 있습니다")
        
        # 5. 실시간 분석
        self.logger.info("🎯 준비 완료! 풀업→런지 교체 5종목 실시간 분석을 시작합니다...")
        return self.run_realtime_analysis()

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='🏋️ BLAZE: 풀업→런지 교체 5종목 운동 자세 분석 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 풀업→런지 교체 지원 5종목:
  🏋️‍♀️ 스쿼트      💪 푸쉬업      🏋️‍♂️ 데드리프트
  🔥 벤치프레스    🚀 런지 (풀업 대체)

🔧 주요 변경사항:
  • 운동 교체: 풀업 → 런지
  • 데드리프트 각도 완화 (더 관대한 판정)
  • 벤치프레스 각도 완화 (더 관대한 판정)
  • 스쿼트, 푸쉬업: 기존 설정 유지

📁 데이터 구조:
  data/training_images/lunge_exercise/  ← 런지 이미지 추가
  processed_data/lunge/good/
  processed_data/lunge/bad/  ← 배드 사진도 직접 추가 가능

사용 예시:
  python main.py --mode status         # 시스템 상태 확인
  python main.py --mode process        # 데이터 전처리 (완화된 기준)
  python main.py --mode train          # AI 모델 훈련
  python main.py --mode realtime       # 실시간 분석 (완화된 판정)
  python main.py --mode full           # 전체 파이프라인

실시간 분석 키 조작:
  Q: 종료  |  R: 리셋  |  S: 스크린샷  |  C: 운동 변경  |  H: 도움말
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['status', 'process', 'train', 'realtime', 'full'],
                       help='실행 모드')
    parser.add_argument('--camera', type=int, default=0,
                       help='카메라 ID (기본값: 0)')
    
    args = parser.parse_args()
    
    # BLAZE 시스템 초기화
    blaze = Enhanced5ExerciseBlazeManager()
    
    try:
        if args.mode == 'status':
            blaze.show_status()
            
        elif args.mode == 'process':
            if not blaze.check_dependencies():
                sys.exit(1)
            if not blaze.check_training_data():
                sys.exit(1)
            success = blaze.process_data()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'train':
            if not blaze.check_dependencies():
                sys.exit(1)
            # 전처리된 데이터가 있는지 확인
            processed_path = Path("data/processed_data")
            if not processed_path.exists():
                blaze.logger.info("전처리된 데이터가 없습니다. 먼저 데이터를 처리합니다...")
                if not blaze.process_data():
                    sys.exit(1)
            success = blaze.train_model()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'realtime':
            if not blaze.check_dependencies():
                sys.exit(1)
            success = blaze.run_realtime_analysis(args.camera)
            sys.exit(0 if success else 1)
            
        elif args.mode == 'full':
            success = blaze.run_full_pipeline()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        blaze.logger.info("⏹️ 사용자에 의해 중단됨")
        sys.exit(0)
    except Exception as e:
        blaze.logger.error(f"❌ 예상치 못한 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()