#!/usr/bin/env python3
"""
🏋️ BLAZE - 운동 자세 분석 시스템
간소화된 3단계 워크플로우:
1. 수동으로 운동별 이미지 정리 (500장씩)
2. AI 모델 훈련
3. 실시간 분석

사용법:
    python main.py --help
"""

import sys
import os
import argparse
from pathlib import Path
import logging
from datetime import datetime
import subprocess

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

class BlazeManager:
    """BLAZE 시스템 관리자"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.ensure_directories()
    
    def ensure_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            "data/training_images/squat_exercise",
            "data/training_images/push_up_exercise", 
            "data/training_images/bench_press_exercise",
            "data/training_images/deadlift_exercise",
            "data/training_images/pull_up_exercise",
            "data/processed_data",
            "models",
            "outputs/screenshots",
            "outputs/reports",
            "logs"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ 디렉토리 구조 생성 완료")
    
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
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"❌ 누락된 패키지: {missing_packages}")
            self.logger.info("다음 명령어로 설치하세요:")
            self.logger.info(f"pip install {' '.join(missing_packages)}")
            return False
        
        self.logger.info("✅ 모든 의존성 패키지 확인 완료")
        return True
    
    def check_training_data(self):
        """훈련 데이터 확인"""
        training_dir = Path("data/training_images")
        exercises = ['squat_exercise', 'push_up_exercise', 'bench_press_exercise', 
                    'deadlift_exercise', 'pull_up_exercise']
        
        total_images = 0
        exercise_counts = {}
        
        for exercise in exercises:
            exercise_path = training_dir / exercise
            if not exercise_path.exists():
                exercise_counts[exercise] = 0
                continue
            
            # 이미지 파일 개수 확인
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(list(exercise_path.glob(ext)))
            
            count = len(image_files)
            exercise_counts[exercise] = count
            total_images += count
        
        # 결과 출력
        self.logger.info("📊 훈련 데이터 현황:")
        for exercise, count in exercise_counts.items():
            status = "✅" if count >= 50 else "⚠️" if count > 0 else "❌"
            self.logger.info(f"  {status} {exercise}: {count}개")
        
        self.logger.info(f"📸 총 이미지: {total_images}개")
        
        if total_images == 0:
            self.logger.error("❌ 훈련 데이터가 없습니다!")
            self.logger.info("다음 폴더들에 이미지를 넣어주세요:")
            for exercise in exercises:
                self.logger.info(f"  - data/training_images/{exercise}/")
            return False
        
        if total_images < 250:  # 운동당 평균 50장 미만
            self.logger.warning("⚠️ 훈련 데이터가 부족할 수 있습니다")
            self.logger.info("권장: 각 운동당 100장 이상")
        
        return True
    
    def train_model(self):
        """AI 모델 훈련"""
        try:
            self.logger.info("🧠 AI 모델 훈련 시작...")
            
            # exercise_classifier.py 실행
            result = subprocess.run([
                sys.executable, "exercise_classifier.py", 
                "--mode", "train", 
                "--data_path", "./data/training_images"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ 모델 훈련 완료!")
                
                # 모델 파일 확인
                model_path = Path("models/exercise_classifier.pkl") 
                if model_path.exists():
                    self.logger.info(f"📁 모델 저장 위치: {model_path}")
                    return True
                else:
                    self.logger.error("❌ 모델 파일이 생성되지 않았습니다")
                    return False
            else:
                self.logger.error("❌ 모델 훈련 실패")
                if result.stderr:
                    self.logger.error(f"오류: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 모델 훈련 중 오류: {e}")
            return False
    
    def run_realtime_analysis(self, camera_id=0):
        """실시간 분석 실행"""
        model_path = Path("models/exercise_classifier.pkl")
        
        if not model_path.exists():
            self.logger.error("❌ 훈련된 모델이 없습니다!")
            self.logger.info("먼저 모델을 훈련하세요: python main.py --mode train")
            return False
        
        try:
            self.logger.info("🎮 실시간 분석 시작...")
            
            # simplified_realtime_analyzer.py 실행
            subprocess.run([
                sys.executable, "simplified_realtime_analyzer.py", 
                "--camera", str(camera_id)
            ])
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 실시간 분석 오류: {e}")
            return False
    
    def show_status(self):
        """시스템 상태 표시"""
        print("\n" + "="*70)
        print("🏋️  BLAZE - 운동 자세 분석 시스템 상태")
        print("="*70)
        
        # 의존성 확인
        deps_ok = self.check_dependencies()
        print(f"📦 의존성 패키지: {'✅ 정상' if deps_ok else '❌ 문제'}")
        
        # 훈련 데이터 확인
        data_ok = self.check_training_data()
        print(f"📸 훈련 데이터: {'✅ 준비됨' if data_ok else '❌ 부족'}")
        
        # 모델 상태 확인
        model_path = Path("models/exercise_classifier.pkl")
        model_exists = model_path.exists()
        print(f"🧠 AI 모델: {'✅ 훈련됨' if model_exists else '❌ 없음'}")
        
        print("="*70)
        
        # 다음 단계 안내
        if not deps_ok:
            print("📋 다음 단계: pip install -r requirements.txt")
        elif not data_ok:
            print("📋 다음 단계: data/training_images/ 폴더에 운동별 이미지 넣기")
        elif not model_exists:
            print("📋 다음 단계: python main.py --mode train")
        else:
            print("📋 다음 단계: python main.py --mode realtime")
        
        print()
    
    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        self.logger.info("🚀 BLAZE 전체 파이프라인 시작!")
        
        # 1. 의존성 확인
        if not self.check_dependencies():
            return False
        
        # 2. 훈련 데이터 확인
        if not self.check_training_data():
            return False
        
        # 3. 모델 훈련
        if not self.train_model():
            return False
        
        # 4. 실시간 분석
        self.logger.info("🎯 준비 완료! 실시간 분석을 시작합니다...")
        return self.run_realtime_analysis()

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='🏋️ BLAZE: 운동 자세 분석 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py --mode status         # 시스템 상태 확인
  python main.py --mode train          # AI 모델 훈련
  python main.py --mode realtime       # 실시간 분석
  python main.py --mode full           # 전체 파이프라인
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['status', 'train', 'realtime', 'full'],
                       help='실행 모드')
    parser.add_argument('--camera', type=int, default=0,
                       help='카메라 ID (기본값: 0)')
    
    args = parser.parse_args()
    
    # BLAZE 시스템 초기화
    blaze = BlazeManager()
    
    try:
        if args.mode == 'status':
            blaze.show_status()
            
        elif args.mode == 'train':
            if not blaze.check_dependencies():
                sys.exit(1)
            if not blaze.check_training_data():
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