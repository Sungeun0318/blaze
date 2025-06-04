# Exercise Pose Analysis System

BlazePose를 활용한 운동 자세 분석 및 분류 시스템입니다. 5가지 운동(벤치프레스, 데드리프트, 풀업, 푸시업, 스쿼트)에 대해 자동으로 good/bad 자세를 분류하고 실시간 피드백을 제공합니다.

## 🚀 주요 기능

- **자동 자세 분류**: MediaPipe BlazePose를 사용한 정확한 관절 각도 분석
- **5가지 운동 지원**: 벤치프레스, 데드리프트, 풀업, 푸시업, 스쿼트
- **배치 처리**: 500장씩 이미지를 자동으로 good/bad로 분류
- **실시간 분석**: 웹캠을 통한 실시간 자세 분석 및 피드백
- **비디오 분석**: 비디오 파일 분석 및 주석 추가
- **후처리**: 히스테리시스 및 EMA를 통한 안정적인 결과
- **상세 리포트**: 분석 결과 통계 및 시각화

## 📁 프로젝트 구조

```
BLEEF/
├── data/
│   └── images/
│       ├── bench_press_exercise/
│       ├── deadlift_exercise/
│       ├── pull_up_exercise/
│       ├── push_up_exercise/
│       └── squat_exercise/
├── processed_data/
│   ├── bench_press/
│   │   ├── good/
│   │   └── bad/
│   ├── deadlift/
│   │   ├── good/
│   │   └── bad/
│   ├── pull_up/
│   │   ├── good/
│   │   └── bad/
│   ├── push_up/
│   │   ├── good/
│   │   └── bad/
│   └── squat/
│       ├── good/
│       └── bad/
├── scripts/
│   ├── main.py
│   ├── pose_analysis_system.py
│   ├── realtime_pose_analyzer.py
│   ├── config.py
│   └── utils.py
├── logs/
├── results/
└── README.md
```

## 🛠️ 설치 및 설정

### 1. 의존성 설치

```bash
# 자동 설치
python install_requirements.py

# 또는 수동 설치
pip install opencv-python>=4.8.0
pip install mediapipe>=0.10.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install pandas>=2.0.0
```

### 2. 시스템 초기 설정

```bash
python main.py --mode setup
```

### 3. 데이터 준비

각 운동별로 이미지를 해당 디렉토리에 배치:
- `data/images/bench_press_exercise/` - 벤치프레스 이미지들
- `data/images/deadlift_exercise/` - 데드리프트 이미지들  
- `data/images/pull_up_exercise/` - 풀업 이미지들
- `data/images/push_up_exercise/` - 푸시업 이미지들
- `data/images/squat_exercise/` - 스쿼트 이미지들

## 🎯 사용법

### 1. 데이터셋 검증

```bash
python main.py --mode validate --verbose
```

### 2. 배치 처리 (이미지 분류)

```bash
# 모든 운동 처리 (각 500장 제한)
python main.py --mode batch --exercise all --limit 500

# 특정 운동만 처리
python main.py --mode batch --exercise squat --limit 500

# 지원되는 운동: bench_press, deadlift, pull_up, push_up, squat
```

### 3. 실시간 분석

```bash
# 기본 카메라로 스쿼트 분석
python main.py --mode realtime --exercise squat

# 특정 카메라 사용
python main.py --mode realtime --exercise squat --camera 1
```

**실시간 분석 키 명령어:**
- `q`: 종료
- `r`: 카운터 리셋
- `s`: 스크린샷 저장

### 4. 비디오 분석

```bash
# 비디오 파일 분석
python main.py --mode video --exercise push_up --input video.mp4 --output analyzed_video.mp4
```

### 5. 결과 리포트 생성

```bash
python main.py --mode report
```

## ⚙️ 각도 기준 설정

각 운동별 관절 각도 허용 범위:

### 스쿼트 (Squat)
- **무릎**: 70°-120° (중요도: 높음)
- **엉덩이**: 70°-120° (중요도: 중간)
- **등**: 170°-180° (중요도: 중간)

### 푸시업 (Push-up)
- **팔꿈치**: 80°-120° (중요도: 높음)
- **엉덩이**: 160°-180° (중요도: 매우 높음)
- **무릎**: 170°-180° (중요도: 중간)

### 벤치프레스 (Bench Press)
- **팔꿈치**: 70°-120° (중요도: 높음)
- **어깨**: 60°-100° (중요도: 중간)

### 데드리프트 (Deadlift)
- **무릎**: 160°-180° (중요도: 높음)
- **엉덩이**: 160°-180° (중요도: 중간)
- **등**: 160°-180° (중요도: 매우 높음)

### 풀업 (Pull-up)
- **팔꿈치**: 30°-90° (중요도: 높음)
- **어깨**: 120°-180° (중요도: 중간)

## 🔧 설정 커스터마이징

`config.json` 파일을 수정하여 설정 변경:

```json
{
  "POST_PROCESSING": {
    "hysteresis_threshold": 0.3,
    "ema_alpha": 0.2,
    "window_size": 10,
    "feedback_interval": 2.0
  },
  "EXERCISE_THRESHOLDS": {
    "squat": {
      "left_knee": {
        "points": [23, 25, 27],
        "range": [70, 120],
        "weight": 1.5
      }
    }
  }
}
```

## 📊 출력 결과

### 배치 처리 결과
- `processed_data/` 디렉토리에 운동별 good/bad 분류 이미지
- `*_processing_log.json`: 상세 분석 로그
- `processing_summary.json`: 전체 처리 요약

### 실시간/비디오 분석 결과
- 실시간 시각적 피드백
- 각도 정보 및 위반 사항 표시
- 신뢰도 점수
- 통계 정보 (good/bad 비율)

### 리포트
- `analysis_report.json`: 종합 분석 리포트
- `summary_chart.png`: 시각화 차트
- 운동별 평균 각도 통계
- 일반적인 위반 사항 분석

## 🚨 주의사항

1. **조명**: 충분한 조명에서 촬영된 이미지 사용 권장
2. **자세**: 전신이 잘 보이는 각도에서 촬영
3. **배경**: 복잡하지 않은 배경 권장
4. **의복**: 관절이 잘 보이는 의복 착용

## 🔍 트러블슈팅

### 카메라가 인식되지 않을 때
```bash
# 다른 카메라 ID 시도
python main.py --mode realtime --exercise squat --camera 1
```

### 포즈가 감지되지 않을 때
- 전신이 화면에 잘 보이는지 확인
- 조명이 충분한지 확인
- `min_detection_confidence` 값을 낮춰서 재시도

### 메모리 부족 시
- `--limit` 옵션으로 처리할 이미지 수 제한
- 이미지 해상도 줄이기

## 📈 성능 최적화

1. **실시간 분석**: `model_complexity=1` 사용 (기본값)
2. **배치 처리**: `model_complexity=2` 사용 (정확도 우선)
3. **후처리**: EMA와 히스테리시스로 노이즈 제거

## 🤝 기여하기

1. 새로운 운동 추가
2. 각도 기준 개선
3. 후처리 알고리즘 향상
4. UI/UX 개선

## 📝 라이센스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

## 🎮 실행 예제

### 빠른 시작

```bash
# 1. 시스템 설정
python main.py --mode setup

# 2. 데이터 검증
python main.py --mode validate

# 3. 스쿼트 100장 처리
python main.py --mode batch --exercise squat --limit 100

# 4. 실시간 스쿼트 분석
python main.py --mode realtime --exercise squat

# 5. 결과 확인
python main.py --mode report
```

### 고급 사용법

```bash
# 커스텀 설정으로 실행
python main.py --mode batch --exercise all --config custom_config.json --verbose

# 특정 데이터 경로 지정
python main.py --mode batch --exercise squat --data-path ./my_data --output-path ./my_results
```

이제 시스템이 준비되었습니다! 🎉