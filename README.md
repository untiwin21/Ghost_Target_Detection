# 레이더 고스트 타겟 탐지 시스템 (Modified RaGNNarok)

## 🎯 프로젝트 개요

이 시스템은 **LiDAR를 정답지로 활용하여 레이더의 고스트 타겟을 자동 탐지**하는 딥러닝 시스템입니다.

### 핵심 아이디어
```
LiDAR (실제 물체만 탐지) + Radar (실제 + 고스트 탐지)
                ↓
    SNR + 거리 조합 자동 라벨링
                ↓
    GNN 모델로 고스트 타겟 예측
```

### 중요한 설계 원칙
- **입력 데이터**: 레이더 데이터의 6차원 특징 벡터만 사용
- **LiDAR 역할**: 모델 입력이 아닌 Ground Truth 라벨 생성 전용
- **실제 운용**: 레이더 데이터만으로 고스트 타겟 판단 가능

## 📊 데이터 구조 및 모델 입출력

### 원본 데이터 (Raw Data)
- **RadarMap_v2.txt**: `시간 x y velocity SNR` (레이더 센서 데이터)
- **LiDARMap_v2.txt**: `시간 x y intensity` (Ground Truth 생성용)

### 모델 입력 (Model Input) ⚠️ 중요
- **레이더 데이터만 사용**: x, y 좌표는 **레이더 센서의 탐지 좌표**
- **특징 벡터 (6차원)**: 
  1. `x` - 레이더 X 좌표 (m)
  2. `y` - 레이더 Y 좌표 (m)  
  3. `range` - 거리 (m)
  4. `azimuth` - 방위각 (rad)
  5. `velocity` - 속도 (사용 안함, 0으로 설정)
  6. `SNR` - 신호 대 잡음비 (dB)
- **LiDAR 역할**: 모델 입력이 아닌 Ground Truth 라벨 생성 전용

### 모델 출력 (Model Output)
- **각 레이더 포인트별 예측**: 실제 타겟 확률 (0~1)
- **최종 분류**: 확률 > 0.5 → 실제 타겟, 확률 ≤ 0.5 → 고스트 타겟

### 시간별 프레임 구조
- 각 파일은 시간 순서로 기록된 연속 데이터
- 동일한 시간대의 포인트들이 하나의 **프레임**을 구성
- 예: 0.19초에 31개 레이더 포인트 + 152개 LiDAR 포인트 = Frame 0

## 🎯 라벨링 시스템 (SNR + 거리 조합)

### Ground Truth 생성 규칙
```python
for each 레이더_포인트:
    거리 = min(||레이더_포인트 - LiDAR_포인트|| for LiDAR_포인트 in 프레임)
    SNR = 레이더_포인트.SNR
    
    if 거리 ≤ 0.5m AND SNR ≥ 20.0dB:
        라벨 = 1  # 실제 타겟
    else:
        라벨 = 0  # 고스트 타겟
```

### 라벨링 조건 설명
- **거리 조건**: 레이더 포인트가 가장 가까운 LiDAR 포인트에서 0.5m 이내
- **SNR 조건**: 신호 대 잡음비가 20dB 이상
- **조합 조건**: 두 조건을 **모두** 만족해야 실제 타겟

## 🏗️ 딥러닝 모델 아키텍처

### GNN 모델 구조 (HybridGhostGNN)
```
입력 (6차원) → GraphSAGE → BatchNorm → ReLU → Dropout (0.1)
             → GraphSAGE → BatchNorm → ReLU → Dropout (0.1)
             → GraphSAGE → Sigmoid → 출력 (실제 타겟 확률)
```

### 모델 파라미터
- **모델 타입**: HybridGhostGNN (GraphSAGE 기반)
- **입력 차원**: 6차원 (레이더 특징 벡터)
- **은닉층 차원**: 128
- **레이어 수**: 3층
- **Dropout**: 0.1 (10%)
- **출력**: 1차원 (실제 타겟 확률 0~1)

### 학습 파라미터
- **Epochs**: 50
- **Batch Size**: 16
- **Learning Rate**: 0.005
- **Optimizer**: Adam
- **Loss Function**: BCELoss (Binary Cross Entropy)
- **Scheduler**: ReduceLROnPlateau (patience=10, factor=0.5)
- **Train/Validation Split**: 80%/20%

### 그래프 구성
- **노드**: 각 레이더 포인트
- **엣지**: k-NN (k=8) 기반 공간적 연결
- **목적**: 주변 포인트들의 관계를 학습하여 고스트 탐지

## 🚀 사용법

### 1. 환경 설정 (GPU 권장)
```bash
# CUDA 지원 PyTorch 설치 (GPU 사용)
pip install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 나머지 필요한 패키지 설치
pip install --break-system-packages torch-geometric scipy matplotlib scikit-learn

# GPU 환경 확인
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 2. 모델 학습 (GPU 자동 감지)
```bash
python3 ghost_detector.py
```

### 3. 추론 실행
```bash
python3 inference.py
```

### 4. 결과 시각화
```bash
# 개별 프레임 시각화 (4패널 분석)
python3 visualize_detection.py

# 모든 시간대 통합 시각화
python3 visualize_all_timeframes.py

# 복도 모양 시각화
python3 visualize_corridor_view.py

# Real Target + LiDAR 전용 시각화 (최신)
python3 visualize_real_targets_only.py

# 복도3 데이터 시각화
python3 visualize_bokdo3.py
python3 visualize_bokdo3_zoomed.py
```

### 5. 시각화 결과 해석
#### 색상 의미
- 🔵 **파란 점**: LiDAR 포인트 (실제 물체 위치)
- 🟢 **초록 점**: 실제 타겟으로 분류된 레이더 신호
- 🔴 **빨간 점**: 고스트 타겟으로 분류된 레이더 신호
- 🌈 **색상 그라데이션**: SNR 값 또는 예측 확률

#### 4패널 구성 (visualize_detection.py)
1. **원본 데이터**: LiDAR(파란점) + Radar(SNR 색상 매핑)
2. **Ground Truth**: 실제 타겟(초록) vs 고스트 타겟(빨강)
3. **모델 예측**: 예측된 실제/고스트 분류 + 정확도
4. **예측 확률**: 신뢰도 히트맵

## 📈 성능 결과

### 최종 성능 (GPU 학습)
- **학습 정확도**: 94.42%
- **전체 테스트 정확도**: 98.1% (100개 프레임 기준)
- **데이터 분포**: Real 21.2%, Ghost 78.8%
- **학습 환경**: NVIDIA GeForce RTX 4060 Ti, CUDA 12.1
- **학습 시간**: 약 2-3분

### 데이터 통계
- **총 프레임**: 2,090개
- **총 레이더 포인트**: 65,188개
- **SNR 범위**: 15.0 ~ 45.4 dB
- **거리 범위**: 0.005 ~ 12.066 m

### 혼동 행렬 (100개 프레임 테스트)
- **True Positive**: 661 (실제를 실제로 예측)
- **True Negative**: 2,540 (고스트를 고스트로 예측)
- **False Positive**: 33 (고스트를 실제로 잘못 예측)
- **False Negative**: 30 (실제를 고스트로 잘못 예측)

### 하드웨어 요구사항
- **권장**: NVIDIA GPU (CUDA 지원)
- **최소**: CPU (학습 시간 증가)
- **메모리**: 8GB RAM 이상
- **저장공간**: 1GB 이상

## 📊 시각화 결과

### 생성되는 파일들
- `detection_frame_*.png`: 개별 프레임 분석 (기존 방식)
- `detection_summary.png`: 전체 데이터 요약
- `all_timeframes_detection.png`: 모든 시간대 통합 시각화 (축 범위 넓음)
- `corridor_view_detection.png`: 복도 모양 시각화 (실제 복도 모양이 보임)
- `real_targets_lidar_only.png`: **Real Target + LiDAR 전용 시각화 (깔끔한 결과)**
- `ghost_detector.pth`: 학습된 모델

### 시각화 내용
#### 개별 프레임 분석 (기존)
각 시각화는 4개 패널로 구성:
1. **원본 데이터**: LiDAR(파란점) + Radar(SNR 색상 매핑)
2. **Ground Truth**: 실제 타겟(초록) vs 고스트 타겟(빨강)
3. **모델 예측**: 예측된 실제/고스트 분류 + 정확도
4. **예측 확률**: 신뢰도 히트맵 (초록=실제 확신, 빨강=고스트 확신)

#### Real Target + LiDAR 전용 시각화 (최신 - 깔끔한 결과)
딥러닝 모델이 예측한 Real Target과 LiDAR만 추출하여 시각화:
1. **LiDAR + Real Target (SNR)**: 359,931개 LiDAR + 5,016개 Real Target
2. **LiDAR + Real Target (확률)**: 평균 확률 0.892 (최소 0.501)
3. **Real Target만**: SNR 19.7~54.3 dB 범위
4. **LiDAR만**: 참조용 전체 환경 구조
- **Real Target 분포**: 복도 중앙(82.6%), 복도 중간 구간(54.9%)
- **고품질 결과**: Ghost Target 제거로 깔끔한 시각화

## 🔍 시각화 해석 가이드

### 색상 의미
- 🔵 **파란 점**: LiDAR 포인트 (실제 물체 위치)
- 🟢 **초록 점**: 실제 타겟으로 분류된 레이더 신호
- 🔴 **빨간 점**: 고스트 타겟으로 분류된 레이더 신호
- 🌈 **색상 그라데이션**: SNR 값 또는 예측 확률

### 분석 예시
- **Frame 0**: 모든 레이더 포인트가 LiDAR에서 0.768m 이상 떨어져 있어 모두 고스트
- **Frame 1-10**: 일부 레이더 포인트가 LiDAR 근처에 있어 실제 타겟으로 분류

## 🔧 파일 구조 및 역할

### 📁 핵심 코드 파일
```
modified_ragnnarok/
├── 📊 데이터 관련
│   ├── data_structures.py           # RadarPoint, LiDARPoint 데이터 클래스 정의
│   ├── RadarMap_v2.txt              # 레이더 센서 데이터 (시간 x y velocity SNR)
│   ├── LiDARMap_v2.txt              # LiDAR 센서 데이터 (시간 x y intensity)
│   ├── RadarMap_bokdo3_v6.txt       # 복도3 레이더 데이터
│   └── LiDARMap_bokdo3_v6.txt       # 복도3 LiDAR 데이터
│
├── 🧠 딥러닝 모델
│   ├── hybrid_ghost_gnn.py          # HybridGhostGNN 모델 정의 (GraphSAGE 기반)
│   ├── ghost_detector.py            # 메인 학습 시스템 (GPU 지원)
│   ├── inference.py                 # 학습된 모델로 추론 실행
│   └── ghost_detector.pth           # 학습된 모델 가중치 (94.42% 정확도)
│
├── 📈 시각화 스크립트
│   ├── visualize_detection.py       # 개별 프레임 분석 (4패널 구성)
│   ├── visualize_all_timeframes.py  # 모든 시간대 통합 시각화
│   ├── visualize_corridor_view.py   # 복도 모양 시각화
│   ├── visualize_real_targets_only.py # Real Target + LiDAR 전용 시각화
│   ├── visualize_bokdo3.py          # 복도3 데이터 시각화
│   └── visualize_bokdo3_zoomed.py   # 복도3 확대 시각화
│
├── 📋 설정 및 문서
│   ├── requirements.txt             # Python 패키지 의존성
│   ├── README.md                    # 프로젝트 문서 (이 파일)
│   ├── GITHUB_SETUP.md              # GitHub 설정 가이드
│   └── .gitignore                   # Git 무시 파일 설정
│
├── 📊 결과 이미지
│   ├── detection_frame_*.png        # 개별 프레임 분석 결과
│   ├── detection_summary.png        # 전체 데이터 요약
│   ├── all_timeframes_detection.png # 모든 시간대 통합 결과
│   ├── corridor_view_detection.png  # 복도 모양 시각화 결과
│   ├── real_targets_lidar_only.png  # Real Target + LiDAR 전용 결과
│   ├── bokdo3_detection_results.png # 복도3 탐지 결과
│   └── bokdo3_zoomed_results.png    # 복도3 확대 결과
│
├── 📚 참고 자료
│   ├── RaGNNarok A Light-Weight Graph Neural Network for Enhancing Radar Point Clouds in Unmanned Ground Vehicles.pdf
│   └── Ghost Target Detection in 3D Radar Data using.pdf
│
└── 🔧 기타
    ├── venv/                        # Python 가상환경
    └── __pycache__/                 # Python 캐시 파일
```

### 📝 주요 파일 역할 설명

#### 🧠 딥러닝 모델 파일
- **`hybrid_ghost_gnn.py`**: GraphSAGE 기반 GNN 모델 정의
  - HybridGhostGNN 클래스 (3층 GraphSAGE + BatchNorm + Dropout)
  - 6차원 특징 벡터 추출 함수
  - k-NN 기반 그래프 엣지 생성
  
- **`ghost_detector.py`**: 메인 학습 시스템
  - GhostDetectorDataset 클래스 (SNR + 거리 조합 라벨링)
  - GPU 자동 감지 및 학습
  - 모델 저장 및 성능 평가

- **`inference.py`**: 학습된 모델로 추론
  - 새로운 데이터에 대한 고스트 타겟 예측
  - 실시간 추론 가능

#### 📊 데이터 처리 파일
- **`data_structures.py`**: 기본 데이터 구조
  - RadarPoint, LiDARPoint 데이터클래스
  - RadarFrame, LiDARFrame 타입 정의

#### 📈 시각화 파일
- **`visualize_detection.py`**: 개별 프레임 4패널 분석
- **`visualize_all_timeframes.py`**: 전체 시간대 통합 뷰
- **`visualize_corridor_view.py`**: 실제 복도 모양 시각화
- **`visualize_real_targets_only.py`**: 깔끔한 결과 시각화 (최신)

## 🎯 기술적 특징

### 혁신점
1. **멀티 센서 융합**: LiDAR + Radar 데이터 활용
2. **지능형 라벨링**: 거리 + 신호 세기 조합 판단
3. **공간 관계 학습**: GraphSAGE 기반 GNN으로 주변 포인트 관계 학습
4. **실시간 처리**: GPU 가속으로 빠른 학습/추론
5. **레이더 전용 추론**: 실제 운용 시 레이더 데이터만으로 판단

### 장점
- ✅ **완전 자동화**: 수동 라벨링 불필요
- ✅ **높은 정확도**: 98% 이상의 안정적 성능
- ✅ **실용적**: 실제 센서 데이터로 검증
- ✅ **확장 가능**: 다른 환경/센서에 적용 가능
- ✅ **실시간 가능**: 레이더 데이터만으로 즉시 판단

### 한계점
- ⚠️ **LiDAR 의존성**: Ground Truth 생성을 위해 LiDAR 필수 (학습 시에만)
- ⚠️ **임계값 민감성**: 환경에 따른 임계값 조정 필요
- ⚠️ **정적 환경 특화**: 현재 데이터는 비교적 정적인 환경

## 🤝 활용 분야

### 자율주행
- 레이더 고스트 타겟 제거로 안전성 향상
- 실시간 물체 탐지 정확도 개선

### 로봇 내비게이션
- 실내/실외 환경에서 정확한 장애물 인식
- 고스트 타겟으로 인한 오작동 방지

### 보안 시스템
- 침입 탐지 시 false alarm 감소
- 실제 위협과 고스트 신호 구분

## 📚 참고 문헌
- RaGNNarok: A Lightweight RADAR Graph Neural Network for Real-Time Indoor Autonomous Robot Radar Ghost Detection
- Ghost Target Detection in Automotive Radar Systems using PointNet

## 🔄 업데이트 로그
- **v1.0**: 초기 버전 - SNR + 거리 조합 라벨링 시스템
- **v1.1**: GPU 지원 추가 및 성능 최적화
- **v1.2**: 다양한 시각화 옵션 추가
- **v1.3**: 추론 전용 모듈 추가 및 문서 개선

---
**Modified RaGNNarok** - SNR + 거리 조합 기반 레이더 고스트 타겟 자동 탐지 시스템  
🚀 **실제 운용**: 레이더 데이터만으로 고스트 타겟 실시간 탐지 가능
