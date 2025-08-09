# 레이더 고스트 타겟 탐지 시스템

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

## 📊 데이터 구조

### 입력 데이터
- **RadarMap_v2.txt**: `시간 x y velocity SNR`
- **LiDARMap_v2.txt**: `시간 x y intensity`

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

## 🏗️ 시스템 아키텍처

### GNN 모델 구조
```
입력 (6차원) → GraphSAGE → BatchNorm → ReLU → Dropout
             → GraphSAGE → BatchNorm → ReLU → Dropout  
             → GraphSAGE → Sigmoid → 출력 (실제 타겟 확률)
```

### 특징 벡터 (6차원)
1. `x` - X 좌표 (m)
2. `y` - Y 좌표 (m)
3. `range` - 거리 (m)
4. `azimuth` - 방위각 (rad)
5. `velocity` - 속도 (사용 안함, 0으로 설정)
6. `SNR` - 신호 대 잡음비 (dB)

### 그래프 구성
- **노드**: 각 레이더 포인트
- **엣지**: k-NN (k=8) 기반 공간적 연결
- **목적**: 주변 포인트들의 관계를 학습하여 고스트 탐지

## 🚀 사용법

### 1. 환경 설정
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio torch-geometric scipy matplotlib
```

### 2. 모델 학습
```bash
python ghost_detector.py
```

### 3. 결과 시각화
```bash
python visualize_detection.py
```

## 📈 성능 결과

### 최종 성능
- **학습 정확도**: 94.75%
- **전체 테스트 정확도**: 98.1% (100개 프레임 기준)
- **데이터 분포**: Real 25.0%, Ghost 75.0%

### 데이터 통계
- **총 프레임**: 2,090개
- **총 레이더 포인트**: 65,188개
- **SNR 범위**: 15.0 ~ 45.4 dB
- **거리 범위**: 0.005 ~ 12.066 m

## 📊 시각화 결과

### 생성되는 파일들
- `detection_frame_*.png`: 개별 프레임 분석
- `detection_summary.png`: 전체 데이터 요약
- `ghost_detector.pth`: 학습된 모델

### 시각화 내용
각 시각화는 4개 패널로 구성:
1. **원본 데이터**: LiDAR(파란점) + Radar(SNR 색상 매핑)
2. **Ground Truth**: 실제 타겟(초록) vs 고스트 타겟(빨강)
3. **모델 예측**: 예측된 실제/고스트 분류 + 정확도
4. **예측 확률**: 신뢰도 히트맵 (초록=실제 확신, 빨강=고스트 확신)

## 🔍 시각화 해석 가이드

### 색상 의미
- 🔵 **파란 점**: LiDAR 포인트 (실제 물체 위치)
- 🟢 **초록 점**: 실제 타겟으로 분류된 레이더 신호
- 🔴 **빨간 점**: 고스트 타겟으로 분류된 레이더 신호
- 🌈 **색상 그라데이션**: SNR 값 또는 예측 확률

### 분석 예시
- **Frame 0**: 모든 레이더 포인트가 LiDAR에서 0.768m 이상 떨어져 있어 모두 고스트
- **Frame 1-10**: 일부 레이더 포인트가 LiDAR 근처에 있어 실제 타겟으로 분류

## 🔧 파일 구조

```
modified_ragnnarok/
├── RadarMap_v2.txt              # 레이더 데이터
├── LiDARMap_v2.txt              # LiDAR 데이터
├── data_structures.py           # 데이터 구조 정의
├── hybrid_ghost_gnn.py          # GNN 모델 정의
├── ghost_detector.py            # 메인 시스템 (학습)
├── visualize_detection.py       # 시각화 시스템
├── ghost_detector.pth           # 학습된 모델
├── detection_frame_*.png        # 개별 프레임 시각화
├── detection_summary.png        # 전체 요약 시각화
└── README.md                    # 이 파일
```

## 🎯 기술적 특징

### 혁신점
1. **멀티 센서 융합**: LiDAR + Radar 데이터 활용
2. **지능형 라벨링**: 거리 + 신호 세기 조합 판단
3. **공간 관계 학습**: GraphSAGE 기반 GNN으로 주변 포인트 관계 학습
4. **실시간 처리**: GPU 가속으로 빠른 학습/추론

### 장점
- ✅ **완전 자동화**: 수동 라벨링 불필요
- ✅ **높은 정확도**: 98% 이상의 안정적 성능
- ✅ **실용적**: 실제 센서 데이터로 검증
- ✅ **확장 가능**: 다른 환경/센서에 적용 가능

### 한계점
- ⚠️ **LiDAR 의존성**: Ground Truth 생성을 위해 LiDAR 필수
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

---
**Modified RaGNNarok** - SNR + 거리 조합 기반 레이더 고스트 타겟 자동 탐지 시스템
