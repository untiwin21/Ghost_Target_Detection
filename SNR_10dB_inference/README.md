# SNR 10dB 기준 추론 (기본 설정)

## 📋 **실험 개요**
- **목적**: SNR 10dB 기준 학습된 모델로 bokdo3 데이터 추론
- **추론 데이터**: RadarMap_bokdo3_v6.txt, LiDARMap_bokdo3_v6.txt

## ⚙️ **모델 파라미터**
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **사용 모델** | ghost_detector.pth | SNR 10dB 기준 학습 |
| **학습 정확도** | 91.34% | 검증 데이터 기준 |
| **k-NN 연결** | 8개 | 학습 시와 동일 |
| **추론 프레임** | 1,777개 | bokdo3 데이터 |

## 📊 **추론 결과**
- **총 레이더 포인트**: 14,809개
- **실제 타겟**: 14,649개 (98.9%)
- **고스트 타겟**: 160개 (1.1%)
- **평균 확률**: 0.536
- **처리된 프레임**: 1,777개

## 📁 **파일 구조**
```
SNR_10dB_inference/
├── inference.py                    # 추론 스크립트
├── visualize_10dB_results.py      # 시각화 스크립트
├── ghost_detector.pth             # 학습된 모델
├── hybrid_ghost_gnn.py            # GNN 모델 정의
├── data_structures.py             # 데이터 구조 정의
├── RadarMap_bokdo3_v6.txt         # 추론용 레이더 데이터
├── LiDARMap_bokdo3_v6.txt         # 추론용 LiDAR 데이터
├── SNR_10dB_all_frames.png        # 전체 프레임 시각화
├── SNR_10dB_radar_focused.png     # 레이더 중심 확대 시각화
└── requirements.txt               # 패키지 의존성
```

## 🎨 **시각화 결과**

### 📊 **SNR_10dB_all_frames.png**
4패널 구성으로 전체 데이터 분석:
1. **원본 데이터**: LiDAR(파란점) + Radar(SNR 색상)
2. **모델 예측**: Real(초록) vs Ghost(빨강) 분류
3. **예측 확률**: 0~1 확률 히트맵
4. **통계 정보**: 성능 지표 및 데이터 통계

### 🔍 **SNR_10dB_radar_focused.png**
레이더 중심 5m 마진 확대 뷰:
- LiDAR 노이즈 필터링으로 깔끔한 시각화
- 레이더 패턴 명확한 관찰 가능
- 실제 타겟 분포 집중 분석

## 🚀 **사용법**
```bash
# 추론 실행
python3 inference.py

# 시각화 생성
python3 visualize_10dB_results.py
```

## 📈 **성능 분석**
### ✅ **장점**
- 매우 높은 실제 타겟 탐지율 (98.9%)
- 낮은 False Negative (실제 타겟 놓침 최소)
- 안정적인 추론 성능

### ⚠️ **문제점**
- 과도하게 관대한 판단 (고스트도 Real로 분류)
- 높은 False Positive (고스트 타겟 탐지 부족)
- 실용적 정밀도 부족

## 🔄 **개선 방향**
이 결과를 바탕으로 다음 실험들이 진행됨:
- k-NN 연결 수 감소로 이웃 영향 줄이기
- 거리 임계값 조정으로 더 엄격한 기준 적용
- 정밀도와 재현율의 균형점 찾기
