# GitHub 저장소 생성 및 푸시 가이드

## 1. GitHub 저장소 생성

1. https://github.com 에 로그인
2. 우측 상단 "+" 버튼 클릭 → "New repository"
3. Repository name: `Ghost_Target_Detectection` (또는 원하는 이름)
4. Description: `SNR + Distance based Radar Ghost Target Detection System using GNN`
5. Public/Private 선택
6. **"Add a README file" 체크 해제** (이미 README.md가 있음)
7. **"Add .gitignore" 선택 안함** (이미 .gitignore가 있음)
8. "Create repository" 클릭

## 2. 로컬에서 푸시

저장소 생성 후 다음 명령어 실행:

```bash
cd /mnt/c/Users/user/Desktop/modified_ragnnarok

# 원격 저장소 URL 확인/수정 (필요시)
git remote -v

# 만약 저장소 이름이 다르다면:
# git remote set-url origin https://github.com/untiwin21/새로운저장소이름.git

# 푸시
git push -u origin main
```

## 3. 현재 준비된 파일들

✅ **핵심 코드**
- `ghost_detector.py` - 메인 시스템 (SNR + 거리 조합)
- `hybrid_ghost_gnn.py` - GNN 모델
- `visualize_detection.py` - 시각화 시스템
- `data_structures.py` - 데이터 구조

✅ **데이터**
- `RadarMap_v2.txt` - 레이더 데이터 (2MB)
- `LiDARMap_v2.txt` - LiDAR 데이터 (37MB)
- `ghost_detector.pth` - 학습된 모델 (150KB)

✅ **시각화 결과**
- `detection_frame_*.png` - 개별 프레임 분석 (5개)
- `detection_summary.png` - 전체 요약

✅ **문서**
- `README.md` - 완전한 프로젝트 문서
- `requirements.txt` - 필요한 패키지 목록
- `.gitignore` - Git 무시 파일 설정

✅ **참고 논문**
- `RaGNNarok A Light-Weight Graph Neural Network...pdf`
- `Ghost Target Detection in 3D Radar Data using.pdf`

## 4. 커밋 메시지

이미 다음 메시지로 커밋되어 있습니다:

```
Initial commit: SNR + Distance based Radar Ghost Target Detection System

- Implemented hybrid labeling system using SNR (≥20dB) + Distance (≤0.5m) criteria
- Achieved 94.75% training accuracy and 98.1% test accuracy
- Added comprehensive visualization system for detection results
- Included GNN model with GraphSAGE architecture
- Complete documentation and usage examples
```

## 5. 저장소 설명 추천

**Repository Description:**
```
🎯 Radar Ghost Target Detection using SNR + Distance based labeling with Graph Neural Networks. Achieves 98.1% accuracy by combining LiDAR ground truth with radar signal analysis.
```

**Topics (태그):**
```
radar, lidar, ghost-detection, graph-neural-networks, pytorch, deep-learning, sensor-fusion, autonomous-driving
```
