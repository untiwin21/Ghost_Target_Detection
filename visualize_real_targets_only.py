"""
딥러닝 모델이 예측한 Real target과 LiDAR 값만 추출하여 시각화
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from ghost_detector import GhostDetectorDataset
from hybrid_ghost_gnn import HybridGhostGNN

def visualize_real_targets_only():
    """모델이 예측한 Real target과 LiDAR만 시각화"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Real Target + LiDAR 시각화 시작! 장치: {device}")
    
    # 데이터셋 로드
    dataset = GhostDetectorDataset(
        radar_data_path="RadarMap_v2.txt",
        lidar_data_path="LiDARMap_v2.txt",
        distance_threshold=0.5,
        snr_threshold=20.0
    )
    
    # 학습된 모델 로드
    model = HybridGhostGNN(input_dim=6, hidden_dim=128, dropout=0.1)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('ghost_detector.pth'))
    else:
        model.load_state_dict(torch.load('ghost_detector.pth', map_location='cpu'))
    model.to(device)
    model.eval()
    
    print("📊 Real Target + LiDAR 데이터 수집 중...")
    
    # Real target과 LiDAR 데이터만 수집
    real_radar_x, real_radar_y, real_radar_snr = [], [], []
    real_radar_probs = []  # 예측 확률
    all_lidar_x, all_lidar_y = [], []
    
    # 500개 프레임 처리
    max_frames = min(500, len(dataset.radar_frames))
    
    for frame_idx in range(max_frames):
        if frame_idx % 50 == 0:
            print(f"처리 중: {frame_idx}/{max_frames}")
            
        radar_frame = dataset.radar_frames[frame_idx]
        lidar_frame = dataset.lidar_frames[frame_idx]
        
        if not radar_frame or not lidar_frame:
            continue
        
        # 레이더 데이터
        radar_x = np.array([p.x for p in radar_frame])
        radar_y = np.array([p.y for p in radar_frame])
        radar_snr = np.array([p.rcs for p in radar_frame])
        
        # LiDAR 데이터
        lidar_x = [p[0] for p in lidar_frame]
        lidar_y = [p[1] for p in lidar_frame]
        
        # 모델 예측
        if frame_idx < len(dataset):
            data = dataset[frame_idx].to(device)
            with torch.no_grad():
                predictions = model(data)
                predicted_probs = predictions.squeeze().cpu().numpy()
                predicted_labels = (predicted_probs > 0.5).astype(int)
        else:
            predicted_labels = np.zeros(len(radar_frame))
            predicted_probs = np.zeros(len(radar_frame))
        
        # Real target으로 예측된 레이더 포인트만 추출
        real_mask = predicted_labels == 1
        
        if real_mask.sum() > 0:  # Real target이 있는 경우만
            real_radar_x.extend(radar_x[real_mask])
            real_radar_y.extend(radar_y[real_mask])
            real_radar_snr.extend(radar_snr[real_mask])
            real_radar_probs.extend(predicted_probs[real_mask])
        
        # 모든 LiDAR 데이터 누적
        all_lidar_x.extend(lidar_x)
        all_lidar_y.extend(lidar_y)
    
    # 배열로 변환
    real_radar_x = np.array(real_radar_x)
    real_radar_y = np.array(real_radar_y)
    real_radar_snr = np.array(real_radar_snr)
    real_radar_probs = np.array(real_radar_probs)
    all_lidar_x = np.array(all_lidar_x)
    all_lidar_y = np.array(all_lidar_y)
    
    print(f"📈 Real Target + LiDAR 시각화 생성 중...")
    print(f"  - Real Target 레이더 포인트: {len(real_radar_x)}개")
    print(f"  - LiDAR 포인트: {len(all_lidar_x)}개")
    
    # 시각화 생성 (2x2 레이아웃)
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    
    # 복도 모양이 보이도록 축 범위 설정
    x_min, x_max = -2, 35  # 복도 길이 방향
    y_min, y_max = -15, 15  # 복도 폭 방향
    
    # 1. LiDAR + Real Target (SNR 색상)
    axes[0,0].scatter(all_lidar_x, all_lidar_y, c='lightblue', alpha=0.3, s=1, label=f'LiDAR Points ({len(all_lidar_x)})')
    if len(real_radar_x) > 0:
        scatter = axes[0,0].scatter(real_radar_x, real_radar_y, c=real_radar_snr, cmap='viridis', 
                                   s=25, alpha=0.9, edgecolors='black', linewidth=0.2)
        plt.colorbar(scatter, ax=axes[0,0], label='SNR (dB)')
    axes[0,0].set_title(f'LiDAR + Predicted Real Targets (SNR)\nReal Targets: {len(real_radar_x)} points')
    axes[0,0].set_xlabel('X (m) - Along Corridor')
    axes[0,0].set_ylabel('Y (m) - Across Corridor')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlim(x_min, x_max)
    axes[0,0].set_ylim(y_min, y_max)
    axes[0,0].set_aspect('equal')
    
    # 2. LiDAR + Real Target (예측 확률 색상)
    axes[0,1].scatter(all_lidar_x, all_lidar_y, c='lightblue', alpha=0.3, s=1, label='LiDAR Points')
    if len(real_radar_x) > 0:
        scatter_prob = axes[0,1].scatter(real_radar_x, real_radar_y, c=real_radar_probs, cmap='Reds', 
                                        s=25, alpha=0.9, edgecolors='black', linewidth=0.2,
                                        vmin=0.5, vmax=1.0)
        plt.colorbar(scatter_prob, ax=axes[0,1], label='Real Target Probability')
    axes[0,1].set_title(f'LiDAR + Predicted Real Targets (Confidence)\nAvg Confidence: {real_radar_probs.mean():.3f}')
    axes[0,1].set_xlabel('X (m) - Along Corridor')
    axes[0,1].set_ylabel('Y (m) - Across Corridor')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xlim(x_min, x_max)
    axes[0,1].set_ylim(y_min, y_max)
    axes[0,1].set_aspect('equal')
    
    # 3. Real Target만 (크게 표시)
    if len(real_radar_x) > 0:
        scatter_large = axes[1,0].scatter(real_radar_x, real_radar_y, c=real_radar_snr, cmap='plasma', 
                                         s=50, alpha=0.9, edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter_large, ax=axes[1,0], label='SNR (dB)')
    axes[1,0].set_title(f'Predicted Real Targets Only\n{len(real_radar_x)} points, SNR: {real_radar_snr.min():.1f}~{real_radar_snr.max():.1f} dB')
    axes[1,0].set_xlabel('X (m) - Along Corridor')
    axes[1,0].set_ylabel('Y (m) - Across Corridor')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlim(x_min, x_max)
    axes[1,0].set_ylim(y_min, y_max)
    axes[1,0].set_aspect('equal')
    
    # 4. LiDAR만 (참조용)
    axes[1,1].scatter(all_lidar_x, all_lidar_y, c='blue', alpha=0.4, s=1)
    axes[1,1].set_title(f'LiDAR Points Only (Reference)\n{len(all_lidar_x)} points')
    axes[1,1].set_xlabel('X (m) - Along Corridor')
    axes[1,1].set_ylabel('Y (m) - Across Corridor')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlim(x_min, x_max)
    axes[1,1].set_ylim(y_min, y_max)
    axes[1,1].set_aspect('equal')
    
    plt.tight_layout()
    
    # 저장
    filename = 'real_targets_lidar_only.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 저장: {filename}")
    
    # 통계 출력
    print(f"\n📊 Real Target + LiDAR 통계 ({max_frames}개 프레임):")
    print(f"  총 LiDAR 포인트: {len(all_lidar_x)}개")
    print(f"  예측된 Real Target: {len(real_radar_x)}개")
    if len(real_radar_x) > 0:
        print(f"  Real Target SNR 범위: {real_radar_snr.min():.1f} ~ {real_radar_snr.max():.1f} dB")
        print(f"  Real Target 평균 확률: {real_radar_probs.mean():.3f}")
        print(f"  Real Target 최소 확률: {real_radar_probs.min():.3f}")
        print(f"  Real Target X 범위: {real_radar_x.min():.2f} ~ {real_radar_x.max():.2f} m")
        print(f"  Real Target Y 범위: {real_radar_y.min():.2f} ~ {real_radar_y.max():.2f} m")
    
    # Real Target 밀도 분석
    if len(real_radar_x) > 0:
        print(f"\n🔍 Real Target 분포 분석:")
        print(f"  복도 앞쪽 (X < 10m): {(real_radar_x < 10).sum()}개")
        print(f"  복도 중간 (10m ≤ X < 20m): {((real_radar_x >= 10) & (real_radar_x < 20)).sum()}개")
        print(f"  복도 뒤쪽 (X ≥ 20m): {(real_radar_x >= 20).sum()}개")
        print(f"  복도 중앙 (|Y| < 2m): {(np.abs(real_radar_y) < 2).sum()}개")
        print(f"  복도 가장자리 (|Y| ≥ 2m): {(np.abs(real_radar_y) >= 2).sum()}개")

if __name__ == '__main__':
    print("🎯 Real Target + LiDAR 전용 시각화")
    print("목표: 딥러닝 모델이 예측한 Real target과 LiDAR 값만 추출하여 시각화")
    
    visualize_real_targets_only()
    
    print("\n✅ Real Target + LiDAR 시각화 완료!")
    print("생성된 파일:")
    print("- real_targets_lidar_only.png: Real Target + LiDAR 전용 시각화")
