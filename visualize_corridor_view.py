"""
복도 모양이 보이도록 개선된 시각화 스크립트
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from ghost_detector import GhostDetectorDataset
from hybrid_ghost_gnn import HybridGhostGNN

def visualize_corridor_data():
    """복도 모양이 보이도록 시각화"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 복도 시각화 시작! 장치: {device}")
    
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
    
    print("📊 복도 데이터 수집 중...")
    
    # 모든 시간대의 데이터 수집 (더 많은 프레임으로)
    all_radar_x, all_radar_y, all_radar_snr = [], [], []
    all_lidar_x, all_lidar_y = [], []
    all_ground_truth, all_predictions, all_pred_probs = [], [], []
    
    # 더 많은 프레임 처리 (복도 전체 모양을 보기 위해)
    max_frames = min(500, len(dataset.radar_frames))
    
    for frame_idx in range(max_frames):
        if frame_idx % 50 == 0:
            print(f"처리 중: {frame_idx}/{max_frames}")
            
        radar_frame = dataset.radar_frames[frame_idx]
        lidar_frame = dataset.lidar_frames[frame_idx]
        
        if not radar_frame or not lidar_frame:
            continue
        
        # 레이더 데이터
        radar_x = [p.x for p in radar_frame]
        radar_y = [p.y for p in radar_frame]
        radar_snr = [p.rcs for p in radar_frame]
        
        # LiDAR 데이터
        lidar_x = [p[0] for p in lidar_frame]
        lidar_y = [p[1] for p in lidar_frame]
        
        # Ground Truth 계산
        radar_positions = np.array([[p.x, p.y] for p in radar_frame])
        lidar_positions = np.array(lidar_frame)
        distances = cdist(radar_positions, lidar_positions)
        min_distances = np.min(distances, axis=1)
        
        # SNR + 거리 조합 Ground Truth
        ground_truth = []
        for distance, snr in zip(min_distances, radar_snr):
            if distance <= 0.5 and snr >= 20.0:
                ground_truth.append(1)  # 실제
            else:
                ground_truth.append(0)  # 고스트
        
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
        
        # 데이터 누적
        all_radar_x.extend(radar_x)
        all_radar_y.extend(radar_y)
        all_radar_snr.extend(radar_snr)
        all_lidar_x.extend(lidar_x)
        all_lidar_y.extend(lidar_y)
        all_ground_truth.extend(ground_truth)
        all_predictions.extend(predicted_labels)
        all_pred_probs.extend(predicted_probs)
    
    # 배열로 변환
    all_radar_x = np.array(all_radar_x)
    all_radar_y = np.array(all_radar_y)
    all_radar_snr = np.array(all_radar_snr)
    all_lidar_x = np.array(all_lidar_x)
    all_lidar_y = np.array(all_lidar_y)
    all_ground_truth = np.array(all_ground_truth)
    all_predictions = np.array(all_predictions)
    all_pred_probs = np.array(all_pred_probs)
    
    print(f"📈 복도 시각화 생성 중... (총 {len(all_radar_x)}개 레이더 포인트, {len(all_lidar_x)}개 LiDAR 포인트)")
    
    # 데이터 범위 확인
    print(f"X 범위: {min(all_lidar_x.min(), all_radar_x.min()):.2f} ~ {max(all_lidar_x.max(), all_radar_x.max()):.2f}")
    print(f"Y 범위: {min(all_lidar_y.min(), all_radar_y.min()):.2f} ~ {max(all_lidar_y.max(), all_radar_y.max()):.2f}")
    
    # 시각화 생성 (복도 모양이 보이도록 축 범위 조정)
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    
    # 복도 모양이 보이도록 축 범위 설정 (실제 데이터 범위 기반)
    x_min, x_max = -2, 35  # 복도 길이 방향 (실제 데이터: 0.05 ~ 33.87)
    y_min, y_max = -15, 15  # 복도 폭 방향 (실제 데이터: -13.12 ~ 14.27)
    
    # 1. 원본 데이터 (복도 모양)
    axes[0,0].scatter(all_lidar_x, all_lidar_y, c='blue', alpha=0.4, s=1, label=f'LiDAR Points ({len(all_lidar_x)})')
    scatter = axes[0,0].scatter(all_radar_x, all_radar_y, c=all_radar_snr, cmap='viridis', 
                               s=8, alpha=0.8, edgecolors='black', linewidth=0.1)
    plt.colorbar(scatter, ax=axes[0,0], label='SNR (dB)')
    axes[0,0].set_title(f'Corridor View: Raw Data\nLiDAR: {len(all_lidar_x)} points, Radar: {len(all_radar_x)} points')
    axes[0,0].set_xlabel('X (m) - Along Corridor')
    axes[0,0].set_ylabel('Y (m) - Across Corridor')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlim(x_min, x_max)
    axes[0,0].set_ylim(y_min, y_max)
    axes[0,0].set_aspect('equal')
    
    # 2. Ground Truth (복도 모양)
    real_mask = all_ground_truth == 1
    ghost_mask = all_ground_truth == 0
    
    axes[0,1].scatter(all_lidar_x, all_lidar_y, c='blue', alpha=0.3, s=1, label='LiDAR')
    axes[0,1].scatter(all_radar_x[real_mask], all_radar_y[real_mask], 
                     c='green', s=12, alpha=0.9, edgecolors='black', linewidth=0.1,
                     label=f'Real Targets ({real_mask.sum()})')
    axes[0,1].scatter(all_radar_x[ghost_mask], all_radar_y[ghost_mask], 
                     c='red', s=12, alpha=0.9, edgecolors='black', linewidth=0.1,
                     label=f'Ghost Targets ({ghost_mask.sum()})')
    
    axes[0,1].set_title(f'Ground Truth (Distance ≤ 0.5m AND SNR ≥ 20dB)\nReal: {real_mask.sum()}, Ghost: {ghost_mask.sum()}')
    axes[0,1].set_xlabel('X (m) - Along Corridor')
    axes[0,1].set_ylabel('Y (m) - Across Corridor')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xlim(x_min, x_max)
    axes[0,1].set_ylim(y_min, y_max)
    axes[0,1].set_aspect('equal')
    
    # 3. 모델 예측 결과 (복도 모양)
    pred_real_mask = all_predictions == 1
    pred_ghost_mask = all_predictions == 0
    
    axes[1,0].scatter(all_lidar_x, all_lidar_y, c='blue', alpha=0.3, s=1, label='LiDAR')
    axes[1,0].scatter(all_radar_x[pred_real_mask], all_radar_y[pred_real_mask], 
                     c='green', s=12, alpha=0.9, edgecolors='black', linewidth=0.1,
                     label=f'Predicted Real ({pred_real_mask.sum()})')
    axes[1,0].scatter(all_radar_x[pred_ghost_mask], all_radar_y[pred_ghost_mask], 
                     c='red', s=12, alpha=0.9, edgecolors='black', linewidth=0.1,
                     label=f'Predicted Ghost ({pred_ghost_mask.sum()})')
    
    # 정확도 계산
    accuracy = (all_predictions == all_ground_truth).mean() * 100
    axes[1,0].set_title(f'Model Prediction (Accuracy: {accuracy:.1f}%)\nPred Real: {pred_real_mask.sum()}, Pred Ghost: {pred_ghost_mask.sum()}')
    axes[1,0].set_xlabel('X (m) - Along Corridor')
    axes[1,0].set_ylabel('Y (m) - Across Corridor')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlim(x_min, x_max)
    axes[1,0].set_ylim(y_min, y_max)
    axes[1,0].set_aspect('equal')
    
    # 4. 예측 확률 히트맵 (복도 모양)
    scatter_prob = axes[1,1].scatter(all_radar_x, all_radar_y, c=all_pred_probs, cmap='RdYlGn', 
                                    s=15, alpha=0.9, edgecolors='black', linewidth=0.1,
                                    vmin=0, vmax=1)
    axes[1,1].scatter(all_lidar_x, all_lidar_y, c='blue', alpha=0.2, s=0.5, label='LiDAR')
    plt.colorbar(scatter_prob, ax=axes[1,1], label='Real Target Probability')
    axes[1,1].set_title(f'Prediction Confidence (Corridor View)\n(Green: High confidence Real, Red: High confidence Ghost)')
    axes[1,1].set_xlabel('X (m) - Along Corridor')
    axes[1,1].set_ylabel('Y (m) - Across Corridor')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlim(x_min, x_max)
    axes[1,1].set_ylim(y_min, y_max)
    axes[1,1].set_aspect('equal')
    
    plt.tight_layout()
    
    # 저장
    filename = 'corridor_view_detection.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 저장: {filename}")
    
    # 통계 출력
    print(f"\n📊 복도 데이터 통계 ({max_frames}개 프레임):")
    print(f"  총 레이더 포인트: {len(all_radar_x)}개")
    print(f"  총 LiDAR 포인트: {len(all_lidar_x)}개")
    print(f"  SNR 범위: {all_radar_snr.min():.1f} ~ {all_radar_snr.max():.1f} dB")
    print(f"  X 좌표 범위: {all_radar_x.min():.2f} ~ {all_radar_x.max():.2f} m")
    print(f"  Y 좌표 범위: {all_radar_y.min():.2f} ~ {all_radar_y.max():.2f} m")
    print(f"  Ground Truth - Real: {real_mask.sum()} ({real_mask.mean()*100:.1f}%)")
    print(f"  Model Prediction - Real: {pred_real_mask.sum()} ({pred_real_mask.mean()*100:.1f}%)")
    print(f"  전체 정확도: {accuracy:.1f}%")
    
    # 혼동 행렬
    tp = ((all_predictions == 1) & (all_ground_truth == 1)).sum()
    tn = ((all_predictions == 0) & (all_ground_truth == 0)).sum()
    fp = ((all_predictions == 1) & (all_ground_truth == 0)).sum()
    fn = ((all_predictions == 0) & (all_ground_truth == 1)).sum()
    
    print(f"  혼동 행렬:")
    print(f"    True Positive: {tp} | True Negative: {tn}")
    print(f"    False Positive: {fp} | False Negative: {fn}")

if __name__ == '__main__':
    print("🎯 복도 모양 시각화")
    print("목표: 복도를 지나면서 찍은 센서값들의 복도 모양 시각화")
    
    visualize_corridor_data()
    
    print("\n✅ 복도 시각화 완료!")
    print("생성된 파일:")
    print("- corridor_view_detection.png: 복도 모양 시각화")
