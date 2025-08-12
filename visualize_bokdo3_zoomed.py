"""
Bokdo3 데이터 - 레이더 데이터 중심으로 확대된 시각화
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from data_structures import RadarPoint
from hybrid_ghost_gnn import HybridGhostGNN, create_graph_data

def load_and_visualize_bokdo3_zoomed():
    """bokdo3 데이터를 레이더 중심으로 확대해서 시각화"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Bokdo3 데이터 확대 시각화 시작! 장치: {device}")
    
    # 학습된 모델 로드
    model = HybridGhostGNN(input_dim=6, hidden_dim=128)
    model.load_state_dict(torch.load('ghost_detector.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # 데이터 로드
    radar_data = []
    with open("RadarMap_bokdo3_v6.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    time = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    velocity = float(parts[3])
                    snr = float(parts[4])
                    radar_data.append((time, x, y, velocity, snr))
    
    lidar_data = []
    with open("LiDARMap_bokdo3_v6.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 4:
                    time = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    intensity = float(parts[3])
                    lidar_data.append((time, x, y, intensity))
    
    # 시간별로 그룹화
    radar_by_time = {}
    for time, x, y, velocity, snr in radar_data:
        if time not in radar_by_time:
            radar_by_time[time] = []
        radar_by_time[time].append((x, y, velocity, snr))
    
    lidar_by_time = {}
    for time, x, y, intensity in lidar_data:
        if time not in lidar_by_time:
            lidar_by_time[time] = []
        lidar_by_time[time].append((x, y, intensity))
    
    # 모든 데이터 수집
    all_radar_points = []
    all_lidar_points = []
    
    common_times = set(radar_by_time.keys()) & set(lidar_by_time.keys())
    
    print(f"처리할 시간 프레임: {len(common_times)}개")
    
    with torch.no_grad():
        for i, time in enumerate(sorted(common_times)):
            radar_points_data = radar_by_time[time]
            lidar_points_data = lidar_by_time[time]
            
            if len(radar_points_data) >= 5:  # 최소 포인트 수
                # 레이더 포인트 생성
                radar_points = []
                for x, y, velocity, snr in radar_points_data:
                    point = RadarPoint(x, y, velocity, snr)  # vr, rcs
                    radar_points.append(point)
                
                # LiDAR 포인트 좌표
                lidar_coords = np.array([(x, y) for x, y, _ in lidar_points_data])
                
                # Ground Truth 라벨 생성 (참조용)
                radar_coords = np.array([(p.x, p.y) for p in radar_points])
                if len(lidar_coords) > 0:
                    distances = cdist(radar_coords, lidar_coords)
                    min_distances = np.min(distances, axis=1)
                else:
                    min_distances = np.full(len(radar_points), float('inf'))
                
                labels = []
                for j, point in enumerate(radar_points):
                    distance_condition = min_distances[j] <= 0.5
                    snr_condition = point.rcs >= 20.0
                    if distance_condition and snr_condition:
                        labels.append(1)
                    else:
                        labels.append(0)
                
                # 그래프 데이터 생성 및 예측
                graph_data = create_graph_data(radar_points, labels, k=8)
                graph_data = graph_data.to(device)
                
                output = model(graph_data)
                probabilities = output.cpu().numpy().flatten()
                predictions = (probabilities > 0.5).astype(int)
                
                # 데이터 수집
                for j, (point, pred, prob) in enumerate(zip(radar_points, predictions, probabilities)):
                    all_radar_points.append((point.x, point.y, point.rcs, pred, prob))
                
                for x, y, intensity in lidar_points_data:
                    all_lidar_points.append((x, y, intensity))
            
            if i % 100 == 0:
                print(f"처리 완료: {i+1}/{len(common_times)} 프레임")
    
    # 데이터 분리
    all_radar_points = np.array(all_radar_points)
    all_lidar_points = np.array(all_lidar_points)
    
    # Real target과 Ghost target 분리
    real_target_mask = all_radar_points[:, 3] == 1
    ghost_target_mask = all_radar_points[:, 3] == 0
    
    real_targets = all_radar_points[real_target_mask]
    ghost_targets = all_radar_points[ghost_target_mask]
    
    # 레이더 데이터 범위 계산 (확대를 위해)
    radar_x_min, radar_x_max = np.min(all_radar_points[:, 0]), np.max(all_radar_points[:, 0])
    radar_y_min, radar_y_max = np.min(all_radar_points[:, 1]), np.max(all_radar_points[:, 1])
    
    # 레이더 범위에 여유 공간 추가
    x_margin = (radar_x_max - radar_x_min) * 0.1
    y_margin = (radar_y_max - radar_y_min) * 0.1
    
    zoom_x_min = radar_x_min - x_margin
    zoom_x_max = radar_x_max + x_margin
    zoom_y_min = radar_y_min - y_margin
    zoom_y_max = radar_y_max + y_margin
    
    print(f"📊 데이터 통계:")
    print(f"전체 LiDAR 포인트: {len(all_lidar_points):,}개")
    print(f"전체 레이더 포인트: {len(all_radar_points):,}개")
    print(f"Real Target: {len(real_targets):,}개 ({len(real_targets)/len(all_radar_points)*100:.1f}%)")
    print(f"Ghost Target: {len(ghost_targets):,}개 ({len(ghost_targets)/len(all_radar_points)*100:.1f}%)")
    print(f"레이더 데이터 범위: X({radar_x_min:.1f}~{radar_x_max:.1f}), Y({radar_y_min:.1f}~{radar_y_max:.1f})")
    
    if len(real_targets) > 0:
        print(f"Real Target 평균 확률: {np.mean(real_targets[:, 4]):.3f}")
        print(f"Real Target SNR 범위: {np.min(real_targets[:, 2]):.1f} ~ {np.max(real_targets[:, 2]):.1f} dB")
    
    # 시각화 - 레이더 중심으로 확대
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bokdo3 데이터 - 레이더 중심 확대 시각화', fontsize=16, fontweight='bold')
    
    # 1. 전체 데이터 (레이더 중심 확대)
    ax1 = axes[0, 0]
    # LiDAR 포인트 중 확대 범위 내의 것만 표시
    lidar_in_range = all_lidar_points[
        (all_lidar_points[:, 0] >= zoom_x_min) & (all_lidar_points[:, 0] <= zoom_x_max) &
        (all_lidar_points[:, 1] >= zoom_y_min) & (all_lidar_points[:, 1] <= zoom_y_max)
    ]
    
    if len(lidar_in_range) > 0:
        ax1.scatter(lidar_in_range[:, 0], lidar_in_range[:, 1], 
                   c='lightblue', s=2, alpha=0.4, label=f'LiDAR ({len(lidar_in_range):,}개)')
    
    if len(ghost_targets) > 0:
        ax1.scatter(ghost_targets[:, 0], ghost_targets[:, 1], 
                   c='red', s=15, alpha=0.7, label=f'Ghost Target ({len(ghost_targets):,}개)')
    
    if len(real_targets) > 0:
        scatter1 = ax1.scatter(real_targets[:, 0], real_targets[:, 1], 
                              c=real_targets[:, 2], s=25, cmap='Greens', 
                              label=f'Real Target ({len(real_targets):,}개)')
        plt.colorbar(scatter1, ax=ax1, label='SNR (dB)')
    
    ax1.set_xlim(zoom_x_min, zoom_x_max)
    ax1.set_ylim(zoom_y_min, zoom_y_max)
    ax1.set_title('전체 데이터 (레이더 중심 확대)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Real Target + LiDAR (확대)
    ax2 = axes[0, 1]
    if len(lidar_in_range) > 0:
        ax2.scatter(lidar_in_range[:, 0], lidar_in_range[:, 1], 
                   c='lightblue', s=2, alpha=0.4, label=f'LiDAR ({len(lidar_in_range):,}개)')
    
    if len(real_targets) > 0:
        scatter2 = ax2.scatter(real_targets[:, 0], real_targets[:, 1], 
                              c=real_targets[:, 4], s=25, cmap='Reds', 
                              label=f'Real Target ({len(real_targets):,}개)')
        plt.colorbar(scatter2, ax=ax2, label='예측 확률')
    
    ax2.set_xlim(zoom_x_min, zoom_x_max)
    ax2.set_ylim(zoom_y_min, zoom_y_max)
    ax2.set_title('Real Target + LiDAR (확률 색상)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Real Target만 (확대)
    ax3 = axes[1, 0]
    if len(real_targets) > 0:
        scatter3 = ax3.scatter(real_targets[:, 0], real_targets[:, 1], 
                              c=real_targets[:, 2], s=40, cmap='Reds')
        plt.colorbar(scatter3, ax=ax3, label='SNR (dB)')
    
    ax3.set_xlim(zoom_x_min, zoom_x_max)
    ax3.set_ylim(zoom_y_min, zoom_y_max)
    ax3.set_title(f'Real Target만 ({len(real_targets):,}개)')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 레이더 경로 추적 (시간순 연결)
    ax4 = axes[1, 1]
    if len(real_targets) > 0:
        # Real target들을 시간 순서대로 연결해서 경로 표시
        ax4.scatter(real_targets[:, 0], real_targets[:, 1], 
                   c=real_targets[:, 2], s=30, cmap='Reds', alpha=0.8)
        ax4.plot(real_targets[:, 0], real_targets[:, 1], 
                'b-', alpha=0.3, linewidth=1, label='이동 경로')
        plt.colorbar(scatter3, ax=ax4, label='SNR (dB)')
    
    ax4.set_xlim(zoom_x_min, zoom_x_max)
    ax4.set_ylim(zoom_y_min, zoom_y_max)
    ax4.set_title('Real Target 이동 경로')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bokdo3_zoomed_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 확대 시각화 완료! 'bokdo3_zoomed_results.png' 파일로 저장되었습니다.")
    print(f"📍 레이더 데이터가 잘 보이도록 {zoom_x_min:.1f}~{zoom_x_max:.1f} (X), {zoom_y_min:.1f}~{zoom_y_max:.1f} (Y) 범위로 확대했습니다.")

if __name__ == "__main__":
    load_and_visualize_bokdo3_zoomed()
