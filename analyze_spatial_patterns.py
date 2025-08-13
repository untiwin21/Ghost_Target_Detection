"""
공간적 패턴 분석을 통한 유리벽 영역 감지
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import os

def analyze_spatial_patterns():
    """공간적 패턴 분석으로 유리벽 영역 감지"""
    
    # 데이터 로드
    radar_points = []
    lidar_points = []
    
    print("데이터 로딩 중...")
    
    # 레이더 데이터 로딩 (bokdo3)
    with open("/mnt/c/Users/user/Desktop/modified_ragnnarok/SNR_15dB_k5_0.4m_inference/RadarMap_bokdo3_v6.txt", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x, y = float(parts[1]), float(parts[2])
                snr = float(parts[4])
                radar_points.append((x, y, snr))
    
    # LiDAR 데이터 로딩 (bokdo3)
    with open("/mnt/c/Users/user/Desktop/modified_ragnnarok/SNR_15dB_k5_0.4m_inference/LiDARMap_bokdo3_v6.txt", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                x, y = float(parts[1]), float(parts[2])
                lidar_points.append((x, y))
    
    radar_array = np.array(radar_points)
    lidar_array = np.array(lidar_points)
    
    print(f"레이더 포인트: {len(radar_array):,}개")
    print(f"LiDAR 포인트: {len(lidar_array):,}개")
    
    # LiDAR 클러스터링으로 벽면 감지
    lidar_clustering = DBSCAN(eps=0.5, min_samples=10).fit(lidar_array[:, :2])
    lidar_labels = lidar_clustering.labels_
    
    # 각 LiDAR 클러스터 분석
    unique_labels = set(lidar_labels)
    wall_segments = []
    
    for label in unique_labels:
        if label == -1:  # 노이즈 제외
            continue
        
        cluster_points = lidar_array[lidar_labels == label]
        if len(cluster_points) > 50:  # 충분히 큰 클러스터만
            # 벽면의 방향성 분석
            x_range = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
            y_range = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
            
            # 직선성 분석 (벽면은 직선에 가까움)
            if x_range > y_range:  # 수평 벽면
                wall_type = 'horizontal'
                wall_pos = np.mean(cluster_points[:, 1])
                wall_range = (np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0]))
            else:  # 수직 벽면
                wall_type = 'vertical'
                wall_pos = np.mean(cluster_points[:, 0])
                wall_range = (np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1]))
            
            wall_segments.append({
                'type': wall_type,
                'position': wall_pos,
                'range': wall_range,
                'points': cluster_points,
                'point_count': len(cluster_points)
            })
    
    # 각 레이더 포인트에 대해 벽면과의 관계 분석
    radar_wall_analysis = []
    
    for i, (rx, ry, snr) in enumerate(radar_array):
        # 가장 가까운 LiDAR 포인트까지의 거리
        distances = np.sqrt((lidar_array[:, 0] - rx)**2 + (lidar_array[:, 1] - ry)**2)
        min_distance = np.min(distances)
        
        # 어느 벽면에 가까운지 분석
        closest_wall = None
        min_wall_distance = float('inf')
        
        for wall in wall_segments:
            if wall['type'] == 'vertical':
                if wall['range'][0] <= ry <= wall['range'][1]:
                    wall_distance = abs(rx - wall['position'])
                    if wall_distance < min_wall_distance:
                        min_wall_distance = wall_distance
                        closest_wall = wall
            else:  # horizontal
                if wall['range'][0] <= rx <= wall['range'][1]:
                    wall_distance = abs(ry - wall['position'])
                    if wall_distance < min_wall_distance:
                        min_wall_distance = wall_distance
                        closest_wall = wall
        
        # 레이더 포인트 분류
        is_near_lidar = min_distance <= 0.4
        is_near_wall = min_wall_distance <= 1.0 if closest_wall else False
        
        # SNR 기반 추가 분석
        snr_category = 'high' if snr >= 25 else 'medium' if snr >= 15 else 'low'
        
        radar_wall_analysis.append({
            'position': (rx, ry),
            'snr': snr,
            'snr_category': snr_category,
            'min_lidar_distance': min_distance,
            'is_near_lidar': is_near_lidar,
            'is_near_wall': is_near_wall,
            'closest_wall': closest_wall,
            'wall_distance': min_wall_distance if closest_wall else float('inf')
        })
    
    # 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 전체 데이터 + 벽면 클러스터
    ax1.scatter(lidar_array[:, 0], lidar_array[:, 1], c='blue', s=1, alpha=0.3, label='LiDAR')
    
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, wall in enumerate(wall_segments):
        color = colors[i % len(colors)]
        ax1.scatter(wall['points'][:, 0], wall['points'][:, 1], 
                   c=color, s=3, alpha=0.8, label=f'Wall {i+1} ({wall["point_count"]} pts)')
    
    ax1.scatter(radar_array[:, 0], radar_array[:, 1], c=radar_array[:, 2], 
               s=10, cmap='viridis', alpha=0.6, label='Radar')
    ax1.set_title('1. Wall Segment Detection', fontsize=14)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. LiDAR 인접 vs 벽면 인접 분석
    near_lidar_only = [(r['position'][0], r['position'][1]) for r in radar_wall_analysis 
                       if r['is_near_lidar'] and not r['is_near_wall']]
    near_wall_only = [(r['position'][0], r['position'][1]) for r in radar_wall_analysis 
                      if not r['is_near_lidar'] and r['is_near_wall']]
    near_both = [(r['position'][0], r['position'][1]) for r in radar_wall_analysis 
                 if r['is_near_lidar'] and r['is_near_wall']]
    near_neither = [(r['position'][0], r['position'][1]) for r in radar_wall_analysis 
                    if not r['is_near_lidar'] and not r['is_near_wall']]
    
    if near_lidar_only:
        ax2.scatter(*zip(*near_lidar_only), c='green', s=15, alpha=0.8, label=f'Near LiDAR only ({len(near_lidar_only)})')
    if near_wall_only:
        ax2.scatter(*zip(*near_wall_only), c='red', s=15, alpha=0.8, label=f'Near Wall only ({len(near_wall_only)})')
    if near_both:
        ax2.scatter(*zip(*near_both), c='orange', s=15, alpha=0.8, label=f'Near Both ({len(near_both)})')
    if near_neither:
        ax2.scatter(*zip(*near_neither), c='gray', s=15, alpha=0.8, label=f'Isolated ({len(near_neither)})')
    
    ax2.scatter(lidar_array[:, 0], lidar_array[:, 1], c='blue', s=1, alpha=0.2, label='LiDAR')
    ax2.set_title('2. Radar Point Classification', fontsize=14)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. SNR vs 거리 분석
    lidar_distances = [r['min_lidar_distance'] for r in radar_wall_analysis]
    snr_values = [r['snr'] for r in radar_wall_analysis]
    wall_distances = [r['wall_distance'] if r['wall_distance'] != float('inf') else 10 
                     for r in radar_wall_analysis]
    
    scatter = ax3.scatter(lidar_distances, snr_values, c=wall_distances, 
                         s=20, cmap='coolwarm', alpha=0.7)
    ax3.axvline(x=0.4, color='red', linestyle='--', label='0.4m LiDAR threshold')
    ax3.axhline(y=15, color='green', linestyle='--', label='15dB SNR threshold')
    ax3.set_xlabel('Distance to nearest LiDAR (m)')
    ax3.set_ylabel('SNR (dB)')
    ax3.set_title('3. SNR vs LiDAR Distance (colored by Wall Distance)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Distance to Wall (m)')
    
    # 4. 통계 분석
    ax4.axis('off')
    
    # 문제 영역 식별
    problem_points = [r for r in radar_wall_analysis 
                     if r['is_near_lidar'] and r['snr'] >= 15 and r['is_near_wall']]
    
    stats_text = f"""
    🔍 Spatial Pattern Analysis Results
    
    📊 Wall Segments Detected: {len(wall_segments)}
    {chr(10).join([f"  • Wall {i+1}: {w['type']} at {w['position']:.1f}m ({w['point_count']} pts)" 
                   for i, w in enumerate(wall_segments)])}
    
    📈 Radar Point Classification:
    • Near LiDAR only: {len(near_lidar_only):,} ({len(near_lidar_only)/len(radar_array)*100:.1f}%)
    • Near Wall only: {len(near_wall_only):,} ({len(near_wall_only)/len(radar_array)*100:.1f}%)
    • Near Both: {len(near_both):,} ({len(near_both)/len(radar_array)*100:.1f}%)
    • Isolated: {len(near_neither):,} ({len(near_neither)/len(radar_array)*100:.1f}%)
    
    🚨 Problem Area (Near LiDAR + Wall + High SNR):
    • Count: {len(problem_points):,} points
    • Percentage: {len(problem_points)/len(radar_array)*100:.1f}%
    • These might be glass wall reflections!
    
    💡 Recommendations:
    • Use wall-aware labeling for glass areas
    • Apply different SNR thresholds near walls
    • Consider multi-path reflection patterns
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('spatial_pattern_analysis.png', dpi=300, bbox_inches='tight')
    print("공간 패턴 분석 결과 저장: spatial_pattern_analysis.png")
    
    return wall_segments, radar_wall_analysis

if __name__ == "__main__":
    wall_segments, radar_analysis = analyze_spatial_patterns()
    
    print(f"\n🎯 분석 완료!")
    print(f"벽면 세그먼트: {len(wall_segments)}개")
    print(f"레이더 포인트 분석: {len(radar_analysis):,}개")
