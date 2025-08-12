"""
Bokdo3 데이터의 SNR 분포 분석 및 최적 임계값 찾기
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def load_radar_data(file_path):
    """레이더 데이터 로딩"""
    radar_points = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                time = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                velocity = float(parts[3])
                snr = float(parts[4])
                radar_points.append((time, x, y, velocity, snr))
    return radar_points

def load_lidar_data(file_path):
    """LiDAR 데이터 로딩"""
    lidar_points = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                time = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                intensity = float(parts[3])
                lidar_points.append((time, x, y, intensity))
    return lidar_points

def group_by_time(points, time_tolerance=0.01):
    """시간별로 포인트 그룹화"""
    frames = []
    current_frame = []
    current_time = None
    
    for point in points:
        time = point[0]
        if current_time is None:
            current_time = time
        
        if abs(time - current_time) > time_tolerance:
            if current_frame:
                frames.append(current_frame)
            current_frame = []
            current_time = time
        
        current_frame.append(point)
    
    if current_frame:
        frames.append(current_frame)
    
    return frames

def analyze_bokdo3_snr_distribution():
    """Bokdo3 SNR 분포 분석"""
    print("📊 Bokdo3 SNR 분포 분석 시작...")
    
    # 데이터 로딩
    radar_data = load_radar_data('RadarMap_bokdo3_v6.txt')
    lidar_data = load_lidar_data('LiDARMap_bokdo3_v6.txt')
    
    print(f"레이더 포인트: {len(radar_data)}개")
    print(f"LiDAR 포인트: {len(lidar_data)}개")
    
    # 시간별 그룹화
    radar_frames = group_by_time(radar_data)
    lidar_frames = group_by_time(lidar_data)
    
    print(f"레이더 프레임: {len(radar_frames)}개")
    print(f"LiDAR 프레임: {len(lidar_frames)}개")
    
    # SNR 분석
    all_snrs = [point[4] for point in radar_data]
    snr_array = np.array(all_snrs)
    
    print(f"\n=== Bokdo3 SNR 통계 ===")
    print(f"평균: {np.mean(snr_array):.2f} dB")
    print(f"표준편차: {np.std(snr_array):.2f} dB")
    print(f"최소값: {np.min(snr_array):.2f} dB")
    print(f"최대값: {np.max(snr_array):.2f} dB")
    print(f"25% 분위수: {np.percentile(snr_array, 25):.2f} dB")
    print(f"50% 분위수: {np.percentile(snr_array, 50):.2f} dB")
    print(f"75% 분위수: {np.percentile(snr_array, 75):.2f} dB")
    
    # 거리 기반 Real/Ghost 분류 분석
    distance_threshold = 0.5
    min_frames = min(len(radar_frames), len(lidar_frames))
    
    real_snrs = []
    ghost_snrs = []
    
    for i in range(min(100, min_frames)):  # 처음 100개 프레임만
        radar_frame = radar_frames[i]
        lidar_frame = lidar_frames[i]
        
        if len(radar_frame) < 5:
            continue
            
        radar_positions = np.array([[p[1], p[2]] for p in radar_frame])  # x, y
        lidar_positions = np.array([[p[1], p[2]] for p in lidar_frame])  # x, y
        radar_snrs_frame = np.array([p[4] for p in radar_frame])  # SNR
        
        if len(lidar_positions) == 0:
            ghost_snrs.extend(radar_snrs_frame)
            continue
        
        # 거리 계산
        distances = cdist(radar_positions, lidar_positions)
        min_distances = np.min(distances, axis=1)
        
        # 거리 기준으로만 분류 (SNR 조건 제외)
        for distance, snr in zip(min_distances, radar_snrs_frame):
            if distance <= distance_threshold:
                real_snrs.append(snr)
            else:
                ghost_snrs.append(snr)
    
    real_snrs = np.array(real_snrs)
    ghost_snrs = np.array(ghost_snrs)
    
    print(f"\n=== 거리 기준 분류 (거리 <= {distance_threshold}m) ===")
    if len(real_snrs) > 0:
        print(f"Real Targets SNR:")
        print(f"  - 개수: {len(real_snrs)}")
        print(f"  - 평균: {np.mean(real_snrs):.2f} dB")
        print(f"  - 표준편차: {np.std(real_snrs):.2f} dB")
        print(f"  - 최소값: {np.min(real_snrs):.2f} dB")
        print(f"  - 최대값: {np.max(real_snrs):.2f} dB")
        print(f"  - 25% 분위수: {np.percentile(real_snrs, 25):.2f} dB")
        print(f"  - 50% 분위수: {np.percentile(real_snrs, 50):.2f} dB")
        print(f"  - 75% 분위수: {np.percentile(real_snrs, 75):.2f} dB")
    
    if len(ghost_snrs) > 0:
        print(f"\nGhost Targets SNR:")
        print(f"  - 개수: {len(ghost_snrs)}")
        print(f"  - 평균: {np.mean(ghost_snrs):.2f} dB")
        print(f"  - 표준편차: {np.std(ghost_snrs):.2f} dB")
        print(f"  - 최소값: {np.min(ghost_snrs):.2f} dB")
        print(f"  - 최대값: {np.max(ghost_snrs):.2f} dB")
    
    # 다양한 SNR 임계값에서의 성능 분석
    print(f"\n=== Bokdo3에 최적화된 SNR 임계값 분석 ===")
    snr_thresholds = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    
    best_f1 = 0
    best_snr = 15.0
    
    for snr_thresh in snr_thresholds:
        if len(real_snrs) == 0 or len(ghost_snrs) == 0:
            continue
            
        # 현재 SNR 임계값으로 분류
        real_as_real = np.sum(real_snrs >= snr_thresh)
        real_as_ghost = np.sum(real_snrs < snr_thresh)
        ghost_as_real = np.sum(ghost_snrs >= snr_thresh)
        ghost_as_ghost = np.sum(ghost_snrs < snr_thresh)
        
        # 성능 지표 계산
        precision = real_as_real / (real_as_real + ghost_as_real) if (real_as_real + ghost_as_real) > 0 else 0
        recall = real_as_real / (real_as_real + real_as_ghost) if (real_as_real + real_as_ghost) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (real_as_real + ghost_as_ghost) / (len(real_snrs) + len(ghost_snrs)) if (len(real_snrs) + len(ghost_snrs)) > 0 else 0
        
        real_retention = real_as_real / len(real_snrs) if len(real_snrs) > 0 else 0
        
        print(f"SNR >= {snr_thresh:4.1f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}, Acc={accuracy:.3f}, Real보존={real_retention:.3f}")
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_snr = snr_thresh
    
    print(f"\n🎯 Bokdo3 추천 SNR 임계값: {best_snr} dB (F1-Score: {best_f1:.3f})")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Bokdo3 SNR Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. 전체 SNR 히스토그램
    ax1 = axes[0, 0]
    ax1.hist(snr_array, bins=50, alpha=0.7, color='blue', density=True)
    ax1.axvline(np.mean(snr_array), color='red', linestyle='--', label=f'Mean: {np.mean(snr_array):.1f}dB')
    ax1.axvline(17.5, color='orange', linestyle='--', label='Current Threshold: 17.5dB')
    ax1.axvline(best_snr, color='green', linestyle='--', label=f'Recommended: {best_snr}dB')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Density')
    ax1.set_title('Bokdo3 Overall SNR Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Real vs Ghost SNR 비교
    ax2 = axes[0, 1]
    if len(real_snrs) > 0 and len(ghost_snrs) > 0:
        bins = np.linspace(min(np.min(real_snrs), np.min(ghost_snrs)), 
                          max(np.max(real_snrs), np.max(ghost_snrs)), 30)
        ax2.hist(real_snrs, bins=bins, alpha=0.7, label=f'Real ({len(real_snrs)})', color='green', density=True)
        ax2.hist(ghost_snrs, bins=bins, alpha=0.7, label=f'Ghost ({len(ghost_snrs)})', color='red', density=True)
        ax2.axvline(best_snr, color='black', linestyle='--', label=f'Recommended: {best_snr}dB')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Density')
    ax2.set_title('Real vs Ghost SNR Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 데이터 보존율 vs SNR 임계값
    ax3 = axes[1, 0]
    thresholds = np.arange(11, 25, 0.5)
    retention_rates = []
    for thresh in thresholds:
        retention = np.sum(snr_array >= thresh) / len(snr_array) * 100
        retention_rates.append(retention)
    
    ax3.plot(thresholds, retention_rates, 'b-', linewidth=2, label='Data Retention')
    ax3.axvline(17.5, color='orange', linestyle='--', label='Current (17.5dB)')
    ax3.axvline(best_snr, color='green', linestyle='--', label=f'Recommended ({best_snr}dB)')
    ax3.set_xlabel('SNR Threshold (dB)')
    ax3.set_ylabel('Data Retention Rate (%)')
    ax3.set_title('Data Retention vs SNR Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 성능 지표 비교
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    current_retention = np.sum(snr_array >= 17.5) / len(snr_array) * 100
    recommended_retention = np.sum(snr_array >= best_snr) / len(snr_array) * 100
    
    comparison_text = f"""
Bokdo3 SNR Analysis Results

Dataset Characteristics:
• Total Points: {len(snr_array):,}
• SNR Range: {np.min(snr_array):.1f} - {np.max(snr_array):.1f} dB
• SNR Mean: {np.mean(snr_array):.1f} ± {np.std(snr_array):.1f} dB

Real vs Ghost Analysis:
• Real Targets: {len(real_snrs)} (distance ≤ 0.5m)
• Ghost Targets: {len(ghost_snrs)}
• Real SNR Mean: {np.mean(real_snrs):.1f} dB
• Ghost SNR Mean: {np.mean(ghost_snrs):.1f} dB

Threshold Comparison:
• Current (17.5dB): {current_retention:.1f}% retention
• Recommended ({best_snr}dB): {recommended_retention:.1f}% retention
• Best F1-Score: {best_f1:.3f}

Recommendation:
Use SNR ≥ {best_snr}dB for Bokdo3 dataset
    """
    
    ax4.text(0.05, 0.95, comparison_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('bokdo3_snr_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_snr, best_f1

if __name__ == '__main__':
    best_threshold, best_score = analyze_bokdo3_snr_distribution()
    print(f"\n✅ 분석 완료!")
    print(f"🎯 Bokdo3 최적 SNR 임계값: {best_threshold}dB")
    print(f"📊 예상 F1-Score: {best_score:.3f}")
    print(f"💾 결과 저장: bokdo3_snr_analysis.png")
