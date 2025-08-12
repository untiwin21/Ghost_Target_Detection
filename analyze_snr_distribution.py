"""
SNR 분포 분석 및 적절한 임계값 찾기
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
# import seaborn as sns  # 사용하지 않음

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

def analyze_snr_by_distance(radar_frames, lidar_frames, distance_thresholds=[0.3, 0.5, 0.7, 1.0]):
    """거리별 SNR 분포 분석"""
    results = {}
    
    min_frames = min(len(radar_frames), len(lidar_frames))
    
    for dist_thresh in distance_thresholds:
        real_snrs = []
        ghost_snrs = []
        
        for i in range(min_frames):
            if len(radar_frames[i]) < 5:  # 최소 포인트 수
                continue
                
            radar_positions = np.array([[p[1], p[2]] for p in radar_frames[i]])  # x, y
            lidar_positions = np.array([[p[1], p[2]] for p in lidar_frames[i]])  # x, y
            radar_snrs = np.array([p[4] for p in radar_frames[i]])  # SNR
            
            if len(lidar_positions) == 0:
                ghost_snrs.extend(radar_snrs)
                continue
            
            # 거리 계산
            distances = cdist(radar_positions, lidar_positions)
            min_distances = np.min(distances, axis=1)
            
            # 거리 기준으로 분류
            real_mask = min_distances <= dist_thresh
            ghost_mask = ~real_mask
            
            real_snrs.extend(radar_snrs[real_mask])
            ghost_snrs.extend(radar_snrs[ghost_mask])
        
        results[dist_thresh] = {
            'real_snrs': np.array(real_snrs),
            'ghost_snrs': np.array(ghost_snrs)
        }
    
    return results

def plot_snr_analysis(results):
    """SNR 분석 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SNR Distribution Analysis by Distance Threshold', fontsize=16)
    
    distance_thresholds = list(results.keys())
    
    for idx, dist_thresh in enumerate(distance_thresholds):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        real_snrs = results[dist_thresh]['real_snrs']
        ghost_snrs = results[dist_thresh]['ghost_snrs']
        
        # 히스토그램
        bins = np.linspace(10, 50, 40)
        ax.hist(real_snrs, bins=bins, alpha=0.7, label=f'Real Targets (n={len(real_snrs)})', 
                color='green', density=True)
        ax.hist(ghost_snrs, bins=bins, alpha=0.7, label=f'Ghost Targets (n={len(ghost_snrs)})', 
                color='red', density=True)
        
        # 통계 정보
        real_mean = np.mean(real_snrs) if len(real_snrs) > 0 else 0
        ghost_mean = np.mean(ghost_snrs) if len(ghost_snrs) > 0 else 0
        real_std = np.std(real_snrs) if len(real_snrs) > 0 else 0
        ghost_std = np.std(ghost_snrs) if len(ghost_snrs) > 0 else 0
        
        ax.axvline(real_mean, color='green', linestyle='--', alpha=0.8, 
                  label=f'Real Mean: {real_mean:.1f}±{real_std:.1f}')
        ax.axvline(ghost_mean, color='red', linestyle='--', alpha=0.8,
                  label=f'Ghost Mean: {ghost_mean:.1f}±{ghost_std:.1f}')
        
        ax.set_title(f'Distance Threshold: {dist_thresh}m')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('snr_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def find_optimal_snr_threshold(results, distance_threshold=0.5):
    """최적 SNR 임계값 찾기"""
    real_snrs = results[distance_threshold]['real_snrs']
    ghost_snrs = results[distance_threshold]['ghost_snrs']
    
    print(f"\n=== SNR 분석 결과 (거리 임계값: {distance_threshold}m) ===")
    
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
        print(f"  - 25% 분위수: {np.percentile(ghost_snrs, 25):.2f} dB")
        print(f"  - 50% 분위수: {np.percentile(ghost_snrs, 50):.2f} dB")
        print(f"  - 75% 분위수: {np.percentile(ghost_snrs, 75):.2f} dB")
    
    # 다양한 SNR 임계값에서의 성능 분석
    print(f"\n=== 다양한 SNR 임계값에서의 분류 성능 ===")
    snr_thresholds = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 25.0]
    
    best_f1 = 0
    best_snr = 20.0
    
    for snr_thresh in snr_thresholds:
        # 현재 SNR 임계값으로 분류
        real_as_real = np.sum(real_snrs >= snr_thresh) if len(real_snrs) > 0 else 0
        real_as_ghost = np.sum(real_snrs < snr_thresh) if len(real_snrs) > 0 else 0
        ghost_as_real = np.sum(ghost_snrs >= snr_thresh) if len(ghost_snrs) > 0 else 0
        ghost_as_ghost = np.sum(ghost_snrs < snr_thresh) if len(ghost_snrs) > 0 else 0
        
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
    
    print(f"\n🎯 추천 SNR 임계값: {best_snr} dB (F1-Score: {best_f1:.3f})")
    
    return best_snr

if __name__ == '__main__':
    print("📊 SNR 분포 분석 시작...")
    
    # 데이터 로딩
    radar_data = load_radar_data('RadarMap_v2.txt')
    lidar_data = load_lidar_data('LiDARMap_v2.txt')
    
    print(f"레이더 포인트: {len(radar_data)}개")
    print(f"LiDAR 포인트: {len(lidar_data)}개")
    
    # 시간별 그룹화
    radar_frames = group_by_time(radar_data)
    lidar_frames = group_by_time(lidar_data)
    
    print(f"레이더 프레임: {len(radar_frames)}개")
    print(f"LiDAR 프레임: {len(lidar_frames)}개")
    
    # SNR 분석
    results = analyze_snr_by_distance(radar_frames, lidar_frames)
    
    # 시각화
    plot_snr_analysis(results)
    
    # 최적 임계값 찾기
    optimal_snr = find_optimal_snr_threshold(results, distance_threshold=0.5)
    
    print(f"\n✅ 분석 완료! 결과는 'snr_distribution_analysis.png'에 저장되었습니다.")
