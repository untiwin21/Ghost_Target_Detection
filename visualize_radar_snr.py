"""
레이더 데이터의 SNR 분포 시각화
"""
import numpy as np
import matplotlib.pyplot as plt
import os

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

def analyze_radar_snr(file_path, dataset_name):
    """레이더 SNR 분석"""
    print(f"\n=== {dataset_name} 분석 ===")
    
    if not os.path.exists(file_path):
        print(f"파일이 존재하지 않습니다: {file_path}")
        return None
    
    radar_data = load_radar_data(file_path)
    
    if not radar_data:
        print(f"데이터가 없습니다: {file_path}")
        return None
    
    # SNR 값 추출
    snr_values = np.array([point[4] for point in radar_data])
    
    # 통계 정보
    print(f"총 레이더 포인트: {len(radar_data):,}개")
    print(f"SNR 통계:")
    print(f"  - 평균: {np.mean(snr_values):.2f} dB")
    print(f"  - 표준편차: {np.std(snr_values):.2f} dB")
    print(f"  - 최소값: {np.min(snr_values):.2f} dB")
    print(f"  - 최대값: {np.max(snr_values):.2f} dB")
    print(f"  - 25% 분위수: {np.percentile(snr_values, 25):.2f} dB")
    print(f"  - 50% 분위수 (중앙값): {np.percentile(snr_values, 50):.2f} dB")
    print(f"  - 75% 분위수: {np.percentile(snr_values, 75):.2f} dB")
    
    # SNR 범위별 분포
    ranges = [(15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 50)]
    print(f"\nSNR 범위별 분포:")
    for min_snr, max_snr in ranges:
        count = np.sum((snr_values >= min_snr) & (snr_values < max_snr))
        percentage = count / len(snr_values) * 100
        print(f"  - {min_snr}-{max_snr} dB: {count:,}개 ({percentage:.1f}%)")
    
    return {
        'data': radar_data,
        'snr_values': snr_values,
        'name': dataset_name,
        'file_path': file_path
    }

def plot_snr_distributions(radar_datasets):
    """SNR 분포 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Radar SNR Distribution Analysis', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # 1. 히스토그램 비교
    ax1 = axes[0, 0]
    for i, dataset in enumerate(radar_datasets):
        if dataset is None:
            continue
        ax1.hist(dataset['snr_values'], bins=50, alpha=0.7, 
                label=f"{dataset['name']} (n={len(dataset['snr_values']):,})",
                color=colors[i], density=True)
    
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Density')
    ax1.set_title('SNR Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 박스플롯
    ax2 = axes[0, 1]
    valid_datasets = [d for d in radar_datasets if d is not None]
    if valid_datasets:
        snr_data = [d['snr_values'] for d in valid_datasets]
        labels = [d['name'] for d in valid_datasets]
        
        box_plot = ax2.boxplot(snr_data, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('SNR (dB)')
    ax2.set_title('SNR Distribution Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. 누적 분포 함수 (CDF)
    ax3 = axes[1, 0]
    for i, dataset in enumerate(radar_datasets):
        if dataset is None:
            continue
        sorted_snr = np.sort(dataset['snr_values'])
        y = np.arange(1, len(sorted_snr) + 1) / len(sorted_snr)
        ax3.plot(sorted_snr, y, label=dataset['name'], color=colors[i], linewidth=2)
    
    # 주요 임계값 표시
    thresholds = [15, 17, 20, 25]
    for thresh in thresholds:
        ax3.axvline(thresh, color='gray', linestyle='--', alpha=0.5)
        ax3.text(thresh, 0.1, f'{thresh}dB', rotation=90, alpha=0.7)
    
    ax3.set_xlabel('SNR (dB)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function (CDF)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. SNR 임계값별 데이터 보존율
    ax4 = axes[1, 1]
    thresholds = np.arange(15, 30, 0.5)
    
    for i, dataset in enumerate(radar_datasets):
        if dataset is None:
            continue
        retention_rates = []
        for thresh in thresholds:
            retention = np.sum(dataset['snr_values'] >= thresh) / len(dataset['snr_values']) * 100
            retention_rates.append(retention)
        
        ax4.plot(thresholds, retention_rates, label=dataset['name'], 
                color=colors[i], linewidth=2, marker='o', markersize=3)
    
    # 현재 임계값 표시
    ax4.axvline(20.0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax4.text(20.2, 50, 'Current\nThreshold\n(20dB)', color='red', fontweight='bold')
    
    ax4.set_xlabel('SNR Threshold (dB)')
    ax4.set_ylabel('Data Retention Rate (%)')
    ax4.set_title('Data Retention vs SNR Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('radar_snr_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def recommend_snr_threshold(radar_datasets):
    """SNR 임계값 추천"""
    print(f"\n🎯 SNR 임계값 추천 분석")
    print("=" * 50)
    
    for dataset in radar_datasets:
        if dataset is None:
            continue
            
        print(f"\n📊 {dataset['name']} 데이터셋:")
        snr_values = dataset['snr_values']
        
        # 다양한 임계값에서의 데이터 보존율
        thresholds = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 25.0]
        
        print("SNR 임계값별 데이터 보존율:")
        for thresh in thresholds:
            retention = np.sum(snr_values >= thresh) / len(snr_values) * 100
            lost = 100 - retention
            print(f"  SNR >= {thresh:4.1f}dB: {retention:5.1f}% 보존, {lost:5.1f}% 손실")
        
        # 추천 임계값 (75% 데이터 보존 기준)
        target_retention = 75.0
        for thresh in np.arange(15.0, 25.0, 0.1):
            retention = np.sum(snr_values >= thresh) / len(snr_values) * 100
            if retention <= target_retention:
                recommended_thresh = thresh - 0.1
                break
        else:
            recommended_thresh = 15.0
        
        final_retention = np.sum(snr_values >= recommended_thresh) / len(snr_values) * 100
        print(f"\n💡 추천 임계값: {recommended_thresh:.1f}dB (데이터 {final_retention:.1f}% 보존)")
        
        # 현재 임계값과 비교
        current_retention = np.sum(snr_values >= 20.0) / len(snr_values) * 100
        print(f"📈 현재 20.0dB vs 추천 {recommended_thresh:.1f}dB:")
        print(f"   - 현재: {current_retention:.1f}% 보존")
        print(f"   - 추천: {final_retention:.1f}% 보존")
        print(f"   - 개선: +{final_retention - current_retention:.1f}% 더 많은 데이터 활용")

if __name__ == '__main__':
    print("📊 레이더 SNR 분포 분석 시작...")
    
    # 레이더 파일들
    radar_files = [
        ('RadarMap_v2.txt', 'RadarMap_v2'),
        ('RadarMap_bokdo3_v6.txt', 'RadarMap_bokdo3_v6')
    ]
    
    # 각 파일 분석
    radar_datasets = []
    for file_path, name in radar_files:
        dataset = analyze_radar_snr(file_path, name)
        radar_datasets.append(dataset)
    
    # 시각화
    plot_snr_distributions(radar_datasets)
    
    # 임계값 추천
    recommend_snr_threshold(radar_datasets)
    
    print(f"\n✅ 분석 완료! 결과는 'radar_snr_distribution_analysis.png'에 저장되었습니다.")
