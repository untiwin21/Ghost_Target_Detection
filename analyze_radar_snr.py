"""
레이더 데이터 SNR 분포 분석
"""
import numpy as np
import matplotlib.pyplot as plt

def analyze_radar_snr(file_path, dataset_name):
    """레이더 파일의 SNR 분포 분석"""
    snr_values = []
    
    print(f"\n=== {dataset_name} 분석 ===")
    print(f"파일: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 빈 줄 건너뛰기
                    continue
                    
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        snr = float(parts[4])  # SNR은 5번째 컬럼
                        snr_values.append(snr)
                    except ValueError:
                        print(f"Warning: Line {line_num}에서 SNR 파싱 오류: {line}")
                        continue
    
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다: {file_path}")
        return None
    
    if not snr_values:
        print("Error: SNR 데이터를 찾을 수 없습니다.")
        return None
    
    snr_array = np.array(snr_values)
    
    # 기본 통계
    print(f"📊 기본 통계:")
    print(f"  • 총 포인트 수: {len(snr_array):,}개")
    print(f"  • 평균 SNR: {np.mean(snr_array):.2f} dB")
    print(f"  • 표준편차: {np.std(snr_array):.2f} dB")
    print(f"  • 최소값: {np.min(snr_array):.1f} dB")
    print(f"  • 최대값: {np.max(snr_array):.1f} dB")
    print(f"  • 중앙값: {np.median(snr_array):.2f} dB")
    
    # 분위수
    print(f"\n📈 분위수:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(snr_array, p)
        print(f"  • {p}%: {value:.2f} dB")
    
    # 임계값별 데이터 보존율
    print(f"\n🎯 임계값별 데이터 보존율:")
    thresholds = [10, 15, 17.5, 20, 25, 30]
    for threshold in thresholds:
        preserved = np.sum(snr_array >= threshold)
        percentage = preserved / len(snr_array) * 100
        print(f"  • SNR ≥ {threshold:4.1f}dB: {preserved:6,}개 ({percentage:5.1f}%)")
    
    return snr_array

def create_comparison_plot(snr_v2, snr_bokdo3):
    """두 데이터셋 SNR 분포 비교 시각화"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. RadarMap_v2 히스토그램
    ax1.hist(snr_v2, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(snr_v2), color='red', linestyle='--', linewidth=2, 
                label=f'평균: {np.mean(snr_v2):.1f}dB')
    ax1.axvline(20, color='orange', linestyle='--', linewidth=2, label='20dB 임계값')
    ax1.axvline(17.5, color='green', linestyle='--', linewidth=2, label='17.5dB 임계값')
    ax1.axvline(10, color='purple', linestyle='--', linewidth=2, label='10dB 임계값')
    ax1.set_title('RadarMap_v2.txt SNR 분포', fontsize=14)
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('빈도')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. RadarMap_bokdo3_v6 히스토그램
    ax2.hist(snr_bokdo3, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(snr_bokdo3), color='red', linestyle='--', linewidth=2, 
                label=f'평균: {np.mean(snr_bokdo3):.1f}dB')
    ax2.axvline(20, color='orange', linestyle='--', linewidth=2, label='20dB 임계값')
    ax2.axvline(17.5, color='green', linestyle='--', linewidth=2, label='17.5dB 임계값')
    ax2.axvline(10, color='purple', linestyle='--', linewidth=2, label='10dB 임계값')
    ax2.set_title('RadarMap_bokdo3_v6.txt SNR 분포', fontsize=14)
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('빈도')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 박스플롯 비교
    ax3.boxplot([snr_v2, snr_bokdo3], labels=['RadarMap_v2', 'RadarMap_bokdo3_v6'])
    ax3.axhline(20, color='orange', linestyle='--', alpha=0.7, label='20dB')
    ax3.axhline(17.5, color='green', linestyle='--', alpha=0.7, label='17.5dB')
    ax3.axhline(10, color='purple', linestyle='--', alpha=0.7, label='10dB')
    ax3.set_title('SNR 분포 박스플롯 비교', fontsize=14)
    ax3.set_ylabel('SNR (dB)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 누적 분포 함수 (CDF)
    sorted_v2 = np.sort(snr_v2)
    sorted_bokdo3 = np.sort(snr_bokdo3)
    y_v2 = np.arange(1, len(sorted_v2) + 1) / len(sorted_v2) * 100
    y_bokdo3 = np.arange(1, len(sorted_bokdo3) + 1) / len(sorted_bokdo3) * 100
    
    ax4.plot(sorted_v2, y_v2, label='RadarMap_v2', linewidth=2)
    ax4.plot(sorted_bokdo3, y_bokdo3, label='RadarMap_bokdo3_v6', linewidth=2)
    ax4.axvline(20, color='orange', linestyle='--', alpha=0.7, label='20dB')
    ax4.axvline(17.5, color='green', linestyle='--', alpha=0.7, label='17.5dB')
    ax4.axvline(10, color='purple', linestyle='--', alpha=0.7, label='10dB')
    ax4.set_title('누적 분포 함수 (CDF)', fontsize=14)
    ax4.set_xlabel('SNR (dB)')
    ax4.set_ylabel('누적 백분율 (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('radar_snr_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 시각화 저장: radar_snr_distribution_comparison.png")

def main():
    print("🎯 레이더 데이터 SNR 분포 분석")
    print("=" * 50)
    
    # RadarMap_v2.txt 분석
    snr_v2 = analyze_radar_snr(
        "/mnt/c/Users/user/Desktop/modified_ragnnarok/SNR_10dB_training/RadarMap_v2.txt",
        "RadarMap_v2 (학습용)"
    )
    
    # RadarMap_bokdo3_v6.txt 분석
    snr_bokdo3 = analyze_radar_snr(
        "/mnt/c/Users/user/Desktop/modified_ragnnarok/SNR_10dB_inference/RadarMap_bokdo3_v6.txt",
        "RadarMap_bokdo3_v6 (추론용)"
    )
    
    if snr_v2 is not None and snr_bokdo3 is not None:
        # 비교 분석
        print(f"\n🔍 두 데이터셋 비교:")
        print(f"  • v2 평균: {np.mean(snr_v2):.2f}dB vs bokdo3 평균: {np.mean(snr_bokdo3):.2f}dB")
        print(f"  • v2 표준편차: {np.std(snr_v2):.2f}dB vs bokdo3 표준편차: {np.std(snr_bokdo3):.2f}dB")
        print(f"  • v2 범위: {np.min(snr_v2):.1f}~{np.max(snr_v2):.1f}dB vs bokdo3 범위: {np.min(snr_bokdo3):.1f}~{np.max(snr_bokdo3):.1f}dB")
        
        # 임계값별 비교
        print(f"\n📊 임계값별 보존율 비교:")
        thresholds = [10, 17.5, 20]
        for threshold in thresholds:
            v2_preserved = np.sum(snr_v2 >= threshold) / len(snr_v2) * 100
            bokdo3_preserved = np.sum(snr_bokdo3 >= threshold) / len(snr_bokdo3) * 100
            print(f"  • SNR ≥ {threshold}dB: v2({v2_preserved:.1f}%) vs bokdo3({bokdo3_preserved:.1f}%)")
        
        # 시각화 생성
        create_comparison_plot(snr_v2, snr_bokdo3)
        
        # 권장사항
        print(f"\n💡 권장사항:")
        print(f"  • RadarMap_v2 (학습용): 평균 {np.mean(snr_v2):.1f}dB → SNR 17.5~20dB 적절")
        print(f"  • RadarMap_bokdo3_v6 (추론용): 평균 {np.mean(snr_bokdo3):.1f}dB → SNR 10~15dB 적절")
        print(f"  • 도메인 차이: {abs(np.mean(snr_v2) - np.mean(snr_bokdo3)):.1f}dB 차이로 도메인 적응 필요")

if __name__ == "__main__":
    main()
