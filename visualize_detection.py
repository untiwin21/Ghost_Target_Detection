"""
실제 LiDAR/Radar 데이터와 고스트 탐지 결과 시각화
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from ghost_detector import GhostDetectorDataset
from hybrid_ghost_gnn import HybridGhostGNN

def visualize_detection_results(frame_indices=[0, 1, 2], save_individual=True):
    """실제 데이터와 탐지 결과 시각화"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    print("🎯 실제 데이터 시각화 생성 중...")
    
    for frame_idx in frame_indices:
        if frame_idx >= len(dataset.radar_frames):
            continue
            
        # 원본 데이터
        radar_frame = dataset.radar_frames[frame_idx]
        lidar_frame = dataset.lidar_frames[frame_idx]
        
        # 좌표 추출
        radar_x = [p.x for p in radar_frame]
        radar_y = [p.y for p in radar_frame]
        radar_snr = [p.rcs for p in radar_frame]
        
        lidar_x = [p[0] for p in lidar_frame]
        lidar_y = [p[1] for p in lidar_frame]
        
        # Ground Truth 계산
        radar_positions = np.array([[p.x, p.y] for p in radar_frame])
        lidar_positions = np.array(lidar_frame)
        distances = cdist(radar_positions, lidar_positions)
        min_distances = np.min(distances, axis=1)
        
        # SNR + 거리 조합 Ground Truth
        ground_truth = []
        for i, (distance, snr) in enumerate(zip(min_distances, radar_snr)):
            if distance <= 0.5 and snr >= 20.0:
                ground_truth.append(1)  # 실제
            else:
                ground_truth.append(0)  # 고스트
        
        ground_truth = np.array(ground_truth)
        
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
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 원본 데이터 (LiDAR + Radar)
        axes[0,0].scatter(lidar_x, lidar_y, c='blue', alpha=0.4, s=8, label='LiDAR Points')
        scatter = axes[0,0].scatter(radar_x, radar_y, c=radar_snr, cmap='viridis', 
                                   s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=axes[0,0], label='SNR (dB)')
        axes[0,0].set_title(f'Frame {frame_idx}: Raw Data\nLiDAR: {len(lidar_frame)} points, Radar: {len(radar_frame)} points')
        axes[0,0].set_xlabel('X (m)')
        axes[0,0].set_ylabel('Y (m)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axis('equal')
        
        # 2. Ground Truth (SNR + 거리 조합)
        real_mask = ground_truth == 1
        ghost_mask = ground_truth == 0
        
        axes[0,1].scatter(lidar_x, lidar_y, c='blue', alpha=0.3, s=8, label='LiDAR')
        axes[0,1].scatter(np.array(radar_x)[real_mask], np.array(radar_y)[real_mask], 
                         c='green', s=80, alpha=0.9, edgecolors='black', linewidth=1,
                         label=f'Real Targets ({real_mask.sum()})')
        axes[0,1].scatter(np.array(radar_x)[ghost_mask], np.array(radar_y)[ghost_mask], 
                         c='red', s=80, alpha=0.9, edgecolors='black', linewidth=1,
                         label=f'Ghost Targets ({ghost_mask.sum()})')
        
        axes[0,1].set_title(f'Ground Truth (Distance ≤ 0.5m AND SNR ≥ 20dB)\nReal: {real_mask.sum()}, Ghost: {ghost_mask.sum()}')
        axes[0,1].set_xlabel('X (m)')
        axes[0,1].set_ylabel('Y (m)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axis('equal')
        
        # 3. 모델 예측 결과
        pred_real_mask = predicted_labels == 1
        pred_ghost_mask = predicted_labels == 0
        
        axes[1,0].scatter(lidar_x, lidar_y, c='blue', alpha=0.3, s=8, label='LiDAR')
        axes[1,0].scatter(np.array(radar_x)[pred_real_mask], np.array(radar_y)[pred_real_mask], 
                         c='green', s=80, alpha=0.9, edgecolors='black', linewidth=1,
                         label=f'Predicted Real ({pred_real_mask.sum()})')
        axes[1,0].scatter(np.array(radar_x)[pred_ghost_mask], np.array(radar_y)[pred_ghost_mask], 
                         c='red', s=80, alpha=0.9, edgecolors='black', linewidth=1,
                         label=f'Predicted Ghost ({pred_ghost_mask.sum()})')
        
        # 정확도 계산
        accuracy = (predicted_labels == ground_truth).mean() * 100
        axes[1,0].set_title(f'Model Prediction (Accuracy: {accuracy:.1f}%)\nPred Real: {pred_real_mask.sum()}, Pred Ghost: {pred_ghost_mask.sum()}')
        axes[1,0].set_xlabel('X (m)')
        axes[1,0].set_ylabel('Y (m)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axis('equal')
        
        # 4. 예측 확률 히트맵
        scatter_prob = axes[1,1].scatter(radar_x, radar_y, c=predicted_probs, cmap='RdYlGn', 
                                        s=100, alpha=0.8, edgecolors='black', linewidth=0.5,
                                        vmin=0, vmax=1)
        axes[1,1].scatter(lidar_x, lidar_y, c='blue', alpha=0.2, s=5, label='LiDAR')
        plt.colorbar(scatter_prob, ax=axes[1,1], label='Real Target Probability')
        axes[1,1].set_title(f'Prediction Confidence\n(Green: High confidence Real, Red: High confidence Ghost)')
        axes[1,1].set_xlabel('X (m)')
        axes[1,1].set_ylabel('Y (m)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axis('equal')
        
        plt.tight_layout()
        
        if save_individual:
            filename = f'detection_frame_{frame_idx}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ 저장: {filename}")
        
        # 상세 분석 출력
        print(f"\n📊 프레임 {frame_idx} 상세 분석:")
        print(f"  LiDAR 포인트: {len(lidar_frame)}개")
        print(f"  Radar 포인트: {len(radar_frame)}개")
        print(f"  SNR 범위: {min(radar_snr):.1f} ~ {max(radar_snr):.1f} dB")
        print(f"  거리 범위: {min_distances.min():.3f} ~ {min_distances.max():.3f} m")
        print(f"  Ground Truth - Real: {real_mask.sum()}, Ghost: {ghost_mask.sum()}")
        print(f"  Model Prediction - Real: {pred_real_mask.sum()}, Ghost: {pred_ghost_mask.sum()}")
        print(f"  정확도: {accuracy:.1f}%")
        
        # 혼동 행렬
        tp = ((predicted_labels == 1) & (ground_truth == 1)).sum()
        tn = ((predicted_labels == 0) & (ground_truth == 0)).sum()
        fp = ((predicted_labels == 1) & (ground_truth == 0)).sum()
        fn = ((predicted_labels == 0) & (ground_truth == 1)).sum()
        
        print(f"  혼동 행렬:")
        print(f"    True Positive: {tp} | True Negative: {tn}")
        print(f"    False Positive: {fp} | False Negative: {fn}")
        
        if not save_individual:
            plt.show()
    
    # 전체 요약 시각화
    create_summary_visualization(dataset, model, device)

def create_summary_visualization(dataset, model, device):
    """전체 데이터 요약 시각화"""
    print("\n📈 전체 데이터 요약 분석 중...")
    
    all_snr = []
    all_distances = []
    all_ground_truth = []
    all_predictions = []
    
    # 처음 100개 프레임 분석
    for i in range(min(100, len(dataset.radar_frames))):
        radar_frame = dataset.radar_frames[i]
        lidar_frame = dataset.lidar_frames[i]
        
        if not radar_frame or not lidar_frame:
            continue
        
        # SNR과 거리 계산
        radar_positions = np.array([[p.x, p.y] for p in radar_frame])
        lidar_positions = np.array(lidar_frame)
        snr_values = [p.rcs for p in radar_frame]
        
        distances = cdist(radar_positions, lidar_positions)
        min_distances = np.min(distances, axis=1)
        
        # Ground Truth
        ground_truth = []
        for distance, snr in zip(min_distances, snr_values):
            if distance <= 0.5 and snr >= 20.0:
                ground_truth.append(1)
            else:
                ground_truth.append(0)
        
        # 모델 예측 (가능한 경우)
        if i < len(dataset):
            data = dataset[i].to(device)
            with torch.no_grad():
                predictions = model(data)
                predicted_labels = (predictions.squeeze() > 0.5).int().cpu().numpy()
        else:
            predicted_labels = np.zeros(len(radar_frame))
        
        all_snr.extend(snr_values)
        all_distances.extend(min_distances)
        all_ground_truth.extend(ground_truth)
        all_predictions.extend(predicted_labels)
    
    all_snr = np.array(all_snr)
    all_distances = np.array(all_distances)
    all_ground_truth = np.array(all_ground_truth)
    all_predictions = np.array(all_predictions)
    
    # 요약 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # SNR vs 거리 (Ground Truth)
    real_mask = all_ground_truth == 1
    ghost_mask = all_ground_truth == 0
    
    axes[0,0].scatter(all_distances[ghost_mask], all_snr[ghost_mask], 
                     c='red', alpha=0.6, s=20, label=f'Ghost ({ghost_mask.sum()})')
    axes[0,0].scatter(all_distances[real_mask], all_snr[real_mask], 
                     c='green', alpha=0.8, s=20, label=f'Real ({real_mask.sum()})')
    axes[0,0].axvline(x=0.5, color='blue', linestyle='--', alpha=0.7, label='Distance threshold')
    axes[0,0].axhline(y=20.0, color='orange', linestyle='--', alpha=0.7, label='SNR threshold')
    axes[0,0].set_xlabel('Distance to nearest LiDAR (m)')
    axes[0,0].set_ylabel('SNR (dB)')
    axes[0,0].set_title('Ground Truth: SNR vs Distance')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # SNR vs 거리 (Model Prediction)
    pred_real_mask = all_predictions == 1
    pred_ghost_mask = all_predictions == 0
    
    axes[0,1].scatter(all_distances[pred_ghost_mask], all_snr[pred_ghost_mask], 
                     c='red', alpha=0.6, s=20, label=f'Pred Ghost ({pred_ghost_mask.sum()})')
    axes[0,1].scatter(all_distances[pred_real_mask], all_snr[pred_real_mask], 
                     c='green', alpha=0.8, s=20, label=f'Pred Real ({pred_real_mask.sum()})')
    axes[0,1].axvline(x=0.5, color='blue', linestyle='--', alpha=0.7)
    axes[0,1].axhline(y=20.0, color='orange', linestyle='--', alpha=0.7)
    axes[0,1].set_xlabel('Distance to nearest LiDAR (m)')
    axes[0,1].set_ylabel('SNR (dB)')
    axes[0,1].set_title('Model Prediction: SNR vs Distance')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # SNR 분포
    axes[1,0].hist(all_snr[real_mask], bins=30, alpha=0.7, color='green', label='Real targets')
    axes[1,0].hist(all_snr[ghost_mask], bins=30, alpha=0.7, color='red', label='Ghost targets')
    axes[1,0].axvline(x=20.0, color='orange', linestyle='--', label='SNR threshold')
    axes[1,0].set_xlabel('SNR (dB)')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('SNR Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 거리 분포
    axes[1,1].hist(all_distances[real_mask], bins=30, alpha=0.7, color='green', label='Real targets')
    axes[1,1].hist(all_distances[ghost_mask], bins=30, alpha=0.7, color='red', label='Ghost targets')
    axes[1,1].axvline(x=0.5, color='blue', linestyle='--', label='Distance threshold')
    axes[1,1].set_xlabel('Distance to LiDAR (m)')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Distance Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detection_summary.png', dpi=300, bbox_inches='tight')
    print("✅ 저장: detection_summary.png")
    
    # 통계 출력
    overall_accuracy = (all_predictions == all_ground_truth).mean() * 100
    print(f"\n📊 전체 통계 (100개 프레임):")
    print(f"  총 레이더 포인트: {len(all_snr)}개")
    print(f"  SNR 범위: {all_snr.min():.1f} ~ {all_snr.max():.1f} dB")
    print(f"  거리 범위: {all_distances.min():.3f} ~ {all_distances.max():.3f} m")
    print(f"  Ground Truth - Real: {real_mask.sum()} ({real_mask.mean()*100:.1f}%)")
    print(f"  Model Prediction - Real: {pred_real_mask.sum()} ({pred_real_mask.mean()*100:.1f}%)")
    print(f"  전체 정확도: {overall_accuracy:.1f}%")

if __name__ == '__main__':
    print("🎯 실제 LiDAR/Radar 데이터 시각화")
    print("목표: 실제 데이터에서 무엇을 실제 물체로 인식했는지 시각화")
    
    visualize_detection_results(frame_indices=[0, 1, 2, 5, 10])
    
    print("\n✅ 시각화 완료!")
    print("생성된 파일:")
    print("- detection_frame_*.png: 개별 프레임 분석")
    print("- detection_summary.png: 전체 데이터 요약")
