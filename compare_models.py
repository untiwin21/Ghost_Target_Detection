"""
기존 모델 vs 개선된 모델 비교 추론 및 시각화
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

from data_structures import RadarPoint, RadarFrame
from hybrid_ghost_gnn import HybridGhostGNN, create_graph_data

class ModelComparison:
    """기존 모델과 개선된 모델 비교"""
    
    def __init__(self, 
                 original_model_path: str = 'ghost_detector.pth',
                 improved_model_path: str = 'ghost_detector_improved.pth',
                 radar_data_path: str = 'RadarMap_v2.txt',
                 lidar_data_path: str = 'LiDARMap_v2.txt'):
        
        self.original_model_path = original_model_path
        self.improved_model_path = improved_model_path
        self.radar_data_path = radar_data_path
        self.lidar_data_path = lidar_data_path
        
        # GPU 사용 가능 여부 확인
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 비교 추론 장치: {self.device}")
        
        # 모델들 로드
        self.original_model = self.load_model(original_model_path, "기존 모델")
        self.improved_model = self.load_model(improved_model_path, "개선된 모델")
        
        # 데이터 로드
        self.frames = self.load_data()
        
    def load_model(self, model_path, model_name):
        """모델 로드"""
        if not os.path.exists(model_path):
            print(f"⚠️ {model_name} 파일이 없습니다: {model_path}")
            return None
            
        model = HybridGhostGNN(input_dim=6, hidden_dim=128)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"✅ {model_name} 로드 완료: {model_path}")
        return model
    
    def load_data(self):
        """데이터 로딩"""
        print("📊 데이터 로딩 중...")
        
        # 레이더 데이터 로딩
        radar_frames = []
        current_frame = []
        current_time = None
        
        with open(self.radar_data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    time = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    velocity = float(parts[3])
                    snr = float(parts[4])
                    
                    if current_time is None:
                        current_time = time
                    
                    if abs(time - current_time) > 0.01:
                        if current_frame:
                            radar_frames.append(current_frame)
                        current_frame = []
                        current_time = time
                    
                    current_frame.append(RadarPoint(x, y, 0.0, snr))
            
            if current_frame:
                radar_frames.append(current_frame)
        
        # LiDAR 데이터 로딩
        lidar_frames = []
        current_frame = []
        current_time = None
        
        with open(self.lidar_data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    time = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    
                    if current_time is None:
                        current_time = time
                    
                    if abs(time - current_time) > 0.01:
                        if current_frame:
                            lidar_frames.append(current_frame)
                        current_frame = []
                        current_time = time
                    
                    current_frame.append((x, y))
            
            if current_frame:
                lidar_frames.append(current_frame)
        
        # 프레임 결합
        frames = []
        min_frames = min(len(radar_frames), len(lidar_frames))
        
        for i in range(min_frames):
            if len(radar_frames[i]) >= 5:
                frames.append({
                    'radar': radar_frames[i],
                    'lidar': lidar_frames[i],
                    'frame_id': i
                })
        
        print(f"✅ 유효한 프레임: {len(frames)}개")
        return frames
    
    def predict_with_model(self, model, radar_points):
        """모델로 예측"""
        if model is None or len(radar_points) < 5:
            return []
        
        graph_data = create_graph_data(radar_points, k=8)
        graph_data = graph_data.to(self.device)
        
        with torch.no_grad():
            predictions = model(graph_data)
            probabilities = predictions.squeeze().cpu().numpy()
        
        return probabilities
    
    def create_ground_truth(self, radar_points, lidar_points, snr_threshold, distance_threshold=0.5):
        """Ground Truth 생성"""
        if not lidar_points:
            return [0] * len(radar_points)
        
        radar_positions = np.array([[p.x, p.y] for p in radar_points])
        lidar_positions = np.array(lidar_points)
        snr_values = np.array([p.rcs for p in radar_points])
        
        distances = cdist(radar_positions, lidar_positions)
        min_distances = np.min(distances, axis=1)
        
        labels = []
        for distance, snr in zip(min_distances, snr_values):
            if distance <= distance_threshold and snr >= snr_threshold:
                labels.append(1)
            else:
                labels.append(0)
        
        return labels
    
    def compare_models(self, num_frames=100):
        """모델 비교 및 시각화"""
        print(f"🔍 모델 비교 중... ({num_frames}개 프레임)")
        
        # 데이터 수집
        all_radar_points = []
        all_lidar_points = []
        original_predictions = []
        improved_predictions = []
        original_gt = []
        improved_gt = []
        
        for i, frame in enumerate(self.frames[:num_frames]):
            radar_points = frame['radar']
            lidar_points = frame['lidar']
            
            # 예측
            orig_pred = self.predict_with_model(self.original_model, radar_points)
            impr_pred = self.predict_with_model(self.improved_model, radar_points)
            
            if len(orig_pred) == 0 or len(impr_pred) == 0:
                continue
            
            # Ground Truth (각각 다른 임계값)
            orig_gt = self.create_ground_truth(radar_points, lidar_points, 20.0)  # 기존
            impr_gt = self.create_ground_truth(radar_points, lidar_points, 17.5)  # 개선
            
            # 데이터 수집
            all_radar_points.extend(radar_points)
            all_lidar_points.extend(lidar_points)
            original_predictions.extend(orig_pred)
            improved_predictions.extend(impr_pred)
            original_gt.extend(orig_gt)
            improved_gt.extend(impr_gt)
        
        # 배열 변환
        original_predictions = np.array(original_predictions)
        improved_predictions = np.array(improved_predictions)
        original_gt = np.array(original_gt)
        improved_gt = np.array(improved_gt)
        
        radar_positions = np.array([[p.x, p.y] for p in all_radar_points])
        radar_snrs = np.array([p.rcs for p in all_radar_points])
        lidar_positions = np.array(all_lidar_points)
        
        # 시각화
        self.visualize_comparison(
            radar_positions, radar_snrs, lidar_positions,
            original_predictions, improved_predictions,
            original_gt, improved_gt
        )
        
        # 성능 비교
        return self.calculate_performance_metrics(
            original_predictions, improved_predictions,
            original_gt, improved_gt
        )
    
    def visualize_comparison(self, radar_pos, radar_snrs, lidar_pos, 
                           orig_pred, impr_pred, orig_gt, impr_gt):
        """비교 시각화"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Original Model vs Improved Model Comparison', fontsize=18, fontweight='bold')
        
        # 1. 원본 데이터
        ax1 = axes[0, 0]
        if len(lidar_pos) > 0:
            ax1.scatter(lidar_pos[:, 0], lidar_pos[:, 1], c='lightblue', s=1, alpha=0.4, label=f'LiDAR ({len(lidar_pos):,})')
        scatter = ax1.scatter(radar_pos[:, 0], radar_pos[:, 1], c=radar_snrs, s=15, cmap='viridis', alpha=0.8)
        plt.colorbar(scatter, ax=ax1, label='SNR (dB)')
        ax1.set_title('Original Data: LiDAR + Radar')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 기존 모델 Ground Truth (SNR ≥ 20dB)
        ax2 = axes[0, 1]
        orig_real = orig_gt == 1
        orig_ghost = ~orig_real
        if np.sum(orig_ghost) > 0:
            ax2.scatter(radar_pos[orig_ghost, 0], radar_pos[orig_ghost, 1], c='red', s=15, alpha=0.7, label=f'Ghost ({np.sum(orig_ghost):,})')
        if np.sum(orig_real) > 0:
            ax2.scatter(radar_pos[orig_real, 0], radar_pos[orig_real, 1], c='green', s=15, alpha=0.7, label=f'Real ({np.sum(orig_real):,})')
        ax2.set_title('Original GT (SNR ≥ 20dB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 개선된 모델 Ground Truth (SNR ≥ 17.5dB)
        ax3 = axes[0, 2]
        impr_real = impr_gt == 1
        impr_ghost = ~impr_real
        if np.sum(impr_ghost) > 0:
            ax3.scatter(radar_pos[impr_ghost, 0], radar_pos[impr_ghost, 1], c='red', s=15, alpha=0.7, label=f'Ghost ({np.sum(impr_ghost):,})')
        if np.sum(impr_real) > 0:
            ax3.scatter(radar_pos[impr_real, 0], radar_pos[impr_real, 1], c='green', s=15, alpha=0.7, label=f'Real ({np.sum(impr_real):,})')
        ax3.set_title('Improved GT (SNR ≥ 17.5dB)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 기존 모델 예측
        ax4 = axes[1, 0]
        orig_pred_real = orig_pred > 0.5
        orig_pred_ghost = ~orig_pred_real
        if np.sum(orig_pred_ghost) > 0:
            ax4.scatter(radar_pos[orig_pred_ghost, 0], radar_pos[orig_pred_ghost, 1], c='red', s=15, alpha=0.7, label=f'Pred Ghost ({np.sum(orig_pred_ghost):,})')
        if np.sum(orig_pred_real) > 0:
            ax4.scatter(radar_pos[orig_pred_real, 0], radar_pos[orig_pred_real, 1], c='green', s=15, alpha=0.7, label=f'Pred Real ({np.sum(orig_pred_real):,})')
        
        orig_acc = np.mean((orig_pred > 0.5) == orig_gt) * 100
        ax4.set_title(f'Original Model Prediction (Acc: {orig_acc:.1f}%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 개선된 모델 예측
        ax5 = axes[1, 1]
        impr_pred_real = impr_pred > 0.5
        impr_pred_ghost = ~impr_pred_real
        if np.sum(impr_pred_ghost) > 0:
            ax5.scatter(radar_pos[impr_pred_ghost, 0], radar_pos[impr_pred_ghost, 1], c='red', s=15, alpha=0.7, label=f'Pred Ghost ({np.sum(impr_pred_ghost):,})')
        if np.sum(impr_pred_real) > 0:
            ax5.scatter(radar_pos[impr_pred_real, 0], radar_pos[impr_pred_real, 1], c='green', s=15, alpha=0.7, label=f'Pred Real ({np.sum(impr_pred_real):,})')
        
        impr_acc = np.mean((impr_pred > 0.5) == impr_gt) * 100
        ax5.set_title(f'Improved Model Prediction (Acc: {impr_acc:.1f}%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Real Target 비교
        ax6 = axes[1, 2]
        if len(lidar_pos) > 0:
            ax6.scatter(lidar_pos[:, 0], lidar_pos[:, 1], c='lightblue', s=1, alpha=0.3, label='LiDAR')
        if np.sum(orig_pred_real) > 0:
            ax6.scatter(radar_pos[orig_pred_real, 0], radar_pos[orig_pred_real, 1], c='blue', s=20, alpha=0.6, label=f'Original Real ({np.sum(orig_pred_real):,})')
        if np.sum(impr_pred_real) > 0:
            ax6.scatter(radar_pos[impr_pred_real, 0], radar_pos[impr_pred_real, 1], c='red', s=15, alpha=0.8, label=f'Improved Real ({np.sum(impr_pred_real):,})')
        ax6.set_title('Real Target Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. 예측 확률 비교 - 기존
        ax7 = axes[2, 0]
        scatter = ax7.scatter(radar_pos[:, 0], radar_pos[:, 1], c=orig_pred, s=15, cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax7, label='Real Probability')
        ax7.set_title('Original Model Confidence')
        ax7.grid(True, alpha=0.3)
        
        # 8. 예측 확률 비교 - 개선
        ax8 = axes[2, 1]
        scatter = ax8.scatter(radar_pos[:, 0], radar_pos[:, 1], c=impr_pred, s=15, cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax8, label='Real Probability')
        ax8.set_title('Improved Model Confidence')
        ax8.grid(True, alpha=0.3)
        
        # 9. 성능 통계
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # 성능 계산
        orig_tp = np.sum((orig_pred > 0.5) & (orig_gt == 1))
        orig_tn = np.sum((orig_pred <= 0.5) & (orig_gt == 0))
        orig_fp = np.sum((orig_pred > 0.5) & (orig_gt == 0))
        orig_fn = np.sum((orig_pred <= 0.5) & (orig_gt == 1))
        
        impr_tp = np.sum((impr_pred > 0.5) & (impr_gt == 1))
        impr_tn = np.sum((impr_pred <= 0.5) & (impr_gt == 0))
        impr_fp = np.sum((impr_pred > 0.5) & (impr_gt == 0))
        impr_fn = np.sum((impr_pred <= 0.5) & (impr_gt == 1))
        
        orig_precision = orig_tp / (orig_tp + orig_fp) if (orig_tp + orig_fp) > 0 else 0
        orig_recall = orig_tp / (orig_tp + orig_fn) if (orig_tp + orig_fn) > 0 else 0
        orig_f1 = 2 * orig_precision * orig_recall / (orig_precision + orig_recall) if (orig_precision + orig_recall) > 0 else 0
        
        impr_precision = impr_tp / (impr_tp + impr_fp) if (impr_tp + impr_fp) > 0 else 0
        impr_recall = impr_tp / (impr_tp + impr_fn) if (impr_tp + impr_fn) > 0 else 0
        impr_f1 = 2 * impr_precision * impr_recall / (impr_precision + impr_recall) if (impr_precision + impr_recall) > 0 else 0
        
        orig_real_retention = np.sum(orig_pred_real) / len(orig_pred) * 100
        impr_real_retention = np.sum(impr_pred_real) / len(impr_pred) * 100
        
        stats_text = f"""
Model Performance Comparison

Original Model (SNR >= 20dB):
• Accuracy: {orig_acc:.1f}%
• Precision: {orig_precision:.3f}
• Recall: {orig_recall:.3f}
• F1-Score: {orig_f1:.3f}
• Real Retention: {orig_real_retention:.1f}%
• Real Targets: {np.sum(orig_pred_real):,}

Improved Model (SNR >= 17.5dB):
• Accuracy: {impr_acc:.1f}%
• Precision: {impr_precision:.3f}
• Recall: {impr_recall:.3f}
• F1-Score: {impr_f1:.3f}
• Real Retention: {impr_real_retention:.1f}%
• Real Targets: {np.sum(impr_pred_real):,}

Improvement:
• Accuracy: {impr_acc - orig_acc:+.1f}%
• F1-Score: {impr_f1 - orig_f1:+.3f}
• Real Retention: {impr_real_retention - orig_real_retention:+.1f}%
• Additional Real Targets: {np.sum(impr_pred_real) - np.sum(orig_pred_real):+,}
        """
        
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_performance_metrics(self, orig_pred, impr_pred, orig_gt, impr_gt):
        """성능 지표 계산"""
        orig_acc = np.mean((orig_pred > 0.5) == orig_gt) * 100
        impr_acc = np.mean((impr_pred > 0.5) == impr_gt) * 100
        
        orig_real_count = np.sum(orig_pred > 0.5)
        impr_real_count = np.sum(impr_pred > 0.5)
        
        return {
            'original': {
                'accuracy': orig_acc,
                'real_targets': orig_real_count
            },
            'improved': {
                'accuracy': impr_acc,
                'real_targets': impr_real_count
            },
            'improvement': {
                'accuracy_gain': impr_acc - orig_acc,
                'additional_real_targets': impr_real_count - orig_real_count
            }
        }

def main():
    """메인 실행 함수"""
    print("🔍 기존 모델 vs 개선된 모델 비교 시작!")
    
    # 모델 비교
    comparison = ModelComparison()
    results = comparison.compare_models(num_frames=100)
    
    print(f"\n📊 비교 결과:")
    print(f"기존 모델 (SNR ≥ 20dB):")
    print(f"  - 정확도: {results['original']['accuracy']:.1f}%")
    print(f"  - Real Targets: {results['original']['real_targets']:,}개")
    
    print(f"\n개선된 모델 (SNR ≥ 17.5dB):")
    print(f"  - 정확도: {results['improved']['accuracy']:.1f}%")
    print(f"  - Real Targets: {results['improved']['real_targets']:,}개")
    
    print(f"\n🎯 개선 효과:")
    print(f"  - 정확도 향상: {results['improvement']['accuracy_gain']:+.1f}%")
    print(f"  - 추가 Real Targets: {results['improvement']['additional_real_targets']:+,}개")
    
    print(f"\n💾 결과 저장: model_comparison_results.png")

if __name__ == '__main__':
    main()
