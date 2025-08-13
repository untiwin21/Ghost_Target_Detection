"""
SNR 18dB + k=4 + 거리 0.4m 기준 추론 및 시각화
유리벽 문제 해결 효과 검증
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

from data_structures import RadarPoint, RadarFrame
from hybrid_ghost_gnn import HybridGhostGNN, create_graph_data

class StricterInference:
    def __init__(self, model_path: str, device: str = 'auto'):
        self.model_path = model_path
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.model = HybridGhostGNN(input_dim=6, hidden_dim=128, num_layers=3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Using device: {self.device}")
        print(f"Stricter model loaded from {model_path}")
        print(f"Model criteria: k=4, SNR=18dB, distance=0.4m")
        print(f"🎯 Glass wall problem solving test!")
    
    def load_data(self, radar_path: str, lidar_path: str):
        """데이터 로딩"""
        print("Loading data...")
        
        # 레이더 데이터 로딩
        self.radar_frames = []
        current_frame = []
        current_time = None
        
        with open(radar_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    time = float(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    velocity = float(parts[3])
                    snr = float(parts[4])
                    
                    if current_time is None:
                        current_time = time
                    
                    if abs(time - current_time) < 1e-6:
                        current_frame.append(RadarPoint(x, y, velocity, snr))
                    else:
                        if len(current_frame) >= 5:
                            self.radar_frames.append(current_frame)
                        current_frame = [RadarPoint(x, y, velocity, snr)]
                        current_time = time
        
        if len(current_frame) >= 5:
            self.radar_frames.append(current_frame)
        
        # LiDAR 데이터 로딩
        self.lidar_frames = []
        current_frame = []
        current_time = None
        
        with open(lidar_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    time = float(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    
                    if current_time is None:
                        current_time = time
                    
                    if abs(time - current_time) < 1e-6:
                        current_frame.append((x, y))
                    else:
                        self.lidar_frames.append(current_frame)
                        current_frame = [(x, y)]
                        current_time = time
        
        if current_frame:
            self.lidar_frames.append(current_frame)
        
        print(f"Loaded {len(self.radar_frames)} radar frames")
        print(f"Loaded {len(self.lidar_frames)} lidar frames")
    
    def run_inference(self):
        """추론 실행 (k=4 사용)"""
        print("Running stricter inference...")
        
        self.results = []
        
        for i, radar_frame in enumerate(self.radar_frames):
            if i % 100 == 0:
                print(f"Processing frame {i+1}/{len(self.radar_frames)}")
            
            if len(radar_frame) < 5:
                continue
            
            # 그래프 데이터 생성 (k=4 사용)
            graph_data = create_graph_data(radar_frame, k=4)
            graph_data = graph_data.to(self.device)
            
            # 추론
            with torch.no_grad():
                predictions = self.model(graph_data)
                probabilities = torch.sigmoid(predictions).cpu().numpy().flatten()
            
            # 결과 저장
            frame_result = {
                'radar_points': radar_frame,
                'lidar_points': self.lidar_frames[i] if i < len(self.lidar_frames) else [],
                'predictions': probabilities,
                'frame_idx': i
            }
            self.results.append(frame_result)
        
        print(f"Stricter inference completed for {len(self.results)} frames")
    
    def visualize_all_frames(self, save_path: str = "stricter_all_frames.png"):
        """모든 프레임을 하나의 그림에 시각화"""
        print("Creating stricter visualization for all frames...")
        
        # 모든 데이터 수집
        all_radar_x, all_radar_y = [], []
        all_lidar_x, all_lidar_y = [], []
        all_real_x, all_real_y = [], []
        all_ghost_x, all_ghost_y = [], []
        all_snr_values = []
        all_probabilities = []
        
        for result in self.results:
            radar_points = result['radar_points']
            lidar_points = result['lidar_points']
            predictions = result['predictions']
            
            # LiDAR 포인트
            if lidar_points:
                lidar_x = [p[0] for p in lidar_points]
                lidar_y = [p[1] for p in lidar_points]
                all_lidar_x.extend(lidar_x)
                all_lidar_y.extend(lidar_y)
            
            # 레이더 포인트 분류
            for j, (point, prob) in enumerate(zip(radar_points, predictions)):
                all_radar_x.append(point.x)
                all_radar_y.append(point.y)
                all_snr_values.append(point.rcs)
                all_probabilities.append(prob)
                
                if prob > 0.5:  # 실제 타겟
                    all_real_x.append(point.x)
                    all_real_y.append(point.y)
                else:  # 고스트 타겟
                    all_ghost_x.append(point.x)
                    all_ghost_y.append(point.y)
        
        # 4개 서브플롯 생성
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 원본 데이터 (LiDAR + Radar with SNR)
        if all_lidar_x:
            ax1.scatter(all_lidar_x, all_lidar_y, c='blue', s=1, alpha=0.3, label=f'LiDAR ({len(all_lidar_x)})')
        scatter1 = ax1.scatter(all_radar_x, all_radar_y, c=all_snr_values, s=20, 
                              cmap='viridis', alpha=0.8, label=f'Radar ({len(all_radar_x)})')
        ax1.set_title('1. Original Data: LiDAR + Radar (SNR colored)', fontsize=14)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='SNR (dB)')
        
        # 2. 엄격한 모델 예측 결과
        if all_real_x:
            ax2.scatter(all_real_x, all_real_y, c='green', s=20, alpha=0.8, 
                       label=f'Real Targets ({len(all_real_x)})')
        if all_ghost_x:
            ax2.scatter(all_ghost_x, all_ghost_y, c='red', s=20, alpha=0.8, 
                       label=f'Ghost Targets ({len(all_ghost_x)})')
        if all_lidar_x:
            ax2.scatter(all_lidar_x, all_lidar_y, c='blue', s=1, alpha=0.2, label='LiDAR')
        ax2.set_title('2. Stricter Model Predictions (k=4, 18dB)', fontsize=14)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 예측 확률 히트맵
        scatter3 = ax3.scatter(all_radar_x, all_radar_y, c=all_probabilities, s=20, 
                              cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1)
        ax3.set_title('3. Stricter Prediction Probabilities', fontsize=14)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='Real Target Probability')
        
        # 4. 통계 정보
        ax4.axis('off')
        stats_text = f"""
        📊 Stricter Model Results (k=4, 18dB)
        
        🎯 Glass Wall Problem Test:
        • Total Radar Points: {len(all_radar_x):,}
        • Real Targets: {len(all_real_x):,} ({len(all_real_x)/len(all_radar_x)*100:.1f}%)
        • Ghost Targets: {len(all_ghost_x):,} ({len(all_ghost_x)/len(all_radar_x)*100:.1f}%)
        
        📈 Data Statistics:
        • Total Frames: {len(self.results)}
        • LiDAR Points: {len(all_lidar_x):,}
        • Avg Probability: {np.mean(all_probabilities):.3f}
        • SNR Range: {min(all_snr_values):.1f} - {max(all_snr_values):.1f} dB
        • Avg SNR: {np.mean(all_snr_values):.1f} dB
        
        🎯 Stricter Criteria Effects:
        • SNR threshold: 18dB (vs 15dB balanced)
        • k-NN connections: 4 (reduced influence)
        • Distance threshold: 0.4m (balanced)
        
        🪟 Glass Wall Solution:
        • Higher SNR threshold rejects weak reflections
        • Reduced k-NN prevents neighbor bias
        • Should improve glass wall ghost detection
        
        🚀 Device: {self.device}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightpink", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stricter all frames visualization saved: {save_path}")
        
        return len(all_real_x), len(all_ghost_x), np.mean(all_probabilities)
    
    def visualize_radar_focused(self, save_path: str = "stricter_radar_focused.png", margin: float = 5.0):
        """레이더 포인트 기준으로 확대된 엄격한 시각화"""
        print("Creating stricter radar-focused visualization...")
        
        # 모든 레이더 포인트 수집
        all_radar_x, all_radar_y = [], []
        all_real_x, all_real_y = [], []
        all_ghost_x, all_ghost_y = [], []
        all_snr_values = []
        all_probabilities = []
        
        for result in self.results:
            radar_points = result['radar_points']
            predictions = result['predictions']
            
            for point, prob in zip(radar_points, predictions):
                all_radar_x.append(point.x)
                all_radar_y.append(point.y)
                all_snr_values.append(point.rcs)
                all_probabilities.append(prob)
                
                if prob > 0.5:
                    all_real_x.append(point.x)
                    all_real_y.append(point.y)
                else:
                    all_ghost_x.append(point.x)
                    all_ghost_y.append(point.y)
        
        # 레이더 포인트 범위 계산
        if all_radar_x:
            x_min, x_max = min(all_radar_x) - margin, max(all_radar_x) + margin
            y_min, y_max = min(all_radar_y) - margin, max(all_radar_y) + margin
        else:
            x_min, x_max, y_min, y_max = -10, 10, -10, 10
        
        # 해당 범위 내의 LiDAR 포인트만 필터링
        filtered_lidar_x, filtered_lidar_y = [], []
        for result in self.results:
            lidar_points = result['lidar_points']
            for lx, ly in lidar_points:
                if x_min <= lx <= x_max and y_min <= ly <= y_max:
                    filtered_lidar_x.append(lx)
                    filtered_lidar_y.append(ly)
        
        # 시각화
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 레이더 중심 원본 데이터
        if filtered_lidar_x:
            ax1.scatter(filtered_lidar_x, filtered_lidar_y, c='blue', s=2, alpha=0.4, 
                       label=f'LiDAR in range ({len(filtered_lidar_x)})')
        scatter1 = ax1.scatter(all_radar_x, all_radar_y, c=all_snr_values, s=30, 
                              cmap='viridis', alpha=0.9, label=f'Radar ({len(all_radar_x)})')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_title('1. Stricter Radar-Focused: Original Data', fontsize=14)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='SNR (dB)')
        
        # 2. 레이더 중심 예측 결과
        if all_real_x:
            ax2.scatter(all_real_x, all_real_y, c='green', s=30, alpha=0.9, 
                       label=f'Real Targets ({len(all_real_x)})')
        if all_ghost_x:
            ax2.scatter(all_ghost_x, all_ghost_y, c='red', s=30, alpha=0.9, 
                       label=f'Ghost Targets ({len(all_ghost_x)})')
        if filtered_lidar_x:
            ax2.scatter(filtered_lidar_x, filtered_lidar_y, c='blue', s=2, alpha=0.3, label='LiDAR')
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_title('2. Stricter Radar-Focused: Model Predictions', fontsize=14)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 레이더 중심 확률 히트맵
        scatter3 = ax3.scatter(all_radar_x, all_radar_y, c=all_probabilities, s=30, 
                              cmap='RdYlGn', alpha=0.9, vmin=0, vmax=1)
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.set_title('3. Stricter Radar-Focused: Prediction Probabilities', fontsize=14)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='Real Target Probability')
        
        # 4. 범위 정보
        ax4.axis('off')
        range_text = f"""
        🎯 Stricter Radar-Focused View
        
        📏 View Range:
        • X: {x_min:.1f} to {x_max:.1f} m
        • Y: {y_min:.1f} to {y_max:.1f} m
        • Margin: {margin} m
        
        📊 Glass Wall Test Results:
        • Radar Points: {len(all_radar_x)} (all)
        • LiDAR in range: {len(filtered_lidar_x):,}
        • Real Targets: {len(all_real_x)} ({len(all_real_x)/len(all_radar_x)*100:.1f}%)
        • Ghost Targets: {len(all_ghost_x)} ({len(all_ghost_x)/len(all_radar_x)*100:.1f}%)
        
        🎯 Stricter Criteria:
        • SNR threshold: 18dB (higher than 15dB)
        • k-NN connections: 4 (reduced from 5/8)
        • Distance threshold: 0.4m (balanced)
        
        🪟 Glass Wall Solution Strategy:
        • Higher SNR rejects weak glass reflections
        • Fewer neighbors reduce false positives
        • Should better distinguish real vs ghost
        
        🚀 Model: Stricter k=4, 18dB, 0.4m
        """
        ax4.text(0.1, 0.9, range_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightpink", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stricter radar-focused visualization saved: {save_path}")

def main():
    # 추론 실행
    inferencer = StricterInference(
        model_path="ghost_detector_18dB_k4.pth",
        device='auto'
    )
    
    # 데이터 로드
    inferencer.load_data(
        radar_path="RadarMap_bokdo3_v6.txt",
        lidar_path="LiDARMap_bokdo3_v6.txt"
    )
    
    # 추론 실행
    inferencer.run_inference()
    
    # 시각화 생성
    print("\n" + "="*50)
    print("🎨 Creating stricter visualizations...")
    
    # 1. 모든 프레임 시각화
    real_count, ghost_count, avg_prob = inferencer.visualize_all_frames()
    
    # 2. 레이더 중심 확대 시각화
    inferencer.visualize_radar_focused()
    
    print("\n" + "="*50)
    print("✅ Stricter visualization completed!")
    print(f"📊 Results: {real_count} real, {ghost_count} ghost targets")
    print(f"📈 Average probability: {avg_prob:.3f}")
    print("🖼️  Generated files:")
    print("   - stricter_all_frames.png")
    print("   - stricter_radar_focused.png")
    print("\n🪟 Glass wall problem test completed!")

if __name__ == "__main__":
    main()
