"""
SNR 18dB + k=4 + 거리 0.4m 기준 고스트 타겟 탐지 시스템
더 엄격한 SNR 기준 + 적은 이웃 영향
"""
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

from data_structures import RadarPoint, RadarFrame
from hybrid_ghost_gnn import HybridGhostGNN, create_graph_data

class StricterGhostDetectorDataset:
    """SNR 18dB + k=4 + 거리 0.4m 더 엄격한 데이터셋"""
    
    def __init__(self, 
                 radar_data_path: str,
                 lidar_data_path: str,
                 distance_threshold: float = 0.4,
                 snr_threshold: float = 18.0,  # 더 엄격한 SNR
                 k: int = 4,                   # 적은 이웃 영향
                 min_points_per_frame: int = 5):
        
        self.radar_data_path = radar_data_path
        self.lidar_data_path = lidar_data_path
        self.distance_threshold = distance_threshold
        self.snr_threshold = snr_threshold
        self.k = k
        self.min_points_per_frame = min_points_per_frame
        
        self.radar_frames = []
        self.lidar_frames = []
        self.processed_data = []
        
        self._load_data()
        self._process_data()
    
    def _load_data(self):
        """데이터 로딩"""
        print("데이터 로딩 중...")
        
        # 레이더 데이터 로딩
        self.radar_frames = []
        current_frame = []
        current_time = None
        
        with open(self.radar_data_path, 'r') as f:
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
                        if len(current_frame) >= self.min_points_per_frame:
                            self.radar_frames.append(current_frame)
                        current_frame = [RadarPoint(x, y, velocity, snr)]
                        current_time = time
        
        if len(current_frame) >= self.min_points_per_frame:
            self.radar_frames.append(current_frame)
        
        print(f"레이더 프레임: {len(self.radar_frames)}개")
        
        # LiDAR 데이터 로딩
        self.lidar_frames = []
        current_frame = []
        current_time = None
        
        with open(self.lidar_data_path, 'r') as f:
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
        
        print(f"LiDAR 프레임: {len(self.lidar_frames)}개")
    
    def _process_data(self):
        """데이터 처리 및 라벨링"""
        print("더 엄격한 기준으로 SNR + 거리 조합 라벨링 중...")
        
        total_frames = len(self.radar_frames)
        total_nodes = 0
        total_real = 0
        total_ghost = 0
        
        for i, radar_frame in enumerate(self.radar_frames):
            if i % 500 == 0:
                print(f"처리 중: {i}/{total_frames}")
            
            if len(radar_frame) < self.min_points_per_frame:
                continue
            
            # 해당하는 LiDAR 프레임 찾기
            lidar_frame = self.lidar_frames[i] if i < len(self.lidar_frames) else []
            
            # 더 엄격한 기준으로 라벨링
            labels = self._label_points_stricter(radar_frame, lidar_frame)
            
            # 그래프 데이터 생성 (k=4 사용)
            graph_data = create_graph_data(radar_frame, labels, k=self.k)
            
            if graph_data.x.size(0) > 0:
                self.processed_data.append(graph_data)
                total_nodes += len(labels)
                total_real += sum(labels)
                total_ghost += len(labels) - sum(labels)
        
        print(f"\n=== 더 엄격한 기준 데이터셋 통계 ===")
        print(f"📏 거리 임계값: {self.distance_threshold}m")
        print(f"📡 SNR 임계값: {self.snr_threshold}dB (더 엄격)")
        print(f"🔗 k-NN 연결: {self.k}개 (적은 이웃 영향)")
        print(f"총 그래프: {len(self.processed_data)}개")
        print(f"총 노드: {total_nodes}개")
        print(f"실제 타겟: {total_real} ({total_real/total_nodes*100:.1f}%)")
        print(f"고스트 타겟: {total_ghost} ({total_ghost/total_nodes*100:.1f}%)")
    
    def _label_points_stricter(self, radar_points, lidar_frame):
        """더 엄격한 SNR + 거리 조합 라벨링"""
        if not lidar_frame:
            return [0] * len(radar_points)
        
        radar_positions = np.array([[p.x, p.y] for p in radar_points])
        lidar_positions = np.array(lidar_frame)
        snr_values = np.array([p.rcs for p in radar_points])
        
        # 거리 계산
        distances = cdist(radar_positions, lidar_positions)
        min_distances = np.min(distances, axis=1)
        
        # 더 엄격한 SNR + 거리 조합 라벨링
        labels = []
        for distance, snr in zip(min_distances, snr_values):
            if distance <= self.distance_threshold and snr >= self.snr_threshold:
                labels.append(1)  # 실제 타겟
            else:
                labels.append(0)  # 고스트 타겟
        
        return labels
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

def create_stricter_training_visualization(dataset, save_path="stricter_training_analysis.png"):
    """더 엄격한 기준 학습 데이터 시각화"""
    print("더 엄격한 기준 학습 데이터 시각화 생성 중...")
    
    # 데이터 수집
    all_radar_x, all_radar_y = [], []
    all_lidar_x, all_lidar_y = [], []
    all_real_x, all_real_y = [], []
    all_ghost_x, all_ghost_y = [], []
    all_snr_values = []
    all_distances = []
    
    # 원본 데이터에서 수집
    for i, radar_frame in enumerate(dataset.radar_frames[:100]):  # 처음 100프레임만
        lidar_frame = dataset.lidar_frames[i] if i < len(dataset.lidar_frames) else []
        
        # LiDAR 포인트
        if lidar_frame:
            lidar_x = [p[0] for p in lidar_frame]
            lidar_y = [p[1] for p in lidar_frame]
            all_lidar_x.extend(lidar_x)
            all_lidar_y.extend(lidar_y)
        
        # 레이더 포인트와 라벨
        labels = dataset._label_points_stricter(radar_frame, lidar_frame)
        
        # 거리 계산
        if lidar_frame:
            radar_positions = np.array([[p.x, p.y] for p in radar_frame])
            lidar_positions = np.array(lidar_frame)
            distances = cdist(radar_positions, lidar_positions)
            min_distances = np.min(distances, axis=1)
        else:
            min_distances = [float('inf')] * len(radar_frame)
        
        for j, (point, label, dist) in enumerate(zip(radar_frame, labels, min_distances)):
            all_radar_x.append(point.x)
            all_radar_y.append(point.y)
            all_snr_values.append(point.rcs)
            all_distances.append(dist)
            
            if label == 1:
                all_real_x.append(point.x)
                all_real_y.append(point.y)
            else:
                all_ghost_x.append(point.x)
                all_ghost_y.append(point.y)
    
    # 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 원본 데이터
    if all_lidar_x:
        ax1.scatter(all_lidar_x, all_lidar_y, c='blue', s=1, alpha=0.3, label=f'LiDAR ({len(all_lidar_x)})')
    scatter1 = ax1.scatter(all_radar_x, all_radar_y, c=all_snr_values, s=20, 
                          cmap='viridis', alpha=0.8, label=f'Radar ({len(all_radar_x)})')
    ax1.set_title('1. Training Data: LiDAR + Radar (SNR colored)', fontsize=14)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='SNR (dB)')
    
    # 2. 더 엄격한 기준 라벨링 결과
    if all_real_x:
        ax2.scatter(all_real_x, all_real_y, c='green', s=20, alpha=0.8, 
                   label=f'Real Targets ({len(all_real_x)})')
    if all_ghost_x:
        ax2.scatter(all_ghost_x, all_ghost_y, c='red', s=20, alpha=0.8, 
                   label=f'Ghost Targets ({len(all_ghost_x)})')
    if all_lidar_x:
        ax2.scatter(all_lidar_x, all_lidar_y, c='blue', s=1, alpha=0.2, label='LiDAR')
    ax2.set_title('2. Stricter Labeling (k=4, 18dB, 0.4m)', fontsize=14)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. SNR 분포 비교
    ax3.hist(all_snr_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=18, color='red', linestyle='--', linewidth=2, label='18dB threshold')
    ax3.axvline(x=15, color='orange', linestyle='--', linewidth=2, label='15dB (previous)')
    ax3.axvline(x=20, color='green', linestyle='--', linewidth=2, label='20dB (original)')
    ax3.set_title('3. SNR Distribution with Thresholds', fontsize=14)
    ax3.set_xlabel('SNR (dB)')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 통계 정보
    ax4.axis('off')
    real_ratio = len(all_real_x) / len(all_radar_x) * 100 if all_radar_x else 0
    ghost_ratio = len(all_ghost_x) / len(all_radar_x) * 100 if all_radar_x else 0
    avg_snr = np.mean(all_snr_values) if all_snr_values else 0
    avg_dist = np.mean([d for d in all_distances if d != float('inf')]) if all_distances else 0
    
    stats_text = f"""
    📊 Stricter Training Data Analysis
    
    🎯 Stricter Criteria:
    • Distance threshold: 0.4m (balanced)
    • k-NN connections: 4 (reduced influence)
    • SNR threshold: 18.0 dB (stricter)
    
    📈 Labeling Results:
    • Total radar points: {len(all_radar_x):,}
    • Real targets: {len(all_real_x):,} ({real_ratio:.1f}%)
    • Ghost targets: {len(all_ghost_x):,} ({ghost_ratio:.1f}%)
    
    📊 Data Statistics:
    • Average SNR: {avg_snr:.1f} dB
    • Average distance: {avg_dist:.3f} m
    • LiDAR points: {len(all_lidar_x):,}
    
    🎯 Expected Benefits:
    • Higher precision (fewer false positives)
    • Better glass wall rejection
    • Reduced neighbor over-influence
    • More conservative predictions
    
    ⚖️ Trade-offs:
    • Lower recall (may miss some real targets)
    • Stricter than domain-adapted 15dB
    • Between 15dB and 20dB thresholds
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Stricter training visualization saved: {save_path}")
    
    return len(all_real_x), len(all_ghost_x)

def train_stricter_ghost_detector():
    """더 엄격한 기준의 고스트 탐지기 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 더 엄격한 기준 GPU 학습 시작! 장치: {device}")
    print(f"📡 SNR 임계값: 18dB (15dB와 20dB 사이)")
    print(f"🔗 k-NN 연결: 4개 (이웃 영향 감소)")
    print(f"📏 거리 임계값: 0.4m (균형점)")
    
    # 더 엄격한 기준 데이터셋
    dataset = StricterGhostDetectorDataset(
        radar_data_path="RadarMap_v2.txt",
        lidar_data_path="LiDARMap_v2.txt",
        distance_threshold=0.4,
        snr_threshold=18.0,  # 더 엄격
        k=4                  # 적은 이웃
    )
    
    # 학습 데이터 시각화
    real_count, ghost_count = create_stricter_training_visualization(dataset)
    
    # 데이터 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 모델 초기화
    model = HybridGhostGNN(input_dim=6, hidden_dim=128, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 학습
    print("🎯 더 엄격한 기준으로 학습 시작...")
    best_val_acc = 0
    
    for epoch in range(50):
        # 훈련
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(torch.sigmoid(out.squeeze()), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 검증
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    pred = (torch.sigmoid(out.squeeze()) > 0.5).float()
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            
            val_acc = correct / total * 100
            scheduler.step(total_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'ghost_detector_18dB_k4.pth')
            
            print(f"Epoch {epoch+1}/50 | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")
    
    print(f"🎉 더 엄격한 기준 학습 완료! 최고 정확도: {best_val_acc:.2f}%")
    return best_val_acc

if __name__ == "__main__":
    train_stricter_ghost_detector()
