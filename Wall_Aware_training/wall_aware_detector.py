"""
벽면 인식 기반 적응적 고스트 탐지 시스템
유리벽 영역에서 다른 기준 적용
"""
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import os

from data_structures import RadarPoint, RadarFrame
from hybrid_ghost_gnn import HybridGhostGNN, create_graph_data

class WallAwareGhostDetectorDataset:
    """벽면 인식 기반 적응적 라벨링 데이터셋"""
    
    def __init__(self, 
                 radar_data_path: str,
                 lidar_data_path: str,
                 distance_threshold: float = 0.4,
                 snr_threshold: float = 15.0,
                 wall_snr_threshold: float = 25.0,  # 벽면 근처 더 엄격한 SNR
                 k: int = 5,
                 min_points_per_frame: int = 5):
        
        self.radar_data_path = radar_data_path
        self.lidar_data_path = lidar_data_path
        self.distance_threshold = distance_threshold
        self.snr_threshold = snr_threshold
        self.wall_snr_threshold = wall_snr_threshold
        self.k = k
        self.min_points_per_frame = min_points_per_frame
        
        self.radar_frames = []
        self.lidar_frames = []
        self.processed_data = []
        self.wall_segments = []
        
        self._load_data()
        self._detect_walls()
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
    
    def _detect_walls(self):
        """LiDAR 데이터로부터 벽면 감지"""
        print("벽면 감지 중...")
        
        # 모든 LiDAR 포인트 수집
        all_lidar = []
        for frame in self.lidar_frames:
            all_lidar.extend(frame)
        
        if not all_lidar:
            return
        
        lidar_array = np.array(all_lidar)
        
        # DBSCAN 클러스터링으로 벽면 감지
        clustering = DBSCAN(eps=0.5, min_samples=20).fit(lidar_array)
        labels = clustering.labels_
        
        unique_labels = set(labels)
        self.wall_segments = []
        
        for label in unique_labels:
            if label == -1:  # 노이즈 제외
                continue
            
            cluster_points = lidar_array[labels == label]
            if len(cluster_points) > 100:  # 충분히 큰 클러스터만
                # 벽면의 방향성 분석
                x_range = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
                y_range = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
                
                # 직선성 분석
                if x_range > y_range * 2:  # 수평 벽면
                    wall_type = 'horizontal'
                    wall_pos = np.mean(cluster_points[:, 1])
                    wall_range = (np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0]))
                elif y_range > x_range * 2:  # 수직 벽면
                    wall_type = 'vertical'
                    wall_pos = np.mean(cluster_points[:, 0])
                    wall_range = (np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1]))
                else:
                    continue  # 명확하지 않은 벽면 제외
                
                self.wall_segments.append({
                    'type': wall_type,
                    'position': wall_pos,
                    'range': wall_range,
                    'points': cluster_points
                })
        
        print(f"감지된 벽면: {len(self.wall_segments)}개")
        for i, wall in enumerate(self.wall_segments):
            print(f"  벽면 {i+1}: {wall['type']} at {wall['position']:.1f}m")
    
    def _is_near_wall(self, x, y, threshold=1.0):
        """레이더 포인트가 벽면 근처인지 판단"""
        for wall in self.wall_segments:
            if wall['type'] == 'vertical':
                if wall['range'][0] <= y <= wall['range'][1]:
                    if abs(x - wall['position']) <= threshold:
                        return True
            else:  # horizontal
                if wall['range'][0] <= x <= wall['range'][1]:
                    if abs(y - wall['position']) <= threshold:
                        return True
        return False
    
    def _process_data(self):
        """벽면 인식 기반 데이터 처리 및 라벨링"""
        print("벽면 인식 기반 라벨링 중...")
        
        total_frames = len(self.radar_frames)
        total_nodes = 0
        total_real = 0
        total_ghost = 0
        wall_area_points = 0
        wall_area_real = 0
        
        for i, radar_frame in enumerate(self.radar_frames):
            if i % 500 == 0:
                print(f"처리 중: {i}/{total_frames}")
            
            if len(radar_frame) < self.min_points_per_frame:
                continue
            
            # 해당하는 LiDAR 프레임 찾기
            lidar_frame = self.lidar_frames[i] if i < len(self.lidar_frames) else []
            
            # 벽면 인식 기반 라벨링
            labels = self._label_points_wall_aware(radar_frame, lidar_frame)
            
            # 벽면 영역 통계
            for j, point in enumerate(radar_frame):
                if self._is_near_wall(point.x, point.y):
                    wall_area_points += 1
                    if labels[j] == 1:
                        wall_area_real += 1
            
            # 그래프 데이터 생성
            graph_data = create_graph_data(radar_frame, labels, k=self.k)
            
            if graph_data.x.size(0) > 0:
                self.processed_data.append(graph_data)
                total_nodes += len(labels)
                total_real += sum(labels)
                total_ghost += len(labels) - sum(labels)
        
        print(f"\n=== 벽면 인식 기반 데이터셋 통계 ===")
        print(f"📏 거리 임계값: {self.distance_threshold}m")
        print(f"📡 일반 SNR 임계값: {self.snr_threshold}dB")
        print(f"🧱 벽면 SNR 임계값: {self.wall_snr_threshold}dB (더 엄격)")
        print(f"🔗 k-NN 연결: {self.k}개")
        print(f"총 그래프: {len(self.processed_data)}개")
        print(f"총 노드: {total_nodes}개")
        print(f"실제 타겟: {total_real} ({total_real/total_nodes*100:.1f}%)")
        print(f"고스트 타겟: {total_ghost} ({total_ghost/total_nodes*100:.1f}%)")
        print(f"\n🧱 벽면 영역 분석:")
        print(f"벽면 근처 포인트: {wall_area_points}개")
        print(f"벽면 근처 실제 타겟: {wall_area_real}개 ({wall_area_real/wall_area_points*100:.1f}%)")
    
    def _label_points_wall_aware(self, radar_points, lidar_frame):
        """벽면 인식 기반 적응적 라벨링"""
        if not lidar_frame:
            return [0] * len(radar_points)
        
        radar_positions = np.array([[p.x, p.y] for p in radar_points])
        lidar_positions = np.array(lidar_frame)
        snr_values = np.array([p.rcs for p in radar_points])
        
        # 거리 계산
        distances = cdist(radar_positions, lidar_positions)
        min_distances = np.min(distances, axis=1)
        
        # 벽면 인식 기반 적응적 라벨링
        labels = []
        for i, (distance, snr) in enumerate(zip(min_distances, snr_values)):
            point = radar_points[i]
            
            # 벽면 근처인지 확인
            near_wall = self._is_near_wall(point.x, point.y)
            
            # 적응적 SNR 임계값 적용
            if near_wall:
                # 벽면 근처: 더 엄격한 SNR 기준 (유리벽 반사 고려)
                snr_thresh = self.wall_snr_threshold
            else:
                # 일반 영역: 기본 SNR 기준
                snr_thresh = self.snr_threshold
            
            # 라벨링
            if distance <= self.distance_threshold and snr >= snr_thresh:
                labels.append(1)  # 실제 타겟
            else:
                labels.append(0)  # 고스트 타겟
        
        return labels
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

def create_wall_aware_visualization(dataset, save_path="wall_aware_training_analysis.png"):
    """벽면 인식 기반 학습 데이터 시각화"""
    print("벽면 인식 기반 학습 데이터 시각화 생성 중...")
    
    # 데이터 수집
    all_radar_x, all_radar_y = [], []
    all_lidar_x, all_lidar_y = [], []
    all_real_x, all_real_y = [], []
    all_ghost_x, all_ghost_y = [], []
    wall_real_x, wall_real_y = [], []
    wall_ghost_x, wall_ghost_y = [], []
    all_snr_values = []
    
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
        labels = dataset._label_points_wall_aware(radar_frame, lidar_frame)
        
        for j, (point, label) in enumerate(zip(radar_frame, labels)):
            all_radar_x.append(point.x)
            all_radar_y.append(point.y)
            all_snr_values.append(point.rcs)
            
            near_wall = dataset._is_near_wall(point.x, point.y)
            
            if label == 1:
                all_real_x.append(point.x)
                all_real_y.append(point.y)
                if near_wall:
                    wall_real_x.append(point.x)
                    wall_real_y.append(point.y)
            else:
                all_ghost_x.append(point.x)
                all_ghost_y.append(point.y)
                if near_wall:
                    wall_ghost_x.append(point.x)
                    wall_ghost_y.append(point.y)
    
    # 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 벽면 감지 결과
    if all_lidar_x:
        ax1.scatter(all_lidar_x, all_lidar_y, c='blue', s=1, alpha=0.3, label=f'LiDAR ({len(all_lidar_x)})')
    
    # 벽면 표시
    colors = ['red', 'green', 'orange', 'purple']
    for i, wall in enumerate(dataset.wall_segments):
        color = colors[i % len(colors)]
        ax1.scatter(wall['points'][:, 0], wall['points'][:, 1], 
                   c=color, s=3, alpha=0.8, label=f'Wall {i+1}')
    
    scatter1 = ax1.scatter(all_radar_x, all_radar_y, c=all_snr_values, s=15, 
                          cmap='viridis', alpha=0.7, label=f'Radar ({len(all_radar_x)})')
    ax1.set_title('1. Wall Detection + Radar Data', fontsize=14)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='SNR (dB)')
    
    # 2. 벽면 인식 기반 라벨링 결과
    if all_real_x:
        ax2.scatter(all_real_x, all_real_y, c='green', s=15, alpha=0.8, 
                   label=f'Real Targets ({len(all_real_x)})')
    if all_ghost_x:
        ax2.scatter(all_ghost_x, all_ghost_y, c='red', s=15, alpha=0.8, 
                   label=f'Ghost Targets ({len(all_ghost_x)})')
    if wall_real_x:
        ax2.scatter(wall_real_x, wall_real_y, c='darkgreen', s=25, alpha=1.0, 
                   marker='^', label=f'Wall Real ({len(wall_real_x)})')
    if wall_ghost_x:
        ax2.scatter(wall_ghost_x, wall_ghost_y, c='darkred', s=25, alpha=1.0, 
                   marker='v', label=f'Wall Ghost ({len(wall_ghost_x)})')
    
    if all_lidar_x:
        ax2.scatter(all_lidar_x, all_lidar_y, c='blue', s=1, alpha=0.2, label='LiDAR')
    ax2.set_title('2. Wall-Aware Labeling Results', fontsize=14)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. SNR 임계값 비교
    wall_snr = [point.rcs for frame in dataset.radar_frames[:100] for point in frame 
                if dataset._is_near_wall(point.x, point.y)]
    normal_snr = [point.rcs for frame in dataset.radar_frames[:100] for point in frame 
                  if not dataset._is_near_wall(point.x, point.y)]
    
    if wall_snr:
        ax3.hist(wall_snr, bins=30, alpha=0.7, color='red', label=f'Wall Area SNR ({len(wall_snr)})', density=True)
    if normal_snr:
        ax3.hist(normal_snr, bins=30, alpha=0.7, color='blue', label=f'Normal Area SNR ({len(normal_snr)})', density=True)
    
    ax3.axvline(x=dataset.snr_threshold, color='blue', linestyle='--', linewidth=2, 
                label=f'Normal threshold: {dataset.snr_threshold}dB')
    ax3.axvline(x=dataset.wall_snr_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Wall threshold: {dataset.wall_snr_threshold}dB')
    ax3.set_title('3. SNR Distribution: Wall vs Normal Areas', fontsize=14)
    ax3.set_xlabel('SNR (dB)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 통계 정보
    ax4.axis('off')
    wall_points = len(wall_real_x) + len(wall_ghost_x)
    normal_points = len(all_radar_x) - wall_points
    
    stats_text = f"""
    🧱 Wall-Aware Training Analysis
    
    🎯 Adaptive Criteria:
    • Normal area SNR: {dataset.snr_threshold}dB
    • Wall area SNR: {dataset.wall_snr_threshold}dB (stricter)
    • Distance threshold: {dataset.distance_threshold}m
    • k-NN connections: {dataset.k}
    
    📊 Wall Detection:
    • Detected walls: {len(dataset.wall_segments)}
    • Wall area points: {wall_points:,} ({wall_points/len(all_radar_x)*100:.1f}%)
    • Normal area points: {normal_points:,} ({normal_points/len(all_radar_x)*100:.1f}%)
    
    📈 Labeling Results:
    • Total real targets: {len(all_real_x):,}
    • Wall area real: {len(wall_real_x):,} ({len(wall_real_x)/wall_points*100:.1f}% of wall area)
    • Normal area real: {len(all_real_x)-len(wall_real_x):,}
    
    💡 Glass Wall Strategy:
    • Higher SNR threshold near walls
    • Reduces false positives from reflections
    • Preserves true targets in normal areas
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightpink", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Wall-aware training visualization saved: {save_path}")

def train_wall_aware_detector():
    """벽면 인식 기반 고스트 탐지기 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 벽면 인식 기반 GPU 학습 시작! 장치: {device}")
    print(f"🧱 유리벽 반사 문제 해결을 위한 적응적 학습")
    
    # 벽면 인식 기반 데이터셋
    dataset = WallAwareGhostDetectorDataset(
        radar_data_path="RadarMap_v2.txt",
        lidar_data_path="LiDARMap_v2.txt",
        distance_threshold=0.4,
        snr_threshold=15.0,      # 일반 영역
        wall_snr_threshold=25.0, # 벽면 영역 (더 엄격)
        k=5
    )
    
    # 학습 데이터 시각화
    create_wall_aware_visualization(dataset)
    
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
    print("🎯 벽면 인식 기반 학습 시작...")
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
                torch.save(model.state_dict(), 'wall_aware_detector.pth')
            
            print(f"Epoch {epoch+1}/50 | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")
    
    print(f"🎉 벽면 인식 기반 학습 완료! 최고 정확도: {best_val_acc:.2f}%")
    return best_val_acc

if __name__ == "__main__":
    train_wall_aware_detector()
