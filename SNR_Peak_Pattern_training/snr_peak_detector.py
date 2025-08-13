"""
SNR 봉우리 패턴 기반 유리벽 감지 시스템
SNR이 봉우리 형태로 나타나면 유리벽으로 인식하여 고스트 처리
"""
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import os

from data_structures import RadarPoint, RadarFrame
from hybrid_ghost_gnn import HybridGhostGNN, create_graph_data

class SNRPeakPatternDataset:
    """SNR 봉우리 패턴 기반 유리벽 감지 데이터셋"""
    
    def __init__(self, 
                 radar_data_path: str,
                 lidar_data_path: str,
                 distance_threshold: float = 0.4,
                 snr_threshold: float = 15.0,
                 k: int = 5,
                 peak_detection_distance: float = 2.0,  # 봉우리 감지 범위
                 min_points_per_frame: int = 5):
        
        self.radar_data_path = radar_data_path
        self.lidar_data_path = lidar_data_path
        self.distance_threshold = distance_threshold
        self.snr_threshold = snr_threshold
        self.k = k
        self.peak_detection_distance = peak_detection_distance
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
    
    def _detect_snr_peaks(self, radar_points):
        """SNR 봉우리 패턴 감지"""
        if len(radar_points) < 5:
            return []
        
        # 레이더 포인트를 거리 순으로 정렬
        points_with_range = [(np.sqrt(p.x**2 + p.y**2), p.rcs, i, p) for i, p in enumerate(radar_points)]
        points_with_range.sort(key=lambda x: x[0])  # 거리순 정렬
        
        ranges = [p[0] for p in points_with_range]
        snrs = [p[1] for p in points_with_range]
        indices = [p[2] for p in points_with_range]
        
        # SNR 봉우리 감지
        glass_wall_indices = []
        
        # 이동 평균으로 스무딩
        if len(snrs) >= 5:
            window_size = min(5, len(snrs))
            smoothed_snrs = np.convolve(snrs, np.ones(window_size)/window_size, mode='same')
            
            # 봉우리 감지 (prominence 기반)
            peaks, properties = find_peaks(smoothed_snrs, 
                                         prominence=5,  # 최소 5dB 돌출
                                         distance=3)    # 최소 3포인트 간격
            
            # 감지된 봉우리 주변 포인트들을 유리벽 후보로 분류
            for peak_idx in peaks:
                # 봉우리 주변 포인트들 확인
                start_idx = max(0, peak_idx - 2)
                end_idx = min(len(indices), peak_idx + 3)
                
                for i in range(start_idx, end_idx):
                    original_idx = indices[i]
                    glass_wall_indices.append(original_idx)
        
        return glass_wall_indices
    
    def _detect_spatial_snr_peaks(self, radar_points):
        """공간적 SNR 봉우리 패턴 감지 (2D)"""
        glass_wall_indices = []
        
        if len(radar_points) < 10:
            return glass_wall_indices
        
        # 공간적 클러스터링
        positions = np.array([[p.x, p.y] for p in radar_points])
        snrs = np.array([p.rcs for p in radar_points])
        
        # DBSCAN으로 공간적 클러스터 형성
        clustering = DBSCAN(eps=1.0, min_samples=3).fit(positions)
        labels = clustering.labels_
        
        # 각 클러스터에서 SNR 패턴 분석
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # 노이즈 제외
                continue
            
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < 3:
                continue
            
            cluster_snrs = snrs[cluster_indices]
            
            # 클러스터 내 SNR 분산이 높으면 (봉우리 패턴) 유리벽 후보
            snr_std = np.std(cluster_snrs)
            snr_max = np.max(cluster_snrs)
            snr_min = np.min(cluster_snrs)
            
            # 봉우리 패턴 조건:
            # 1. 높은 SNR 분산 (다양한 반사 강도)
            # 2. 최대-최소 SNR 차이가 큼
            # 3. 평균 SNR이 높음 (강한 반사)
            if (snr_std > 8 and 
                (snr_max - snr_min) > 15 and 
                np.mean(cluster_snrs) > 20):
                
                glass_wall_indices.extend(cluster_indices.tolist())
        
        return glass_wall_indices
    
    def _process_data(self):
        """SNR 봉우리 패턴 기반 데이터 처리 및 라벨링"""
        print("SNR 봉우리 패턴 기반 라벨링 중...")
        
        total_frames = len(self.radar_frames)
        total_nodes = 0
        total_real = 0
        total_ghost = 0
        glass_wall_detected = 0
        glass_wall_points = 0
        
        for i, radar_frame in enumerate(self.radar_frames):
            if i % 500 == 0:
                print(f"처리 중: {i}/{total_frames}")
            
            if len(radar_frame) < self.min_points_per_frame:
                continue
            
            # 해당하는 LiDAR 프레임 찾기
            lidar_frame = self.lidar_frames[i] if i < len(self.lidar_frames) else []
            
            # SNR 봉우리 패턴 기반 라벨링
            labels = self._label_points_peak_aware(radar_frame, lidar_frame)
            
            # 유리벽 감지 통계
            glass_indices = self._detect_spatial_snr_peaks(radar_frame)
            if glass_indices:
                glass_wall_detected += 1
                glass_wall_points += len(glass_indices)
            
            # 그래프 데이터 생성
            graph_data = create_graph_data(radar_frame, labels, k=self.k)
            
            if graph_data.x.size(0) > 0:
                self.processed_data.append(graph_data)
                total_nodes += len(labels)
                total_real += sum(labels)
                total_ghost += len(labels) - sum(labels)
        
        print(f"\n=== SNR 봉우리 패턴 기반 데이터셋 통계 ===")
        print(f"📏 거리 임계값: {self.distance_threshold}m")
        print(f"📡 SNR 임계값: {self.snr_threshold}dB")
        print(f"🔗 k-NN 연결: {self.k}개")
        print(f"🏔️ 봉우리 감지 범위: {self.peak_detection_distance}m")
        print(f"총 그래프: {len(self.processed_data)}개")
        print(f"총 노드: {total_nodes}개")
        print(f"실제 타겟: {total_real} ({total_real/total_nodes*100:.1f}%)")
        print(f"고스트 타겟: {total_ghost} ({total_ghost/total_nodes*100:.1f}%)")
        print(f"\n🪟 유리벽 감지 통계:")
        print(f"유리벽 패턴 감지 프레임: {glass_wall_detected}개")
        print(f"유리벽 후보 포인트: {glass_wall_points}개")
    
    def _label_points_peak_aware(self, radar_points, lidar_frame):
        """SNR 봉우리 패턴 인식 기반 라벨링"""
        if not lidar_frame:
            return [0] * len(radar_points)
        
        radar_positions = np.array([[p.x, p.y] for p in radar_points])
        lidar_positions = np.array(lidar_frame)
        snr_values = np.array([p.rcs for p in radar_points])
        
        # 거리 계산
        distances = cdist(radar_positions, lidar_positions)
        min_distances = np.min(distances, axis=1)
        
        # 유리벽 후보 포인트 감지
        glass_wall_indices = self._detect_spatial_snr_peaks(radar_points)
        
        # 라벨링
        labels = []
        for i, (distance, snr) in enumerate(zip(min_distances, snr_values)):
            # 기본 조건 확인
            basic_condition = distance <= self.distance_threshold and snr >= self.snr_threshold
            
            # 유리벽 패턴 확인
            is_glass_wall = i in glass_wall_indices
            
            if basic_condition and not is_glass_wall:
                labels.append(1)  # 실제 타겟
            else:
                labels.append(0)  # 고스트 타겟 (유리벽 포함)
        
        return labels
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

def create_peak_pattern_visualization(dataset, save_path="snr_peak_pattern_analysis.png"):
    """SNR 봉우리 패턴 분석 시각화"""
    print("SNR 봉우리 패턴 분석 시각화 생성 중...")
    
    # 데이터 수집
    all_radar_x, all_radar_y = [], []
    all_lidar_x, all_lidar_y = [], []
    all_real_x, all_real_y = [], []
    all_ghost_x, all_ghost_y = [], []
    glass_wall_x, glass_wall_y = [], []
    all_snr_values = []
    
    # 원본 데이터에서 수집
    for i, radar_frame in enumerate(dataset.radar_frames[:50]):  # 처음 50프레임만
        lidar_frame = dataset.lidar_frames[i] if i < len(dataset.lidar_frames) else []
        
        # LiDAR 포인트
        if lidar_frame:
            lidar_x = [p[0] for p in lidar_frame]
            lidar_y = [p[1] for p in lidar_frame]
            all_lidar_x.extend(lidar_x)
            all_lidar_y.extend(lidar_y)
        
        # 유리벽 후보 감지
        glass_indices = dataset._detect_spatial_snr_peaks(radar_frame)
        
        # 레이더 포인트와 라벨
        labels = dataset._label_points_peak_aware(radar_frame, lidar_frame)
        
        for j, (point, label) in enumerate(zip(radar_frame, labels)):
            all_radar_x.append(point.x)
            all_radar_y.append(point.y)
            all_snr_values.append(point.rcs)
            
            if j in glass_indices:
                glass_wall_x.append(point.x)
                glass_wall_y.append(point.y)
            
            if label == 1:
                all_real_x.append(point.x)
                all_real_y.append(point.y)
            else:
                all_ghost_x.append(point.x)
                all_ghost_y.append(point.y)
    
    # 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. SNR 분포 + 유리벽 감지
    if all_lidar_x:
        ax1.scatter(all_lidar_x, all_lidar_y, c='blue', s=1, alpha=0.3, label=f'LiDAR ({len(all_lidar_x)})')
    
    scatter1 = ax1.scatter(all_radar_x, all_radar_y, c=all_snr_values, s=15, 
                          cmap='viridis', alpha=0.7, label=f'Radar ({len(all_radar_x)})')
    
    if glass_wall_x:
        ax1.scatter(glass_wall_x, glass_wall_y, c='red', s=50, alpha=0.8, 
                   marker='x', label=f'Glass Wall Pattern ({len(glass_wall_x)})')
    
    ax1.set_title('1. SNR Distribution + Glass Wall Detection', fontsize=14)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='SNR (dB)')
    
    # 2. 봉우리 패턴 기반 라벨링 결과
    if all_real_x:
        ax2.scatter(all_real_x, all_real_y, c='green', s=15, alpha=0.8, 
                   label=f'Real Targets ({len(all_real_x)})')
    if all_ghost_x:
        ax2.scatter(all_ghost_x, all_ghost_y, c='red', s=15, alpha=0.8, 
                   label=f'Ghost Targets ({len(all_ghost_x)})')
    if glass_wall_x:
        ax2.scatter(glass_wall_x, glass_wall_y, c='orange', s=30, alpha=1.0, 
                   marker='s', label=f'Glass Wall Ghost ({len(glass_wall_x)})')
    
    if all_lidar_x:
        ax2.scatter(all_lidar_x, all_lidar_y, c='blue', s=1, alpha=0.2, label='LiDAR')
    ax2.set_title('2. Peak Pattern-Aware Labeling', fontsize=14)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. SNR 봉우리 패턴 예시
    # 대표적인 프레임에서 거리별 SNR 패턴 보기
    example_frame = dataset.radar_frames[10] if len(dataset.radar_frames) > 10 else dataset.radar_frames[0]
    points_with_range = [(np.sqrt(p.x**2 + p.y**2), p.rcs) for p in example_frame]
    points_with_range.sort(key=lambda x: x[0])
    
    ranges = [p[0] for p in points_with_range]
    snrs = [p[1] for p in points_with_range]
    
    ax3.plot(ranges, snrs, 'b-o', alpha=0.7, label='SNR vs Range')
    ax3.set_xlabel('Range (m)')
    ax3.set_ylabel('SNR (dB)')
    ax3.set_title('3. SNR Pattern Example (Range vs SNR)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 봉우리 감지 표시
    if len(snrs) >= 5:
        from scipy.signal import find_peaks
        smoothed_snrs = np.convolve(snrs, np.ones(3)/3, mode='same')
        peaks, _ = find_peaks(smoothed_snrs, prominence=3, distance=2)
        if len(peaks) > 0:
            ax3.plot([ranges[i] for i in peaks], [smoothed_snrs[i] for i in peaks], 
                    'ro', markersize=8, label='Detected Peaks')
            ax3.legend()
    
    # 4. 통계 정보
    ax4.axis('off')
    glass_ratio = len(glass_wall_x) / len(all_radar_x) * 100 if all_radar_x else 0
    
    stats_text = f"""
    🏔️ SNR Peak Pattern Analysis
    
    🎯 Detection Strategy:
    • SNR peak detection in spatial clusters
    • High SNR variance indicates glass walls
    • Peak prominence threshold: 5dB
    • Cluster analysis with DBSCAN
    
    📊 Pattern Detection Results:
    • Total radar points: {len(all_radar_x):,}
    • Glass wall patterns: {len(glass_wall_x):,} ({glass_ratio:.1f}%)
    • Real targets: {len(all_real_x):,} ({len(all_real_x)/len(all_radar_x)*100:.1f}%)
    • Ghost targets: {len(all_ghost_x):,} ({len(all_ghost_x)/len(all_radar_x)*100:.1f}%)
    
    🪟 Glass Wall Characteristics:
    • SNR variance > 8dB in clusters
    • Max-Min SNR difference > 15dB
    • Average SNR > 20dB
    • Multiple reflection peaks
    
    💡 Benefits:
    • Automatic glass wall detection
    • Reduces false positives from reflections
    • Physics-based pattern recognition
    • No manual wall mapping needed
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightsteelblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SNR peak pattern visualization saved: {save_path}")

def train_snr_peak_detector():
    """SNR 봉우리 패턴 기반 고스트 탐지기 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 SNR 봉우리 패턴 기반 GPU 학습 시작! 장치: {device}")
    print(f"🏔️ 유리벽 SNR 봉우리 패턴 자동 감지 및 고스트 처리")
    
    # SNR 봉우리 패턴 기반 데이터셋
    dataset = SNRPeakPatternDataset(
        radar_data_path="RadarMap_v2.txt",
        lidar_data_path="LiDARMap_v2.txt",
        distance_threshold=0.4,
        snr_threshold=15.0,
        k=5,
        peak_detection_distance=2.0
    )
    
    # 학습 데이터 시각화
    create_peak_pattern_visualization(dataset)
    
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
    print("🎯 SNR 봉우리 패턴 기반 학습 시작...")
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
                torch.save(model.state_dict(), 'snr_peak_detector.pth')
            
            print(f"Epoch {epoch+1}/50 | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")
    
    print(f"🎉 SNR 봉우리 패턴 기반 학습 완료! 최고 정확도: {best_val_acc:.2f}%")
    return best_val_acc

if __name__ == "__main__":
    train_snr_peak_detector()
