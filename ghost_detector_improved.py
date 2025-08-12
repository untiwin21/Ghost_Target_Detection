"""
개선된 SNR 임계값을 사용한 고스트 타겟 탐지 시스템
SNR 임계값: 20.0dB → 17.5dB로 변경
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

class ImprovedGhostDetectorDataset:
    """개선된 SNR 임계값을 사용한 데이터셋"""
    
    def __init__(self, 
                 radar_data_path: str,
                 lidar_data_path: str,
                 distance_threshold: float = 0.5,
                 snr_threshold: float = 17.5,  # 20.0 → 17.5로 변경
                 k: int = 8,
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
        
        # 레이더 데이터 로딩: time x y velocity SNR
        self.radar_frames = []
        current_frame = []
        current_time = None
        
        with open(self.radar_data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    time = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    velocity = float(parts[3])  # 사용 안함
                    snr = float(parts[4])
                    
                    if current_time is None:
                        current_time = time
                    
                    if abs(time - current_time) > 0.01:
                        if current_frame:
                            self.radar_frames.append(current_frame)
                        current_frame = []
                        current_time = time
                    
                    current_frame.append(RadarPoint(x, y, 0.0, snr))  # vr=0, rcs=snr
            
            if current_frame:
                self.radar_frames.append(current_frame)
        
        # LiDAR 데이터 로딩: time x y intensity
        self.lidar_frames = []
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
                            self.lidar_frames.append(current_frame)
                        current_frame = []
                        current_time = time
                    
                    current_frame.append((x, y))
            
            if current_frame:
                self.lidar_frames.append(current_frame)
        
        print(f"레이더 프레임: {len(self.radar_frames)}개")
        print(f"LiDAR 프레임: {len(self.lidar_frames)}개")
    
    def _process_data(self):
        """개선된 SNR + 거리 조합 라벨링으로 데이터 처리"""
        print(f"개선된 SNR + 거리 조합 라벨링 중... (SNR >= {self.snr_threshold}dB)")
        
        min_frames = min(len(self.radar_frames), len(self.lidar_frames))
        
        for frame_idx in range(min_frames):
            if frame_idx % 500 == 0:
                print(f"처리 중: {frame_idx}/{min_frames}")
            
            radar_frame = self.radar_frames[frame_idx]
            lidar_frame = self.lidar_frames[frame_idx]
            
            if len(radar_frame) < self.min_points_per_frame:
                continue
            
            labels = self._label_points_combined(radar_frame, lidar_frame)
            graph_data = create_graph_data(radar_frame, labels, k=self.k)
            
            if graph_data.x.size(0) > 0:
                self.processed_data.append(graph_data)
        
        # 통계 출력
        total_nodes = sum(data.x.size(0) for data in self.processed_data)
        total_real = sum(data.y.sum().item() for data in self.processed_data)
        total_ghost = total_nodes - total_real
        
        print(f"\n=== 개선된 데이터셋 통계 (SNR >= {self.snr_threshold}dB + 거리 <= {self.distance_threshold}m) ===")
        print(f"총 그래프: {len(self.processed_data)}개")
        print(f"총 노드: {total_nodes}개")
        print(f"실제 타겟: {total_real} ({total_real/total_nodes*100:.1f}%)")
        print(f"고스트 타겟: {total_ghost} ({total_ghost/total_nodes*100:.1f}%)")
        
        # 기존 20dB와 비교
        print(f"\n📈 개선 효과 예상:")
        print(f"- 기존 20dB: 약 64.5% 데이터 보존")
        print(f"- 개선 {self.snr_threshold}dB: 약 77% 데이터 보존")
        print(f"- 예상 개선: +12.5% 더 많은 실제 타겟 보존")
    
    def _label_points_combined(self, radar_points, lidar_frame):
        """개선된 SNR + 거리 조합 라벨링"""
        if not lidar_frame:
            return [0] * len(radar_points)
        
        radar_positions = np.array([[p.x, p.y] for p in radar_points])
        lidar_positions = np.array(lidar_frame)
        snr_values = np.array([p.rcs for p in radar_points])
        
        # 거리 계산
        distances = cdist(radar_positions, lidar_positions)
        min_distances = np.min(distances, axis=1)
        
        # 개선된 SNR + 거리 조합 라벨링
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

def train_improved_ghost_detector():
    """개선된 고스트 탐지기 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 개선된 모델 학습 시작! 장치: {device}")
    print(f"📊 주요 개선사항: SNR 임계값 20.0dB → 17.5dB")
    
    # 개선된 데이터셋
    dataset = ImprovedGhostDetectorDataset(
        radar_data_path="RadarMap_v2.txt",
        lidar_data_path="LiDARMap_v2.txt",
        distance_threshold=0.5,
        snr_threshold=17.5  # 개선된 임계값
    )
    
    # 데이터 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 모델
    model = HybridGhostGNN(input_dim=6, hidden_dim=128, dropout=0.1)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    print("🎯 개선된 모델 학습 시작...")
    best_val_acc = 0
    
    for epoch in range(50):
        # 학습
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_batches += 1
        
        # 검증
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct_preds, total_nodes = 0, 0
            
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    output = model(data)
                    preds = (output.squeeze() > 0.5).int()
                    correct_preds += (preds == data.y.int()).sum().item()
                    total_nodes += data.num_nodes
            
            val_acc = (correct_preds / total_nodes) * 100
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'ghost_detector_improved.pth')
            
            avg_train_loss = total_train_loss / train_batches
            print(f"Epoch {epoch+1:02d}/50 | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")
    
    print(f"🎉 개선된 모델 학습 완료!")
    print(f"📈 최고 정확도: {best_val_acc:.2f}%")
    print(f"💾 모델 저장: ghost_detector_improved.pth")
    
    return dataset, model

def compare_thresholds():
    """기존 vs 개선된 임계값 비교"""
    print("\n🔍 임계값 비교 분석...")
    
    # 레이더 데이터 로딩
    radar_data = []
    with open("RadarMap_v2.txt", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                snr = float(parts[4])
                radar_data.append(snr)
    
    radar_snrs = np.array(radar_data)
    
    thresholds = [17.5, 18.0, 18.5, 19.0, 19.5, 20.0]
    
    print("임계값별 데이터 보존율 비교:")
    for thresh in thresholds:
        retention = np.sum(radar_snrs >= thresh) / len(radar_snrs) * 100
        lost = 100 - retention
        status = "🆕 개선" if thresh < 20.0 else "📊 기존"
        print(f"  SNR >= {thresh:4.1f}dB: {retention:5.1f}% 보존, {lost:5.1f}% 손실 {status}")

if __name__ == '__main__':
    print("🎯 개선된 SNR 임계값 고스트 탐지 시스템")
    
    # 임계값 비교
    compare_thresholds()
    
    # 개선된 모델 학습
    dataset, model = train_improved_ghost_detector()
