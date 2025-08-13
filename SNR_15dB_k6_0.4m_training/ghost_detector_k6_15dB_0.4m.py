#!/usr/bin/env python3
"""
레이더 고스트 타겟 탐지 시스템 - SNR 15dB + k=6 + 거리 0.4m 학습
Modified RaGNNarok v2.0 - GPU 가속 지원

설정:
- SNR 임계값: 15.0 dB (도메인 적응 균형 기준)
- 거리 임계값: 0.4 m (중간 지점)
- k-NN 연결: 6개 (적절한 이웃 영향)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, BatchNorm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time
import os

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 사용 중인 디바이스: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

@dataclass
class RadarPoint:
    """레이더 포인트 데이터 구조"""
    timestamp: float
    x: float
    y: float
    velocity: float
    snr: float

@dataclass
class LiDARPoint:
    """LiDAR 포인트 데이터 구조"""
    timestamp: float
    x: float
    y: float
    intensity: float

# 타입 정의
RadarFrame = List[RadarPoint]
LiDARFrame = List[LiDARPoint]

class HybridGhostGNN(nn.Module):
    """GraphSAGE 기반 고스트 타겟 탐지 GNN 모델"""
    
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=3, dropout=0.1):
        super(HybridGhostGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE 레이어들
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 첫 번째 레이어
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # 중간 레이어들
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # 출력 레이어
        self.convs.append(SAGEConv(hidden_dim, 1))
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        # 중간 레이어들
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropout_layer(x)
        
        # 출력 레이어
        x = self.convs[-1](x, edge_index)
        x = torch.sigmoid(x)
        
        return x.squeeze()

def extract_features(radar_points: List[RadarPoint]) -> np.ndarray:
    """레이더 포인트에서 6차원 특징 벡터 추출"""
    features = []
    for point in radar_points:
        # 극좌표 변환
        range_val = np.sqrt(point.x**2 + point.y**2)
        azimuth = np.arctan2(point.y, point.x)
        
        # 6차원 특징 벡터: [x, y, range, azimuth, velocity, snr]
        feature_vector = [
            point.x,
            point.y,
            range_val,
            azimuth,
            point.velocity,
            point.snr
        ]
        features.append(feature_vector)
    
    return np.array(features, dtype=np.float32)

def create_knn_edges(positions: np.ndarray, k: int = 6) -> np.ndarray:
    """k-NN 기반 그래프 엣지 생성"""
    n_points = len(positions)
    edges = []
    
    for i in range(n_points):
        # 현재 포인트와 다른 모든 포인트 간의 거리 계산
        distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
        
        # 자기 자신을 제외하고 가장 가까운 k개 포인트 찾기
        nearest_indices = np.argsort(distances)[1:k+1]  # 첫 번째는 자기 자신이므로 제외
        
        # 양방향 엣지 추가
        for j in nearest_indices:
            edges.append([i, j])
            edges.append([j, i])
    
    # 중복 제거
    edges = list(set(tuple(edge) for edge in edges))
    return np.array(edges).T if edges else np.array([[], []])

def load_radar_data(filename: str) -> Dict[float, RadarFrame]:
    """레이더 데이터 로드"""
    radar_frames = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                timestamp = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                velocity = float(parts[3])
                snr = float(parts[4])
                
                point = RadarPoint(timestamp, x, y, velocity, snr)
                
                if timestamp not in radar_frames:
                    radar_frames[timestamp] = []
                radar_frames[timestamp].append(point)
    
    return radar_frames

def load_lidar_data(filename: str) -> Dict[float, LiDARFrame]:
    """LiDAR 데이터 로드"""
    lidar_frames = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                timestamp = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                intensity = float(parts[3])
                
                point = LiDARPoint(timestamp, x, y, intensity)
                
                if timestamp not in lidar_frames:
                    lidar_frames[timestamp] = []
                lidar_frames[timestamp].append(point)
    
    return lidar_frames

def create_labels_snr_distance(radar_frame: RadarFrame, lidar_frame: LiDARFrame, 
                              snr_threshold: float = 15.0, distance_threshold: float = 0.4) -> List[int]:
    """SNR + 거리 조합 기반 라벨 생성 (15dB + 0.4m)"""
    labels = []
    
    # LiDAR 포인트들의 좌표 배열
    lidar_positions = np.array([[p.x, p.y] for p in lidar_frame])
    
    for radar_point in radar_frame:
        radar_pos = np.array([radar_point.x, radar_point.y])
        
        # 가장 가까운 LiDAR 포인트까지의 거리 계산
        if len(lidar_positions) > 0:
            distances = np.sqrt(np.sum((lidar_positions - radar_pos)**2, axis=1))
            min_distance = np.min(distances)
        else:
            min_distance = float('inf')
        
        # SNR + 거리 조합 조건 (15dB + 0.4m)
        if min_distance <= distance_threshold and radar_point.snr >= snr_threshold:
            labels.append(1)  # 실제 타겟
        else:
            labels.append(0)  # 고스트 타겟
    
    return labels

class GhostDetectorDataset:
    """고스트 탐지 데이터셋"""
    
    def __init__(self, radar_file: str, lidar_file: str, k: int = 6):
        self.k = k
        self.radar_frames = load_radar_data(radar_file)
        self.lidar_frames = load_lidar_data(lidar_file)
        self.graphs = []
        self.labels = []
        
        self._create_graphs()
    
    def _create_graphs(self):
        """그래프 데이터 생성"""
        print("📊 그래프 데이터 생성 중...")
        
        total_real_targets = 0
        total_ghost_targets = 0
        
        for timestamp in self.radar_frames:
            if timestamp in self.lidar_frames:
                radar_frame = self.radar_frames[timestamp]
                lidar_frame = self.lidar_frames[timestamp]
                
                if len(radar_frame) < 2:  # 그래프를 만들기에 포인트가 너무 적음
                    continue
                
                # 특징 추출
                features = extract_features(radar_frame)
                
                # 라벨 생성 (SNR 15dB + 거리 0.4m)
                frame_labels = create_labels_snr_distance(radar_frame, lidar_frame, 
                                                        snr_threshold=15.0, distance_threshold=0.4)
                
                # k-NN 엣지 생성
                positions = features[:, :2]  # x, y 좌표만 사용
                edge_index = create_knn_edges(positions, k=self.k)
                
                if edge_index.size > 0:
                    # PyTorch Geometric 데이터 생성
                    graph_data = Data(
                        x=torch.FloatTensor(features),
                        edge_index=torch.LongTensor(edge_index),
                        y=torch.FloatTensor(frame_labels)
                    )
                    
                    self.graphs.append(graph_data)
                    self.labels.extend(frame_labels)
                    
                    # 통계 업데이트
                    total_real_targets += sum(frame_labels)
                    total_ghost_targets += len(frame_labels) - sum(frame_labels)
        
        print(f"✅ 총 {len(self.graphs)}개 그래프 생성 완료")
        print(f"📊 실제 타겟: {total_real_targets:,}개 ({total_real_targets/(total_real_targets+total_ghost_targets)*100:.1f}%)")
        print(f"👻 고스트 타겟: {total_ghost_targets:,}개 ({total_ghost_targets/(total_real_targets+total_ghost_targets)*100:.1f}%)")
        print(f"🔗 k-NN 연결: k={self.k}")

def train_model():
    """모델 학습 실행"""
    print("🚀 레이더 고스트 타겟 탐지 모델 학습 시작")
    print("⚙️  설정: SNR ≥ 15dB + 거리 ≤ 0.4m + k=6")
    
    # 데이터셋 생성
    dataset = GhostDetectorDataset('RadarMap_v2.txt', 'LiDARMap_v2.txt', k=6)
    
    if len(dataset.graphs) == 0:
        print("❌ 그래프 데이터가 없습니다!")
        return
    
    # 학습/검증 분할
    train_graphs, val_graphs = train_test_split(dataset.graphs, test_size=0.2, random_state=42)
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
    
    # 모델 초기화
    model = HybridGhostGNN(input_dim=6, hidden_dim=128, num_layers=3, dropout=0.1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 학습 기록
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 15
    
    print(f"🎯 학습 시작 - 총 {len(train_graphs)}개 학습 그래프, {len(val_graphs)}개 검증 그래프")
    
    for epoch in range(50):
        start_time = time.time()
        
        # 학습 단계
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(outputs, batch.y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 예측 결과 저장
            predictions = (outputs > 0.5).float()
            train_predictions.extend(predictions.cpu().numpy())
            train_targets.extend(batch.y.cpu().numpy())
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(outputs, batch.y)
                
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(batch.y.cpu().numpy())
        
        # 정확도 계산
        train_acc = accuracy_score(train_targets, train_predictions)
        val_acc = accuracy_score(val_targets, val_predictions)
        
        # 기록 저장
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # 학습률 스케줄링
        scheduler.step(val_loss)
        
        # 조기 종료 체크
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 최고 모델 저장
            torch.save(model.state_dict(), 'ghost_detector_k6_15dB_0.4m.pth')
        else:
            patience_counter += 1
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:2d}/50 | "
              f"Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        if patience_counter >= max_patience:
            print(f"🛑 조기 종료 (patience={max_patience})")
            break
    
    print(f"🏆 최고 검증 정확도: {best_val_acc:.4f}")
    
    # 학습 결과 시각화
    create_training_visualization(train_losses, val_losses, train_accuracies, val_accuracies, dataset)

def create_training_visualization(train_losses, val_losses, train_accuracies, val_accuracies, dataset):
    """학습 결과 시각화"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 손실 함수 그래프
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss (k=6, SNR≥15dB, dist≤0.4m)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 정확도 그래프
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 데이터 분포 히스토그램
    all_labels = dataset.labels
    real_count = sum(all_labels)
    ghost_count = len(all_labels) - real_count
    
    categories = ['Real Targets', 'Ghost Targets']
    counts = [real_count, ghost_count]
    colors = ['green', 'red']
    
    bars = ax3.bar(categories, counts, color=colors, alpha=0.7)
    ax3.set_title('Target Distribution (k=6 neighbors)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count')
    
    # 막대 위에 숫자 표시
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}\n({count/len(all_labels)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # 4. 모델 설정 정보
    ax4.axis('off')
    info_text = f"""
    🎯 모델 설정 (k=6 균형 기준)
    
    📊 데이터 통계:
    • 총 그래프: {len(dataset.graphs):,}개
    • 총 노드: {len(all_labels):,}개
    • 실제 타겟: {real_count:,}개 ({real_count/len(all_labels)*100:.1f}%)
    • 고스트 타겟: {ghost_count:,}개 ({ghost_count/len(all_labels)*100:.1f}%)
    
    ⚙️ 하이퍼파라미터:
    • SNR 임계값: 15.0 dB (도메인 적응)
    • 거리 임계값: 0.4 m (중간 지점)
    • k-NN 연결: 6개 (균형잡힌 이웃)
    • 은닉층 차원: 128
    • 드롭아웃: 0.1
    • 학습률: 0.005
    
    🚀 성능:
    • 최고 검증 정확도: {max(val_accuracies):.4f}
    • 디바이스: {device}
    """
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('k6_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 학습 분석 결과가 'k6_training_analysis.png'에 저장되었습니다.")

if __name__ == "__main__":
    train_model()
