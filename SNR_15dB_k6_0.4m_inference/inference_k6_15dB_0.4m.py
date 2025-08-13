#!/usr/bin/env python3
"""
레이더 고스트 타겟 탐지 추론 시스템 - SNR 15dB + k=6 + 거리 0.4m
Modified RaGNNarok v2.0 - GPU 가속 지원

추론 및 시각화:
- 전체 뷰: 모든 데이터 포인트 표시
- 레이더 중심 확대 뷰: 레이더 포인트 중심으로 확대
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import os

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 사용 중인 디바이스: {device}")

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

def load_radar_data(filename: str) -> Dict[float, List[RadarPoint]]:
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

def load_lidar_data(filename: str) -> Dict[float, List[LiDARPoint]]:
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

def run_inference():
    """추론 실행 및 시각화"""
    print("🔍 레이더 고스트 타겟 탐지 추론 시작")
    print("⚙️  설정: SNR ≥ 15dB + k=6 + 거리 0.4m")
    
    # 모델 로드
    model = HybridGhostGNN(input_dim=6, hidden_dim=128, num_layers=3, dropout=0.1).to(device)
    
    # 학습된 가중치 로드
    if os.path.exists('ghost_detector_k6_15dB_0.4m.pth'):
        model.load_state_dict(torch.load('ghost_detector_k6_15dB_0.4m.pth', map_location=device))
        print("✅ 학습된 모델 로드 완료")
    else:
        print("❌ 모델 파일을 찾을 수 없습니다!")
        return
    
    model.eval()
    
    # 데이터 로드
    radar_frames = load_radar_data('RadarMap_v2.txt')
    lidar_frames = load_lidar_data('LiDARMap_v2.txt')
    
    print(f"📊 레이더 프레임: {len(radar_frames)}개")
    print(f"📊 LiDAR 프레임: {len(lidar_frames)}개")
    
    # 모든 데이터 수집
    all_radar_points = []
    all_lidar_points = []
    all_predictions = []
    all_probabilities = []
    
    processed_frames = 0
    
    with torch.no_grad():
        for timestamp in sorted(radar_frames.keys()):
            if timestamp in lidar_frames:
                radar_frame = radar_frames[timestamp]
                lidar_frame = lidar_frames[timestamp]
                
                if len(radar_frame) < 2:
                    continue
                
                # 특징 추출
                features = extract_features(radar_frame)
                
                # k-NN 엣지 생성
                positions = features[:, :2]  # x, y 좌표만 사용
                edge_index = create_knn_edges(positions, k=6)
                
                if edge_index.size > 0:
                    # PyTorch Geometric 데이터 생성
                    graph_data = Data(
                        x=torch.FloatTensor(features),
                        edge_index=torch.LongTensor(edge_index)
                    ).to(device)
                    
                    # 추론 실행
                    outputs = model(graph_data.x, graph_data.edge_index)
                    predictions = (outputs > 0.5).float()
                    
                    # 결과 저장
                    for i, (radar_point, pred, prob) in enumerate(zip(radar_frame, predictions.cpu().numpy(), outputs.cpu().numpy())):
                        all_radar_points.append(radar_point)
                        all_predictions.append(pred)
                        all_probabilities.append(prob)
                    
                    # LiDAR 포인트 저장
                    all_lidar_points.extend(lidar_frame)
                    
                    processed_frames += 1
    
    print(f"✅ {processed_frames}개 프레임 처리 완료")
    print(f"📊 총 레이더 포인트: {len(all_radar_points):,}개")
    print(f"📊 총 LiDAR 포인트: {len(all_lidar_points):,}개")
    
    # 통계 계산
    real_targets = sum(all_predictions)
    ghost_targets = len(all_predictions) - real_targets
    
    print(f"🎯 실제 타겟: {real_targets:,}개 ({real_targets/len(all_predictions)*100:.1f}%)")
    print(f"👻 고스트 타겟: {ghost_targets:,}개 ({ghost_targets/len(all_predictions)*100:.1f}%)")
    
    # 시각화 생성
    create_visualizations(all_radar_points, all_lidar_points, all_predictions, all_probabilities)

def create_visualizations(radar_points, lidar_points, predictions, probabilities):
    """시각화 생성 (전체 뷰 + 레이더 중심 확대 뷰)"""
    
    # 데이터 준비
    radar_x = [p.x for p in radar_points]
    radar_y = [p.y for p in radar_points]
    radar_snr = [p.snr for p in radar_points]
    
    lidar_x = [p.x for p in lidar_points]
    lidar_y = [p.y for p in lidar_points]
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # 실제/고스트 타겟 분리
    real_mask = predictions == 1
    ghost_mask = predictions == 0
    
    # 1. 전체 뷰 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1-1. 원본 데이터 (LiDAR + Radar SNR)
    scatter1 = ax1.scatter(lidar_x, lidar_y, c='blue', s=1, alpha=0.3, label='LiDAR')
    scatter2 = ax1.scatter(radar_x, radar_y, c=radar_snr, s=20, cmap='viridis', alpha=0.8, label='Radar')
    ax1.set_title('Original Data: LiDAR + Radar (SNR colored)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax1, label='SNR (dB)')
    
    # 1-2. 예측 결과 (실제 vs 고스트)
    ax2.scatter(lidar_x, lidar_y, c='blue', s=1, alpha=0.3, label='LiDAR')
    ax2.scatter(np.array(radar_x)[real_mask], np.array(radar_y)[real_mask], 
               c='green', s=20, alpha=0.8, label=f'Real Targets ({sum(real_mask):,})')
    ax2.scatter(np.array(radar_x)[ghost_mask], np.array(radar_y)[ghost_mask], 
               c='red', s=20, alpha=0.8, label=f'Ghost Targets ({sum(ghost_mask):,})')
    ax2.set_title('Prediction Results: Real vs Ghost Targets', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 1-3. 예측 확률 히트맵
    scatter3 = ax3.scatter(radar_x, radar_y, c=probabilities, s=20, cmap='RdYlGn', alpha=0.8)
    ax3.set_title('Prediction Probabilities (Green=Real, Red=Ghost)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Real Target Probability')
    
    # 1-4. 통계 정보
    ax4.axis('off')
    stats_text = f"""
    🎯 추론 결과 통계 (k=6, SNR≥15dB)
    
    📊 데이터 개수:
    • 총 레이더 포인트: {len(radar_points):,}개
    • 총 LiDAR 포인트: {len(lidar_points):,}개
    • 실제 타겟: {sum(real_mask):,}개 ({sum(real_mask)/len(predictions)*100:.1f}%)
    • 고스트 타겟: {sum(ghost_mask):,}개 ({sum(ghost_mask)/len(predictions)*100:.1f}%)
    
    📈 예측 확률 분석:
    • 평균 확률: {np.mean(probabilities):.3f}
    • 실제 타겟 평균 확률: {np.mean(probabilities[real_mask]):.3f}
    • 고스트 타겟 평균 확률: {np.mean(probabilities[ghost_mask]):.3f}
    
    ⚙️ 모델 설정:
    • SNR 임계값: 15.0 dB
    • 거리 임계값: 0.4 m
    • k-NN 연결: 6개
    • 디바이스: {device}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('k6_15dB_all_frames.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 레이더 중심 확대 뷰
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 레이더 포인트 범위 계산
    radar_x_min, radar_x_max = min(radar_x), max(radar_x)
    radar_y_min, radar_y_max = min(radar_y), max(radar_y)
    
    # 여백 추가
    margin = 2.0
    x_min, x_max = radar_x_min - margin, radar_x_max + margin
    y_min, y_max = radar_y_min - margin, radar_y_max + margin
    
    # 2-1. 확대된 원본 데이터
    # LiDAR 포인트 필터링 (확대 범위 내)
    lidar_in_range_x = [x for x, y in zip(lidar_x, lidar_y) if x_min <= x <= x_max and y_min <= y <= y_max]
    lidar_in_range_y = [y for x, y in zip(lidar_x, lidar_y) if x_min <= x <= x_max and y_min <= y <= y_max]
    
    ax1.scatter(lidar_in_range_x, lidar_in_range_y, c='blue', s=2, alpha=0.5, label='LiDAR')
    scatter1 = ax1.scatter(radar_x, radar_y, c=radar_snr, s=30, cmap='viridis', alpha=0.9, label='Radar')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title('Radar-Focused View: Original Data', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='SNR (dB)')
    
    # 2-2. 확대된 예측 결과
    ax2.scatter(lidar_in_range_x, lidar_in_range_y, c='blue', s=2, alpha=0.5, label='LiDAR')
    ax2.scatter(np.array(radar_x)[real_mask], np.array(radar_y)[real_mask], 
               c='green', s=30, alpha=0.9, label=f'Real Targets ({sum(real_mask):,})')
    ax2.scatter(np.array(radar_x)[ghost_mask], np.array(radar_y)[ghost_mask], 
               c='red', s=30, alpha=0.9, label=f'Ghost Targets ({sum(ghost_mask):,})')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title('Radar-Focused View: Prediction Results', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 2-3. 확대된 확률 히트맵
    scatter3 = ax3.scatter(radar_x, radar_y, c=probabilities, s=30, cmap='RdYlGn', alpha=0.9)
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_title('Radar-Focused View: Prediction Probabilities', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Real Target Probability')
    
    # 2-4. 확대 뷰 정보
    ax4.axis('off')
    zoom_text = f"""
    🔍 레이더 중심 확대 뷰 (k=6, SNR≥15dB)
    
    📏 확대 범위:
    • X: {x_min:.1f}m ~ {x_max:.1f}m
    • Y: {y_min:.1f}m ~ {y_max:.1f}m
    • 범위 내 LiDAR: {len(lidar_in_range_x):,}개
    
    🎯 레이더 포인트 분석:
    • 총 레이더 포인트: {len(radar_points):,}개
    • 실제 타겟: {sum(real_mask):,}개
    • 고스트 타겟: {sum(ghost_mask):,}개
    
    📊 SNR 분포:
    • 최소 SNR: {min(radar_snr):.1f} dB
    • 최대 SNR: {max(radar_snr):.1f} dB
    • 평균 SNR: {np.mean(radar_snr):.1f} dB
    
    🔬 확대 효과:
    • 레이더-LiDAR 관계 명확히 표시
    • 고스트 타겟 패턴 분석 용이
    • 실제 타겟 검증 가능
    """
    
    ax4.text(0.05, 0.95, zoom_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('k6_15dB_radar_focused.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 시각화 완료:")
    print("  - k6_15dB_all_frames.png: 전체 뷰")
    print("  - k6_15dB_radar_focused.png: 레이더 중심 확대 뷰")

if __name__ == "__main__":
    run_inference()
