#!/usr/bin/env python3
"""
HybridGhostGNN 모델 정의
GraphSAGE 기반 레이더 고스트 타겟 탐지 GNN
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm
import numpy as np
from dataclasses import dataclass
from typing import List

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
