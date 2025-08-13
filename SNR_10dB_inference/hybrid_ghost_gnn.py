"""
프롬프트 3 (수정): GNN 모델 아키텍처 정의
- 입력 특징 벡터에서 p_det 제외 (6차원)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.data import Data
import numpy as np
from typing import List, Optional

from data_structures import RadarPoint, RadarFrame

class HybridGhostGNN(nn.Module):
    """정적/동적 구분 없이 모든 포인트를 입력받는 GNN 모델"""
    def __init__(self, 
                 input_dim: int = 6,  # x, y, range, azimuth, vr, rcs
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super(HybridGhostGNN, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.batch_norms.append(BatchNorm(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.convs.append(SAGEConv(hidden_dim, 1))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return torch.sigmoid(x)

def extract_node_features(radar_points: RadarFrame) -> torch.Tensor:
    """GNN의 각 노드(점) 특징들을 계산합니다. (p_det 제외)"""
    if not radar_points:
        return torch.empty(0, 6)

    features = []
    for point in radar_points:
        range_val = np.sqrt(point.x**2 + point.y**2)
        azimuth = np.arctan2(point.y, point.x)
        features.append([point.x, point.y, range_val, azimuth, point.vr, point.rcs])

    return torch.tensor(features, dtype=torch.float32)

def create_edges_knn(positions: torch.Tensor, k: int = 8) -> torch.Tensor:
    """k-NN 기반으로 그래프 엣지를 생성합니다."""
    if len(positions) <= 1: return torch.empty(2, 0, dtype=torch.long)
    dist_matrix = torch.cdist(positions, positions)
    dist_matrix.fill_diagonal_(float('inf'))
    k = min(k, len(positions) - 1)
    _, knn_indices = torch.topk(dist_matrix, k, dim=1, largest=False)
    source_nodes = torch.arange(len(positions)).repeat_interleave(k)
    target_nodes = knn_indices.flatten()
    return torch.stack([source_nodes, target_nodes])

def create_graph_data(radar_points: RadarFrame, 
                     labels: Optional[List[int]] = None,
                     k: int = 8) -> Data:
    """레이더 포인트들로부터 PyTorch Geometric Data 객체를 생성합니다."""
    if not radar_points:
        return Data(x=torch.empty(0, 6), 
                   edge_index=torch.empty(2, 0, dtype=torch.long),
                   y=torch.empty(0, dtype=torch.long))

    node_features = extract_node_features(radar_points)
    positions = node_features[:, :2] # x, y 좌표
    edge_index = create_edges_knn(positions, k)

    y = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    return Data(x=node_features, edge_index=edge_index, y=y)
