#!/usr/bin/env python3
"""
데이터 구조 정의
레이더 및 LiDAR 포인트 데이터 클래스
"""

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

# 타입 정의
RadarFrame = List[RadarPoint]
LiDARFrame = List[LiDARPoint]
