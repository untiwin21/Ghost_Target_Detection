"""
기본 데이터 구조 정의 - LiDAR와 Radar 포인트 모두 포함
"""
from dataclasses import dataclass
from typing import List

@dataclass
class RadarPoint:
    """하나의 레이더 점을 표현하는 데이터 클래스"""
    x: float  # 미터
    y: float  # 미터
    vr: float  # m/s, 도플러 속도
    rcs: float  # dB

    def __repr__(self) -> str:
        return f"RadarPoint(x={self.x:.3f}, y={self.y:.3f}, vr={self.vr:.3f}, rcs={self.rcs:.1f})"

@dataclass
class LiDARPoint:
    """하나의 LiDAR 점을 표현하는 데이터 클래스"""
    x: float  # 미터
    y: float  # 미터
    intensity: float  # 강도

    def __repr__(self) -> str:
        return f"LiDARPoint(x={self.x:.3f}, y={self.y:.3f}, intensity={self.intensity:.1f})"

# 프레임 타입 정의
RadarFrame = List[RadarPoint]
LiDARFrame = List[LiDARPoint]
