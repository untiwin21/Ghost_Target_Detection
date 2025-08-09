"""
프롬프트 1 (수정): 기본 데이터 구조 정의
- EgoVelocity 클래스는 더 이상 필요하지 않습니다.
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

# RadarFrame 타입 정의
RadarFrame = List[RadarPoint]
