"""
학습된 모델로 새로운 데이터에 대한 추론 실행
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

from data_structures import RadarPoint, RadarFrame
from hybrid_ghost_gnn import HybridGhostGNN, create_graph_data

class GhostDetectorInference:
    """학습된 모델로 추론만 수행하는 클래스"""
    
    def __init__(self, 
                 model_path: str,
                 radar_data_path: str,
                 lidar_data_path: str,
                 distance_threshold: float = 0.5,
                 snr_threshold: float = 10.0,
                 k: int = 8,
                 min_points_per_frame: int = 5):
        
        self.model_path = model_path
        self.radar_data_path = radar_data_path
        self.lidar_data_path = lidar_data_path
        self.distance_threshold = distance_threshold
        self.snr_threshold = snr_threshold
        self.k = k
        self.min_points_per_frame = min_points_per_frame
        
        # GPU 사용 가능 여부 확인
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 모델 로드
        self.model = self.load_model()
        
        # 데이터 로드
        self.frames = self.load_data()
        
    def load_model(self):
        """학습된 모델 로드"""
        model = HybridGhostGNN(input_dim=6, hidden_dim=128)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"Model loaded from {self.model_path}")
        return model
        
    def load_data(self):
        """레이더와 LiDAR 데이터 로드 및 프레임 생성"""
        # 레이더 데이터 로드
        radar_data = []
        with open(self.radar_data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        time = float(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        velocity = float(parts[3])
                        snr = float(parts[4])
                        radar_data.append((time, x, y, velocity, snr))
        
        # LiDAR 데이터 로드
        lidar_data = []
        with open(self.lidar_data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 4:
                        time = float(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        intensity = float(parts[3])
                        lidar_data.append((time, x, y, intensity))
        
        # 시간별로 그룹화
        radar_by_time = {}
        for time, x, y, velocity, snr in radar_data:
            if time not in radar_by_time:
                radar_by_time[time] = []
            radar_by_time[time].append((x, y, velocity, snr))
        
        lidar_by_time = {}
        for time, x, y, intensity in lidar_data:
            if time not in lidar_by_time:
                lidar_by_time[time] = []
            lidar_by_time[time].append((x, y, intensity))
        
        # 프레임 생성
        frames = []
        common_times = set(radar_by_time.keys()) & set(lidar_by_time.keys())
        
        for time in sorted(common_times):
            radar_points_data = radar_by_time[time]
            lidar_points_data = lidar_by_time[time]
            
            if len(radar_points_data) >= self.min_points_per_frame:
                # 레이더 포인트 생성
                radar_points = []
                for x, y, velocity, snr in radar_points_data:
                    point = RadarPoint(x, y, velocity, snr)  # vr, rcs 순서
                    radar_points.append(point)
                
                # LiDAR 포인트 좌표
                lidar_coords = np.array([(x, y) for x, y, _ in lidar_points_data])
                
                # Ground Truth 라벨 생성 (SNR + 거리 조합)
                labels = self.generate_labels(radar_points, lidar_coords)
                
                # 프레임 데이터 저장 (RadarFrame은 List[RadarPoint])
                frame_data = {
                    'time': time,
                    'radar_points': radar_points,  # List[RadarPoint]
                    'labels': labels
                }
                frames.append(frame_data)
        
        print(f"Loaded {len(frames)} frames for inference")
        return frames
    
    def generate_labels(self, radar_points, lidar_coords):
        """SNR + 거리 조합으로 Ground Truth 라벨 생성"""
        labels = []
        radar_coords = np.array([(p.x, p.y) for p in radar_points])
        
        if len(lidar_coords) > 0:
            distances = cdist(radar_coords, lidar_coords)
            min_distances = np.min(distances, axis=1)
        else:
            min_distances = np.full(len(radar_points), float('inf'))
        
        for i, point in enumerate(radar_points):
            distance_condition = min_distances[i] <= self.distance_threshold
            snr_condition = point.rcs >= self.snr_threshold  # rcs 속성 사용
            
            # 두 조건을 모두 만족해야 실제 타겟
            if distance_condition and snr_condition:
                labels.append(1)  # 실제 타겟
            else:
                labels.append(0)  # 고스트 타겟
        
        return labels
    
    def run_inference(self):
        """모든 프레임에 대해 추론 실행"""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print("Running inference...")
        
        with torch.no_grad():
            for i, frame_data in enumerate(self.frames):
                radar_points = frame_data['radar_points']
                labels = frame_data['labels']
                
                if len(radar_points) < self.min_points_per_frame:
                    continue
                
                # 그래프 데이터 생성
                graph_data = create_graph_data(radar_points, labels, k=self.k)
                graph_data = graph_data.to(self.device)
                
                # 모델 예측
                output = self.model(graph_data)
                probabilities = output.cpu().numpy().flatten()
                predictions = (probabilities > 0.5).astype(int)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                all_probabilities.extend(probabilities)
                
                if i % 100 == 0:
                    print(f"Processed {i+1}/{len(self.frames)} frames")
        
        # 결과 통계
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        accuracy = np.mean(all_predictions == all_labels)
        
        print(f"\n=== Inference Results ===")
        print(f"Total points: {len(all_predictions)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Real targets (GT): {np.sum(all_labels)} ({np.mean(all_labels)*100:.1f}%)")
        print(f"Real targets (Pred): {np.sum(all_predictions)} ({np.mean(all_predictions)*100:.1f}%)")
        print(f"Average probability: {np.mean(all_probabilities):.4f}")
        
        return {
            'frames': self.frames,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'accuracy': accuracy
        }

def main():
    """메인 실행 함수"""
    # 파일 경로 설정
    model_path = "ghost_detector.pth"
    radar_data_path = "RadarMap_bokdo3_v6.txt"
    lidar_data_path = "LiDARMap_bokdo3_v6.txt"
    
    # 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    
    if not os.path.exists(radar_data_path):
        print(f"Error: Radar data file {radar_data_path} not found!")
        return
        
    if not os.path.exists(lidar_data_path):
        print(f"Error: LiDAR data file {lidar_data_path} not found!")
        return
    
    # 추론 실행
    detector = GhostDetectorInference(
        model_path=model_path,
        radar_data_path=radar_data_path,
        lidar_data_path=lidar_data_path
    )
    
    results = detector.run_inference()
    
    print(f"\nInference completed! Results saved in results dictionary.")
    print(f"You can now run visualization scripts to see the results.")

if __name__ == "__main__":
    main()
