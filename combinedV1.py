import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque
import matplotlib.pyplot as plt

def compute_angle(start, middle, end):
    try:
        vector1 = middle - start
        vector2 = end - middle
        dot_product = np.dot(vector1, vector2)
        magnitude_v1 = np.linalg.norm(vector1)
        magnitude_v2 = np.linalg.norm(vector2)
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    except:
        return None

class CombinedTracker:
    def __init__(
        self,
        source_video_path: str,
        target_video_path: str,
        known_real_world_distance_meters: float = 10,
        measured_pixel_distance: float = 100,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        model_resolution: int = 1280,
        show_angle: bool = True
    ):
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.pixel_to_meter = known_real_world_distance_meters / measured_pixel_distance
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_resolution = model_resolution
        self.show_angle = show_angle
        
        # Initialize models
        self.pose_model = YOLO('yolov8n-pose.pt')
        self.detection_model = YOLO('yolov8x.pt')
        self.active_keypoints = [11, 13, 15]  # Shoulder, elbow, wrist
        
        # Initialize video info
        self.video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
        
        # Initialize trackers and annotators
        self.initialize_trackers()
        
        # Initialize tracking data structures
        self.coordinates = defaultdict(lambda: deque(maxlen=self.video_info.fps))
        
        if show_angle:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.angles = []
            self.times = []

    def initialize_trackers(self):
        self.byte_track = sv.ByteTrack(
            frame_rate=self.video_info.fps,
            track_activation_threshold=self.confidence_threshold
        )
        
        thickness = sv.calculate_dynamic_line_thickness(resolution_wh=self.video_info.resolution_wh)
        text_scale = sv.calculate_dynamic_text_scale(resolution_wh=self.video_info.resolution_wh)
        
        self.box_annotator = sv.BoxAnnotator(thickness=thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=self.video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER
        )

    def process_speed_tracking(self, frame, labels):
        results = self.detection_model(frame, imgsz=self.model_resolution, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence > self.confidence_threshold]
        detections = detections[detections.class_id == 0]  # person class
        detections = detections.with_nms(self.iou_threshold)
        detections = self.byte_track.update_with_detections(detections=detections)
        
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        
        for tracker_id, [x, y] in zip(detections.tracker_id, points):
            self.coordinates[tracker_id].append(x)
        
        for tracker_id in detections.tracker_id:
            if len(self.coordinates[tracker_id]) < self.video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = self.coordinates[tracker_id][-1]
                coordinate_end = self.coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(self.coordinates[tracker_id]) / self.video_info.fps
                distance_in_meters = distance * self.pixel_to_meter
                speed = (distance_in_meters / time) * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")
        
        return detections

    def process_pose_estimation(self, frame):
        results = self.pose_model(frame)
        if len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy.cpu().numpy()[0]
            
            try:
                # Draw lines between keypoints
                color = (255, 255, 0)
                for i in range(len(self.active_keypoints) - 1):
                    pt1 = tuple(keypoints[self.active_keypoints[i]].astype(int))
                    pt2 = tuple(keypoints[self.active_keypoints[i + 1]].astype(int))
                    cv2.line(frame, pt1, pt2, color, 8)
                    cv2.circle(frame, pt1, 5, color, -1)
                
                final_pt = tuple(keypoints[self.active_keypoints[-1]].astype(int))
                cv2.circle(frame, final_pt, 5, color, -1)
                
                if self.show_angle:
                    angle = compute_angle(
                        keypoints[self.active_keypoints[0]],
                        keypoints[self.active_keypoints[1]],
                        keypoints[self.active_keypoints[2]]
                    )
                    if angle is not None and not np.isnan(angle):
                        angle_text = f"{round(angle)}"
                        cv2.putText(frame, angle_text, (270, 1500),
                                  cv2.FONT_HERSHEY_SIMPLEX, 5, color, 5, cv2.LINE_AA)
                        
                        self.angles.append(angle)
                        self.times.append(len(self.times))
                        self.ax.clear()
                        self.ax.plot(self.times, self.angles, marker='o', color='orange')
                        self.ax.set_xlabel('Frame')
                        self.ax.set_ylabel('Angle (degrees)')
                        self.ax.set_title('Joint Angle Over Time')
                        plt.draw()
                        plt.pause(0.01)
                        
            except Exception as e:
                print(f"Error processing pose: {str(e)}")

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)
        
        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                labels = []
                
                # Process speed tracking
                detections = self.process_speed_tracking(frame, labels)
                
                # Process pose estimation
                self.process_pose_estimation(frame)
                
                # Annotate frame with speed tracking
                annotated_frame = frame.copy()
                annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                
                sink.write_frame(annotated_frame)
        
        if self.show_angle:
            plt.ioff()
            plt.close()

def run_combined_tracking(
    source_video_path: str,
    target_video_path: str,
    known_real_world_distance_meters: float = 10,
    measured_pixel_distance: float = 100,
    show_angle: bool = True
):
    tracker = CombinedTracker(
        source_video_path=source_video_path,
        target_video_path=target_video_path,
        known_real_world_distance_meters=known_real_world_distance_meters,
        measured_pixel_distance=measured_pixel_distance,
        show_angle=show_angle
    )
    tracker.process_video()

if __name__ == "__main__":
    run_combined_tracking(
        source_video_path="./vid/tarun_running.mp4",
        target_video_path="./tarun_combined_tracking.mp4",
        show_angle=True
    )