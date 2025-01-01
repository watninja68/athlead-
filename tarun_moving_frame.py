import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque

def track_athlete_speed(
    source_video_path: str,
    target_video_path: str,
    known_real_world_distance_meters: float = 10,
    measured_pixel_distance: float = 100,
    model_name: str = "yolov8x.pt",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    model_resolution: int = 1280
):
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    model = YOLO(model_name)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, 
        track_activation_threshold=confidence_threshold
    )

    thickness = sv.calculate_dynamic_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)
    
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER
    )

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    pixel_to_meter = known_real_world_distance_meters / measured_pixel_distance

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            result = model(frame, imgsz=model_resolution, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > confidence_threshold]
            detections = detections[detections.class_id == 0]
            detections = detections.with_nms(iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

            for tracker_id, [x, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(x)

            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    distance_in_meters = distance * pixel_to_meter
                    speed = (distance_in_meters / time) * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            sink.write_frame(annotated_frame)
if __name__ =="__main__":
    track_athlete_speed(source_video_path="./vid/stock.mp4",target_video_path="./tarun_moving_frame.mp4")