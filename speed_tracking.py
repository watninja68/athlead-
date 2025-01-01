import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque

# Paths to video files
SOURCE_VIDEO_PATH = "./vid/usain.mp4"
TARGET_VIDEO_PATH = "./athlete_speed_output.mp4"

# Constants
CONFIDENCE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5
MODEL_NAME = "yolov8x"  # YOLO model
MODEL_RESOLUTION = 1280

# Frame processing setup
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
frame_iterator = iter(frame_generator)
frame = next(frame_iterator)

# Load the YOLO model
model = YOLO(MODEL_NAME)

# Get video information and initialize annotators
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# Tracker initialization
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps, track_activation_threshold=CONFIDENCE_THRESHOLD
)

# Annotator settings
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

# Store coordinates for speed calculation
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

# Pixel-to-meter conversion (must be calibrated based on video)
# Measure pixel distance for a known real-world distance
known_real_world_distance_meters = 10  # e.g., 10 meters
measured_pixel_distance = 100  # Example: measured from video
pixel_to_meter = known_real_world_distance_meters / measured_pixel_distance

# Open the output video file
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # Loop over video frames
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        # Run YOLO detection on the frame
        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Filter detections by confidence and class (only track people, class ID 0)
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[detections.class_id == 0]  # Class 0 = 'person'

        # Refine detections using non-max suppression
        detections = detections.with_nms(IOU_THRESHOLD)

        # Track detections across frames
        detections = byte_track.update_with_detections(detections=detections)

        # Get bottom-center points of each detected person for speed calculation
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

        # Store tracked coordinates to calculate speed
        for tracker_id, [x, y] in zip(detections.tracker_id, points):
            # Store the X-axis position (horizontal movement for running) instead of Y-axis
            coordinates[tracker_id].append(x)

        # Format labels (ID and speed)
        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # Speed calculation based on X-axis movement
                coordinate_start = coordinates[tracker_id][-1]  # Most recent position
                coordinate_end = coordinates[tracker_id][0]    # Older position
                distance = abs(coordinate_start - coordinate_end)  # Displacement in pixels
                time = len(coordinates[tracker_id]) / video_info.fps  # Time in seconds

                # Convert pixel distance to meters using calibrated scale
                distance_in_meters = distance * pixel_to_meter

                # Calculate speed in meters per second (m/s) and convert to km/h
                speed = (distance_in_meters / time) * 3.6  # Convert m/s to km/h

                labels.append(f"#{tracker_id} {int(speed)} km/h")

        # Annotate the frame with bounding boxes, labels, and traces
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Add the annotated frame to the output video
        sink.write_frame(annotated_frame)