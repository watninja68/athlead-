import socket
import threading
import json
from datetime import datetime
import pandas as pd
from collections import defaultdict, deque
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from queue import Queue
import re
import signal
import sys
import time
import csv
from pathlib import Path
import os
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import supervision as sv

data_queue = Queue()
device_data = defaultdict(lambda: defaultdict(list))
angle_data = deque(maxlen=100)
max_data_points = 100
server_socket = None
running = True

pose_model = YOLO('yolov8n-pose.pt')
detection_model = YOLO('yolov8x.pt')
active_keypoints = [11, 13, 15]

byte_track = sv.ByteTrack(frame_rate=30, track_activation_threshold=0.3)

cap = cv2.VideoCapture(0)

def cleanup():
    global running, server_socket, cap
    running = False
    if server_socket:
        server_socket.close()
    if cap:
        cap.release()
    sys.exit(0)

def signal_handler(sig, frame):
    cleanup()

def compute_angle(start, middle, end):
    try:
        vector1 = middle - start
        vector2 = end - middle
        
        magnitude_v1 = np.linalg.norm(vector1)
        magnitude_v2 = np.linalg.norm(vector2)
        
        # Check for zero magnitudes to avoid division by zero
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return None
            
        dot_product = np.dot(vector1, vector2)
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        
        # Handle numerical instability
        if cos_angle > 1:
            cos_angle = 1
        elif cos_angle < -1:
            cos_angle = -1
            
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        # Validate the final angle
        if np.isnan(angle_deg) or np.isinf(angle_deg):
            return None
            
        return angle_deg
    except Exception as e:
        return None

def process_frame(frame):
    if frame is None:
        return frame, None

    pose_frame = frame.copy()
    speed_frame = frame.copy()
    
    pose_results = pose_model(pose_frame, verbose=False)
    current_angle = None
    
    if len(pose_results) > 0 and len(pose_results[0].keypoints.xy) > 0:
        keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]
        
        try:
            color = (255, 255, 0)
            for i in range(len(active_keypoints) - 1):
                pt1 = tuple(keypoints[active_keypoints[i]].astype(int))
                pt2 = tuple(keypoints[active_keypoints[i + 1]].astype(int))
                cv2.line(pose_frame, pt1, pt2, color, 2)
                cv2.circle(pose_frame, pt1, 5, color, -1)
            
            final_pt = tuple(keypoints[active_keypoints[-1]].astype(int))
            cv2.circle(pose_frame, final_pt, 5, color, -1)
            
            current_angle = compute_angle(
                keypoints[active_keypoints[0]],
                keypoints[active_keypoints[1]],
                keypoints[active_keypoints[2]]
            )
            
            if current_angle is not None and not np.isnan(current_angle):
                angle_text = f"Angle: {round(current_angle)}°"
                cv2.putText(pose_frame, angle_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                angle_data.append(current_angle)
        except:
            pass
    
    detect_results = detection_model(speed_frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(detect_results)
    detections = detections[detections.confidence > 0.3]
    detections = detections[detections.class_id == 0]
    detections = detections.with_nms(0.5)
    detections = byte_track.update_with_detections(detections=detections)
    
    for box, confidence, class_id, tracker_id in zip(
        detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id
    ):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(speed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person #{tracker_id} ({confidence:.2f})"
        cv2.putText(speed_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    combined_frame = cv2.addWeighted(pose_frame, 0.5, speed_frame, 0.5, 0)
    return combined_frame, current_angle

def parse_sensor_data(data_str):
    try:
        device_id_match = re.search(r"Device ID: (\w+)", data_str)
        device_name_match = re.search(r"Name: (\w+)", data_str)
        
        if not device_id_match or not device_name_match:
            return None
            
        device_id = device_id_match.group(1)
        device_name = device_name_match.group(1)
        
        accel_x = float(re.search(r"Acceleration X: ([-\d.]+)", data_str).group(1))
        accel_y = float(re.search(r"Y: ([-\d.]+)", data_str).group(1))
        accel_z = float(re.search(r"Z: ([-\d.]+)", data_str).group(1))
        
        gyro_x = float(re.search(r"Rotation X: ([-\d.]+)", data_str).group(1))
        gyro_y = float(re.search(r"Y: ([-\d.]+)", data_str).group(1))
        gyro_z = float(re.search(r"Z: ([-\d.]+)", data_str).group(1))
        
        temp = float(re.search(r"Temperature: ([-\d.]+)", data_str).group(1))
        
        return {
            'device_id': device_id,
            'device_name': device_name,
            'timestamp': datetime.now(),
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
            'temperature': temp
        }
    except:
        return None

def handle_client(client_socket, addr):
    try:
        data = client_socket.recv(1024).decode()
        if data:
            parsed_data = parse_sensor_data(data)
            if parsed_data:
                data_queue.put(parsed_data)
    finally:
        client_socket.close()

def find_available_port(start_port, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    raise OSError("No available ports found")

def server_thread():
    global server_socket, running
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    port = find_available_port(6969)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(5)
    
    while running:
        try:
            server_socket.settimeout(1.0)
            client_socket, addr = server_socket.accept()
            threading.Thread(target=handle_client, args=(client_socket, addr)).start()
        except socket.timeout:
            continue
        except:
            if running:
                continue

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Real-time Pose and Speed Tracking Dashboard", 
            style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    html.Div([
        html.Div([
            html.H3("Video Feed with Tracking", style={'textAlign': 'center'}),
            html.Img(id='video-feed', 
                    style={'width': '100%', 
                           'maxWidth': '640px', 
                           'border': '1px solid #ddd',
                           'borderRadius': '8px'})
        ], className='six columns'),
        
        html.Div([
            html.H3("Joint Angle Over Time", style={'textAlign': 'center'}),
            dcc.Graph(id='angle-graph')
        ], className='six columns'),
    ], className='row', style={'marginBottom': '20px'}),
    
    html.Div([
        html.H3("Sensor Data", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='device-selector',
            placeholder='Select a device',
            style={'marginBottom': '20px'}
        ),
        dcc.Graph(id='accel-graph'),
        dcc.Graph(id='gyro-graph'),
        dcc.Graph(id='temp-graph')
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=100,
        n_intervals=0
    )
], style={'padding': '20px'})

@app.callback(
    Output('device-selector', 'options'),
    Input('interval-component', 'n_intervals')
)
def update_device_list(n):
    devices = list(device_data.keys())
    return [{'label': f'Device {device_id}', 'value': device_id} for device_id in devices]

@app.callback(
    Output('video-feed', 'src'),
    Input('interval-component', 'n_intervals')
)
def update_video_feed(n):
    if cap is None or not cap.isOpened():
        return ''
    
    ret, frame = cap.read()
    if not ret:
        return ''
    
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)
    
    processed_frame, current_angle = process_frame(frame)
    if processed_frame is None:
        return ''
    
    _, buffer = cv2.imencode('.jpg', processed_frame)
    frame_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{frame_str}"

@app.callback(
    Output('angle-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_angle_graph(n):
    return {
        'data': [{
            'x': list(range(len(angle_data))),
            'y': list(angle_data),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Joint Angle',
            'line': {'color': 'orange'}
        }],
        'layout': {
            'title': 'Joint Angle History',
            'xaxis': {'title': 'Frame'},
            'yaxis': {'title': 'Angle (degrees)'}
        }
    }

@app.callback(
    [Output('accel-graph', 'figure'),
     Output('gyro-graph', 'figure'),
     Output('temp-graph', 'figure')],
    [Input('interval-component', 'n_intervals'),
     Input('device-selector', 'value')]
)
def update_sensor_graphs(n, selected_device):
    while not data_queue.empty():
        data = data_queue.get()
        device_id = data['device_id']
        
        for key, value in data.items():
            if key != 'device_id':
                device_data[device_id][key].append(value)
                if len(device_data[device_id][key]) > max_data_points:
                    device_data[device_id][key].pop(0)

    if not selected_device or not device_data:
        empty_fig = {'data': [], 'layout': {'title': 'No Data Available'}}
        return empty_fig, empty_fig, empty_fig

    device = device_data[selected_device]
    
    accel_fig = {
        'data': [
            go.Scatter(x=list(range(len(device['accel_x']))), y=device['accel_x'], name='X'),
            go.Scatter(x=list(range(len(device['accel_y']))), y=device['accel_y'], name='Y'),
            go.Scatter(x=list(range(len(device['accel_z']))), y=device['accel_z'], name='Z')
        ],
        'layout': {
            'title': 'Acceleration (m/s²)',
            'xaxis': {'title': 'Samples'},
            'yaxis': {'title': 'm/s²'}
        }
    }

    gyro_fig = {
        'data': [
            go.Scatter(x=list(range(len(device['gyro_x']))), y=device['gyro_x'], name='X'),
            go.Scatter(x=list(range(len(device['gyro_y']))), y=device['gyro_y'], name='Y'),
            go.Scatter(x=list(range(len(device['gyro_z']))), y=device['gyro_z'], name='Z')
        ],
        'layout': {
            'title': 'Gyroscope (rad/s)',
            'xaxis': {'title': 'Samples'},
            'yaxis': {'title': 'rad/s'}
        }
    }

    temp_fig = {
        'data': [
            go.Scatter(x=list(range(len(device['temperature']))), 
                      y=device['temperature'], 
                      name='Temperature')
        ],
        'layout': {
            'title': 'Temperature (°C)',
            'xaxis': {'title': 'Samples'},
            'yaxis': {'title': '°C'}
        }
    }

    return accel_fig, gyro_fig, temp_fig

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    
    server_thread = threading.Thread(target=server_thread)
    server_thread.daemon = True
    server_thread.start()
    
    try:
        app.run_server(debug=False, host='0.0.0.0', port=8050)
    finally:
        cleanup()