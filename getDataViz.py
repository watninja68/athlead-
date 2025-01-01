import socket
import threading
import json
from datetime import datetime
import pandas as pd
from collections import defaultdict
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
import json
from datetime import datetime
import os

# Global data structure to store readings from all devices
data_queue = Queue()
device_data = defaultdict(lambda: defaultdict(list))
max_data_points = 100  # Maximum number of data points to keep per device
server_socket = None
running = True

def cleanup():
    """Cleanup function to close socket and exit gracefully"""
    global running, server_socket
    running = False
    if server_socket:
        try:
            server_socket.close()
        except:
            pass
    print("\nServer shutdown complete")

def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print("\nShutting down server...")
    cleanup()
    sys.exit(0)

def parse_sensor_data(data_str):
    """Parse the incoming data string into a structured format."""
    try:
        # Extract device ID
        device_id_match = re.search(r"Device ID: (\w+)", data_str)
        device_name_match = re.search(r"Name: (\w+)", data_str)
        
        if not device_id_match or not device_name_match:
            return None
            
        device_id = device_id_match.group(1)
        device_name = device_name_match.group(1)
        
        # Extract sensor readings using regex
        accel_x = float(re.search(r"Acceleration X: ([-\d.]+)", data_str).group(1))
        accel_y = float(re.search(r"Y: ([-\d.]+)", data_str).group(1))
        accel_z = float(re.search(r"Z: ([-\d.]+)", data_str).group(1))
        
        gyro_x = float(re.search(r"Rotation X: ([-\d.]+)", data_str).group(1))
        gyro_y = float(re.search(r"Y: ([-\d.]+)", data_str).group(1))
        gyro_z = float(re.search(r"Z: ([-\d.]+)", data_str).group(1))
        
        temp = float(re.search(r"Temperature: ([-\d.]+)", data_str).group(1))
        
        timestamp = datetime.now()
        
        return {
            'device_id': device_id,
            'device_name': device_name,
            'timestamp': timestamp,
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
            'temperature': temp
        }
    except Exception as e:
        print(f"Error parsing data: {e}")
        return None

def handle_client(client_socket, addr):
    """Handle incoming client connections."""
    try:
        data = client_socket.recv(1024).decode()
        if data:
            parsed_data = parse_sensor_data(data)
            if parsed_data:
                data_queue.put(parsed_data)
    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        client_socket.close()

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
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
    """Run the TCP server in a separate thread."""
    global server_socket, running
    
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Find an available port
        port = find_available_port(6969)
        server_socket.bind(('0.0.0.0', port))
        server_socket.listen(5)
        print(f"Server listening on port {port}")
        
        while running:
            try:
                server_socket.settimeout(1.0)  # 1 second timeout
                client_socket, addr = server_socket.accept()
                threading.Thread(target=handle_client, args=(client_socket, addr)).start()
            except socket.timeout:
                continue
            except Exception as e:
                if running:  # Only print error if we're not shutting down
                    print(f"Error accepting connection: {e}")
                    
    except Exception as e:
        print(f"Server thread error: {e}")
    finally:
        if server_socket:
            server_socket.close()

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("MPU6050 Sensor Dashboard"),
    html.Div([
        dcc.Dropdown(
            id='device-selector',
            placeholder='Select a device'
        ),
    ]),
    html.Div([
        html.Div([
            html.H3("Accelerometer Data"),
            dcc.Graph(id='accel-graph'),
        ], className='six columns'),
        html.Div([
            html.H3("Gyroscope Data"),
            dcc.Graph(id='gyro-graph'),
        ], className='six columns'),
    ], className='row'),
    html.Div([
        html.H3("Temperature"),
        dcc.Graph(id='temp-graph'),
    ]),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # in milliseconds
        n_intervals=0
    )
])

@app.callback(
    Output('device-selector', 'options'),
    Input('interval-component', 'n_intervals')
)
def update_device_list(n):
    devices = list(device_data.keys())
    return [{'label': f'Device {device_id}', 'value': device_id} for device_id in devices]

@app.callback(
    [Output('accel-graph', 'figure'),
     Output('gyro-graph', 'figure'),
     Output('temp-graph', 'figure')],
    [Input('interval-component', 'n_intervals'),
     Input('device-selector', 'value')]
)
def update_graphs(n, selected_device):
    # Process any new data in the queue
    while not data_queue.empty():
        data = data_queue.get()
        device_id = data['device_id']
        
        # Store the data
        for key, value in data.items():
            device_data[device_id][key].append(value)
            if len(device_data[device_id][key]) > max_data_points:
                device_data[device_id][key].pop(0)

    if not selected_device or not device_data:
        # Return empty figures if no device is selected
        empty_fig = {
            'data': [],
            'layout': {
                'title': 'No data available',
                'xaxis': {'title': 'Samples'},
                'yaxis': {'title': 'Values'}
            }
        }
        return empty_fig, empty_fig, empty_fig

    device = device_data[selected_device]
    
    # Create accelerometer figure
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

    # Create gyroscope figure
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

    # Create temperature figure
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
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the server thread
    server_thread = threading.Thread(target=server_thread)
    server_thread.daemon = True
    server_thread.start()
    
    try:
        # Run the Dash app
        app.run_server(debug=False, host='0.0.0.0', port=8050)
    finally:
        cleanup()