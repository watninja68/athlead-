import cv2
import mediapipe as mp
import csv
import torch 
import torch.nn as nn
import json
class Neural(nn.Module):
    def __init__(self, input_size):
        super(Neural, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x
model = Neural(132)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model.load_state_dict(torch.load("./models/RunningSitting_v2.pth"))
model.to(device=device)
model.eval()

def write_landmarks_to_csv(landmarks, frame_number, csv_data):
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
    print("\n")
def load_data(json_data, device):
    try:
        frame_data = json.loads(json_data)
        data = []
        for _, landmark_data in frame_data.items():
            data.extend([
                landmark_data['x'],
                landmark_data['y'],
                landmark_data['z'],
                landmark_data['visibility']
            ])
        data_tensor = torch.tensor([data], device=device, dtype=torch.float32)
        return data_tensor
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Problematic JSON data: {json_data}")
        return None
video_path = './vid/usain.mp4'
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(video_path)

frame_number = 1
csv_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        dic = {}
        for mark, data_point in zip(mp_pose.PoseLandmark, result.pose_landmarks.landmark):
            dic[mark.value] = {
                "landmark": mark.name,
                "x": data_point.x,
                "y": data_point.y,
                "z": data_point.z,
                "visibility": data_point.visibility
            }
        json_object = json.dumps(dic)
        
        tensor_data = load_data(json_object, device=device)
        if tensor_data is not None:
            y = model(tensor_data)
            print(f"Frame {frame_number}: Prediction = {y.item()}")
            if y.item() > 0.5:
                text = "Running"
            else:
                text = "Sitting"
        
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),  2)

    cv2.imshow('MediaPipe Pose', frame)

    frame_number += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Uncomment the following lines if you want to write the data to a CSV file
# with open(output_csv, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Frame', 'Landmark', 'X', 'Y', 'Z'])
#     writer.writerows(csv_data)