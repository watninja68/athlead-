import json
import cv2
import mediapipe as mp
import glob
import os
file_path = './Database/Running and sitting/1/1_1.json'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
count = 0
import glob
image_list = []
count = 0
for filename in glob.glob('./Database/Running and sitting/1/*.jpg'): 
    count += 1
    image_list.append(filename)
    if count == 250:
        break
IMAGE_FILES = image_list
BG_COLOR = (192, 192, 192) 
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    try :
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        dic = {}
        for mark, data_point in zip(mp_pose.PoseLandmark, results.pose_landmarks.landmark):
            dic[mark.value] = dict(landmark = mark.name, 
                x = data_point.x,
                y = data_point.y,
                z = data_point.z,
                visibility = data_point.visibility)

        json_object = json.dumps(dic, indent=2)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, 'r') as f:
                obj = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            obj = []

        obj.append(json_object) 

        with open(file_path, 'w') as outfile:
            json.dump(obj, outfile, indent=4)
    except  Exception as e:
        count += 1
print(count)
   