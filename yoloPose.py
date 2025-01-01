import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

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

class PoseEstimation:
    def __init__(self, video_name):
        self.model = YOLO('yolov8n-pose.pt')
        self.active_keypoints = [11, 13, 15]
        self.video_path = video_name
        self.scale = 1/2
        current_fps = 24
        desired_fps = 10
        self.skip_factor = current_fps // desired_fps

    def analyze_pose(self, show_angle=False, save_video=True):
        if show_angle:
            plt.ion()
            fig, ax = plt.subplots()
            angles = []
            times = []

        frame_count = 0
        color = (255, 255, 0)
        cv2.namedWindow("KeyPoints on Video", cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties for saving
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) // self.skip_factor
        
        # Initialize video writer
        if save_video:
            output_path = self.video_path.rsplit('.', 1)[0] + '_analyzed.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % self.skip_factor != 0:
                continue

            height, width, _ = frame.shape
            window_width = int(width * self.scale)
            window_height = int(height * self.scale)
            cv2.resizeWindow("KeyPoints on Video", window_width, window_height)

            results = self.model(frame)
            if len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy.cpu().numpy()[0]

                try:
                    # Draw lines between keypoints
                    for i in range(len(self.active_keypoints) - 1):
                        pt1 = tuple(keypoints[self.active_keypoints[i]].astype(int))
                        pt2 = tuple(keypoints[self.active_keypoints[i + 1]].astype(int))
                        cv2.line(frame, pt1, pt2, color, 8)
                        cv2.circle(frame, pt1, 5, color, -1)
                    
                    # Draw final keypoint
                    final_pt = tuple(keypoints[self.active_keypoints[-1]].astype(int))
                    cv2.circle(frame, final_pt, 5, color, -1)

                    if show_angle:
                        angle = compute_angle(
                            keypoints[self.active_keypoints[0]],
                            keypoints[self.active_keypoints[1]],
                            keypoints[self.active_keypoints[2]]
                        )
                        if angle is not None:
                            angle_text = f"{round(angle)}" if not np.isnan(angle) else "N/A"
                            cv2.putText(frame, angle_text, (270, 1500), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 5, color, 5, cv2.LINE_AA)
                            
                            if not np.isnan(angle):
                                angles.append(angle)
                                times.append(frame_count)
                                ax.clear()
                                ax.plot(times, angles, marker='o', color='orange')
                                ax.set_xlabel('Time')
                                ax.set_ylabel('Angle (degrees)')
                                ax.set_title('Angle vs. Time')
                                plt.draw()
                                plt.pause(0.05)
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    continue

            if save_video:
                out.write(frame)
                
            cv2.imshow("KeyPoints on Video", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close()

def run_analyze_pose(video_path, show_angle=False, save_video=True):
    pe = PoseEstimation(video_path)
    pe.analyze_pose(show_angle=show_angle, save_video=save_video)

if __name__ == "__main__":
    run_analyze_pose('./vid/stock.mp4', show_angle=True, save_video=True)