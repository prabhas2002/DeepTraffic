from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='Tracking Demo')
parser.add_argument('--trained_weights', type=str, default='./runs/detect/yolov8n_idd/weights/best.pt', help='Path to trained weights file')
parser.add_argument('--video_path',type=str, default='./videos/demo.mp4',help='enter video path')
args = parser.parse_args()


model = YOLO(args.trained_weights)  # Load a trained model


# Perform tracking with the model
results = model.track(source=args.video_path, tracker = 'bytetrack.yaml', save=True)  # Tracking with ByteTracker


# results will be saved in runs/detect/trackx/videoname.avi

# trackx means like track , track2, track3,...
