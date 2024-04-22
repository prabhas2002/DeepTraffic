from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import argparse


def tracking(args):

    model = YOLO(args.trained_weights)

    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)
    output_txt_file = args.save_results

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    with open(output_txt_file, 'w') as file:
        frame_number = 1
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, tracker=args.type_tracker, persist=True)

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    x_center = x
                    y_center = y
                    box_width = w
                    box_height = h
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)


                    object_id = track_id
                    bb_left = x_center - box_width / 2
                    bb_top = y_center - box_height / 2
                    confidence = 1  # Assuming confidence is always 1
                    x, y, z = -1, -1, -1  # Set world coordinates to -1

                    # Write line to file in MOT16 format
                    file.write(f"{frame_number},{object_id},{bb_left},{bb_top},{box_width},{box_height},{confidence},{x},{y},{z}\n")

                    
                frame_number += 1
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    

def main():
    parser = argparse.ArgumentParser(description='Tracking using  YOLO model')
    parser.add_argument('--trained_weights', type=str, default='./runs/detect/yolov8s_gram/weights/best.pt', help='Path to trained weights file')
    parser.add_argument('--video_path',type=str, default='./videos/M-30.mp4', help='Path to video for tracking')
    parser.add_argument('--type_tracker', type=str, default='bytetrack.yaml', help='file to access type of tracker')
    parser.add_argument('--save_results',type=str, default='./results/M-30.txt',help='results saved in MOT challenge format')
    

    args = parser.parse_args()
    tracking(args)


if __name__ == '__main__':
    main()
