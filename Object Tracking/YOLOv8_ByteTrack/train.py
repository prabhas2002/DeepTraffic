
import argparse
from ultralytics import YOLO

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training')
    parser.add_argument('--data', type=str, default='gram.yaml', help='Path to data configuration file')
    parser.add_argument('--pretrained_weights', type=str, default='./yolov8n.pt', help='Path to coco pretrained weights file')
    parser.add_argument('--device', type=str, default='5,7', help='Device IDs for training')
    parser.add_argument('--epoch',type=int, default=200, help='Number of epochs for model training')
    parser.add_argument('--img_size',type=int, default=1280, help='Size of the image')
    parser.add_argument('--save_results',type=str, default = 'yolov8n_gram',help='Path to save model weights and results')

    args = parser.parse_args()

    device = [int(x) for x in args.device.split(',')]

    # Load the model
    model = YOLO(args.pretrained_weights)

    # Training
    results = model.train(data=args.data, imgsz=args.img_size, epochs=args.epoch, batch=args.batch, name=args.save_results, device=device)

if __name__ == '__main__':
    main()
