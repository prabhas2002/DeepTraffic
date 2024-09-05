
# DeepTraffic


The DeepTraffic repository encompasses object detection and object tracking folders, featuring various algorithms and resources for traffic flow prediction and optimization and Vehicle Tracking.

##

# Object Detection

We evaluated three object detection techniques: DETR, CornerNet, and YOLOv8, across three distinct datasets: the msCOCO Dataset, Kaggle’s Traffic Detection Dataset, and the Indian Driving Dataset(IDD).

To compare the performance of these algorithms, we utilized the following metrics:

• Mean Intersection over Union (mIOU): Calculated as the mean IOU of all detected bounding boxes relative to their corresponding ground truth bounding boxes, employing the Hungarian algorithm for association.

• Mean Average Precision (mAP): Computed as the average of Average Precision values across various object categories, providing a comprehensive evaluation of the model's detection performance across different classes.

• Average Inference Time: Represents the average time taken for inference on images. Lower inference times indicate faster model performance.

#### Performance on Above Datasets:

| Model     | msCOCO mAP | msCOCO mIOU | msCOCO AT | Kaggle mAP | Kaggle mIOU | Kaggle AT | IDD mAP | IDD mIOU | IDD AT |
|-----------|--------------------|-------------|-----------|--------------------|-------------|-----------|-----------------|----------|--------|
| DETR      | 0.426              | 0.84        | 0.036     | 0.397              | 0.83        | 0.036     | 0.265           | 0.84     | 0.036  |
| CornerNet | 0.406              | 0.79        | 0.036     | 0.297              | 0.78        | 0.038     | 0.254           | 0.78     | 0.037  |
| Yolov8    | 0.521              | 0.859       | 0.013     | 0.603              | 0.831       | 0.015     | 0.333           | 0.845    | 0.015  |




##



# Object Tracking 

We delve into two multi-object tracking algorithms: FairMOT and YOLOv8 with ByteTrack. Beginning with pre-trained COCO models, we trained our models on the Indian Driving Dataset (IDD) and subsequently tested them on the Gram Dataset. Our evaluation includes an in-depth analysis of performance metrics and insights gained from this experimentation process.


## 1. YOLOv8_ByteTrack

### Data Preparation 

Datasets are availavle on the following links: [Indian Driving Dataset(IDD)](https://idd.insaan.iiit.ac.in/dataset/details/) and [GRAM Road-Traffic Monitoring](https://gram.web.uah.es/data/datasets/rtm/index.html)

Dataset should be in following format.

#### IDD Dataset
```bash
Object Tracking/
    |-- YOLOv8_ByteTrack/
        |-- IDD/
            |-- images/
                |-- train/
                    |-- 1.jpg
                    ...
                |-- val/
                    |-- 1.jpg
                    ...
                |-- test/
                    |-- 1.jpg
                    ...
            |-- labels/
                |-- train/
                    |-- 1.txt
                    ...
                |-- val/
                    |-- 1.txt
                    ...
            |-- train.txt
            |-- val.txt
            |-- test.txt
            |-- IDD_to_YOLOv8.ipynb
```
#### GRAM Dataset

```bash
Object Tracking/
    |-- YOLOv8_ByteTrack/
        |-- GRAM/
            |-- GRAM-RTMv4/
            |-- images/
                |-- M-30/
                    |-- 1.jpg
                    ...
                |-- M-30-HD/
                    |-- 1.jpg
                    ...
                |-- Urban1/
                    |-- 1.jpg
                    ...
            |-- labels/
                |-- M-30/
                    |-- 1.txt
                    ...
                |-- M-30-HD/
                    |-- 1.txt
                    ...
                |-- Urban1/
                    |-- 1.txt
                    ...
        
            |-- test.txt
            |-- train.txt
            |-- val.txt
            |-- GRAM_to_YOLOv8.ipynb

  ```

  


### Installation & Run Code


In order to use this repository, you need to create an environment:
1. Clone the github repository:
```bash
git clone https://github.com/anujiisc/DeepTraffic.git
```
2. Before training model on these datasets, download everything from [YOLOv8_ByteTrack Onedrive](https://onedrive.com) and put these things inside DeepTraffic/Object Tracking/YOLOv8 ByteTrack/ folder.

3. Now you need to create a conda environment:
```bash
cd DeepTraffic/
cd Object\ Tracking/
cd YOLOv8_ByteTrack/
conda env create -f environment.yml
conda activate yolov8
```
4. Train model on IDD dataset using following command:
```bash
python train.py --batch 64 --data idd.yaml --pretrained_weights yolov8n.pt \
--device 0,1,2,3,4,5,6,7 --epoch 100 --img_size 1280 --save_results yolov8n_idd
```
You can adjust these arguments as required. The results and model weights are saved in directory
runs/detect/yolov8n_idd. 

5. Finetune IDD trained model on GRAM dataset using following command:
```bash
python train.py --batch 32 --data gram.yaml \
--pretrained_weights ./runs/detect/yolov8n_idd/weights/best.pt \
--device 3,4,5,6 --epoch 50 --img_size 1280 --save_results yolov8n_gram
```
You can adjust these arguments as required. The results and model weights are saved in directory runs/detect/yolov8n_gram.


6. Track videos using trained model using following command:
```bash
python track.py --trained_weights ./runs/detect/yolov8n_gram/weights/best.pt \
--video_path ./videos/M-30.mp4 --save_results ./results/M-30.txt \
--type_tracker bytetrack.yaml
```
You can adjust these arguments as required. Tracking results will be saved in directory results/M-30.txt as in format required for MOT challenge.

7. Calculate evaluation metrics on tracked results received from previous command:
```bash
python eval.py
```
Actual MOT challenge format files are inside GRAM_MOT/ and results files are inside results/. The code is available in mot_format.ipynb . Tracking code is also available in this file.

8. If you want to run tracking on demo video, use the below command.
```bash
python demo.py --video_path ./videos/demo.mp4 
```
Resulted video will be saved in folder runs/detect/track/demo.avi

### Performance on GRAM Dataset using YOLOv8_ByteTrack


| Dataset   | FPS   | MOTA | IDF1 | MT | ML |
|-----------|-------|------|------|----|----|
| M-30      | 31.35 | 0.90 | 0.95 | 32 | 0  |
| M-30-HD   | 29.85 | 0.87 | 0.93 | 30 | 0  |
| Urban1    | 32.05 | 0.88 | 0.94 | 0  | 0  |


    



















##



## 2. FairMOT


### Data Preparation 

Dataset should be in following format.

#### IDD Dataset

```bash
Object Tracking/
    |-- FairMOT/
        |-- IDD/
            |-- images/
                |-- train/
                    |-- 1.jpg
                    |-- 2.jpg
                    ...
                |-- val/
                    |-- 1.jpg
                    |-- 2.jpg
                    ...
            |-- labels_with_ids/
                |-- train/
                    |-- 1.txt
                    |-- 2.txt
                    ...
                |-- val/
                    |-- 1.txt
                    |-- 2.txt
                    ...   

  ```

#### GRAM Dataset
  
```bash
Object Tracking/
    |-- FairMOT/
        |-- GRAM/
            |-- images/
                |-- test/
                    |-- M-30/
                        |-- 1.jpg
                        ...
                    |-- M-30-HD/
                        |-- 1.jpg
                        ...
                    |-- Urban1/
                        |-- 1.jpg
                        ...
                |-- train/
                    |-- M-30/
                        |-- 1001.jpg
                        ...
                    |-- M-30-HD/
                        |-- 1001.jpg
                        ...
                    |-- Urban1/
                        |-- 1001.jpg
                        ...
            |-- labels_with_ids/
                |-- test/
                    |-- M-30/
                        |-- 1.txt
                        ...
                    |-- M-30-HD/
                        |-- 1.txt
                        ...
                    |-- Urban1/
                        |-- 1.txt
                        ...
                |-- train/
                    |-- M-30/
                        |-- 1001.txt
                        ...
                    |-- M-30-HD/
                        |-- 1001.txt
                        ...
                    |-- Urban1/
                        |-- 1001.txt
                        ...
        
            |-- M-30/
                |-- gt/
                    |-- gt.txt
            |-- M-30-HD/
                |-- gt/
                    |-- gt.txt
            |-- Urban1/
                |-- gt/
                    |-- gt.txt
  ```

### Installation & Run Code


In order to use this repository, you need to follow these steps:
1. Clone the GitHub repository, if not cloned yet:
```bash
git clone https://github.com/anujiisc/DeepTraffic.git
```

2. Navigate to the FairMOT directory:
```bash
cd DeepTraffic/
cd Object\ Tracking/
cd FairMOT/
conda env create -f environment.yml
conda activate fairmot
```
3. We use [DCNv2 pytorch 1.7](https://github.com/ifzhang/DCNv2/tree/pytorch_1.7) in our backbone network (pytorch 1.7 branch). Run the following commands to clone it.
```
git clone https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh
cd ..
```
4. Before training model on these datasets, download everything from FairMOT Onedrive and put these things inside DeepTraffic/Object Tracking/FairMOT/ folder.
5. Our baseline FairMOT model (DLA-34 backbone) is pretrained on the IDD(Indian Driving Dataset) for 30 epochs with the self-supervised learning approach and then trained on the GRAM dataset for 60 epochs.
6. Pretrained models are inside models/ directory eg: models/coco dla.pth .
7. Train the model using IDD dataset with the following command:
```bash
cd src/
python train.py mot --exp_id IDD --gpus 1,3 --batch_size 40 \
--load_model ../models/coco_dla.pth --num_epochs 30 --lr_step 50 \
--data_cfg ../src/lib/cfg/IDD.json
```
Adjust the arguments as required. Model weights will be saved in exp/mot/ with exp_id.

8. We fine-tuned the model using the Gram dataset for an additional 60 epochs using the following command:
```bash
python train.py mot --exp_id GRAM --gpus 1,3 --batch_size 40 \
--load_model ../exp/mot/IDD/model_last.pth --num_epochs 60 --lr_step 50 \
--data_cfg ../src/lib/cfg/GRAM.json
```
Adjust the arguments as required. Model weights will be saved in exp/mot/ with exp_id.

9. Now perform tracking on GRAM test dataset:
```bash
python track.py mot --load_model ../exp/mot/GRAM/model_last.pth --conf_thres 0.4
```
Tracking results will be saved in results/GRAM/ folder with (MOT challenge format) and tracked images will be saved in folder outputs/GRAM/ . We are getting FPS 26.66 .

10. Calculate evaluation metrics on tracked results received from previous command:
```bash
cd ..
python eval.py 
```
Actual MOT challenge format files are inside GRAM_MOT/ and results files are inside results/GRAM/ folder.

### Performance on GRAM Dataset using FairMOT

| Dataset   | FPS   | MOTA | IDF1 | MT | ML |
|-----------|-------|------|------|----|----|
| M-30      | 16.45 | 0.87 | 0.93 | 33 | 0  |
| M-30-HD   | 16.56 | 0.86 | 0.93 | 30 | 0  |
| Urban1    | 16.29 | 0.59 | 0.75 | 1  | 6  |
