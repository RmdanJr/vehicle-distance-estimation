# Vehicle Distance Estimation

The project is about vehicle object detection & distance estimation using thermal imaging.

The main idea is to make object detection using Yolov5 after fine-tuning it on the FLIR dataset to enable the model to accurately detect objects on thermal images and videos. Then using object detection results as an input to the distance estimation model - trained on the KITTI dataset - we estimate the distance. Finally, we visualize predictions.

![distance_estimation_sample](https://user-images.githubusercontent.com/47370980/176803779-1c676b4a-5a89-4afc-b135-d4ac8ff6eaae.gif)

## Usage

### Object Detection
Detect objects on images/video frames.

#### Setup

- clone repo ```git clone https://github.com/RmdanJr/vehicle-distance-estimation.git```
- navigate to object detector directory ```cd vehicle-distance-estimation/object-detector/```
- install requirements ```pip install -r requirements.txt```
- setup environment ```python setup.py```

#### Training

##### Dataset

- download dataset ```bash scripts/download-flir-dataset.sh```
- format dataset as YOLOv5 format ```python format-dataset.py```

##### Fine-tuning YOLOv5s

- create YAML configuration file ```python create-yaml.py```
- modify yolovs YAML file ```python configure-yolo-yaml.py```
- train model on our custom dataset ```python train.py --epochs 50 --data dataset.yaml --cfg yolov5s.yaml```

 #### Detection
 
 make detections on images/videoes using our training weights.
 
- download our pre-trained model's weights ```gdown --folder 10jpVGSHGILDt85QGf5KwHji0sUjZXbWR```
- detect objects on input video ```python detect.py --save-txt --weights training-results/weights/best.pt --conf 0.4 --source video.mp4```

#### Generate Objects Coordinates Sheet

Frames on rows & classes on columns. Each cell has all center coordinates of the detected objects in a frame on a row from class on a column.

- generate sheet from text labels ```python generate-coordinates-sheet.py```

### Distance Estimation
Estimate the distance of objects using object detection results.

#### Setup

- navigate to distance estimator directory ```cd vehicle-distance-estimation/distance-estimator/```
- install requirements ```pip install -r requirements.txt```

#### Training

##### Dataset

- download dataset ```bash scripts/download-kitti-dataset.sh```
- format dataset ```bash scripts/organize-dataset-format.sh```

##### Generate CSV File

- generate train and test csv files from dataset annotations ```python generate-csv.py --input=kitti-dataset/train_annots/ --filename=annotations.csv --results .```

##### Train Model

- continue training a pre-trained model ```python training_continuer.py --model models/model@1535477330.json --weights models/model@1535477330.h5 --results models/ --train train.csv --test test.csv```

<ins> OR </ins>

- train model from scratch ```python train.py --results models/ --train train.csv --test test.csv```

#### Estimation

- estimate distance ```python inference.py --data annotations.csv --model models/model@1535470106.json --weights models/model@1535470106.h5 --results .```
- visualize predections ```python visualizer.py --data ../results/data/data.csv --frames ../object-detector/results/frames/ -fps 90 --results results```


## Appendices

- [Docs](docs)
- [Tutorials](tutorials)
- [FLIR Dataset](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
