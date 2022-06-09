# object detection & distance estimation
## 1. introduction - project description
#### the project mainly is all about distance estimation and, object detection using the thermal imaging.
for more details [the project text documentaion](https://docs.google.com/document/d/1ZbCR8RsUdPyrVYYg5FUxKb3aYmQieXIAirAn9aIEj7o/edit)

## 2.Usage
first of all we need to clone the main repository```vehicle-distance-estimation``` using the upcomming command

```
!git clone https://github.com/RmdanJr/vehicle-distance-estimation.git
```


### 2.1 object detection

![object detection sample_1](https://user-images.githubusercontent.com/47370980/172878533-16ed416a-a51f-4929-ab17-b69ac912e631.jpg)

we used YOLOv5 on the detecton YOLOv5 rocket is a family of object detection architectures and models pretrained on the COCO dataset.
you can visit the formal docs [here](https://docs.ultralytics.com/#yolov5).
we have made some modifications on YOLOv5 to be applicaple with our own dataset.

first you need to be inside the Yolo directory/folder
```
%cd vehicle-distance-estimation/object-detector/
```
#### Packages & Setup

installing the initial requirements.

```
!pip install -r requirements.txt
```
#### Preparing the requirement

```
!python setup.py
```
#### Downloading the dataset

you can click [here](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset) to work with FLIR Dataset  
```
!bash scripts/download-flir-dataset.sh
```

#### Formating the Dataset as YOLOv5 Format

yolo architecture depends on specific format for the dataset so you can use the follwoing comman to format it
```
!python format-dataset.py
```
#### Fine-tuning YOLOv5s

yolo have alot of models but here we will configure the the YOLOv5s model and yaml files


first you need to configure the yaml file of the classes and the paths of trainig and validation
so you can write the following command
```
!python create-yaml.py
```
#### Model's YAML Configuration File

then you need to configure the configuration file that have the layres architecture.
```
!python configure-yolo-yaml.py
```

#### Training YOLOv5

now you able to retrain the YOLO model on our dataset(flir)
```
!python train.py --epochs 2 --data dataset.yaml --cfg yolov5s.yaml
```

#### Downloading the trained wieghts

 ```
 !gdown --folder 10jpVGSHGILDt85QGf5KwHji0sUjZXbWR
 ```
 
 #### Detection
 
 make some detection on images/videoes by
 
 note that
 **best.pt** is the weights that we have downladed but after training it on YoloV5s model
 
 ```
 !python detect.py --save-txt --weights training-results/weights/best.pt --conf 0.4 --source 'https://youtu.be/CvI5nvUdbsM'
 ```

#### Generate Objects Coordinates Sheet

The sheet is expected to have a row for each frame & a column for each category. Each cell must have all center coordinates of detected objects of the column (category) on the row (frame).
```
!python generate-coordinates-sheet.py
```
#### Detection Examples

```
!python display-examples.py

```

#### Results

to see the model results like reacal, precision metrics and object loss for testing and validating
```
import cv2
import matplotlib.pyplot as plt

# display results image
# imread() reads image as grayscale, second argument is one => grayscale, zero => RGB 
img = cv2.imread('training-results/results.png', 1)
plt.imshow(img)
plt.axis('off')
```
### 2.2 distance estimation
To make Distance estimation to objects (cars, pedestrians, bycycle) over the detection information you can use the ```distance-estimator```  all what you need is 

### Training
1. (Optional) Use ```hyperopti.py``` for hyperparameter optimization. Choose the hyperparameters you would like to try out.
2. You can use result of 1. and edit ```train.py``` accordingly. Otherwise, use ```train.py``` to define your own model, choose hyperparameters, and start training!
### Inference
1. Use ```inference.py``` to generate predictions for the test set.
```
python inference.py --modelname=models/model@1535470106.json --weights=models/model@1535470106.h5
```
2. Use ```visualizer.py``` to visualize the predictions.
```
cd distance-estimator/
python visualizer.py
```


## 3. conclusion
## brief content about ther result (will aded later after the final testeing)
