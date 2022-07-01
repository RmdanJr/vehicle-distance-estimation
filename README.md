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
![object_detection_sample](https://user-images.githubusercontent.com/47370980/176802901-4c005dbe-57b6-4566-8ae5-26a6d1f7ec85.gif)

#### Results

to see the model results like recall, precision metrics and object loss for testing and validating
```
import cv2
import matplotlib.pyplot as plt

# display results image
# imread() reads image as grayscale, second argument is one => grayscale, zero => RGB 
img = cv2.imread('training-results/results.png', 1)
plt.imshow(img)
plt.axis('off')
```
![trainnig_result](https://raw.githubusercontent.com/RmdanJr/vehicle-distance-estimation/main/object-detector/training-results/results.png)

#### Code tutorial

you can also visit obeject detection [tutorial](https://github.com/RmdanJr/vehicle-distance-estimation/blob/main/tutorials/object_detection.ipynb) 

### 2.2 Distance Estimation
To make Distance estimation to objects (cars, pedestrians, bycycle) over the detection information you can use the ```distance-estimator```  

first you need to 

#### Cloning Repo

```
!git clone https://github.com/RmdanJr/vehicle-distance-estimation.git
```

then you must be inside the ```distance-estimator```

```
%cd vehicle-distance-estimation/distance-estimator/
```

#### Packages & Setup

installing the initial requirements.

```
!pip install -r requirements.txt
```

#### KITTI Dataset
Download Dataset

```
!bash scripts/download-kitti-dataset.sh
```

#### Format Dataset as YOLOv5 Format

as sayed before the data-set labels and images must be formated like the Yolo architecture works with.

```
!bash scripts/organize-dataset-format.sh
```

#### Generate CSV File

generating csv files that the model will deal with that have some information about the training annotations

```
!python generate-csv.py --input=kitti-dataset/train_annots/ --filename=annotations.csv --results .
```

#### Training

first you need to prerain the model on our dataset.

```
!python training_continuer.py --model models/model@1535477330.json --weights models/model@1535477330.h5 --results models/ --train train.csv --test test.csv
```

then you need to train the model another time to fine tuning the weights between training and testing dataset.

```
!python training_continuer.py --results models/ --train train.csv --test test.csv
```

#### Making Predictions

```
!python inference.py --data annotations.csv --model models/model@1535470106.json --weights models/model@1535470106.h5 --results .
```

#### Visualizing

now you can see the predection result.

```
!python visualizer.py --data ../results/data/data.csv --frames ../object-detector/results/frames/ -fps 90 --results results
```
![distance_estimation_sample](https://user-images.githubusercontent.com/47370980/176803779-1c676b4a-5a89-4afc-b135-d4ac8ff6eaae.gif)

#### Results

to see the model result graph write the upcomming model. 

```
from IPython.display import HTML
from base64 import b64encode
mp4 = open('results/output.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
```

[HASSAN](https://www.google.com/webhp?hl=en&sa=X&ved=0ahUKEwiA7Mf86aD4AhXIwQIHHZtcBHUQPAgI)

#### Code tutorial

you can also visit distance estimation [tutorial](https://github.com/RmdanJr/vehicle-distance-estimation/blob/main/tutorials/distance_estimation.ipynb) 

## 3. Conclusion

let's summarize what is the main idea and what we had used to work on it.
we need to make object detection so we have used Yolov5 to have some annotations that needed to passed throug the distance estimation files to bound the the original images with bounding Boxes and passes the bounding boxes coordinates to deal with distance estimator to calculate the distance for the objects after traing the distance estimator and finally we make some predections and visulize them.
