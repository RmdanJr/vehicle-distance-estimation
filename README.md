# distance and speed estimation
## 1. introduction - project description
#### the project mainly is all about distance, speed estimation and, object detection using the thermal imaging, we are going here to build deep learning model that can do distance and speed estimation that works on photoes or videoes on thermal imaging.
for more details [the project text documentaion](https://docs.google.com/document/d/1ZbCR8RsUdPyrVYYg5FUxKb3aYmQieXIAirAn9aIEj7o/edit)

## 2.Usage
### 2.1 object detection
![sample of thermal object detection](https://user-images.githubusercontent.com/47370980/172619765-3ab6f4b2-49cd-41ad-a245-3ae385aa59b7.png)

we used YOLOv5 on the detecton YOLOv5 rocket is a family of object detection architectures and models pretrained on the COCO dataset.
you can visit the formal docs [here](https://docs.ultralytics.com/#yolov5).
we have made some modifications on YOLOv5 to be applicaple with our own dataset.

first of all you need to dounload YOLOv5 and install the initial requirements.
```
git clone https://github.com/Ahmedabdelalem61/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```
then if you need to train the model so you can run this command.
#### note that 
***dataset.yaml*** is a file inside the Yolo folder that have some specification like the path of your dataset labels && the dataset itself.

***yolov5s.yaml*** is a file also inside the Yolo folder that that all about the configrations of the models like the number of classes/objects the layers architecture

***weightsFile*** is the wights that the model should gives to me after the training but initialy should be dounloaded before the training if u need to dounload it from google drive you can write the upcomig command. you should now also that weghts specified on 3 classes initially that needed for our project
```
import gdown
gdown --id 1-4gQP0YzFkkYCZoCaIokl9eM94TaiVAu
```
```
!python train.py --epochs 5 --data dataset.yaml --cfg yolov5s.yaml --weights weightsFile
```

then if you need to detect the images/videoes all what you can do is whriting the upcomming command.
#### note that
***weights*** is whare is the  file of the best weights after the training proccess happen.

***source*** is the images path or specified videoes link or path
```
python detect.py --save-txt --weights /content/best.pt --conf 0.4 --source 'https://youtu.be/CvI5nvUdbsM'
```
### 2.2 distance estimation

### 2.3 speed estimation

## 3. conclusion
## small brief about this context
