# Graduation-Project

## Related Work
---
## Related Work

## The current solutions and their shortcoming
---

## 1. Speed Estimation, and Classification System Based on Virtual Detection Zone and YOLO
There is already a model that uses normal images to estimate  speed but in the presence of clouds and blurred vision here is the problem

- ## 1.1. Data Set Preparation 

~~~
The data set used in this study was prepared by collecting traffic videos recorded with online cameras installed along various roads in Taiwan. Image data were extracted from the traffic videos using a script, and labeling was performed using an open-source software application called “labeling”. According to the common types of vehicles on the road are announced by the Directorate General of Highways, Ministry of Transportation and Communications (MOTC) in Taiwan, this study divides six different sizes, such as sedans, trucks, scooters, buses, hlinkcars, and flinkcars, in the training process, and the vehicle lengths of these six vehicle classes are listed in Table 1. In this study, we used YOLO to perform vehicle classification without using the length of the vehicle.

~~~
- [Dataset](https://github.com/tzutalin/labelImg)

- ## 2.1.  Vehicle Detection and Classification
~~~
This study uses the YOLO algorithm to classify vehicles into six classes. The validation method is used for verifying the vehicle classification in the collected videos. A visual classifier based on the YOLO algorithm is used to verify the vehicle classification capability.In the training process, when a vehicle belonging to one of the six classes is detected, all bounding boxes are extracted, their classes are manually labeled, and the labeled data are passed to the YOLO model for classifying the vehicle.
~~~
~~~
The YOLOv3 model architecture displayed in Figure 4 was used in this study. Images of size 416 × 416 px were input into the Darknet-53 network. This feature extraction network comprises 53 convolutional layers, and thus, it is called Darknet-53 [11]. In Darknet-53, alternating convolution kernels are used, and after each convolution layer, a batch normalization layer is used for normalization. The leaky rectified linear unit function is used as the activation function, the pooling layer is discarded, and the step size of the convolution kernel is increased to reduce the size of the feature map. The YOLOv3 model uses ResNet for feature extraction and subsequently uses the feature pyramid top-down and lateral connections to generate three features with sizes of 13 × 13 × 1024, 26 × 26 × 512, and 52 × 52 × 256 px. The final output depth is (5 + class) × 3, which indicates that the following parameters are predicted: four basic parameters and the credibility of a box across three regression bounding boxes as well as the possibility of each class being contained in the bounding box. YOLOv3 uses the sigmoid function to score each class. When the class score is higher than the threshold, the object is considered to belong to a given category, and any object can simultaneously have multiple class identities without conflict.
~~~

<img src="https://static-01.hindawi.com/articles/mpe/volume-2021/1577614/figures/1577614.fig.004.svgz">

- ## 2.2. Speed Estimation
~~~
In this subsection, the vehicle speed can be estimated using the proposed method. Table 5 lists the actual and the estimated speeds of the vehicles. The results indicate that the average absolute percentage error of vehicle speed estimation was about 7.6%. The use of online video for vehicle speed estimation will cause large speed errors due to network delays. Therefore, network stability is essential to reduce the percentage error in the speed estimation.
~~~
[table](https://www.hindawi.com/journals/mpe/2021/1577614/tab5/)

### 2. Detection, Tracking, and Estimation using Normal Imaging

- There is already a model that uses normal images to estimate  distance and speed but in the presence of clouds and blurred vision here is the problem.[ref](https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/)<br>
- they used OpenCV and Deep Learning to detect vehicles in video streams, track them, and apply speed estimation to detect the MPH/KPH of the moving vehicle.

### 3. Detection, Tracking, and Estimation using thermal imaging
~~~
In fact, there is a company that solved the problem of distance estimation between cars in bad weather through a thermal camera,  
but it did not provide speed estimation, and this is very important to reduce accidents. We'll discuss that in the our proposed method.
~~~

### 4. Arduino Ultrasonic Sensor

~~~
Estimates the distance only. But can’t detect objects whose distance more than 20m. And can’t detect objects which move with high speed.

~~~
