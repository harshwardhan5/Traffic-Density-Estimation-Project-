
# Traffic Density Estimation using YOLOv8
## Introduction

This project focuses on traffic density estimation using the YOLOv8 object detection model. The project involves detecting and analyzing the number of vehicles in specified lanes of a traffic video. The implementation includes loading a pre-trained YOLOv8 model, performing inferences on images and videos, visualizing the results, and analyzing traffic intensity.



# Project Description

## Dataset Preparation
- The dataset should be organized with separate folders for    training and validation images.
- A YAML file (data.yaml) should specify the paths to the training and validation datasets.
## Model Training
- The model is trained using the YOLOv8 architecture.
- Custom training parameters include epochs, batch size, learning rate, and optimization strategy.
## Traffic Density Analysis
- Define regions of interest (ROIs) in the video frames to analyze traffic in specific lanes.
- Use the trained model to detect vehicles in each frame of the video.
- Count vehicles in each lane and determine traffic intensity based on a threshold.
## Installation

To get started with this project, follow these step


1. Clone this project

```bash
git clone https://github.com/harshwardhan5/traffic-density-estimation.git
cd traffic-density-estimation
```

2. Install the required dependencies:
```bash
pip install ultralytics opencv-python-headless pillow matplotlib seaborn pyyaml pandas
```

3. Mount Google Drive (if using Collab):
```bash
from google.colab import drive
drive.mount('/content/drive')
```


    
## Usage

1. Load the YOLOv8 pre-trained model:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
```

2. Perform inference on an image:

```python
Perform inference on an image:
results = model.predict(source='/path/to/image.jpg', imgsz=640, conf=0.5)
```

3. Visualize results:

```python
import cv2
import matplotlib.pyplot as plt

sample_image = results[0].plot(line_width=2)
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
plt.imshow(sample_image)
plt.title('Detected Objects')
plt.axis('off')
plt.show()
```

4. Train the model on a custom dataset:
 ```python
 results = model.train(data='/path/to/data.yaml', epochs=100, imgsz=640, device=0, patience=50, batch=32)
```

5. Perform traffic density analysis on a video:

```python
video_path = '/path/to/sample_video.mp4'
best_model = YOLO('/path/to/best_model.pt')
best_model.predict(source=video_path, save=True)
```




# Results
The project outputs include:

- Annotated images and videos with detected vehicles.
- Traffic density analysis displayed on the video with vehicle counts and traffic intensity (Smooth/Heavy) for each lane.
# Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.


# License
This project is licensed under the MIT License.


[MIT](https://choosealicense.com/licenses/mit/)

