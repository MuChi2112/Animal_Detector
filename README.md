# Animal Detector Application
## Introduction
The Animal Detector is a desktop application designed to identify animal species from images. This application employs a convolutional neural network (CNN) model for image classification and is built using Python with libraries such as Tkinter, OpenCV, PIL, NumPy, and TensorFlow.


## Key Features
- User-Friendly Interface: Simple GUI for uploading and analyzing images.
- Deep Learning Powered: Utilizes a pre-trained CNN for accurate animal species classification.
- Supports Various Image Formats: Compatible with common image file formats.
## Requirements
-Python 3.x
-OpenCV
-PIL
-NumPy
-TensorFlow
-imutils
-Tkinter

## Installation
Ensure you have Python installed on your system. Install required libraries using pip:

```
pip install numpy opencv-python pillow tensorflow imutils
```

## Usage
- Launch the application.
- Click on "Upload an image" to select an image file.
- The application displays the image and predicts the animal species.

## Running the Application
Execute the script from your terminal or command prompt:

```
python app.py
```

## Note
- Ensure the CNN model file (animal.model) and label file (lb.pickle) are in the same directory as the script.
