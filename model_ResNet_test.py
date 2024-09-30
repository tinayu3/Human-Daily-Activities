# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from statistics import mode


# Normalize face data and map pixel values ​​from 0-255 to 0-1
def preprocess_input(images):
    """ preprocess input by substracting the train mean
    # Arguments: images or image of any shape
    # Returns: images or image with substracted train mean (129)
    """
    images = images/255.0
    return images




class ResNet(nn.Module):
    def __init__(self, *args):
        super(ResNet, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0],-1)


class GlobalAvgPool2d(nn.Module):
    # The global average pooling layer can be implemented by setting the pooling window shape to the height and width of the input
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# Residual Neural Network
class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

    
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # The number of channels of the first module is the same as the number of input channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

resnet = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7 , stride=2, padding=3),
    nn.BatchNorm2d(64), 
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
resnet.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
resnet.add_module("resnet_block2", resnet_block(64, 128, 2))
resnet.add_module("resnet_block3", resnet_block(128, 256, 2))
resnet.add_module("resnet_block4", resnet_block(256, 512, 2))
resnet.add_module("global_avg_pool", GlobalAvgPool2d()) # Output of GlobalAvgPool2d: (Batch, 512, 1, 1)
resnet.add_module("fc", nn.Sequential(ResNet(), nn.Linear(512, 7))) 


#A facial recognition classifier that comes with opencv
detection_model_path = 'model/haarcascade_frontalface_default.xml'

classification_model_path = 'model/model_resnet.pkl'

# Loading the face detection model
face_detection = cv2.CascadeClassifier(detection_model_path)

# Loading the expression recognition model
emotion_classifier = torch.load(classification_model_path)


frame_window = 10

#Emoticon Tags
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

emotion_window = []

# Turn on the camera, 0 is the laptop's built-in camera
video_capture = cv2.VideoCapture(0)
# Video file recognition
# video_capture = cv2.VideoCapture("video/example_dsh.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.startWindowThread()
cv2.namedWindow('window_frame')

while True:
    # Read a frame
    _, frame = video_capture.read()
    frame = frame[:,::-1,:]#Horizontal flip, in line with selfie habits
    frame = frame.copy()
    # Get the grayscale image and create an image object in memory
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get all faces in the current frame
    faces = face_detection.detectMultiScale(gray,1.3,5)
    # For all faces found
    for (x, y, w, h) in faces:
        # Draw a rectangular box around the face, (255,0,0) is the color, 2 is the line width
        cv2.rectangle(frame,(x,y),(x+w,y+h),(84,255,159),2)

        # Get face image
        face = gray[y:y+h,x:x+w]

        try:
            # The shape becomes (48,48)
            face = cv2.resize(face,(48,48))
        except:
            continue

        # Expand the dimension and the shape becomes (1,48,48,1)
        # Convert (1, 48, 48, 1) to (1, 1, 48, 48)
        face = np.expand_dims(face,0)
        face = np.expand_dims(face,0)

        # Normalize face data and map pixel values ​​from 0-255 to 0-1
        face = preprocess_input(face)
        new_face=torch.from_numpy(face)
        new_new_face = new_face.float().requires_grad_(False)
        
        # Call our trained expression recognition model to predict the classification
        emotion_arg = np.argmax(emotion_classifier.forward(new_new_face).detach().numpy())
        emotion = emotion_labels[emotion_arg]

        emotion_window.append(emotion)

        if len(emotion_window) >= frame_window:
            emotion_window.pop(0)

        try:
            # Get the category with the most occurrences
            emotion_mode = mode(emotion_window)
        except:
            continue

        # In the upper part of the rectangular box, output the classification text
        cv2.putText(frame,emotion_mode,(x,y-30), font, .7,(0,0,255),1,cv2.LINE_AA)

    try:
        # Display the image from memory to the screen
        cv2.imshow('window_frame', frame)
    except:
        continue

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

