# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
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



class VGG(nn.Module):
    def __init__(self, *args):
        super(VGG, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0],-1)


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # This will halve the width and height
    return nn.Sequential(*blk)
    
    
conv_arch = ((2, 1, 32), (3, 32, 64), (3, 64, 128))
# After 5 vgg_blocks, the width and height will be halved 5 times, becoming 224/32 = 7
fc_features = 128 * 6* 6 # c * w * h
fc_hidden_units = 4096 

def vgg(conv_arch, fc_features, fc_hidden_units):
    net = nn.Sequential()
    # Convolutional layer
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # Each time a vgg_block is passed, the width and height will be halved
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # Fully connected layer
    net.add_module("fc", nn.Sequential(
                                 VGG(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 7)
                                ))
    return net


# A facial recognition classifier that comes with opencv
detection_model_path = 'model/haarcascade_frontalface_default.xml'

classification_model_path = 'model/model_vgg.pkl'

# Loading the face detection model
face_detection = cv2.CascadeClassifier(detection_model_path)

# Loading the expression recognition model
emotion_classifier = torch.load(classification_model_path)


frame_window = 10

# Emoticon Tags
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

