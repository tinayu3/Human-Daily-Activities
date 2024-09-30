import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2

class Dataset(object):
	def __getitem__(self, index):
		raise NotImplementedError
	def __len__(self):
		raise NotImplementedError
	def __add__(self, other):
		return ConcatDataset([self, other])

# Parameter initialization
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # Find string, if not found, return -1, if not -1, the string contains the character
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# Verify the accuracy of the model on the validation set
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        pred = model.forward(images)
        pred = np.argmax(pred.data.numpy(), axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc

class FaceDataset(data.Dataset):
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        df_path = pd.read_csv(root + '\\image_emotion.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '\\image_emotion.csv', header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    def __getitem__(self, item):
        face = cv2.imread(self.root + '\\' + self.path[item])
        # Read single-channel grayscale image
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Gaussian Blur
        # face_Gus = cv2.GaussianBlur(face_gray, (3,3), 0)
        # Histogram Equalization
        face_hist = cv2.equalizeHist(face_gray)
        # Pixel value normalization
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0 # To adapt to the design of the convolutional neural network API in pytorch, the original image needs to be reshaped
        # The data used for training must be of tensor type
        face_tensor = torch.from_numpy(face_normalized) # Convert the numpy data type in python to the tensor data type in pytorch
        face_tensor = face_tensor.type('torch.FloatTensor') # Specify the 'torch.FloatTensor' type. Otherwise, an error will be reported due to data type mismatch after it is sent to the model.
        label = self.label[item]
        return face_tensor, label

    # Get the number of samples in the dataset
    def __len__(self):
        return self.path.shape[0]

class FaceCNN(nn.Module):
    # Initialize the network structure
    def __init__(self):
        super(FaceCNN, self).__init__()

        # First convolution, pooling
        self.conv1 = nn.Sequential(
            # The number of input channels is in_channels, the number of output channels (i.e. the number of channels of the convolution kernel) is out_channels, the convolution kernel size is kernel_size, the step length is stride, and the number of rows and columns filled with 0 is padding.
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), # Convolutional Layer
            nn.BatchNorm2d(num_features=64), # Normalization
            nn.RReLU(inplace=True), # Activation Function
            # output(bitch_size, 64, 24, 24)
            nn.MaxPool2d(kernel_size=2, stride=2), # Max Pooling
        )

        # Second convolution, pooling
        self.conv2 = nn.Sequential(
            # input:(bitch_size, 64, 24, 24), output:(bitch_size, 128, 24, 24), (24-3+2*1)/1+1 = 24
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 128, 12 ,12)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # The third convolution and pooling
        self.conv3 = nn.Sequential(
            # input:(bitch_size, 128, 12, 12), output:(bitch_size, 256, 12, 12), (12-3+2*1)/1+1 = 12
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 256, 6 ,6)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Parameter initialization
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    # Forward Propagation
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Data flattening
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y

def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    # Load data and split batches
    train_loader = data.DataLoader(train_dataset, batch_size)
    # Build the model
    model = FaceCNN()
    # Loss Function
    loss_function = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # Learning rate decay
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # Round-by-round training
    for epoch in range(epochs):
        # Record loss value
        loss_rate = 0
        # scheduler.step() # Learning rate decay
        model.train() # train model
        for images, emotion in train_loader:
            # Gradient zeroing
            optimizer.zero_grad()
            # Forward Propagation
            output = model.forward(images)
            # Error calculation
            loss_rate = loss_function(output, emotion)
            # Back Propagation of Error
            loss_rate.backward()
            # Update Parameters
            optimizer.step()

        # Print the loss of each round
        print('After {} epochs , the loss_rate is : '.format(epoch+1), loss_rate.item())
        if epoch % 5 == 0:
            model.eval() # Model Evaluation
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print('After {} epochs , the acc_train is : '.format(epoch+1), acc_train)
            print('After {} epochs , the acc_val is : '.format(epoch+1), acc_val)

    return model

def main():
    # Dataset instantiation (creating a dataset)
    train_dataset = FaceDataset(root='face_images/train_set')
    val_dataset = FaceDataset(root='face_images/verify_set')
    # Hyperparameters can be specified by yourself
    model = train(train_dataset, val_dataset, batch_size=128, epochs=100, learning_rate=0.1, wt_decay=0)
    # Save the model
    torch.save(model, 'model/model_cnn.pkl')


if __name__ == '__main__':
    main()