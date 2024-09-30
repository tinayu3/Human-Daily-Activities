
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm



BATCH_SIZE = 128
LR = 0.01
EPOCH = 60
DEVICE = torch.device('cpu')


path_train = 'The dataset of your chosen model'
path_vaild = 'The validation set for your chosen model'

transforms_train = transforms.Compose([
    transforms.Grayscale(),# Use ImageFolder to expand to three channels by default, just change it back
    transforms.RandomHorizontalFlip(),# Random Flip
    transforms.ColorJitter(brightness=0.5, contrast=0.5),# Randomly adjust brightness and contrast
    transforms.ToTensor()
])
transforms_vaild = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

data_train = torchvision.datasets.ImageFolder(root=path_train,transform=transforms_train)
data_vaild = torchvision.datasets.ImageFolder(root=path_vaild,transform=transforms_vaild)

train_set = torch.utils.data.DataLoader(dataset=data_train,batch_size=BATCH_SIZE,shuffle=True)
vaild_set = torch.utils.data.DataLoader(dataset=data_vaild,batch_size=BATCH_SIZE,shuffle=False)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0],-1)

        
class GlobalAvgPool2d(nn.Module):
    # The global average pooling layer can be implemented by setting the pooling window shape to the height and width of the input
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])



CNN = nn.Sequential(
    nn.Conv2d(1,64,3),
    nn.ReLU(True),
    nn.MaxPool2d(2,2),
    nn.Conv2d(64,256,3),
    nn.ReLU(True),
    nn.MaxPool2d(3,3),
    Reshape(),
    nn.Linear(256*7*7,4096),
    nn.ReLU(True),
    nn.Linear(4096,1024),
    nn.ReLU(True),
    nn.Linear(1024,7)
    )

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
                                 Reshape(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 7)
                                ))
    return net



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
resnet.add_module("fc", nn.Sequential(Reshape(), nn.Linear(512, 7))) 

# Use that model to switch the annotation
model = CNN
#model = resnet
#model = vgg(conv_arch, fc_features, fc_hidden_units)
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
            #optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

print(model)

train_loss = []
train_ac = []
vaild_loss = []
vaild_ac = []
y_pred = []

def train(model,device,dataset,optimizer,epoch):
    model.train()
    correct = 0
    for i,(x,y) in tqdm(enumerate(dataset)):
        x , y  = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1,keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss = criterion(output,y) 
        loss.backward()
        optimizer.step()   
        
    train_ac.append(correct/len(data_train))   
    train_loss.append(loss.item())
    print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(epoch,loss,correct,len(data_train),100*correct/len(data_train)))

def vaild(model,device,dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i,(x,y) in tqdm(enumerate(dataset)):
            x,y = x.to(device) ,y.to(device)
            output = model(x)
            loss = criterion(output,y)
            pred = output.max(1,keepdim=True)[1]
            global  y_pred 
            y_pred += pred.view(pred.size()[0]).cpu().numpy().tolist()
            correct += pred.eq(y.view_as(pred)).sum().item()
            
    vaild_ac.append(correct/len(data_vaild)) 
    vaild_loss.append(loss.item())
    print("Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(loss,correct,len(data_vaild),100.*correct/len(data_vaild)))


def RUN():
    for epoch in range(1,EPOCH+1):
        '''if epoch==15 :
            LR = 0.1
            optimizer=optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
        if(epoch>30 and epoch%15==0):
            LR*=0.1
            optimizer=optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
        '''
        #Try dynamic learning rate
        train(model,device=DEVICE,dataset=train_set,optimizer=optimizer,epoch=epoch)
        vaild(model,device=DEVICE,dataset=vaild_set)
        torch.save(model,'m0.pth')

RUN()        
#vaild(model,device=DEVICE,dataset=vaild_set)
 
def print_plot(train_plot,vaild_plot,train_text,vaild_text,ac,name):  
    x= [i for i in range(1,len(train_plot)+1)]
    plt.plot(x,train_plot,label=train_text)
    plt.plot(x[-1],train_plot[-1],marker='o')
    plt.annotate("%.2f%%"%(train_plot[-1]*100) if ac else "%.4f"%(train_plot[-1]),xy=(x[-1],train_plot[-1]))
    plt.plot(x,vaild_plot,label=vaild_text)
    plt.plot(x[-1],vaild_plot[-1],marker='o')
    plt.annotate("%.2f%%"%(vaild_plot[-1]*100) if ac else "%.4f"%(vaild_plot[-1]),xy=(x[-1],vaild_plot[-1]))
    plt.legend()
    plt.savefig(name)
    
#print_plot(train_loss,vaild_loss,"train_loss","vaild_loss",False,"loss.jpg")
#print_plot(train_ac,vaild_ac,"train_ac","vaild_ac",True,"ac.jpg")

import seaborn as sns
from sklearn.metrics import confusion_matrix

emotion = ["angry","disgust","fear","happy","sad","surprised","neutral"]
sns.set()
f,ax=plt.subplots()
y_true = [ emotion[i] for _,i in data_vaild]
y_pred = [emotion[i] for i in y_pred]
C2= confusion_matrix(y_true, y_pred, labels=["angry","disgust","fear","happy","sad","surprised","neutral"])#[0, 1, 2,3,4,5,6])

sns.heatmap(C2,annot=True ,fmt='.20g',ax=ax) #Heatmap

ax.set_title('confusion matrix') #Title
ax.set_xlabel('predict') #x-axis
ax.set_ylabel('true') #y-axis
plt.savefig('matrix.jpg')

