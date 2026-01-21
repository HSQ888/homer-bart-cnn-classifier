import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader , Dataset
import torch.optim as optim
import torch.functional as F
import pandas as pd
import numpy as np
import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

current_dir = os.path.dirname(__file__)
homer_bart_dir = os.path.join(current_dir , 'homer_bart_1')

homer_bart_files = os.listdir(homer_bart_dir)

class HomerBartDataset(Dataset):
    def __init__(self , data_dir , transform):
        super().__init__()
        self.data = ImageFolder(data_dir , transform=transform)
        
    def __len__(self):
        return len(self.data)
    
    def  __getitem__(self, index):
        return self.data[index]
    
    @property
    def classes(self):
        return self.data.classes
    
class HomerBartClassifier(nn.Module):
    # def __init__(self, num_classes : int = 2):
    #     super(HomerBartClassifier , self).__init__()
        
    #     #Conv2d calculation ->out_size = (input_size - Kernel)/stride + 1 -> Output shape -> (out_channels , out_size , out_size)
    #     self.conv1 = nn.Conv2d(3 , 12 , 5) # (12,28,28)
    #     self.pool = nn.MaxPool2d(2,2) #(12,14,14)
    #     self.conv2 = nn.Conv2d(12 , 64 , 5) # (64 , 28 , 28) -> Pool -> (64 , 14 , 14)
    #     #(24 , 10 , 10) -> Pool -> (24,5,5) -> Flatten -> (24*5*5)
    #     self.conv3 = nn.Conv2d(64 , 32 , 5) # (32 , 28 , 28) -> Pool ->  (32 , 14 , 14)  -> Flatten -> (32*14*14)
    #     self.relu = nn.ReLU()
    #     self.fc1 = nn.Linear(32*14*14 , 84)
    #     self.fc2 = nn.Linear(84 , 12)
    #     self.fc3 = nn.Linear(12 , num_classes)
        
    def __init__(self, num_classes : int = 2):
        super(HomerBartClassifier , self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=12 , kernel_size=5) #(12,  28 , 28)
        self.pool = nn.MaxPool2d(2,2) #12,14,14 -> flatten -> 12*14*14
        self.fc1 = nn.Linear(12*14*14 , 120)
        self.fc2 = nn.Linear(120 , 84)
        self.fc3 = nn.Linear(84 , num_classes)
        
    
    def forward(self , X): 
        out = self.conv1(X)
        out = torch.relu(self.pool(out))
        out = torch.flatten(out , 1)
        out = self.fc1(out)
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out    
    
    # def forward(self , X):
    #     out = self.relu((self.conv1(X)))
    #     out = self.pool(out)
    #     out = self.relu(self.conv2(out))
    #     out = self.pool(out)
    #     # out = self.relu(self.conv3(out))
    #     out = self.pool(out)
    #     out = torch.flatten(out , 1 )
    #     out = self.relu(self.fc1(out))
    #     out = self.relu(self.fc2(out))
    #     out = self.fc3(out)
        
    #     return out            

#Get the class and index of the category folders
index_class = {idx : category for category , idx in ImageFolder(homer_bart_dir).class_to_idx.items()}

#Transform the images into a similar form
compose_transform = transforms.Compose([
    transforms.Resize((32,32)) ,
    transforms.ToTensor() , 
    transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))
])

homer_bart_dataset = HomerBartDataset(data_dir=homer_bart_dir , transform=compose_transform)

homer_bart_dataloader = DataLoader(homer_bart_dataset , batch_size=32 , shuffle=True)

hb_model = HomerBartClassifier()

epochs , learning_rate = 100 , 0.001

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = hb_model.parameters() , lr = learning_rate , maximize=True , weight_decay=5.0)

for epoch in range(epochs):
    for image , label in homer_bart_dataloader:
        optimizer.zero_grad()
        pred_label = hb_model(image)
        
        # pred_label = torch.tensor(pred_label , dtype=torch.float , requires_grad=True)
        # label = torch.tensor(label , dtype = torch.float , requires_grad = True)
        
        loss = loss_fn(pred_label , label)
        
        loss.backward()
        
        optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs} : Loss = {loss:.2f}')




 
 
 