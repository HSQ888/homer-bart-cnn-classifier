import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader , Dataset
import torch.optim as optim
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
    def __init__(self, num_classes : int = 2):
        super(HomerBartClassifier , self).__init__()
        
        #Conv2d calculation ->out_size = (input_size - Kernel)/stride + 1 -> Output shape -> (out_channels , out_size , out_size)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=12,kernel_size=5)#(12,28,28)
        self.fc1 = nn.Linear(in_features=12 , out_features=2)
        
    def forward(self , X):
        out = self.conv1(X)
        # out = self.fc1(out)
        
        return out            

#Get the class and index of the category folders
index_class = {idx : category for category , idx in ImageFolder(homer_bart_dir).class_to_idx.items()}

#Transform the images into a similar form
compose_transform = transforms.Compose([
    transforms.Resize((32,32)) ,
    transforms.ToTensor()
])

homer_bart_dataset = HomerBartDataset(data_dir=homer_bart_dir , transform=compose_transform)

homer_bart_dataloader = DataLoader(homer_bart_dataset , batch_size=1 , shuffle=True)

hb_model = HomerBartClassifier()

epochs , learning_rate = 10 , 0.001

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = hb_model.parameters() , lr = learning_rate)

for epoch in range(epochs):
    for image , label in homer_bart_dataloader:
        pred_label = hb_model(image)
        break
        # loss = loss_fn(pred_label , label)
        
        # loss.backward()
        
        # optimizer.zero_grad()
        
        # optimizer.step()
        
    break

    print(f'Epoch {epoch}/{epochs} : Loss = {loss:.2f}')
print(np.shape(pred_label))
        

        




# print(np.shape(image))


 
 
 