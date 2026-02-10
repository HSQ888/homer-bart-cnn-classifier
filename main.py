import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader , Dataset
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from PIL import Image
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
        self.conv1 = nn.Conv2d(3 ,12 ,5) # (12 , 28 , 28)
        self.pool = nn.MaxPool2d(2,2) # (12 , 14 , 14)
        self.conv2 = nn.Conv2d(12 , 24 , 5) # (8 , 10 , 10) -> Pool -> (8,5,5) -> Flatten -> 8*5*5
        self.fc1 = nn.Linear(24*5*5 , 84)
        self.fc2 = nn.Linear(84 , 12)
        self.fc3 = nn.Linear(12 , 2)
        self.relu = nn.ReLU()
    
    def forward(self , X):
        out = self.pool(self.relu(self.conv1(X)))
        out = self.pool(self.relu(self.conv2(out)))
        out = torch.flatten(out , 1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out
    
#Get the class and index of the category folders
index_class = {idx : category for category , idx in ImageFolder(homer_bart_dir).class_to_idx.items()}

#Transform the images into a similar form
compose_transform = transforms.Compose([
    transforms.Resize((32,32)) ,
    transforms.ToTensor() , 
    # transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))
])

homer_bart_dataset = HomerBartDataset(data_dir=homer_bart_dir , transform=compose_transform)

homer_bart_dataloader = DataLoader(homer_bart_dataset , batch_size=1 , shuffle=True)

hb_model = HomerBartClassifier()

epochs , learning_rate = 20 , 0.001

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = hb_model.parameters() , lr = learning_rate)

# for epoch in range(epochs):
#     for image , label in homer_bart_dataloader:
        
#         # optimizer.zero_grad()
#         pred_label = hb_model(image)
        
#         loss = loss_fn(pred_label , label)
#         optimizer.zero_grad()
#         loss.backward()
        
#         optimizer.step()

#     print(f'Epoch {epoch}/{epochs} : Loss = {loss:.2f}')
    
def save_model():
    torch.save(hb_model.state_dict() , 'homer_bart_model.pth')
    
    torch.save(index_class , 'indices_to_class.pkl')

def load_model():
    net = HomerBartClassifier()
    
    net.load_state_dict(torch.load('homer_bart_model.pth'))
    
    return net

hb_model = load_model()

my_index = torch.load(r'C:\Users\Admin\Desktop\Henry\Projects\Homer_Bart_Classification\indices_to_class.pkl')

#Transform image for testing
def transform(image , transform = compose_transform):
    transformed_image = transform(image)
    
    return transformed_image

def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    
    return image

image = [r'C:\Users\Admin\Desktop\Henry\Projects\Homer_Bart_Classification\homer104.bmp']

images = [load_image(img) for img in image]

def predict(images):
    hb_model.eval()
    
    with torch.no_grad():
        for image in images :
            output = hb_model(image)
            
            _ , predicted = torch.max(output , 1)
            
            print(f'Prediction : {index_class[predicted.item()]}')
            

        
predict(images)
    
    
    

 
 