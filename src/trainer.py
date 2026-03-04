###Import necessary packages
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

outer_dir = os.getcwd()

homer_bart_1 = "data/homer_bart_1"

homer_bart_dir = os.path.join(outer_dir , homer_bart_1)

#Get all the images in the dataset directory
homer_bart_files = os.listdir(homer_bart_1)

model_folder = os.path.join(outer_dir , "models")

#Get the models directory
model_dir = os.path.join(outer_dir , "models/homer_bart_model.pth")
index_dir = os.path.join(outer_dir , "models/indices_to_class.pkl")

test_images_dir = os.path.join(outer_dir , "data/test_images")

#Get the dataset files images
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
    
#Create the classifier neural network
class HomerBartClassifier(nn.Module):
    def __init__(self, num_classes : int = 2):
        super(HomerBartClassifier , self).__init__()
        
        #Conv2d calculation ->out_size = (input_size - Kernel)/stride + 1 -> Output shape -> (out_channels , out_size , out_size)
        self.conv1 = nn.Conv2d(3 ,12 ,5) # (12 , 28 , 28)
        self.pool = nn.MaxPool2d(2,2) # (12 , 14 , 14)
        self.conv2 = nn.Conv2d(12 , 24 , 5) # (24 , 10 , 10) -> Pool -> (24,5,5) -> Flatten -> 24*5*5
        self.fc1 = nn.Linear(24*5*5 , 84)
        self.fc2 = nn.Linear(84 , 12)
        self.fc3 = nn.Linear(12 , 2)
        self.celu = nn.CELU()
    
    def forward(self , X):
        out = self.pool(self.celu(self.conv1(X)))
        out = self.pool(self.celu(self.conv2(out)))
        out = torch.flatten(out , 1)
        out = self.celu(self.fc1(out))
        out = self.celu(self.fc2(out))
        out = self.fc3(out)
        out = self.celu(out)
        
        return out

#Get the class and index of the category folders
index_class = {idx : category for category , idx in ImageFolder(homer_bart_dir).class_to_idx.items()}

#Transform the images into a similar form
compose_transform = transforms.Compose([
    transforms.Resize((32,32)) ,
    transforms.ToTensor() , 
    transforms.Normalize((0.5,0.5,0.5) , (0.5,0.5,0.5))
])

#Create the dataset instance
homer_bart_dataset = HomerBartDataset(data_dir=homer_bart_dir , transform=compose_transform)

#Create the dataloader
print('Loading the data...')
homer_bart_dataloader = DataLoader(homer_bart_dataset , batch_size=1 , shuffle=True)

#Load the classifier network
hb_model = HomerBartClassifier()

#Define Hyperparameters
epochs , learning_rate = 15 , 0.001

#Define the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = hb_model.parameters() , lr = learning_rate)

'''Train the model in the training loop'''
print('Training model ...')
hb_model.train() #Set the model to training mode

for epoch in range(epochs):
    #Load an image and label for training 
    for image , label in homer_bart_dataloader:
        pred_label = hb_model(image)
        
        loss = loss_fn(pred_label , label)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs} : Loss = {loss:.2f}')

def save_model():
    torch.save(hb_model.state_dict() , model_dir)
    
    torch.save(index_class , index_dir)

#Save the model if not saved
print('Saving model...')
# save_model()
print('Model Saved.')

#Function to transform image for testing
def transform(image , transform = compose_transform):
    transformed_image = transform(image)
    
    return transformed_image

#Fuction to load the trained model
def load_model():
    net = HomerBartClassifier()
    state_dict = torch.load(model_dir)
    net.load_state_dict(state_dict)
    
    return net

#Load the trained model
hb_model = load_model()
#Load the models indices
my_index = torch.load(index_dir)

#Function to load test images
def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    
    return image

test_images = [os.path.join(test_images_dir , image_dir) for image_dir in os.listdir(test_images_dir)]

# Calculate test metrics
def calculate_metrics(image_paths):
        true_labels = []
        predicted_labels = []
        
        hb_model.eval()
        
        with torch.no_grad():
            for image_path in image_paths:
                image = load_image(image_path)
                output = hb_model(image)
                _, predicted = torch.max(output, 1)
                
                predicted_labels.append(predicted.item())
                
                # Determine true label from filename
                true_label = 1 if 'homer' in os.path.basename(image_path) else 0
                true_labels.append(true_label)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)
        
        print("\n=== Test Metrics ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

calculate_metrics(test_images) 








