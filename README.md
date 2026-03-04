
# Homer and Bart Classification Project

## Overview
This project implements a PyTorch CNN machine learning classifier to distinguish between Homer and Bart Simpson characters.

## Goal
Learn the CNN architecture and how to implement Images using torchvision library in PyTorch.

## Project Structure
```
Homer_Bart_Classification/
├── data.zip           # Zip file containing the full dataset
├── models/            # Trained model and index files
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Source code
│   ├── trainer.py      # Model training script
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── LICENSE.txt        # MIT License of the project
└── extractor.py       # Extracts the compressed dataset in the same folder
```

## Dataset
The dataset is obtained from Kaggle and cleaned to obtain the current structure. It is a dataset containing two folders (after extraction) : homer_bart_1 (training) with 215 files and test_images (testing) with 53 files. All images are of Bitmap (.bmp). The zipped dataset is 7.2 MiB in size and the unzipped dataset is 126 MiB in size. 

## Prerequisites
Before cloning or using the code , ensure you have the following :

1. Python 3.10 or higher
2. pip (Python Package Installer)

## Framework
PyTorch

# Installation

## Clone this repo
git clone https://github.com/HSQ888/homer-bart-cnn-classifier.git
cd homer-bart-cnn-classifier

## Install dependencies

```bash
pip install -r requirements.txt
```

# How to Use :

## Extract the dataset from zip
```bash
python extractor.py
```

## Usage

```bash
python src/trainer.py
```

## Model Training
1. Optimizer - Adam Optimizer.
2. Loss Function - Cross Entropy Loss.
3. Epochs - 15 epochs.
4. Train-test ratio - 4:1

## Results
1. Accuracy : 0.77
2. Precision : 0.74
3. Recall : 0.67
4. F1 Score : 0.70
5. Confusion Matrix : ([[True Positives False Negatives] [False Positives True Negatives]]) -> [[27  5]
                        [ 7 14]]

## Contributing
Contributions welcome.Please open issues or submit pull requests.

## License
MIT License
