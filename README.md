# Fruit Image Classification Using CNN

This project implements a Convolutional Neural Network (CNN) for multi-class fruit image classification. The model is trained using data augmentation techniques to enhance generalization and robustness.

## Features

- Data augmentation during training to improve model performance  
- CNN architecture with multiple convolutional and pooling layers  
- Model evaluation using accuracy, confusion matrix, and ROC curves  
- Visualization of training and validation loss and accuracy over epochs  

## Usage

1. Prepare your dataset with images organized in folders by class (one folder per fruit type).  
2. Adjust the dataset paths in the training script to point to your training and validation directories.  
3. Run the training script to train and validate the CNN model.  
4. Evaluate the model performance on the validation set, including detailed metrics and visualizations.  

## Requirements

- Python 3.x  
- TensorFlow  
- scikit-learn  
- pandas  
- matplotlib  
- numpy  

Install dependencies with:

```bash
pip install tensorflow scikit-learn pandas matplotlib numpy
