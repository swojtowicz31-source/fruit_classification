Fruit Image Classification with CNN
This project implements a Convolutional Neural Network (CNN) to classify fruit images into 5 categories. It uses data augmentation to improve generalization and provides comprehensive evaluation including confusion matrix and ROC curves.

Features
CNN model with 3 convolutional layers, max pooling, dropout, and fully connected layers
Data augmentation during training (rotation, shifts, shear, zoom, horizontal flip)
Training and validation with real-time data generators
Evaluation with accuracy, confusion matrix, and ROC curves per class
Visualization of training and validation loss and accuracy curves
Requirements
Python 3.x
TensorFlow 2.x
scikit-learn
matplotlib
pandas
numpy
Usage
Organize your dataset in the following folder structure:
project_folder/
│
├── trening/       # Training images, organized in subfolders per class
├── test/          # Validation/test images, organized similarly
Update the paths in the code for train_generator and validation_generator.
Run the script to train the model and visualize results.
Results
Prints test accuracy after training
Shows confusion matrix
Plots ROC curves for each class
Displays training/validation loss and accuracy graphs
