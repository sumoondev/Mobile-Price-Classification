# Mobile-Price-Classification
This app leverages a Support Vector Machine (SVM) classifier with optimized hyperparameters to predict mobile phone price ranges based on specifications.

# Mobile Price Classification

This app leverages a Support Vector Machine (SVM) classifier with optimized hyperparameters to predict mobile phone price ranges based on specifications.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)

## Project Overview
The Mobile Price Classification project aims to classify mobile phones into different price ranges using their specifications. The model is built using a Support Vector Machine (SVM) classifier, which has been optimized for better performance.

## Features
- Predicts mobile phone price ranges based on various specifications.
- Utilizes a Support Vector Machine (SVM) for classification.
- Optimized hyperparameters for improved accuracy.

## Dataset
The dataset used for this project includes various features of mobile phones, such as:
- Battery capacity
- Screen size
- RAM
- Internal memory
- Camera resolution
- Mobile weight
- Connectivity features (Bluetooth, Wi-Fi, 4G support)

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/PhantomBenz/Mobile-Price-Classification.git
   cd Mobile-Price-Classification/FDS
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn

## Usage
To use the application, run the main script or Jupyter Notebook provided in the repository. The steps include:

## Load the dataset.
Preprocess the data (handle missing values, normalize features).
Train the SVM model with the training data.
Evaluate the model on the test data.
Model Evaluation
The model's performance is evaluated using metrics such as:

Accuracy
Precision
Recall
F1-score
Hyperparameter tuning is performed to optimize the SVM classifier for better results.

## Results
The SVM classifier achieved a high accuracy rate in predicting mobile phone price ranges. Detailed results, including confusion matrices and classification reports, are available in the results section of the notebook.
