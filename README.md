# Credit Card Fraud Detection with Transformer Model

## Overview

This project demonstrates a workflow for detecting credit card fraud using a Transformer-based model. The dataset used is the Credit Card Fraud Detection Dataset, which provides transactional data for training and evaluating the model.

## Project Steps

1. **Dataset Download and Loading**
   - Download the Credit Card Fraud Detection Dataset.
   - Load the dataset into your working environment.

2. **Data Preprocessing**
   - Separate features (`X`) and target labels (`y`).
   - Normalize the data to ensure consistency.
   - Split the dataset into training and testing sets.

3. **Transformer Model Construction**
   - Build a Transformer model with an encoder.
   - Add Dense layers for classification purposes.
   - Ensure the model can extract important features from the data.

4. **Model Training**
   - Train the model using the training dataset.
   - Utilize a suitable binary classification loss function.
   - Implement regularization and dropout techniques to prevent overfitting.

5. **Model Evaluation**
   - Evaluate the trained model using the test dataset.
   - Report performance metrics such as F1-score, Precision, and Recall.
   - Plot the ROC-AUC curve to assess the model's performance visually.

## Requirements

- Python 3.x
- TensorFlow / PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn



## Acknowledgments

- The dataset is provided by Kaggle.
- The Transformer model architecture is inspired by research and implementations in the field of NLP.
