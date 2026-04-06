# Streaming Service Customer Behaviour Analysis

A comparative machine learning study predicting customer spending, churn, and segments in a 5,000-user streaming service dataset.

## Overview

This project builds a complete ML pipeline spanning regression, classification, and clustering to analyse customer behaviour in a streaming service context. The emphasis is on fair comparison: shared preprocessing, fixed train-test splits, and consistent evaluation criteria across all methods.

## What This Project Covers

**Regression (Monthly Spend Prediction)**
- Single-variable regression (linear and polynomial) for each numeric feature
- Multi-variable linear regression with all numeric predictors
- Random Forest regression with mixed numeric and categorical features
- Artificial Neural Network regression (2 hidden layers, ReLU, Adam optimiser)
- Comparative evaluation by RMSE and R-squared on held-out test data

**Classification (Churn Prediction)**
- Logistic Regression with regularisation and class weights
- Random Forest Classifier
- Support Vector Machine (RBF kernel)
- Evaluation using accuracy, precision, recall, F1, and ROC-AUC

**Clustering (Customer Segmentation)**
- k-Means clustering with elbow method and silhouette score selection
- Agglomerative (hierarchical) clustering for comparison
- Cluster profiling and dendrogram visualisation

## Technical Stack

- Python, scikit-learn, TensorFlow/Keras
- NumPy, Pandas, Matplotlib, Seaborn
- ColumnTransformer pipelines for consistent preprocessing

## Key Design Decisions

- **Shared preprocessing pipelines** ensuring fair cross-method comparison
- **Fixed train-test indices** reused across all models for each task
- **Random Forest for mixed features** rather than encoding categoricals into linear regression
- **ANN included despite small dataset** to verify empirically when neural networks help

## Project Structure

```
Streaming_Customer_Analysis.ipynb   # Complete pipeline notebook
README.md                           # This file
```

## Case Study

A detailed reflective case study is available on my [portfolio site](https://chimezie-ai-portfolio.netlify.app/case-study/streaming-customer-analysis).

## Author

Onuchukwu Joseph Chimezie
- [Portfolio](https://chimezie-ai-portfolio.netlify.app)
- [LinkedIn](https://www.linkedin.com/in/onuchukwu-joseph-589912148)
