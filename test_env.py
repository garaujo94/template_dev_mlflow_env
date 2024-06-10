# Step 1: Import necessary libraries
import os

# Import MLflow
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("abcd")


# Step 2: Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Preprocess the data (Standardizing the features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 5: Train a simple machine learning model (Logistic Regression)
model = LogisticRegression()

# Start an MLflow run
with mlflow.start_run():
    # Train the model
    model.fit(X_train, y_train)

    # Step 6: Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

    # Log parameters (in this case, Logistic Regression doesn't have hyperparameters to log, but we'll log model class)
    mlflow.log_param("model_class", "LogisticRegression")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log classification report
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                mlflow.log_metric(f"{key}_{sub_key}", sub_value)
        else:
            mlflow.log_metric(key, value)

    # Log the trained model
    mlflow.sklearn.log_model(model, "logistic_regression_model")
