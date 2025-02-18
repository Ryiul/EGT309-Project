#!/usr/bin/env python3
"""
Docker-ready script for training and saving model
Usage:
    python train_model.py --csv-data Processed_Employee_Data.csv
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import argparse
import pandas as pd
import numpy as np
import os

def load_data(csv_file, test_size=0.25):
    # Load dataset
    df = DATA.to_csv("Processed_Employee_Data.csv", index=False)

    # Assuming the last column is the target variable
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=10, shuffle=True)

    return X_train, X_test, y_train, y_test

def ANN_train(X_train, X_test, y_train, y_test):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    model.save("model.h5")
    print("Model saved as model.h5")

def main():
    parser = argparse.ArgumentParser(description="Train an ANN model and save it as model.h5")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training dataset CSV")
    parser.add_argument("--test-data", type=str, required=True, help="Path to testing dataset CSV")
    
    args = parser.parse_args()

    if not os.path.exists(args.train_data) or not os.path.exists(args.test_data):
        print("Error: Dataset files not found!")
        return
    
    X_train, X_test, y_train, y_test = load_data(args.train_data, args.test_data)
    ANN_train(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
