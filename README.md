# Hybrid Stock Price Prediction Application

## Overview

This project is a Hybrid Stock Price Prediction Application that leverages advanced machine learning models to predict stock prices. The app is built with Streamlit and integrates LSTM, Linear Regression, and a Hybrid Model for accurate predictions. It provides an interactive platform to upload datasets, train models, and visualize predictions.

## Features

1. Data Upload: Users can upload CSV files containing stock data (Date, Open, High, Low, Close, Volume).

2. Model Options: Long Short-Term Memory (LSTM), Linear Regression, Hybrid Model combining LSTM and Linear Regression

3. Visualization: Stock price trends, Feature importance for Linear Regression, Model predictions with confidence intervals.
   
4. Interactive Interface: User-friendly UI for selecting models and exploring results.


## How to Use
1. Upload a CSV file with the following columns: Date (YYYY-MM-DD), Open, High, Low, Close, Volume

2. Select the desired model and view predictions and visualizations.

## Models

1. Long Short-Term Memory (LSTM)

  Description: A recurrent neural network architecture suitable for time-series data.

  Training: Processes sequential data in windows of 60 days.

  Output: Predicts stock prices based on historical data.

2. Linear Regression

  Description: A basic statistical approach to model relationships between variables.

  Training: Uses lagged variables (e.g., Lag_1, Lag_2) for prediction.

  Feature Importance: Displays coefficients for insights into feature relevance.

3. Hybrid Model

  Description: Combines LSTM and Linear Regression outputs using weighted averages.

  Advantage: Merges the strengths of both models for improved accuracy.

## Visualizations

Stock Trends: Displays uploaded stock data.

Feature Importance: Highlights influential variables for Linear Regression.

Model Predictions: Plots LSTM, Linear Regression, and Hybrid predictions alongside actual values.

Confidence Intervals: Provides error margins for hybrid predictions.

**Technologies Used:** Python, Streamlit, TensorFlow, Scikit-learn, Plotly

## Future Enhancements

1. Include additional features like technical indicators.
2. Extend predictions to multi-day forecasts.
3. Optimize the hybrid model for better accuracy.

