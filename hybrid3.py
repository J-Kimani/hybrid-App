import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model


# Page Title
st.title('Stock Price Predictor (Hybrid LSTM + Linear Regression)')

# File Uploader
uploaded = st.file_uploader("Upload CSV File (Date, Open, High, Low, Close, Volume)", type=["csv"])

if uploaded:
    data = pd.read_csv(uploaded)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    st.write("### Uploaded Data", data.tail())

    close = data[[' Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    close[' Close'] = scaler.fit_transform(close)

    # Model Selection
    model_choice = st.selectbox("Select Model", ["LSTM", "Linear Regression", "Hybrid Model"])

    def sequences(data, length=60):
        X, y = [], []
        for i in range(len(data) - length):
            X.append(data[i:i + length])
            y.append(data[i + length])
        return np.array(X), np.array(y)

    seq_length = 60
    X, y = sequences(close[' Close'].values, seq_length)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train LSTM Model
    if model_choice in ["LSTM", "Hybrid Model"]:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        
        # Save the LSTM model
        model.save('lstm_model.h5')
        
        # Save the scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        lstm_predictions = model.predict(X_test)
        lstm_predictions = scaler.inverse_transform(lstm_predictions)

    # Train Linear Regression
    if model_choice in ["Linear Regression", "Hybrid Model"]:
        lagged_data = pd.DataFrame(close[' Close'])
        lagged_data['Lag_1'] = lagged_data[' Close'].shift(1)
        lagged_data['Lag_2'] = lagged_data[' Close'].shift(2)
        lagged_data['Lag_3'] = lagged_data[' Close'].shift(3)
        lagged_data.dropna(inplace=True)

        X_lin = lagged_data[['Lag_1', 'Lag_2', 'Lag_3']]
        y_lin = lagged_data[' Close']
        train_size = int(len(X_lin) * 0.8)
        X_train_lin, X_test_lin = X_lin[:train_size], X_lin[train_size:]
        y_train_lin, y_test_lin = y_lin[:train_size], y_lin[train_size:]

        lin_model = LinearRegression()
        lin_model.fit(X_train_lin, y_train_lin)
        
        # Save the Linear Regression model
        with open('linear_regression_model.pkl', 'wb') as f:
            pickle.dump(lin_model, f)
        
        lin_predictions = lin_model.predict(X_test_lin)
        lin_predictions = scaler.inverse_transform(lin_predictions.reshape(-1, 1))

    # Hybrid Model Combination
    if model_choice == "Hybrid Model":
        min_length = min(len(lstm_predictions), len(lin_predictions))
        lstm_predictions = lstm_predictions[:min_length]
        lin_predictions = lin_predictions[:min_length]
        hybrid_predictions = (0.7 * lstm_predictions) + (0.3 * lin_predictions)

        combined_preds = pd.DataFrame({
            'Date': data.index[-min_length:],
            'Actual': scaler.inverse_transform(y_test[-min_length:].reshape(-1, 1)).flatten(),
            'LSTM': lstm_predictions.flatten(),
            'Linear Regression': lin_predictions.flatten(),
            'Hybrid': hybrid_predictions.flatten()
        })
        combined_preds.set_index('Date', inplace=True)
        st.write("### Combined Model Predictions")
        
        # Plot with Matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(combined_preds.index, combined_preds['LSTM'], label="LSTM", color='blue')
        plt.plot(combined_preds.index, combined_preds['Linear Regression'], label="Linear Regression", color='green')
        plt.plot(combined_preds.index, combined_preds['Hybrid'], label="Hybrid", color='red')
        plt.plot(combined_preds.index, combined_preds['Actual'], label="Actual", color='black', linestyle='--')
        plt.title("Combined Model Predictions")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        st.pyplot(plt)
        st.write(combined_preds.tail(10))

    if model_choice == "LSTM":
        lstm_comparison = pd.DataFrame({
            'Actual': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
            'Date': data.index[-len(lstm_predictions):],
            'LSTM': lstm_predictions.flatten()            
        })
        lstm_comparison.set_index('Date', inplace=True)
        st.write("### LSTM Model Predictions")
        
        # Plot with Matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(lstm_comparison.index, lstm_comparison['LSTM'], label="LSTM", color='blue')
        plt.plot(lstm_comparison.index, lstm_comparison['Actual'], label="Actual", color='black', linestyle='--')
        plt.title("LSTM Model Predictions")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        st.pyplot(plt)
        st.write(lstm_comparison.tail(10))

    # Linear Regression Model Prediction
    if model_choice == "Linear Regression":
        lin_comparison = pd.DataFrame({
        'Actual': scaler.inverse_transform(y_test_lin.values.reshape(-1, 1)).flatten(),
        'Date': data.index[-len(y_test_lin):],
        'Linear Regression': lin_predictions.flatten()
        })
        lin_comparison.set_index('Date', inplace=True)
        st.write("### Linear Regression Predictions")

        # Plot with Matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(lin_comparison.index, lin_comparison['Linear Regression'], label="Linear Regression", color='green')
        plt.plot(lin_comparison.index, lin_comparison['Actual'], label="Actual", color='black', linestyle='--')
        plt.title("Linear Regression Predictions")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        st.pyplot(plt)
        st.write(lin_comparison.tail())
else:
    st.write("Please upload a CSV file to proceed.")
