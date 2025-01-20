import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# Page Title
st.title('Stock Price Predictor (Hybrid LSTM + Linear Regression)')

# Upload CSV
uploaded = st.file_uploader("Upoad CSV File (Date, Open, High, Low, Close, Volume)", type=["csv"])

if uploaded:
    data = pd.read_csv(uploaded)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    st.write("### Uploaded Data", data.tail())

    close = data[[' Close']]

    # Scaling
    scaler = MinMaxScaler(feature_range= (0, 1))
    close[' Close'] = scaler.fit_transform(close)

    # Creating Sequences
    def sequences(data, length= 60):
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

    # LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences= True, input_shape= (X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer= 'adam', loss= 'mean_squared_error')

    model.fit(X_train, y_train, epochs= 10, batch_size= 32, verbose= 1)
    lstm_predictions = model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    # Linear Regression Model
    lagged_data = pd.DataFrame(close[' Close'])
    lagged_data['Lag_1'] = lagged_data[' Close'].shift(1)
    lagged_data['Lag_2'] = lagged_data[' Close'].shift(2)
    lagged_data['Lag_3'] = lagged_data[' Close'].shift(3)
    lagged_data.dropna(inplace= True)

    X_lin = lagged_data[['Lag_1', 'Lag_2', 'Lag_3']]
    y_lin = lagged_data[' Close']

    X_train_lin, X_test_lin = X_lin[:train_size], X_lin[train_size:]
    y_train_lin, y_test_lin = y_lin[:train_size], y_lin[train_size:]

    lin_model = LinearRegression()
    lin_model.fit(X_train_lin, y_train_lin)
    lin_predictions = lin_model.predict(X_test_lin)
    lin_predictions = scaler.inverse_transform(lin_predictions.reshape(-1, 1))

    # Hybrid Model
    min_length = min(len(lstm_predictions), len(lin_predictions))
    lstm_predictions = lstm_predictions[:min_length]
    lin_predictions = lin_predictions[:min_length]

    hybrid_predictions = (0.7 * lstm_predictions) + (0.3 * lin_predictions)

    future_dates = pd.date_range(start=data.index[-1] + datetime.timedelta(days=1), periods=10)
    future_preds = pd.DataFrame({
        'Date': future_dates,
        'LSTM': lstm_predictions[-10:].flatten(),
        'Linear Regression': lin_predictions[-10:].flatten(),
        'Hybrid': hybrid_predictions[-10:].flatten()
    }).set_index('Date')

    # Display Results
    st.write("### Future Predictins")
    st.line_chart(future_preds)
    st.write(future_preds)