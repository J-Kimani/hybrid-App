import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
!pip install plotly
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import base64

# Function to add a background image
def add_bg(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image (replace 'background.png' with your image file path)
add_bg("background.jpg")

# Page Title
st.title("Hybrid Stock Price Prediction")
st.write("Welcome to the stock price prediction app. Please upload your CSV file below.")

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

        # Feature Importance Visualization (using Plotly)
        feature_importance = lin_model.coef_

        # Feature Importance Bar Chart for Linear Regression
        fig = go.Figure()

        # Add bar chart trace
        fig.add_trace(go.Bar(
            x=['Lag_1', 'Lag_2', 'Lag_3'], 
            y=feature_importance, 
            marker=dict(color='skyblue'),  # Set bar color
            text=feature_importance,  # Display values on bars
            textposition='outside'  # Position text outside bars
        ))

        # Update layout for better styling and readability
        fig.update_layout(
            title=dict(
                text="Feature Importance - Linear Regression",
                font=dict(size=18, color="black"),
            ),
            xaxis=dict(
                title=dict(text="Features", font=dict(size=14, color="black")),
                tickfont=dict(size=12, color="black"),
                showline=True,
                linecolor="black",  # Set x-axis border color
                mirror=True,  # Mirror the x-axis border on the top
            ),
            yaxis=dict(
                title=dict(text="Importance", font=dict(size=14, color="black")),
                tickfont=dict(size=12, color="black"),
                showline=True,
                linecolor="black",  # Set y-axis border color
                mirror=True,  # Mirror the y-axis border on the right
                gridcolor="lightgray",  # Set gridlines for y-axis
            ),
            template="plotly_white",
            plot_bgcolor="white",  # Set plot area background to white
            paper_bgcolor="white",  # Set the overall chart background to white
            margin=dict(
                l=50,  # Left margin
                r=50,  # Right margin
                t=50,  # Top margin
                b=50   # Bottom margin
            ),
            legend=dict(
                x=0.5,  # Center the legend horizontally
                y=1.02,  # Place the legend just above the chart
                xanchor='center',
                yanchor='bottom',
                orientation='h',  # Horizontal legend
                bgcolor="white",  # Set legend background color
                bordercolor="black",  # Add border around the legend
                borderwidth=1
            )
        )


        st.plotly_chart(fig)

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

        # Interactive Plot for Combined Model Predictions
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=combined_preds.index, y=combined_preds['LSTM'], mode='lines', name="LSTM", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=combined_preds.index, y=combined_preds['Linear Regression'], mode='lines', name="Linear Regression", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=combined_preds.index, y=combined_preds['Hybrid'], mode='lines', name="Hybrid", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=combined_preds.index, y=combined_preds['Actual'], mode='lines', name="Actual", line=dict(color='black', dash='dash')))

        fig.update_layout(
            title=dict(
                text="Combined Model Predictions",
                font=dict(size=18, color="black")  # Ensure the title is visible
            ),
            xaxis=dict(
                title=dict(text="Date", font=dict(size=14, color="black")),  # X-axis title
                tickfont=dict(size=12, color="black"),  # X-axis tick labels
                showgrid=True,
                gridcolor="lightgray",  # X-axis gridlines
                linecolor="black",  # Bottom border
                showline=True,  # Enable bottom and top borders
                mirror=True  # Enable top border
            ),
            yaxis=dict(
                title=dict(text="Stock Price", font=dict(size=14, color="black")),  # Y-axis title
                tickfont=dict(size=12, color="black"),  # Y-axis tick labels
                showgrid=True,
                gridcolor="lightgray",  # Y-axis gridlines
                linecolor="black",  # Left border
                showline=True,  # Enable left and right borders
                mirror=True  # Enable right border
            ),
            template="plotly_white",  # Use the white template for a clean look
            plot_bgcolor="white",  # Set the plot background to white
            paper_bgcolor="white",  # Set the overall chart background to white
            legend=dict(
                font=dict(size=12, color="black"),  # Ensure the legend text is readable
                bgcolor="white",  # Legend background color
                bordercolor="black",  # Legend border color
                borderwidth=1,  # Legend border width
                x=0.02,  # Adjust x-position inside the grid
                y=0.98,  # Adjust y-position inside the grid
                xanchor="left",
                yanchor="top"
            ),
            margin=dict(
                l=40,  # Left margin
                r=40,  # Right margin
                t=60,  # Top margin
                b=40   # Bottom margin
            )
        )


        st.plotly_chart(fig)
        st.write(combined_preds.tail(10))

        # Confidence Interval for Hybrid Model
        error_margin = np.std(hybrid_predictions) * 0.1  # 10% margin for visualization

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=combined_preds.index,
            y=hybrid_predictions.flatten(),
            mode='lines',
            name="Hybrid Prediction",
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=combined_preds.index,
            y=hybrid_predictions.flatten() + error_margin,
            fill='tonexty',
            mode='none',
            name='Upper Confidence Interval',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        fig.add_trace(go.Scatter(
            x=combined_preds.index,
            y=hybrid_predictions.flatten() - error_margin,
            fill='tonexty',
            mode='none',
            name='Lower Confidence Interval',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))

        fig.update_layout(
            title=dict(
                text="Combined Model Predictions",
                font=dict(size=18, color="black")  # Ensure the title is visible
            ),
            xaxis=dict(
                title=dict(text="Date", font=dict(size=14, color="black")),  # X-axis title
                tickfont=dict(size=12, color="black"),  # X-axis tick labels
                showgrid=True,
                gridcolor="lightgray",  # X-axis gridlines
                linecolor="black",  # Bottom border
                showline=True,  # Enable bottom and top borders
                mirror=True  # Enable top border
            ),
            yaxis=dict(
                title=dict(text="Stock Price", font=dict(size=14, color="black")),  # Y-axis title
                tickfont=dict(size=12, color="black"),  # Y-axis tick labels
                showgrid=True,
                gridcolor="lightgray",  # Y-axis gridlines
                linecolor="black",  # Left border
                showline=True,  # Enable left and right borders
                mirror=True  # Enable right border
            ),
            template="plotly_white",  # Use the white template for a clean look
            plot_bgcolor="white",  # Set the plot background to white
            paper_bgcolor="white",  # Set the overall chart background to white
            legend=dict(
                font=dict(size=12, color="black"),  # Ensure the legend text is readable
                bgcolor="white",  # Legend background color
                bordercolor="black",  # Legend border color
                borderwidth=1,  # Legend border width
                x=0.02,  # Adjust x-position inside the grid
                y=0.98,  # Adjust y-position inside the grid
                xanchor="left",
                yanchor="top"
            ),
            margin=dict(
                l=40,  # Left margin
                r=40,  # Right margin
                t=60,  # Top margin
                b=40   # Bottom margin
            )
        )


        st.plotly_chart(fig)

        # Hybrid Model Accuracy
        hybrid_mae = mean_absolute_error(y_test[-min_length:], hybrid_predictions)
        hybrid_rmse = np.sqrt(mean_squared_error(y_test[-min_length:], hybrid_predictions))
        st.write(f"**Hybrid Model Accuracy** - MAE: {hybrid_mae:.2f}, RMSE: {hybrid_rmse:.2f}")

    if model_choice == "LSTM":
        lstm_comparison = pd.DataFrame({
            'Actual': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
            'Date': data.index[-len(lstm_predictions):],
            'LSTM': lstm_predictions.flatten()            
        })
        lstm_comparison.set_index('Date', inplace=True)
        st.write("### LSTM Model Predictions")

        # Interactive Plot for LSTM Predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lstm_comparison.index, y=lstm_comparison['LSTM'], mode='lines', name="LSTM", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=lstm_comparison.index, y=lstm_comparison['Actual'], mode='lines', name="Actual", line=dict(color='black', dash='dash')))

        fig.update_layout(
            title=dict(
                text="LSTM Model Predictions",
                font=dict(size=18, color="black")  # Make title readable
            ),
            xaxis=dict(
                title=dict(text="Date", font=dict(size=14, color="black")),  # X-axis title
                tickfont=dict(size=12, color="black"),  # X-axis tick labels
                showgrid=True,
                gridcolor="lightgray",  # Gridlines for x-axis
                linecolor="black",  # Bottom border
                showline=True,  # Enable bottom and top borders
                mirror=True  # Enable top border
            ),
            yaxis=dict(
                title=dict(text="Stock Price", font=dict(size=14, color="black")),  # Y-axis title
                tickfont=dict(size=12, color="black"),  # Y-axis tick labels
                showgrid=True,
                gridcolor="lightgray",  # Gridlines for y-axis
                linecolor="black",  # Left border
                showline=True,  # Enable left and right borders
                mirror=True  # Enable right border
            ),
            template="plotly_white",  # Set to white template
            plot_bgcolor="white",  # Plot area background to white
            paper_bgcolor="white",  # Overall chart background to white
            legend=dict(
                font=dict(size=12, color="black"),  # Legend font
                bgcolor="white",  # Legend background
                bordercolor="black",  # Legend border
                borderwidth=1,  # Legend border width
                x=0.1,  # Adjust x-position inside the grid
                y=0.98,  # Adjust y-position inside the grid
                xanchor="left",
                yanchor="top"
            ),
            margin=dict(
                l=40,  # Left margin
                r=40,  # Right margin
                t=60,  # Top margin
                b=40   # Bottom margin
            )
        )

        st.plotly_chart(fig)
        st.write(lstm_comparison.tail(10))

        # LSTM Accuracy
        lstm_mae = mean_absolute_error(y_test[-len(lstm_predictions):], scaler.inverse_transform(lstm_predictions))
        lstm_rmse = np.sqrt(mean_squared_error(y_test[-len(lstm_predictions):], scaler.inverse_transform(lstm_predictions)))
        st.write(f"**LSTM Model Accuracy** - MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}")

    if model_choice == "Linear Regression":
        lin_comparison = pd.DataFrame({
            'Actual': scaler.inverse_transform(y_test_lin.values.reshape(-1, 1)).flatten(),
            'Date': data.index[-len(y_test_lin):],
            'Linear Regression': lin_predictions.flatten()
        })
        lin_comparison.set_index('Date', inplace=True)
        st.write("### Linear Regression Predictions")

        # Interactive Plot for Linear Regression Predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lin_comparison.index, y=lin_comparison['Linear Regression'], mode='lines', name="Linear Regression", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=lin_comparison.index, y=lin_comparison['Actual'], mode='lines', name="Actual", line=dict(color='black', dash='dash')))

        fig.update_layout(
            title=dict(
                text="Linear Regression Predictions",
                font=dict(size=18, color="black"),  # Ensure the title is visible
            ),
            xaxis=dict(
                title=dict(text="Date", font=dict(size=14, color="black")),
                showgrid=True,
                gridcolor="lightgray",
                linecolor="black",  # Bottom border
                showline=True,  # Enable bottom and top borders
                mirror=True,  # Enable top border
                tickfont=dict(size=12, color="black"),  # Ensure tick labels are visible
            ),
            yaxis=dict(
                title=dict(text="Stock Price", font=dict(size=14, color="black")),
                showgrid=True,
                gridcolor="lightgray",
                linecolor="black",  # Left border
                showline=True,  # Enable left and right borders
                mirror=True,  # Enable right border
                tickfont=dict(size=12, color="black"),  # Ensure tick labels are visible
            ),
            template="plotly_white",
            legend=dict(
                font=dict(size=12, color="black"),  # Ensure legend text is visible
                bgcolor="white",  # Set legend background to white
                bordercolor="black",
                borderwidth=1,
                x=0.1,  # Position legend inside the grid (adjust for your layout)
                y=0.98,  # Position legend inside the grid (adjust for your layout)
                xanchor="left",
                yanchor="top",
            ),
            plot_bgcolor="white",  # Set plot area background to white
            paper_bgcolor="white",  # Set the overall chart background to white
            margin=dict(
                l=40,  # Left margin
                r=40,  # Right margin
                t=60,  # Top margin
                b=40,  # Bottom margin
            ),
        )


        st.plotly_chart(fig)
        st.write(lin_comparison.tail())

        # Linear Regression Accuracy
        lin_mae = mean_absolute_error(y_test_lin, scaler.inverse_transform(lin_predictions))
        lin_rmse = np.sqrt(mean_squared_error(y_test_lin, scaler.inverse_transform(lin_predictions)))
        st.write(f"**Linear Regression Accuracy** - MAE: {lin_mae:.2f}, RMSE: {lin_rmse:.2f}")

else:
    st.write("Please upload a CSV file to proceed.")
