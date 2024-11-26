from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.neural_network import MLPRegressor
import os
from io import BytesIO
import base64
import joblib
from keras.models import load_model

app = Flask(_name_)

# Function to preprocess the dataset
def preprocess_dataset(dataset):
    dataset['date_time'] = pd.to_datetime(dataset['date_time'])
    dataset['day_of_week'] = dataset['date_time'].dt.dayofweek
    dataset['hour'] = dataset['date_time'].dt.hour
    dataset['month'] = dataset['date_time'].dt.month
    
    Y = dataset['passenger_flow'].to_numpy()
    dataset.drop(['date_time', 'is_holiday', 'passenger_flow'], axis=1, inplace=True)

    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values

    for i in range(len(types)):
        if types[i] == 'object': 
            le = LabelEncoder()
            dataset[columns[i]] = le.fit_transform(dataset[columns[i]].astype(str))
            label_encoder.append([columns[i], le])

    dataset.fillna(0, inplace=True)
    X = dataset.values
    Y = Y.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    Y = scaler1.fit_transform(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, Y, scaler1

# Function to split the data
def split_data(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to calculate metrics and visualize results
def calculate_metrics(algorithm, predict, test_labels, scaler):
    rmse_value = sqrt(mean_squared_error(test_labels, predict))
    map_value = mean_absolute_error(test_labels, predict)
    print(f"{algorithm} - RMSE: {rmse_value:.4f}, MAE: {map_value:.4f}")

    predict = scaler.inverse_transform(predict.reshape(-1, 1))
    test_labels = scaler.inverse_transform(test_labels)

    plt.figure(figsize=(10, 6))
    plt.plot(test_labels[:200], color='red', label='True Passenger Flow')
    plt.plot(predict[:200], color='green', label='Predicted Passenger Flow')
    plt.title(f'{algorithm} Passenger Flow Prediction')
    plt.xlabel('Samples')
    plt.ylabel('Passenger Flow')
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.close()
    return data, rmse_value, map_value

# Function to train MLP model
def train_mlp_model(X_train, y_train, X_test, y_test):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    mlp = MLPRegressor(hidden_layer_sizes=(256,), max_iter=500)
    mlp.fit(X_train_flat, y_train.ravel())
    predict = mlp.predict(X_test_flat)
    return predict

# Function to train Bidirectional LSTM model
def train_bidirectional_lstm_model(X_train, y_train, X_test, y_test):
    bidirectional_lstm_model = Sequential()
    bidirectional_lstm_model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    bidirectional_lstm_model.add(Dropout(0.2))
    bidirectional_lstm_model.add(LSTM(64, activation='relu', return_sequences=False))
    bidirectional_lstm_model.add(Dropout(0.2))
    bidirectional_lstm_model.add(Dense(64, activation='relu'))
    bidirectional_lstm_model.add(Dense(1))

    bidirectional_lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("best_bidirectional_lstm.keras", save_best_only=True, verbose=1)

    if not os.path.exists("best_bidirectional_lstm.keras"):
        bidirectional_lstm_model.fit(
            X_train, y_train,
            batch_size=32, epochs=100, validation_data=(X_test, y_test),
            callbacks=[early_stopping, model_checkpoint]
        )
    else:
        bidirectional_lstm_model.load_weights("best_bidirectional_lstm.keras")

    predict = bidirectional_lstm_model.predict(X_test)
    return predict

# Function to train Gradient Boosting Regressor model
def train_gradient_boosting_model(X_train, y_train, X_test, y_test):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    gbr.fit(X_train_flat, y_train.ravel())
    predict_gbr = gbr.predict(X_test_flat)
    return predict_gbr

# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred, scaler, tolerance=1.0):
    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    correct = np.sum(np.abs(y_true - y_pred) <= tolerance * y_true)
    accuracy = correct / len(y_true)
    return accuracy * 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            dataset = pd.read_csv(file, nrows=5000)
            X, Y, scaler1 = preprocess_dataset(dataset)
            X_train, X_test, y_train, y_test = split_data(X, Y)

            # Train MLP model
            predict_mlp = train_mlp_model(X_train, y_train, X_test, y_test)
            plot_mlp, rmse_mlp, mae_mlp = calculate_metrics("MLP", predict_mlp, y_test, scaler1)
            acc_mlp_100 = calculate_accuracy(y_test, predict_mlp, scaler1, tolerance=1.0)

            # Train Bidirectional LSTM model
            predict_lstm = train_bidirectional_lstm_model(X_train, y_train, X_test, y_test)
            plot_lstm, rmse_lstm, mae_lstm = calculate_metrics("Bidirectional LSTM", predict_lstm, y_test, scaler1)
            acc_lstm_100 = calculate_accuracy(y_test, predict_lstm, scaler1, tolerance=1.0)

            # Train Gradient Boosting Regressor model
            predict_gbr = train_gradient_boosting_model(X_train, y_train, X_test, y_test)
            plot_gbr, rmse_gbr, mae_gbr = calculate_metrics("Gradient Boosting Regressor", predict_gbr, y_test, scaler1)
            acc_gbr_100 = calculate_accuracy(y_test, predict_gbr.reshape(-1, 1), scaler1, tolerance=1.0)

            return render_template('results.html', 
                                   plot_mlp=plot_mlp, rmse_mlp=rmse_mlp, mae_mlp=mae_mlp, acc_mlp_100=acc_mlp_100,
                                   plot_lstm=plot_lstm, rmse_lstm=rmse_lstm, mae_lstm=mae_lstm, acc_lstm_100=acc_lstm_100,
                                   plot_gbr=plot_gbr, rmse_gbr=rmse_gbr, mae_gbr=mae_gbr, acc_gbr_100=acc_gbr_100)

    return redirect(url_for('index'))

if _name_ == '_main_':
    app.run(debug=True)