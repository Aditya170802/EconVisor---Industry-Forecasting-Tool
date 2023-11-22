from datetime import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class TimeSeriesPredictor:
    def __init__(self, file, selected_column, seq_length=3,  epochs=100, batch_size=1):
        self.file = file
        self.selected_column = selected_column-1
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.df = None
        self.scaler = MinMaxScaler()
        self.model = None

    def preprocess_data(self):
        # Read CSV file and preprocess data
        self.df = pd.read_csv(self.file)
        self.df.iloc[:, 0] = pd.to_datetime(self.df.iloc[:, 0])

        for i in range(1, 5):
            if self.df.iloc[:, i].dtype == 'object':
                self.df.iloc[:, i] = self.df.iloc[:, i].str.replace(',', '')
            self.df.iloc[:, i] = self.df.iloc[:, i].astype(float)

        self.df.set_index(self.df.columns[0], inplace=True)

        selected_data = self.df[self.df.columns[self.selected_column]].values.reshape(-1, 1)
        self.df_scaled = self.scaler.fit_transform(selected_data)
        return self.df

    def create_sequences(self, future_steps):
        # Create sequences and labels for predicting future steps
        sequences = []
        labels = []

        for i in range(len(self.df_scaled) - self.seq_length - future_steps + 1):
            seq = self.df_scaled[i:i + self.seq_length]
            label = self.df_scaled[i + self.seq_length:i + self.seq_length + future_steps]
            sequences.append(seq)
            labels.append(label)

        self.X = np.array(sequences)
        self.y = np.array(labels)

        # Reshape for LSTM input (samples, time steps, features)
        self.X = self.X.reshape((self.X.shape[0], self.X.shape[1], 1))

    def build_model(self, future_steps):
        # Define and train the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(self.seq_length, 1)))
        self.model.add(Dense(future_steps))
        self.model.compile(optimizer='adam', loss='mse')

        self.model.fit(self.X, self.y, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

    def predict_future(self):
        # Predict for future steps
        future_sequence = self.df_scaled[-self.seq_length:]
        future_sequence = future_sequence.reshape((1, self.seq_length, 1))

        future_prediction = self.model.predict(future_sequence)

        # Inverse transform the prediction
        future_prediction = self.scaler.inverse_transform(future_prediction)
        return future_prediction[0][-1]
    
    def convert_to_datetime(self, year, quarter):
        # Map quarter to the corresponding month
        year = int(year)
        quarter = int(quarter[-1])
        month = (quarter - 1) * 3 + 1
        # Create a datetime object for the first day of the quarter
        datetime_object = datetime(year, month, 1, 0, 0, 0)
        
        return datetime_object
    
    def calculate_month_difference(self, datetime1, datetime2):
        difference = relativedelta(datetime2, datetime1)
        months_difference = difference.years * 12 + difference.months
        return months_difference
    
    def forecast_data(self, year, quarter):
        datetime2 = self.convert_to_datetime(year, quarter)
        result = int(self.calculate_month_difference(self.df.index[-1], datetime2)/3)
        self.create_sequences(result)
        self.build_model(result)
        value = self.predict_future()

        present_value = self.df[self.df.columns[self.selected_column]][-1]
        perc_change = ((value - present_value) / present_value) * 100
        if perc_change > 0:
            perc_change = f"+{perc_change:.2f}%"
        else:
            perc_change = f"{perc_change:.2f}%"
        return value.round(3), perc_change
    
    def process_dataset(self, file):
        df = pd.read_csv(file)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        for i in range(1, 5):
            if df.iloc[:, i].dtype == 'object':
                df.iloc[:, i] = df.iloc[:, i].str.replace(',', '')
            df.iloc[:, i] = df.iloc[:, i].astype(float)
        df.set_index(df.columns[0], inplace=True)
        return df


# file_path = file
# selected_col = 1
# predictor = TimeSeriesPredictor(file=file_path, selected_column=selected_col)
# predictor.preprocess_data()
# future_predictions = predictor.forecast_data('2024', 'Q1')
# print("Predicted Production for the next 5 quarters:", future_predictions)
