import numpy as np
from datetime import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

class RFPredictor:
    def __init__(self, file):
        self.df = self.process_dataset(file)

    def process_dataset(self, file):
        df = pd.read_csv(file)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        if(len(df.columns)>2):
            for i in range(1, 5):
                if df.iloc[:, i].dtype == 'object':
                    df.iloc[:, i] = df.iloc[:, i].str.replace(',', '')
                df.iloc[:, i] = df.iloc[:, i].astype(float)
            df.set_index(df.columns[0], inplace=True)
        else:
            if df.iloc[:, 1].dtype == 'object':
                df.iloc[:, 1] = df.iloc[:, 1].str.replace(',', '')
            df.iloc[:, 1] = df.iloc[:, 1].astype(float)
            df.set_index(df.columns[0], inplace=True)
        
        return df

    def train_Rf_model(self, target_index):

        self.df['Year'] = self.df.index.year
        self.df['Quarter'] = self.df.index.quarter

        # Create a lag feature to use past values
        self.df['Lag_Value'] = self.df[self.df.columns[target_index]].shift(1)

        features = ['Year', 'Quarter', 'Lag_Value']
        target = self.df.columns[target_index]

        # Split the data into features and target variable
        X = self.df[features]
        y = self.df[target]

        # Create a Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        X =X.fillna(0)
        rf_model.fit(X, y)
        return rf_model

    def predict_future_value(self, future_date_str, target_index):
        target_index = target_index - 1
        Q1 = self.df.iloc[-4][self.df.columns[target_index]]
        Q2 = self.df.iloc[-3][self.df.columns[target_index]]
        Q3 = self.df.iloc[-2][self.df.columns[target_index]]
        Q4 = self.df.iloc[-1][self.df.columns[target_index]]
        compare = 0
        year = int(future_date_str.split('-')[0])
        qtr = future_date_str.split('-')[1]
        qtr_val = 1
        if qtr == 'Q1':
            compare = Q1
            qtr_val = 1
        elif qtr == 'Q2':
            compare = Q2
            qtr_val = 2

        elif qtr == 'Q3':
            compare = Q3
            qtr_val = 3

        elif qtr == 'Q4':
            compare = Q4
            qtr_val = 4

        model = self.train_Rf_model(target_index)
        future_data = pd.DataFrame({'Year': [year], 'Quarter': [qtr_val], 'Lag_Value': [self.df.iloc[-1][self.df.columns[2]]]})
        predicted_value = model.predict(future_data)[0]

        perc_change = self.calculate_percentage_change(compare, predicted_value)
        if perc_change > 0:
            perc_change = f"+{perc_change:.2f}%"
        else:
            perc_change = f"{perc_change:.2f}%"

        return predicted_value.round(4), perc_change

    @staticmethod
    def convert_quarter_to_date(quarter_str):
        match = re.match(r'(\d+)-Q(\d+)', quarter_str)

        if match:
            year = int(match.group(1))
            quarter = int(match.group(2))
            month = (quarter - 1) * 3 + 1
            date_object = datetime(year, month, 1)
            result_str = date_object.strftime('%Y-%m-%d')

            return result_str

    @staticmethod
    def calculate_percentage_change(y1, y2):
        return ((y2 - y1) / y1) * 100
    





