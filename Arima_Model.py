import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt 


class ARIMAModel:
    def __init__(self, file, column_index):
        self.dataframe = self.process_dataset(file)
        self.column_index = column_index-1

    def process_dataset(self, file):
        df = pd.read_csv(file)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

        if(len(df.columns)>2):
            for i in range(1, 5):
                if df.iloc[:, i].dtype == 'object':
                    df.iloc[:, i] = df.iloc[:, i].str.replace(',', '')
                df.iloc[:, i] = df.iloc[:, i].astype(float)
        else:
            if df.iloc[:, 1].dtype == 'object':
                df.iloc[:, 1] = df.iloc[:, 1].str.replace(',', '')
            df.iloc[:, 1] = df.iloc[:, 1].astype(float)

        df.set_index(df.columns[0], inplace=True)
        return df

    def stationarity_tests(self, timeseries):
        result_adf_trend = adfuller(timeseries, autolag='AIC')
        adf_pvalue_trend = result_adf_trend[1]

        num_diff_trend = 0
        while adf_pvalue_trend > 0.05:
            timeseries = timeseries.diff().dropna()
            result_adf_trend = adfuller(timeseries, autolag='AIC')
            adf_pvalue_trend = result_adf_trend[1]
            num_diff_trend += 1

        return timeseries, num_diff_trend

    def print_acf_pacf_lags(self, timeseries, trend_diff):
        acf_orig = acf(timeseries, fft=True, nlags=20)
        pacf_orig = pacf(timeseries, nlags=20)

        acf_diff_trend = acf(timeseries.diff().dropna(), fft=True, nlags=20)
        pacf_diff_trend = pacf(timeseries.diff().dropna(), nlags=20)

        def find_significant_lags(acf_values):
            conf_interval = 1.96 / len(timeseries)**0.5
            return [i for i, val in enumerate(acf_values) if abs(val) > conf_interval]

        return [find_significant_lags(acf_orig), find_significant_lags(pacf_orig),
                find_significant_lags(acf_diff_trend), find_significant_lags(pacf_diff_trend)]

    def get_PDQ(self, my_list):
        break_index = next((i for i, (a, b) in enumerate(zip(my_list, my_list[1:]), start=1) if b != a + 1), None)
        if break_index is not None:
            return len(my_list[:break_index])
        else:
            return 0

    def train_arima_model(self):
        timeseries, trend_diff = self.stationarity_tests(self.dataframe.iloc[:, self.column_index])
        lags_data = self.print_acf_pacf_lags(timeseries, trend_diff)

        P = []
        Q = []

        if trend_diff != 0:
            P = lags_data[3]
            Q = lags_data[2]
        else:
            P = lags_data[1]
            Q = lags_data[0]

        if 0 in P:
            P.remove(0)

        if 0 in Q:
            Q.remove(0)

        p = self.get_PDQ(P)
        q = self.get_PDQ(Q)
        d = trend_diff

        new_df = pd.DataFrame({self.dataframe.columns[self.column_index]: self.dataframe.iloc[:, self.column_index].values},
                              index=self.dataframe.index)
        
        train_size = int(0.9375 * len(new_df))
        train_df, _ = new_df[:train_size], new_df[train_size:]
        print(train_df.index[-1])

        arima_model = ARIMA(train_df[train_df.columns[0]], order=(p, d, q)).fit()

        return arima_model, train_df

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
        trained_model, train_df = self.train_arima_model()
        result = int(self.calculate_month_difference(train_df.index[-1], datetime2)/3)
        future_predictions = trained_model.forecast(result)
        value = future_predictions.iloc[-1]
        present_value = self.dataframe[self.dataframe.columns[self.column_index]][-1]
        perc_change = ((value - present_value) / present_value) * 100
        if perc_change > 0:
            perc_change = f"+{perc_change:.2f}%"
        else:
            perc_change = f"{perc_change:.2f}%"
        return value.round(3), perc_change
    
def convert_to_datetime(year, quarter):
    # Map quarter to the corresponding month
    year = int(year)
    quarter = int(quarter[-1])
    month = (quarter - 1) * 3 + 1
    
    # Create a datetime object for the first day of the quarter
    datetime_object = datetime(year, month, 1, 0, 0, 0)
    
    return datetime_object
