from flask import Flask, render_template, request
from RF_Model import RFPredictor
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from Arima_Model import ARIMAModel
from Arima_Model import convert_to_datetime
from lstmmodel import TimeSeriesPredictor
from rnnmodel import RNNPredictor

app = Flask(__name__)


def predict_economic_data(form_data):
    # Perform prediction here
    # This is a dummy response, replace it with the actual prediction logic
    industry_type = form_data['industry-type']
    subsector =  form_data.get('subsector')
    year = form_data['year']
    quarter = form_data['quarter']
    selected_model =  form_data['select-model']


    if industry_type!='India':
        file=f'EconVisor---Industry-Forecasting-Tool/Datasets/{industry_type}/{subsector}.csv'
    else:
        file=f'EconVisor---Industry-Forecasting-Tool/Datasets/India/{industry_type}.csv'

    xl1 = file.split('/')[-1].split('.')[0]
    features = pd.read_csv(file).columns

    if selected_model == 'RandomForest':
        if industry_type!='India':
            linear_reg_predictor = RFPredictor(file=file)
            future_date_str = f'{year}-{quarter}'
            predictions = {f'{features[1]}': [0, 0], f'{features[2]}': [0, 0], f'{features[3]}': [0, 0], f'{features[4]}': [0, 0]}
            imgs_names = ['Production (Number)','Economy (Revenues)','Employment','GDP Contribution']

            for i in range(1, 5):
                predicted_value, perc = linear_reg_predictor.predict_future_value(future_date_str, i)
                key = list(predictions.keys())[i-1]
                predictions[key][0] = predicted_value
                predictions[key][1] = perc
                xl = f'{xl1} {key} Forecast Plot'
                yl = f'{key}'
                plot_data_with_prediction(dataframe = linear_reg_predictor.process_dataset(file), 
                    save_path=f'EconVisor---Industry-Forecasting-Tool/Static/ReportPlots/{imgs_names[i-1]}.png',
                    target_index=i-1, xl=xl, yl=yl, 
                    year=year, quarter=quarter, 
                    predicted_value=predicted_value
                )
        else: 
            linear_reg_predictor = RFPredictor(file=file)
            future_date_str = f'{year}-{quarter}'
            predictions = {f'{features[1]}': [0, 0]}
            imgs_names = ['GDP']

            predicted_value, perc = linear_reg_predictor.predict_future_value(future_date_str, 1)
            key = list(predictions.keys())[0]
            predictions[key][0] = predicted_value
            predictions[key][1] = perc
            xl = f'{xl1} {key} Forecast Plot'
            yl = f'{key}'
            plot_data_with_prediction(dataframe = linear_reg_predictor.process_dataset(file), 
                save_path=f'EconVisor---Industry-Forecasting-Tool/Static/ReportPlots/{imgs_names[0]}.png',
                target_index=0, xl=xl, yl=yl, 
                year=year, quarter=quarter, 
                predicted_value=predicted_value
            )



        return predictions

    elif selected_model == 'ARIMA':
        if industry_type!='India':
            predictions = {f'{features[1]}': [0, 0], f'{features[2]}': [0, 0], f'{features[3]}': [0, 0], f'{features[4]}': [0, 0]}
            imgs_names = ['Production (Number)','Economy (Revenues)','Employment','GDP Contribution']
            for i in range(1, 5):
                arima_model_instance = ARIMAModel(file, i)
                predicted_value, perc = arima_model_instance.forecast_data(year, quarter)
                key = list(predictions.keys())[i-1]
                predictions[key][0] = predicted_value
                predictions[key][1] = perc
                xl = f'{xl1} {key} Forecast Plot'
                yl = f'{key}'
                plot_data_with_prediction(dataframe = arima_model_instance.process_dataset(file), 
                    save_path=f'EconVisor---Industry-Forecasting-Tool/Static/ReportPlots/{imgs_names[i-1]}.png',
                    target_index=i-1, xl=xl, yl=yl, 
                    year=year, quarter=quarter, 
                    predicted_value=predicted_value
                )
        else:
            predictions = {f'{features[1]}': [0, 0]}
            imgs_names = ['GDP']
            arima_model_instance = ARIMAModel(file, 1)
            predicted_value, perc = arima_model_instance.forecast_data(year, quarter)
            key = list(predictions.keys())[0]
            predictions[key][0] = predicted_value
            predictions[key][1] = perc
            xl = f'{xl1} {key} Forecast Plot'
            yl = f'{key}'
            plot_data_with_prediction(dataframe = arima_model_instance.process_dataset(file), 
                save_path=f'EconVisor---Industry-Forecasting-Tool/Static/ReportPlots/{imgs_names[0]}.png',
                target_index=0, xl=xl, yl=yl, 
                year=year, quarter=quarter, 
                predicted_value=predicted_value
                )

        return predictions
    
    elif selected_model == 'LSTM':
        if industry_type!='India':
            predictions = {f'{features[1]}': [0, 0], f'{features[2]}': [0, 0], f'{features[3]}': [0, 0], f'{features[4]}': [0, 0]}
            imgs_names = ['Production (Number)','Economy (Revenues)','Employment','GDP Contribution']
            for i in range(1, 5):
                lstm_model_instance = TimeSeriesPredictor(file, i)
                lstm_model_instance.preprocess_data()
                predicted_value, perc = lstm_model_instance.forecast_data(year, quarter)
                key = list(predictions.keys())[i-1]
                predictions[key][0] = predicted_value
                predictions[key][1] = perc
                xl = f'{xl1} {key} Forecast Plot'
                yl = f'{key}'
                plot_data_with_prediction(dataframe = lstm_model_instance.process_dataset(file), 
                    save_path=f'EconVisor---Industry-Forecasting-Tool/Static/ReportPlots/{imgs_names[i-1]}.png',
                    target_index=i-1, xl=xl, yl=yl, 
                    year=year, quarter=quarter, 
                    predicted_value=predicted_value
                )
        else:
            predictions = {f'{features[1]}': [0, 0]}
            imgs_names = ['GDP']
            lstm_model_instance = TimeSeriesPredictor(file, 1)
            lstm_model_instance.preprocess_data()
            predicted_value, perc = lstm_model_instance.forecast_data(year, quarter)
            key = list(predictions.keys())[0]
            predictions[key][0] = predicted_value
            predictions[key][1] = perc
            xl = f'{xl1} {key} Forecast Plot'
            yl = f'{key}'
            plot_data_with_prediction(dataframe = lstm_model_instance.process_dataset(file), 
                save_path=f'EconVisor---Industry-Forecasting-Tool/Static/ReportPlots/{imgs_names[0]}.png',
                target_index=0, xl=xl, yl=yl, 
                year=year, quarter=quarter, 
                predicted_value=predicted_value
                )

        return predictions
    


    elif selected_model == 'RNN':
        if industry_type!='India':
            predictions = {f'{features[1]}': [0, 0], f'{features[2]}': [0, 0], f'{features[3]}': [0, 0], f'{features[4]}': [0, 0]}
            imgs_names = ['Production (Number)','Economy (Revenues)','Employment','GDP Contribution']
            for i in range(1, 5):
                rnn_instance = RNNPredictor(file, i)
                rnn_instance.preprocess_data()
                predicted_value, perc = rnn_instance.forecast_data(year, quarter)
                key = list(predictions.keys())[i-1]
                predictions[key][0] = predicted_value
                predictions[key][1] = perc
                xl = f'{xl1} {key} Forecast Plot'
                yl = f'{key}'
                plot_data_with_prediction(dataframe = rnn_instance.process_dataset(file), 
                    save_path=f'EconVisor---Industry-Forecasting-Tool/Static/ReportPlots/{imgs_names[i-1]}.png',
                    target_index=i-1, xl=xl, yl=yl, 
                    year=year, quarter=quarter, 
                    predicted_value=predicted_value
                )
        else:
            predictions = {f'{features[1]}': [0, 0]}
            imgs_names = ['GDP']

            rnn_instance = RNNPredictor(file, 1)
            rnn_instance.preprocess_data()
            predicted_value, perc = rnn_instance.forecast_data(year, quarter)
            key = list(predictions.keys())[0]
            predictions[key][0] = predicted_value
            predictions[key][1] = perc
            xl = f'{xl1} {key} Forecast Plot'
            yl = f'{key}'
            plot_data_with_prediction(dataframe = rnn_instance.process_dataset(file), 
                save_path=f'EconVisor---Industry-Forecasting-Tool/Static/ReportPlots/{imgs_names[0]}.png',
                target_index=0, xl=xl, yl=yl, 
                year=year, quarter=quarter, 
                predicted_value=predicted_value
                )

        return predictions

    
def plot_data_with_prediction(dataframe, save_path, target_index, xl, yl, year, quarter, predicted_value):
    # Use a dark theme
    matplotlib.pyplot.style.use('dark_background')

    # Plot historical data
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.plot(dataframe.index, dataframe[dataframe.columns[target_index]], label='Historical Data', marker='o', color='lightblue', linestyle='-')

    future_date = convert_to_datetime(year, quarter)

    matplotlib.pyplot.scatter(future_date, predicted_value, color='red', label='Predicted Future Value', zorder=5)
    matplotlib.pyplot.text(future_date, predicted_value, f'{predicted_value:.2f}', color='red', ha='left', va='bottom', fontsize=10, bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.3'))

    # Customize plot aesthetics
    matplotlib.pyplot.title(xl, fontsize=14, color='white')
    matplotlib.pyplot.xlabel('Year', fontsize=12, color='white')
    matplotlib.pyplot.ylabel(yl, fontsize=12, color='white')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.grid(True, color='gray', linestyle='--', alpha=0.5)

    # Customize tick parameters
    matplotlib.pyplot.tick_params(axis='both', which='both', colors='white')
    plot = matplotlib.pyplot.gcf()
    plot.savefig(save_path)


@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/', methods=['GET', 'POST'])
def economic_predictor():
    if request.method == 'POST':
        form_data = request.form
        prediction_result = predict_economic_data(form_data)

        # Pass the prediction result to the results page
        return render_template('results.html', predictions=prediction_result)

    return render_template('index.html')  # Assuming your HTML file is named index.html

if __name__ == '__main__':
    app.run(debug=True)



