from flask import Flask, render_template, request
from RF_Model import RFPredictor
import pandas as pd
from Arima_Model import ARIMAModel
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
        file=f'EconVisor---Industry-Forecasting-Tool/Datasets/{industry_type}.csv'

    xl1 = file.split('/')[-1].split('.')[0]
    features = pd.read_csv(file).columns

    if selected_model == 'RandomForest':
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
            linear_reg_predictor.create_and_save_plot(target_index=i-1, xl=xl, yl=yl, save_path=f'EconVisor---Industry-Forecasting-Tool/Static/ReportPlots/{imgs_names[i-1]}.png', future_date_str=future_date_str, predicted_value=predicted_value)

        return predictions

    elif selected_model == 'ARIMA':
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
            arima_model_instance.create_and_save_plot(target_index=i-1, xl=xl, yl=yl, save_path=f'EconVisor---Industry-Forecasting-Tool/Static/ReportPlots/{imgs_names[i-1]}.png', year=year, quarter=quarter, predicted_value=predicted_value)  

        return predictions
    
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
