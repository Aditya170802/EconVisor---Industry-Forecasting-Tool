from flask import Flask, render_template, request

app = Flask(__name__)


def predict_economic_data(form_data):
    # Perform prediction here
    # This is a dummy response, replace it with the actual prediction logic
    return {
        'industry_type': form_data['industry-type'],
        'subsector': form_data.get('subsector', ''),
        'year': form_data['year'],
        'selected_model': form_data['select-model']
    }

@app.route('/', methods=['GET', 'POST'])
def economic_predictor():
    if request.method == 'POST':
        form_data = request.form
        prediction_result = predict_economic_data(form_data)

        # Pass the prediction result to the results page
        return render_template('results.html', result=prediction_result)

    return render_template('index.html')  # Assuming your HTML file is named index.html

if __name__ == '__main__':
    app.run(debug=True)
