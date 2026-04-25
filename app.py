from flask import Flask, request
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

model_path = 'model.pkl'
with open(model_path, 'rb') as file: 
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict(): 
    data = request.get_json()

    required_fields = {
        "store_ID": int,
        "day_of_week": int,
        "nb_customers_on_day": int,
        "open": int,
        "promotion": int,
        "school_holiday": int,
        "state_holiday": str, 
        "date": str
    }

    validation_data = {}

    for field, expected_type in required_fields.items():
        val = data.get(field)

        if val is None:
            
            return f"Error: Missing field {field}", 400

        try:
            validation_data[field] = expected_type(val)
        except (ValueError, TypeError):
            return f"Error: Field {field} must be of type {expected_type.__name__}", 400

    raw_date = validation_data['date']
    standard_format = "%d/%m/%Y"
    accepted_formats = [standard_format, "%Y-%m-%d", "%d-%m-%Y"]

    clean_date_obj =None

    for fmt in accepted_formats:
        try:
            clean_date_obj = datetime.strptime(raw_date, fmt)
            break
        except ValueError:
            continue
    if clean_date_obj is None:
        return "Error: Invalid date format. Please use DD/MM/YYYY", 400
    validation_data['month'] = clean_date_obj.month
    validation_data['day'] = clean_date_obj.day
    validation_data['year'] = clean_date_obj.year
    validation_data['week_of_year'] = int(clean_date_obj.isocalendar()[1])

    validation_data['sales_lag_7'] = 0
    validation_data['sales_lag_14'] = 0
    validation_data['rolling_mean_7'] = 0
    validation_data['Unnamed: 0'] = 0 # The model expects this index column!

    if validation_data['open'] == 0:
        return {"prediction": 0.0}
    input_df = pd.DataFrame([validation_data])
    model_columns = [
        'day_of_week', 'promotion', 'school_holiday', 'month', 'day', 
        'state_holiday', 'store_ID', 'year', 'week_of_year', 
        'sales_lag_7', 'sales_lag_14', 'rolling_mean_7', 'Unnamed: 0'
    ]
    input_df = input_df[model_columns]
    try:
        log_prediction = model.predict(input_df)[0]
        final_prediction = np.expm1(log_prediction)
        return {"prediction": float(final_prediction)}
    except Exception as e:
        return {"Error": str(e)}, 500
    
if __name__ == "__main__":
    # Running on 0.0.0.0 allows the container to be accessed from the host
    app.run(host='0.0.0.0', port=5000)