from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the model, label encoders, and scaler
model = joblib.load('medical_condition_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')
le_target = joblib.load('target_label_encoder.pkl')

# Function to preprocess the input data
def preprocess_input(data, label_encoders, scaler):
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Remove 'Insurance Provider', 'Admission Type', and 'Billing Amount' columns if they exist
    columns_to_drop = ['Insurance Provider', 'Admission Type', 'Billing Amount']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Encode categorical variables
    for column in ['Gender', 'Blood Type', 'Medication']:
        if column in df.columns:
            df[column] = df[column].map(lambda s: label_encoders[column].transform([s])[0] if s in label_encoders[column].classes_ else -1)
    
    # Standardize the features
    df = scaler.transform(df)
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    
    # Preprocess the input data
    try:
        preprocessed_data = preprocess_input(data, label_encoders, scaler)
    except Exception as e:
        return jsonify({'error': str(e)})
    
    # Make a prediction
    prediction = model.predict(preprocessed_data)
    predicted_condition = le_target.inverse_transform(prediction)
    
    return jsonify({'predicted_condition': predicted_condition[0]})

if __name__ == '__main__':
    app.run(debug=True)
