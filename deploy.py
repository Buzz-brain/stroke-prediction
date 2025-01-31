from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model
model = joblib.load('stroke_prediction_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request body
    input_data = request.get_json()
    
    # Convert input data to a 2D array
    input_data = pd.DataFrame([input_data])
    
    # Make predictions using the trained model
    predictions = model.predict(input_data)
    
    # Return the predicted results as JSON
    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
