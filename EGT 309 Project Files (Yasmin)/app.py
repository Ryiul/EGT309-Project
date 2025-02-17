from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)

INFERENCE_URL = "http://model-inference-service:80/predict/" # The model inference service URL

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            features = request.form  # Get features from the form
            feature_list = list(features.values()) # Convert to list, order is important!
            headers = {'Content-Type': 'application/json'}
            response = requests.post(INFERENCE_URL, data=json.dumps(feature_list), headers=headers)
            response.raise_for_status()
            prediction_data = response.json()
            prediction = prediction_data.get("prediction")
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            error_message = f"Error: {e}"
            return render_template('index.html', error=error_message) # Render with error
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error processing response: {e}")
            error_message = f"Error: {e}"
            return render_template('index.html', error=error_message) # Render with error

    return render_template('index.html', prediction=prediction)  # Render the template

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)