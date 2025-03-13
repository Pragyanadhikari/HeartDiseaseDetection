from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('stacked_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Renders the form on the webpage

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Create the feature array
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Scale the input features
        scaled_input = scaler.transform(features)

        # Make the prediction using the model
        prediction = model.predict(scaled_input)

        # Interpret prediction result
        if prediction == 1:
            prediction_text = "Heart Disease: Yes"
        else:
            prediction_text = "Heart Disease: No"

        # Return the result to the template
        return render_template('index.html', prediction=prediction_text)

    except Exception as e:
        # Log the error and send a more specific message
        print(f"Error: {e}")
        return render_template('index.html', error=str(e), prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
