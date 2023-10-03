from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load your SVR model
model_path = 'model/SVR_model.joblib'  # Replace with the correct relative path
loaded_model = joblib.load(model_path)

# Function to predict using the loaded model
def predict_value(overall, potential, wage_eur, release_clause_eur):
    try:
        # Use the loaded model to make predictions
        predicted_value = loaded_model.predict([[
            overall, potential, wage_eur, release_clause_eur
        ]])[0]
        return predicted_value
    except Exception as e:
        return str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        overall = float(request.form['overall'])
        potential = float(request.form['potential'])
        wage_eur = float(request.form['wage_eur'])
        release_clause_eur = float(request.form['release_clause_eur'])
        # Make a prediction using the predict_value function
        predicted_value = predict_value(
            overall, potential, wage_eur, release_clause_eur
        )

        return jsonify({'predicted_value': predicted_value})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)
