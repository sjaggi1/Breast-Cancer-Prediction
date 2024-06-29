from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1, -1)
    prediction = model.predict(final_features)
    prediction_prob = model.predict_proba(final_features)
    
    prediction_class = prediction[0]
    probability_class_4 = '{0:.2f}'.format(prediction_prob[0][1])

    if prediction_class == 2:
        output = "Benign"
    elif prediction_class == 4:
        output = "Malignant"
    else:
        output = "Unknown"

    return render_template('index.html', pred='Predicted Class: {}, Predicted Diagnosis: {}, Probability of Class 4: {}'.format(prediction_class, output, probability_class_4))

if __name__ == '__main__':
    app.run(debug=True)
