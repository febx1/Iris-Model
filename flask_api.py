from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app=Flask(__name__)

model = joblib.load("model/model.pkl")

class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}


@app.route('/')
def home():
    return "Welcome to the Iris Flower Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        feature_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        df = pd.DataFrame(features, columns=feature_names)

        prediction = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]
        print(probabilities)
        result = {
            'prediction': prediction,
            'label': class_names[prediction],
            'probabilities': {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
        }
        print(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)