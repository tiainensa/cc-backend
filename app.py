from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

import numpy as np

app = Flask(__name__)
CORS(app, origins=["https://cc-front-render-r7sq.onrender.com"])  # Replace with your frontend's URL

# Load the pre-trained pipeline (model + vectorizer)
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return model

@app.route('/')
def hello_world():
    return "<h1>Hello, World! Sentiment Analysis Backend</h1>"

@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    input_data = request.json
    text = input_data.get("text", "")

    # Load the pipeline (model + vectorizer)
    model = load_model()

    # Use the pipeline to make a prediction
    prediction = model.predict([text])[0]  # Get the predicted label (e.g., 'positive', 'negative', 'neutral')
    decision_function = model.decision_function([text])[0]  # Extract the first row of the 2D array

    # Exponentiate and normalize to get probabilities
    probabilities = np.exp(decision_function) / np.sum(np.exp(decision_function))

    # Get the confidence score for the predicted class
    predicted_index = list(model.classes_).index(prediction)  # Find the index of the predicted class
    confidence = probabilities[predicted_index]  # Get the probability for the predicted class

    print(f"DEBUG: prediction: {prediction}")  # Debugging
    print(f"DEBUG: decision_function: {decision_function}")  # Debugging
    print(f"DEBUG: probabilities: {probabilities}")  # Debugging
    print(f"DEBUG: confidence: {confidence}")  # Debugging

    # Map prediction to sentiment
    sentiment = prediction.capitalize()  # Capitalize the prediction (e.g., 'positive' -> 'Positive')

    return jsonify({"text": text, "sentiment": sentiment, "score": round(confidence, 4)})  # Round confidence to 4 decimal places

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)