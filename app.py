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
    confidence = model.decision_function([text])[0]  # Extract the first row of the 2D array

    print(f"DEBUG: prediction: {prediction}")  # Debugging
    print(f"DEBUG: confidence: {confidence}")  # Debugging

    # Map prediction to sentiment and extract the confidence score
    if prediction == 'positive':
        sentiment = "Positive"
        score = float(confidence[1])  # Assuming index 1 corresponds to 'positive'
    elif prediction == 'negative':
        sentiment = "Negative"
        score = float(confidence[0])  # Assuming index 0 corresponds to 'negative'
    elif prediction == 'neutral':
        sentiment = "Neutral"
        score = 0.0  # Neutral sentiment can have a score of 0
    else:
        return jsonify({"error": "Unexpected prediction value"}), 500

    return jsonify({"text": text, "sentiment": sentiment, "score": score})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)