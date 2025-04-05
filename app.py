from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

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
    prediction = model.predict([text])[0]
    confidence = model.decision_function([text])[0]

    # Map prediction to sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"
    score = confidence if prediction == 1 else -confidence

    return jsonify({"text": text, "sentiment": sentiment, "score": score})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)