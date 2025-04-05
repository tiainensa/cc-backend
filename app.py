from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model and vectorizer
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

@app.route('/')
def hello_world():
    return "<h1>Hello, World! Sentiment Analysis Backend</h1>"

@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    input_data = request.json
    text = input_data.get("text", "")

    # Load the model and vectorizer
    model, vectorizer = load_model()

    # Transform the input text and make a prediction
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    # Map prediction to sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"
    score = probability[1] if prediction == 1 else -probability[0]

    return jsonify({"text": text, "sentiment": sentiment, "score": score})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)