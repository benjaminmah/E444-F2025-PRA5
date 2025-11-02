from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

application = Flask(__name__)

def load_model():
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)

    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)

    return loaded_model, vectorizer

model, vectorizer = load_model()

@application.route('/')
def index():
    return "✅ Fake News Detection API is running! Use POST /predict with JSON {'text': '...'}"

@application.route('/health')
def health():
    return jsonify({"status": "ok"})

@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': "Missing 'text'"}), 400

    pred = model.predict(vectorizer.transform([text]))[0]

    try:
        import numpy as np
        if isinstance(pred, (int, np.integer)):
            result = {0: 'FAKE', 1: 'REAL'}.get(int(pred), str(pred).upper())
        else:
            result = str(pred).upper()
    except Exception:
        result = str(pred).upper()

    return jsonify({'prediction': result})

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000, debug=True)