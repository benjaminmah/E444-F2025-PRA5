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

@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    prediction = model.predict(vectorizer.transform([text]))[0]
    result = 'FAKE' if prediction == 0 else 'REAL'

    return jsonify({'prediction': result})

if __name__ == '__main__':
    application.run(debug=True)