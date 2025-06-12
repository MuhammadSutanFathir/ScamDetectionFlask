from flask import Flask, request, jsonify, send_from_directory
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__, static_folder='static')

# Fungsi preprocessing sama seperti sebelumnya
def preprocess_email(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)  # hapus email
    text = re.sub(r'http\S+|www\S+', '', text)  # hapus link
    text = re.sub(r'\d+', '', text)  # hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # hapus tanda baca
    words = text.split()
    stop_words_en = set(stopwords.words('english'))
    stop_words_id = set(stopwords.words('indonesian'))
    stop_words = stop_words_en.union(stop_words_id)
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Load model dan vectorizer
model = joblib.load('naive_bayes_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Route untuk halaman utama
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')  # asumsi index.html di folder sama dengan script.py

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    email_text = data.get('email', '')

    if not email_text:
        return jsonify({'error': 'Email text is required'}), 400

    cleaned = preprocess_email(email_text)
    vector = tfidf_vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "SCAM"
        message = "Email ini terdeteksi sebagai spam. Harap berhati-hati."
    else:
        result = "HAM"
        message = "Email ini tidak terdeteksi sebagai spam."

    return jsonify({'prediction': result, 'message': message})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
