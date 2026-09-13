import io
import os
import ssl
import sys
import time
import traceback

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
import psutil
from transformers import pipeline

from src.preprocess import clean_tweet

app = Flask(__name__)

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

model_tfidf = None
vectorizer_tfidf = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model_tfidf = joblib.load(MODEL_PATH)
        vectorizer_tfidf = joblib.load(VECTORIZER_PATH)
        print("✅ Custom TF-IDF model loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load TF-IDF model: {e}")

def predict_tfidf(tweet):
    if model_tfidf is None or vectorizer_tfidf is None:
        return {
            "sentiment": "unknown",
            "confidence": {"error": "TF-IDF model files not loaded"},
            "metrics": {
                "latency_ms": 0,
                "memory_mb": get_memory_usage_mb()
            }
        }

    start_time = time.perf_counter()
    cleaned = clean_tweet(tweet)
    vec = vectorizer_tfidf.transform([cleaned])
    pred = model_tfidf.predict(vec)[0]
    probs = model_tfidf.predict_proba(vec)[0]
    classes = model_tfidf.classes_
    end_time = time.perf_counter()

    conf = {str(cls).lower(): round(float(p) * 100, 2) for cls, p in zip(classes, probs)}
    return {
        "sentiment": str(pred).lower(),
        "confidence": conf,
        "metrics": {
            "latency_ms": round((end_time - start_time) * 1000, 2),
            "memory_mb": get_memory_usage_mb()
        }
    }

roberta_classifier = None
try:
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    roberta_classifier = pipeline("sentiment-analysis", model=MODEL_NAME, top_k=None)
    print("✅ RoBERTa Transformer loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load RoBERTa model: {e}")

def predict_roberta(tweet):
    if roberta_classifier is None:
        return {
            "sentiment": "unknown",
            "confidence": {"error": "RoBERTa model not loaded"},
            "metrics": {
                "latency_ms": 0,
                "memory_mb": get_memory_usage_mb()
            }
        }

    start_time = time.perf_counter()
    raw_outputs = roberta_classifier(tweet)

    if isinstance(raw_outputs, list) and len(raw_outputs) > 0 and isinstance(raw_outputs[0], list):
        scores_list = raw_outputs[0]
    else:
        scores_list = raw_outputs

    conf = {}
    top_label = None
    max_score = -1.0

    for item in scores_list:
        label = item["label"].lower()
        score = item["score"]
        conf[label] = round(score * 100, 2)
        if score > max_score:
            max_score = score
            top_label = label

    end_time = time.perf_counter()

    return {
        "sentiment": top_label,
        "confidence": conf,
        "metrics": {
            "latency_ms": round((end_time - start_time) * 1000, 2),
            "memory_mb": get_memory_usage_mb()
        }
    }

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Handle missing headers & latin1 encoding fallback for Sentiment140 CSVs
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin1')

        # Check if 'text' column exists; fallback for headerless CSVs
        if 'text' not in [c.lower() for c in df.columns]:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin1', header=None)
            df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

        # Dynamic column detection
        text_col = None
        for col in ['tweet', 'text', 'review', 'comment', 'feedback']:
            if col in [c.lower() for c in df.columns]:
                text_col = [c for c in df.columns if c.lower() == col][0]
                break

        if not text_col:
            text_col = df.columns[-1]

        # Cleaning filters
        drop_empty = request.form.get('drop_empty') == 'true'
        remove_duplicates = request.form.get('remove_duplicates') == 'true'
        min_length = int(request.form.get('min_length', 0))

        if drop_empty:
            df = df.dropna(subset=[text_col])

        if remove_duplicates:
            df = df.drop_duplicates(subset=[text_col])

        if min_length > 0:
            df = df[df[text_col].astype(str).str.len() >= min_length]

        if df.empty:
            return jsonify({'error': 'No rows remaining after applying filters'}), 400

        model_type = request.form.get('model', 'tfidf').lower()
        texts = df[text_col].astype(str).tolist()

        if model_type == 'roberta':
            if roberta_classifier is None:
                return jsonify({'error': 'RoBERTa model is not loaded'}), 500

            raw_outputs = roberta_classifier(texts, batch_size=32, truncation=True)
            predictions = []
            for item in raw_outputs:
                scores = item if isinstance(item, list) else [item]
                top_item = max(scores, key=lambda x: x['score'])
                predictions.append(top_item['label'].lower())

        else:
            if model_tfidf is None or vectorizer_tfidf is None:
                return jsonify({'error': 'TF-IDF model is not loaded'}), 500

            cleaned_texts = [clean_tweet(t) for t in texts]
            vec = vectorizer_tfidf.transform(cleaned_texts)
            predictions = [str(p).lower() for p in model_tfidf.predict(vec)]

        df['predicted_sentiment'] = predictions

        # Compute aggregate metrics
        total = len(df)
        counts = df['predicted_sentiment'].value_counts().to_dict()

        pos_count = counts.get('positive', 0)
        neu_count = counts.get('neutral', 0)
        neg_count = counts.get('negative', 0)

        metrics = {
            "total": total,
            "positive": {"count": pos_count, "pct": round((pos_count / total) * 100, 1)},
            "neutral": {"count": neu_count, "pct": round((neu_count / total) * 100, 1)},
            "negative": {"count": neg_count, "pct": round((neg_count / total) * 100, 1)}
        }

        # Save processed CSV temporarily to memory for download
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Check if requested as JSON summary (for web UI display)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.args.get('format') == 'json':
            csv_data = df.to_csv(index=False)
            return jsonify({
                "metrics": metrics,
                "csv_data": csv_data
            })

        # Standard file download fallback
        return send_file(
            output,
            mimetype='text/csv',
            download_name='sentiment_predictions.csv',
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_all", methods=["POST"])
def predict_all():
    data = request.get_json() or {}
    tweet = data.get("tweet", "")

    if not tweet.strip():
        return jsonify({"error": "No tweet text provided"}), 400

    return jsonify({
        "results": {
            "TF-IDF + Random Forest": predict_tfidf(tweet),
            "RoBERTa Transformer": predict_roberta(tweet)
        }
    })

if __name__ == "__main__":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.run(host='0.0.0.0', port=5001)
