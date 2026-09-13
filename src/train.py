import pandas as pd
import re
import joblib
import ssl
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Bypass SSL verification for NLTK on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    stop_words = set(stopwords.words('english'))
    # Custom rule: Do NOT remove negation words or critical sentiment terms from stopwords
    negation_words = {'not', 'no', 'nor', 'neither', 'never', 'was', 'wasn'}
    stop_words = stop_words - negation_words
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'neither', 'never', 'was', 'wasn'}

def clean_tweet(text):
    """Clean tweet text while preserving numbers and sentiment-bearing contractions."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove @mentions
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation/numbers
    text = text.lower()
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def main():
    print("Loading data...")
    df = pd.read_csv('data/Tweets.csv')
    df = df[['text', 'airline_sentiment']].dropna()
    
    print("Preprocessing tweets...")
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    
    X = df['cleaned_text']
    y = df['airline_sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Vectorizing text using TF-IDF (Unigrams + Bigrams)...")
    # min_df=2 includes more words like 'canceled'/'arrived' instead of dropping them
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=250,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    print("Saving model and vectorizer...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("Done! Model updated successfully.")

if __name__ == '__main__':
    main()
