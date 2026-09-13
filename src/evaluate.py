import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocess import clean_tweet

def evaluate_models():
    data_path = "data/Tweets.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found.")
        return

    # Load dataset
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text', 'airline_sentiment'])
    
    # Preprocess text
    df['clean_text'] = df['text'].apply(clean_tweet)
    X = df['clean_text']
    y = df['airline_sentiment'].str.lower()

    # Train / Test Split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load saved TF-IDF model & vectorizer
    if not (os.path.exists("model.joblib") and os.path.exists("vectorizer.joblib")):
        print("❌ Model files (model.joblib, vectorizer.joblib) not found. Run train.py first.")
        return

    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")

    # Predict
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    # Print Classification Report
    print("\n================ Classification Report (TF-IDF + Random Forest) ================")
    print(classification_report(y_test, y_pred))

    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive'])
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix - TF-IDF Model')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Actual Sentiment')
    
    # Save plot for README / UI inclusion
    os.makedirs("static", exist_ok=True)
    plot_path = "static/confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"✅ Confusion Matrix plot saved to {plot_path}")

if __name__ == "__main__":
    evaluate_models()
