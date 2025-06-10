"""
IMDb Sentiment Analysis with Multinomial Naïve Bayes
Usage:
    python imdb_sentiment.py \
        --data_path IMDB_Dataset.csv \
        --test_size 0.2 \
        --max_df 0.7 \
        --ngram_min 1 \
        --ngram_max 2 \
        --random_state 42
"""

import argparse
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def clean_text(text: str) -> str:
    """Remove HTML tags and non-ASCII characters."""
    text = re.sub(r'<.*?>', '', text)
    text = text.encode('ascii', errors='ignore').decode()
    return text

def load_and_split(data_path: str, test_size: float, random_state: int):
    df = pd.read_csv(data_path)
    df['review'] = df['review'].apply(clean_text)
    return train_test_split(df['review'], df['sentiment'],
                            test_size=test_size, random_state=random_state)

def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center')
    fig.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='IMDb Sentiment Analysis')
    parser.add_argument('--data_path',    type=str,   default='IMDB-Dataset.csv',
                        help='CSV file with columns ["review","sentiment"]')
    parser.add_argument('--test_size',    type=float, default=0.2,
                        help='Fraction of data to reserve for testing')
    parser.add_argument('--max_df',       type=float, default=0.7,
                        help='Max document frequency for TF-IDF')
    parser.add_argument('--ngram_min',    type=int,   default=1,
                        help='Minimum n-gram size for TF-IDF')
    parser.add_argument('--ngram_max',    type=int,   default=2,
                        help='Maximum n-gram size for TF-IDF')
    parser.add_argument('--random_state', type=int,   default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Load and split
    X_train, X_test, y_train, y_test = load_and_split(
        args.data_path, args.test_size, args.random_state
    )

    # Feature extraction
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=args.max_df,
        ngram_range=(args.ngram_min, args.ngram_max)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # Model training
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Evaluation
    y_pred = model.predict(X_test_tfidf)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['positive','negative'])
    plot_confusion(cm, labels=['positive','negative'])

    # Sample predictions
    print("\nSample Predictions:")
    for review, actual, pred in zip(X_test.iloc[:5], y_test.iloc[:5], y_pred[:5]):
        print("-" * 40)
        snippet = review.replace('\n', ' ')[:200]
        print(f"Review: {snippet}...")
        print(f"Actual: {actual} | Predicted: {pred}")

if __name__ == "__main__":
    main()


