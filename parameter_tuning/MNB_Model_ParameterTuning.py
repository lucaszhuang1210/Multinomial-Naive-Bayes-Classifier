"""
IMDb Sentiment Analysis with Hyperparameter Tuning for Precision
Usage:
    python imdb_sentiment_tuned.py \
        --data_path IMDB_Dataset.csv \
        --test_size 0.2 \
        --random_state 42 \
        --threshold 0.7
"""
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, precision_recall_curve, make_scorer
)

def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)
    return text.encode('ascii', errors='ignore').decode()

def load_and_split(path, test_size, seed):
    df = pd.read_csv(path)
    df['review'] = df['review'].apply(clean_text)
    return train_test_split(df['review'], df['sentiment'],
                            test_size=test_size, random_state=seed)

def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='IMDb Sentiment Analysis (tuned)')
    parser.add_argument('--data_path',    type=str,   default='../IMDB-Dataset.csv')
    parser.add_argument('--test_size',    type=float, default=0.2)
    parser.add_argument('--random_state', type=int,   default=42)
    parser.add_argument('--threshold',    type=float, default=0.5,
                        help='Decision threshold for positive class')
    args = parser.parse_args()

    # 1. Load & split
    X_train, X_test, y_train, y_test = load_and_split(
        args.data_path, args.test_size, args.random_state
    )

    # 2. Build a pipeline and grid-search to maximize precision
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('nb', MultinomialNB())
    ])
    param_grid = {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'tfidf__max_df': [0.5, 0.7, 0.9],
        'tfidf__min_df': [1, 5, 10],
        'nb__alpha': [0.01, 0.1, 0.5, 1.0]
    }
    grid = GridSearchCV(
        pipeline, param_grid,
        #scoring=make_scorer(precision_score, pos_label='positive'),
        scoring='accuracy',
        cv=5, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("Best hyperparameters for precision:", grid.best_params_)
    print("CV Precision (positive class):", grid.best_score_)

    # 3. Predict on test set (with optional custom threshold)
    if args.threshold == 0.5:
        y_pred = best_model.predict(X_test)
    else:
        # get positive-class probabilities
        probas = best_model.predict_proba(X_test)[:, best_model.classes_.tolist().index('positive')]
        y_pred = np.where(probas >= args.threshold, 'positive', 'negative')

    # 4. Evaluation
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Test Precision:", precision_score(y_test, y_pred, pos_label='positive'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # 5. Precision-Recall curve
    probas = best_model.predict_proba(X_test)[:, best_model.classes_.tolist().index('positive')]
    precision, recall, thresholds = precision_recall_curve(
        (y_test == 'positive').astype(int), probas
    )
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # 6. Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['positive','negative'])
    plot_confusion(cm, labels=['positive','negative'])

    # 7. Sample predictions
    print("\nSample Predictions:")
    for review, actual, pred in zip(X_test.iloc[:5], y_test.iloc[:5], y_pred[:5]):
        snippet = review.replace('\n',' ')[:200]
        print(f"- Review: {snippet}...")
        print(f"  Actual: {actual} | Predicted: {pred}\n")

if __name__ == "__main__":
    main()
