import argparse
import re
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)
    return text.encode('ascii', errors='ignore').decode()


def load_split(path, test_size, rnd):
    df = pd.read_csv(path)
    df['review'] = df['review'].apply(clean_text)
    return train_test_split(df['review'], df['sentiment'], test_size=test_size, random_state=rnd)


def measure_latency(model, texts, n=1000):
    idx = np.random.choice(len(texts), size=min(n, len(texts)), replace=False)
    samples = [texts[i] for i in idx]
    times = []
    for t in samples:
        start = time.perf_counter()
        if hasattr(model, 'predict_proba'):
            model.predict_proba([t])
        elif hasattr(model, 'decision_function'):
            model.decision_function([t])
        else:
            model.predict([t])
        times.append((time.perf_counter() - start) * 1000)
    arr = np.array(times)
    return arr.mean(), np.percentile(arr, 99)


def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=labels, yticklabels=labels,
           xlabel='Predicted', ylabel='True', title='Confusion Matrix')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i,j], ha='center', va='center')
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()


def plot_tradeoff(results):
    fig, ax = plt.subplots()
    for name,(acc,lat) in results.items():
        ax.scatter(lat, acc, label=name)
    ax.set_xlabel('Latency (ms/sample)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Latency Tradeoff')
    ax.legend()
    plt.show()


def plot_latency_comparison(results):
    """Bar chart comparing latency across all models."""
    fig, ax = plt.subplots()
    names = list(results.keys())
    lats = [results[n][1] for n in names]
    bars = ax.bar(names, lats, width=0.6)
    ax.set_ylabel('Latency (ms/sample)')
    ax.set_title('Latency Comparison Across Models')
    plt.xticks(rotation=45, ha='right')
    # annotate values
    for bar, lat in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width()/2, lat, f'{lat:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


class TextModel:
    def __init__(self, vec, clf):
        self.vec = vec
        self.clf = clf
    def predict(self, texts):
        X = self.vec.transform(texts)
        return self.clf.predict(X)
    def predict_proba(self, texts):
        X = self.vec.transform(texts)
        return self.clf.predict_proba(X)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default='IMDB-Dataset.csv')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--random_state', type=int, default=42)
    p.add_argument('--vectorizer', choices=['count','tfidf','hashing'], default='tfidf')
    p.add_argument('--max_df', type=float, default=0.7)
    p.add_argument('--min_df', type=int, default=1)
    p.add_argument('--ngram_min', type=int, default=1)
    p.add_argument('--ngram_max', type=int, default=2)
    p.add_argument('--alpha', type=float, default=1.0,
                   help='Base NB smoothing parameter')
    p.add_argument('--imp_alpha', type=float, default=None,
                   help='Improved NB smoothing parameter (defaults to base alpha)')
    p.add_argument('--k_features', type=int, default=5000,
                   help='Number of χ²-selected features')
    p.add_argument('--use_grid', action='store_true',
                   help='Enable GridSearch for base NB only')
    args = p.parse_args()

    X_train, X_test, y_train, y_test = load_split(
        args.data_path, args.test_size, args.random_state)

    # choose base vectorizer
    if args.vectorizer == 'count':
        base_vec = CountVectorizer(stop_words='english', min_df=args.min_df,
                                   ngram_range=(args.ngram_min, args.ngram_max))
    elif args.vectorizer == 'hashing':
        base_vec = HashingVectorizer(stop_words='english', ngram_range=(args.ngram_min,args.ngram_max),
                                     alternate_sign=False, norm=None)
    else:
        base_vec = TfidfVectorizer(stop_words='english', max_df=args.max_df,
                                   min_df=args.min_df,
                                   ngram_range=(args.ngram_min, args.ngram_max))

    # train base NB (optional GridSearch)
    if args.use_grid and args.vectorizer != 'hashing':
        pipe0 = Pipeline([('vec', base_vec), ('nb', MultinomialNB())])
        grid = GridSearchCV(pipe0,
                            {'vec__min_df': [1,2,5],
                             'vec__ngram_range': [(1,1),(1,2)],
                             'nb__alpha': [0.1,0.5,1.0]},
                            cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        print("Best params:", grid.best_params_)
        model_base = grid.best_estimator_
        base_model = model_base
    else:
        Xv = base_vec.fit_transform(X_train)
        base_model = MultinomialNB(alpha=args.alpha)
        base_model.fit(Xv, y_train)
        base_model = TextModel(base_vec, base_model)

    # evaluate base NB
    yb = base_model.predict(X_test)
    acc_b = accuracy_score(y_test, yb)
    print(f"Base NB Acc: {acc_b:.4f}")
    print(classification_report(y_test, yb, digits=4))
    cm0 = confusion_matrix(y_test, yb, labels=['positive','negative'])
    plot_confusion(cm0, ['positive','negative'])
    lat_b, lat99_b = measure_latency(base_model, list(X_test))
    print(f"Base NB Latency: {lat_b:.2f} ms, 99th: {lat99_b:.2f} ms")

    # offline χ² selection and retrain NB
    imp_alpha = args.imp_alpha if args.imp_alpha is not None else args.alpha
    acc_imp, lat_imp = None, None
    if args.vectorizer != 'hashing':
        Xv_train = base_vec.transform(X_train)
        selector = SelectKBest(chi2, k=args.k_features)
        selector.fit(Xv_train, y_train)
        mask = selector.get_support()
        feats = np.array(base_vec.get_feature_names_out())[mask]

        # rebuild vectorizer with limited vocabulary
        if args.vectorizer == 'count':
            imp_vec = CountVectorizer(vocabulary=feats)
        else:
            imp_vec = TfidfVectorizer(vocabulary=feats)

        Xtr = imp_vec.fit_transform(X_train)
        Xte = imp_vec.transform(X_test)
        nb_imp = MultinomialNB(alpha=imp_alpha)
        nb_imp.fit(Xtr, y_train)
        yimp = nb_imp.predict(Xte)
        acc_imp = accuracy_score(y_test, yimp)
        print(f"\nImproved NB Acc: {acc_imp:.4f}")
        print(classification_report(y_test, yimp, digits=4))
        imp_model = TextModel(imp_vec, nb_imp)
        lat_imp, lat99_imp = measure_latency(imp_model, list(X_test))
        print(f"Improved NB (k={args.k_features}, α={imp_alpha}) Latency: {lat_imp:.2f} ms, 99th: {lat99_imp:.2f} ms")
    else:
        print("Skipping offline χ² for hashing vectorizer")

    # compare four models
    results = {'BaseNB': (acc_b, lat_b)}
    if acc_imp is not None:
        results[f'NB_χ2_{args.k_features}'] = (acc_imp, lat_imp)
    for name, clf in [('LogisticRegression', LogisticRegression(max_iter=1000)),
                      ('LinearSVC', LinearSVC())]:
        pipe = Pipeline([('vec', base_vec), ('clf', clf)])
        pipe.fit(X_train, y_train)
        yc = pipe.predict(X_test)
        ac = accuracy_score(y_test, yc)
        lt, _ = measure_latency(pipe, list(X_test))
        results[name] = (ac, lt)
        print(f"{name} - Acc: {ac:.4f}, Lat: {lt:.2f} ms")

    # plot tradeoff & latency comparison
    plot_tradeoff(results)
    plot_latency_comparison(results)


if __name__ == '__main__':
    main()
