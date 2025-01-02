import pandas as pd
from transformers import pipeline
from tqdm.notebook import tqdm
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple


class BaselineModel:
    """
    Presents the methods to work with baseline models.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=20000
        )

    def model_id(self) -> str:
        """
        Return the baseline model_id.
        """
        return "baseline"

    def _prep(self, review) -> str:
        """
        Format text for further processing.
        """
        # Lower case
        review = review.lower()

        # Tokenize to each word.
        token = nltk.word_tokenize(review)

        # Stemming
        review = [nltk.stem.SnowballStemmer('russian').stem(w) for w in token]

        # Join the words back into one string separated by space, and return
        # the result.
        return " ".join(review)

    def fit(self, df, request_hyperparams):
        """
        Assign labels with transformers.pipeline and use them to train the
        baseline model.
        """
        sentiment_pipeline = pipeline(
            model='sismetanin/sbert-ru-sentiment-rureviews')
        labels = sentiment_pipeline([text[:2000]
                                    for text in df['text'].values])
        labels = [label['label'] for label in labels]
        df['label'] = labels

        df['target'] = (df['label'] == 'LABEL_1').astype(int)

        X_train, y_train = df['text'].apply(self._prep), df['target']

        train_tv = self.vectorizer.fit_transform(X_train)

        kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        lr = LogisticRegression(random_state=42)

        lr2_param = {
            'penalty': ['l2'],
            'dual': [False],
            'C': [6],
            'class_weight': [{1: 1}]
        }

        lr_CV = GridSearchCV(
            lr,
            param_grid=[lr2_param],
            cv=kfold,
            scoring='roc_auc',
            n_jobs=1,
            verbose=1)
        lr_CV.fit(train_tv, y_train)

        return lr_CV.best_estimator_

    def predict(self, model, texts: List[str]) -> Tuple[str, int, List[float]]:
        """
        Preprocess the given text and make predictions using the given model
        """
        texts_preprocessed = [self._prep(text) for text in texts]
        features = self.vectorizer.transform(texts_preprocessed)

        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        return zip(texts, predictions, probabilities)
