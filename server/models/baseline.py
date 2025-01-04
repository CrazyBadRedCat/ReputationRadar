import nltk
import pickle
import os
import logging
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from app.schemas import BaselineHyperparameters
from typing import List, Tuple


class BaselineModel:
    """
    Presents the methods to work with baseline models.
    """

    def __init__(self, fit_id: str):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=20000
        )
        self.fit_id = fit_id
        self.estimator = None
        self.logger = logging.getLogger("baseline_model")

        nltk.data.path.append('/data/nltk_data')
        nltk.download('punkt_tab', download_dir='/data/nltk_data')

    def model_id(self) -> str:
        """
        Return the baseline model_id.
        """
        return f"baseline_{self.fit_id}"

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

    def fit(self, df, request_hyperparams: BaselineHyperparameters):
        """
        Assign labels with transformers.pipeline and use them to train the
        baseline model.
        """
        self.logger.info("Baseline model fitting start")
        sentiment_pipeline = pipeline(
            model='sismetanin/sbert-ru-sentiment-rureviews')
        df['text'] = df['text'].astype(str)
        labels = sentiment_pipeline([text[:2000]
                                    for text in df['text'].values])
        self.logger.info("Done running sentiment_pipeline")
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

        if request_hyperparams:
            lr2_param = {
                'penalty': request_hyperparams.penalty,
                'dual': request_hyperparams.dual,
                'C': request_hyperparams.C,
                'class_weight': request_hyperparams.class_weight
            }

        lr_CV = GridSearchCV(
            lr,
            param_grid=[lr2_param],
            cv=kfold,
            scoring='roc_auc',
            n_jobs=1,
            verbose=1)
        lr_CV.fit(train_tv, y_train)

        self.logger.info("Baseline model fitted")
        self.estimator = lr_CV.best_estimator_

    def predict(self, texts: List[str]) -> Tuple[str, int, List[float]]:
        """
        Preprocess the given text and make predictions using the given model
        """
        texts_preprocessed = [self._prep(text) for text in texts]
        features = self.vectorizer.transform(texts_preprocessed)

        predictions = self.estimator.predict(features)
        probabilities = self.estimator.predict_proba(features)

        return texts, predictions, probabilities

    def save(self, models_storage_path: str):
        """
        Save the trained model and vectorizer to the models storage.
        """
        if not self.estimator:
            self.logger.error("No trained model found to save.")
            return

        directory = os.path.join(models_storage_path, self.model_id())
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, 'model.pkl'), 'wb') as model_file:
            pickle.dump(self.estimator, model_file)
        with open(os.path.join(directory, 'vectorizer.pkl'), 'wb') as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)

        self.logger.info(f"Model and vectorizer saved to {directory}")

    def load(self, directory: str):
        """
        Load the trained model and vectorizer from disk.
        """
        model_path = os.path.join(directory, 'model.pkl')
        vectorizer_path = os.path.join(directory, 'vectorizer.pkl')

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            self.logger.error("Model or vectorizer file not found in the specified directory.")
            return

        with open(model_path, 'rb') as model_file:
            self.estimator = pickle.load(model_file)
        with open(vectorizer_path, 'rb') as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)

        self.logger.info(f"Model and vectorizer loaded from {directory}")
