import os
import argparse
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from utils import load_dataset
from utils import print_scores
from utils import get_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('-s', '--save_model', action='store_true')
    parser.add_argument('-l', '--load_model', action='store_true')
    args = parser.parse_args()

    if args.load_model:
        x_test, y_test = load_dataset(args.test_data)
        vectorizer = joblib.load(os.path.join(get_model_path(), 'ML_models/vectorizer.pkl'))
        predictor = joblib.load(os.path.join(get_model_path(), 'ML_models/svm_tfidf.pkl'))
        x_test_vectorized = vectorizer.transform(x_test)
        y_test_pred = predictor.predict(x_test_vectorized)
        print_scores(y_test, y_test_pred)

    else:
        x_train, y_train = load_dataset(args.train_data)
        x_valid, y_valid = load_dataset(args.valid_data)
        x_test, y_test = load_dataset(args.test_data)

        n_train, n_valid = len(x_train), len(x_valid)
        x_train = np.concatenate([x_train, x_valid])
        y_train = np.concatenate([y_train, y_valid])

        vectorizer = TfidfVectorizer()
        vectorizer.fit(x_train)

        x_train_vectorized = vectorizer.transform(x_train)
        x_test_vectorized = vectorizer.transform(x_test)

        steps = [
            ('decomposer', TruncatedSVD(random_state=42)),
            ('classifier', SVC(kernel='linear'))
        ]
        pipeline = Pipeline(steps)

        params = {
            'decomposer__n_components': [128, 256, 512],
            'classifier__C': [1e0, 1e1, 1e2, 1e3]
        }

        splitter = [list(range(0, n_train))], [list(range(n_train, n_train + n_valid))]

        predictor = GridSearchCV(
            pipeline,
            params,
            cv=zip(*splitter),
            n_jobs=args.n_jobs,
            verbose=3
        )

        predictor.fit(x_train_vectorized, y_train)
        y_test_pred = predictor.predict(x_test_vectorized)
        print_scores(y_test, y_test_pred)
        print(predictor.best_params_)

        if args.save_model:
            joblib.dump(
                vectorizer,
                os.path.join(get_model_path(), 'ML_models/vectorizer.pkl'),
                compress=1
            )
            joblib.dump(
                predictor.best_estimator_,
                os.path.join(get_model_path(), 'ML_models/svm_tfidf.pkl'),
                compress=1
            )


if __name__ == '__main__':
    main()
