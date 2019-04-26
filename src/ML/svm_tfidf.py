import os
import argparse
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from utils import load_dataset
from utils import tokenize
from utils import print_scores
from utils import get_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help='path to data directory')
    parser.add_argument('--n-jobs', type=int, default=1, help='parallelism')
    parser.add_argument('-s', '--save-model', action='store_true', help='whether to save model')
    parser.add_argument('-l', '--load-model', action='store_true', help='whether to load model')
    args = parser.parse_args()

    if args.load_model:
        x_test, y_test = load_dataset(os.path.join(args.data_dir, 'test.txt'))
        vectorizer = joblib.load(os.path.join(get_model_path(), 'ML_models/vectorizer.pkl'))
        predictor = joblib.load(os.path.join(get_model_path(), 'ML_models/svm_tfidf.pkl'))
        x_test_vectorized = vectorizer.transform(x_test)
        y_test_pred = predictor.predict(x_test_vectorized)
        print_scores(y_test, y_test_pred)

    else:
        x_train, y_train = load_dataset(os.path.join(args.data_dir, 'train.txt'))
        x_valid, y_valid = load_dataset(os.path.join(args.data_dir, 'valid.txt'))
        x_test, y_test = load_dataset(os.path.join(args.data_dir, 'test.txt'))

        n_train, n_valid = len(x_train), len(x_valid)
        x_train = np.concatenate([x_train, x_valid])
        y_train = np.concatenate([y_train, y_valid])

        vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df=0.5, min_df=5)
        vectorizer.fit(x_train)

        x_train_vectorized = vectorizer.transform(x_train)
        x_test_vectorized = vectorizer.transform(x_test)

        steps = [
            ('decomposer', TruncatedSVD(random_state=42)),
            ('classifier', LinearSVC())
        ]
        pipeline = Pipeline(steps)

        params = {
            'decomposer__n_components': [256, 512, 1024],
            'classifier__C': [1e-3, 1e-2, 1e-1, 1e0, 1e1]
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
