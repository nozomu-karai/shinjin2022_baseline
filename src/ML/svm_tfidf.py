import argparse
import numpy as np
import utils
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()

    x_train, y_train = utils.load_dataset(args.train_data)
    x_valid, y_valid = utils.load_dataset(args.valid_data)
    x_test, y_test = utils.load_dataset(args.test_data)

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
        'decomposer__n_components': [128, 256],
        'classifier__C': [1e-1, 1e0, 1e1, 1e2, 1e3]
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
    utils.print_scores(y_test, y_test_pred)


if __name__ == '__main__':
    main()
