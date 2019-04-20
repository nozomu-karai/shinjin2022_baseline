import argparse
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def load_dataset(path: str):
    """

    :param path: path to dataset(train, valid, test)

    """
    x_list, y_list = [], []
    with open(path, 'r') as f:
        for line in f:
            y, x = line.strip().split('\t')
            x_list.append(x)
            y_list.append(y)
    return np.array(x_list), np.array(y_list)


def print_scores(y_test, y_test_pred):
    """

    :param y_test: true labels
    :param y_test_pred: predicted labels

    """
    f_score = f1_score(y_test, y_test_pred, average='macro')
    accuracy = accuracy_score(y_test, y_test_pred)
    print('F-score : %f' % f_score)
    print('Accuracy: %f' % accuracy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()

    x_train, y_train = load_dataset(args.train_data)
    x_valid, y_valid = load_dataset(args.valid_data)
    x_test, y_test = load_dataset(args.test_data)

    x_train = np.concatenate([x_train, x_valid])
    y_train = np.concatenate([y_train, y_valid])

    vectorizer = TfidfVectorizer()
    vectorizer.fit(x_train)

    x_train_vectorized = vectorizer.transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    steps = [
        ('decomposer', TruncatedSVD(random_state=42)),
        ('classifier', SVC(gamma='auto'))
    ]
    pipeline = Pipeline(steps)

    params = {
        'decomposer__n_components': [128, 256],
        'classifier__C': [1e1, 1e2]
    }
    predictor = GridSearchCV(
        pipeline,
        params,
        cv=5,
        n_jobs=args.n_jobs
    )

    predictor.fit(x_train_vectorized, y_train)
    y_test_pred = predictor.predict(x_test_vectorized)
    print_scores(y_test, y_test_pred)


if __name__ == '__main__':
    main()
