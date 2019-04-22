import argparse
import numpy as np
import utils
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from gensim import corpora, matutils


def get_word_split_array(x_train, x_test) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param x_train: not splitted train data
    :param x_test: not splitted test data

    """

    x_train_split, x_test_split = [], []

    for x in x_train.tolist():
        x_train_split.append(x.split(' '))
    for x in x_test.tolist():
        x_test_split.append(x.split(' '))

    return np.array(x_train_split), np.array(x_test_split)


def get_bow_vec(splitted_train, splitted_test) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param splitted_train: splitted train data
    :param splitted_test: splitted test data

    """
    dictionary = corpora.Dictionary(splitted_train)
    dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=100000)

    bow_sentences_train = []
    for x in splitted_train:
        bow_sentence_train = dictionary.doc2bow(x)
        bow_sentences_train.append(list(matutils.corpus2dense([bow_sentence_train], num_terms=len(dictionary)).T[0]))

    bow_sentences_test = []
    for x in splitted_test:
        bow_sentence_test = dictionary.doc2bow(x)
        bow_sentences_test.append(list(matutils.corpus2dense([bow_sentence_test], num_terms=len(dictionary)).T[0]))

    return np.array(bow_sentences_train), np.array(bow_sentences_test)


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

    x_train_split, x_test_split = get_word_split_array(x_train, x_test)

    x_train_vectorized, x_test_vectorized = get_bow_vec(x_train_split, x_test_split)

    steps = [
        ('decomposer', TruncatedSVD(random_state=42)),
        ('classifier', SVC(kernel='linear'))
    ]
    pipeline = Pipeline(steps)

    params = {
        'decomposer__n_components': [128, 256],
        'classifier__C': [1e1, 1e2]
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
