import argparse
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from gensim import corpora, matutils


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


def get_word_split_array(x_train, x_test):
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


def get_bow_vec(splitted_train, splitted_test):
    """

    :param splitted_train: splitted train data
    :param splitted_test: splitted test data

    """
    dictionary = corpora.Dictionary(splitted_train)
    dictionary.filter_extremes(no_below=3, no_above=0.3)

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

    x_train, y_train = load_dataset(args.train_data)
    x_valid, y_valid = load_dataset(args.valid_data)
    x_test, y_test = load_dataset(args.test_data)

    x_train = np.concatenate([x_train, x_valid])
    y_train = np.concatenate([y_train, y_valid])

    x_train_split, x_test_split = get_word_split_array(x_train, x_test)

    x_train_vectorized, x_test_vectorized = get_bow_vec(x_train_split, x_test_split)

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
    )

    predictor.fit(x_train_vectorized, y_train)
    y_test_pred = predictor.predict(x_test_vectorized)
    print_scores(y_test, y_test_pred)


if __name__ == '__main__':
    main()
