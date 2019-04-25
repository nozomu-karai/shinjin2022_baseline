import os
import argparse
import numpy as np

from gensim import corpora, matutils

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from utils import load_dataset
from utils import print_scores
from utils import get_model_path


def get_word_split_array(x) -> np.ndarray:
    """

    :param x: not splitted data

    """
    x_split = []
    for x in x.tolist():
        x_split.append(x.split(' '))

    return np.array(x_split)


def get_bow_vec(splitted_x, dictionary) -> np.ndarray:
    """

    :param splitted_train: splitted train data
    :param splitted_test: splitted test data

    """

    bow_sentences = []
    for x in splitted_x:
        bow_sentence_train = dictionary.doc2bow(x)
        bow_sentences.append(list(matutils.corpus2dense([bow_sentence_train], num_terms=len(dictionary)).T[0]))

    return np.array(bow_sentences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', help='path to train data')
    parser.add_argument('--valid-data', help='path to valid data')
    parser.add_argument('--test-data', help='path to test data')
    parser.add_argument('--n-jobs', type=int, default=1, help='parallelism')
    parser.add_argument('-s', '--save-model', action='store_true', help='whether to save model')
    parser.add_argument('-l', '--load-model', action='store_true', help='whether to load model')
    args = parser.parse_args()

    if args.load_model:
        x_test, y_test = load_dataset(args.test_data)
        dictionary = joblib.load(os.path.join(get_model_path(), 'ML_models/bow_dic.pkl'))
        predictor = joblib.load(os.path.join(get_model_path(), 'ML_models/svm_bow.pkl'))
        x_test_split = get_word_split_array(x_test)
        x_test_vectorized = get_bow_vec(x_test_split, dictionary)
        y_test_pred = predictor.predict(x_test_vectorized)
        print_scores(y_test, y_test_pred)

    else:
        x_train, y_train = load_dataset(args.train_data)
        x_valid, y_valid = load_dataset(args.valid_data)
        x_test, y_test = load_dataset(args.test_data)

        n_train, n_valid = len(x_train), len(x_valid)
        x_train = np.concatenate([x_train, x_valid])
        y_train = np.concatenate([y_train, y_valid])

        x_train_split = get_word_split_array(x_train)
        x_test_split = get_word_split_array(x_test)

        dictionary = corpora.Dictionary(x_train_split)
        dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=100000)

        x_train_vectorized = get_bow_vec(x_train_split, dictionary)
        x_test_vectorized = get_bow_vec(x_test_split, dictionary)

        steps = [
            ('decomposer', TruncatedSVD(random_state=42)),
            ('classifier', SVC(max_iter=10000, cache_size=2000))
        ]
        pipeline = Pipeline(steps)

        params = {
            'decomposer__n_components': [64, 128, 256, 512],
            'classifier__C': [1e1, 1e2, 1e3]
        }
        splitter = [list(range(0, n_train))], [list(range(n_train, n_train + n_valid))]
        predictor = GridSearchCV(
            pipeline,
            params,
            cv=zip(*splitter),
            n_jobs=args.n_jobs,
            verbose=1
        )

        predictor.fit(x_train_vectorized, y_train)
        y_test_pred = predictor.predict(x_test_vectorized)
        print_scores(y_test, y_test_pred)
        print(predictor.best_params_)

        if args.save_model:
            joblib.dump(
                dictionary,
                os.path.join(get_model_path(), 'ML_models/bow_dic.pkl'),
                compress=1
            )
            joblib.dump(
                predictor.best_estimator_,
                os.path.join(get_model_path(), 'ML_models/svm_bow.pkl'),
                compress=1
            )


if __name__ == '__main__':
    main()
