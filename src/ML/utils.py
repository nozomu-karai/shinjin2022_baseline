import numpy as np
from typing import Tuple
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from pathlib import Path


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
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


def print_scores(y_test, y_test_pred) -> None:
    """

    :param y_test: true labels
    :param y_test_pred: predicted labels

    """
    f_score = f1_score(y_test, y_test_pred, average='macro')
    accuracy = accuracy_score(y_test, y_test_pred)
    print('F-score : %f' % f_score)
    print('Accuracy: %f' % accuracy)


def get_model_path() -> str:
    return Path(__file__).resolve().parents[2]
