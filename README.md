# Positive/Negative classification on ACP Corpus

## Development Environment

- Python 3.6.5
- PyTorch 1.0.1.post2
- scikit-learn 0.20.3
- Pipenv 2018.11.26

To setup environment, run
```
$ pipenv sync
```

## Models

### Random Forest
```
$ python src/ML/random_forest.py --train_data path/to/train/file --valid_data path/to/valid/file --test_data path/to/test/file
```

### SVM (BoW)
```
$ python src/ML/svm_bow.py --train_data path/to/train/file --valid_data path/to/valid/file --test_data path/to/test/file
```

### SVM (tf-idf)
```
$ python src/ML/svm_tfidf.py --train_data path/to/train/file --valid_data path/to/valid/file --test_data path/to/test/file
```

### Neural Network Models (MLP, BiLSTM)
 - train (MLP)
    ```
    $ python src/nn/train.py --model MLP --batch-size 2048 --epochs 20 --save-path result/mlp.pth --device <gpu-id>
    ```
- test (MLP)
    ```
    $ python src/nn/test.py --model MLP --batch-size 2048 --load-path result/mlp.pth --device <gpu-id>
    ```

## Dataset
[ACP Corpus: Automatically Constructed Polarity-tagged Corpus](http://www.tkl.iis.u-tokyo.ac.jp/~kaji/acp/)

- tagged and splitted data is located at `/mnt/hinoki/ueda/shinjin2019/acp-2.0`
