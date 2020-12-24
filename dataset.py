import numpy as np
import random


def load_mnist():
    train_data_file = './data/mnist/train-images.idx3-ubyte'
    train_label_file = './data/mnist/train-labels.idx1-ubyte'
    test_data_file = './data/mnist/t10k-images.idx3-ubyte'
    test_label_file = './data/mnist/t10k-labels.idx1-ubyte'

    with open(train_data_file) as f:
        train_data = np.fromfile(f, dtype=np.uint8)[16:]

    with open(train_label_file) as f:
        train_label = np.fromfile(f, dtype=np.uint8)[8:]

    with open(test_data_file) as f:
        test_data = np.fromfile(f, dtype=np.uint8)[16:]

    with open(test_label_file) as f:
        test_label = np.fromfile(f, dtype=np.uint8)[8:]

    n_train_samples = train_label.shape[0]
    n_test_samples  = test_label.shape[0]

    train_data = train_data.reshape(n_train_samples, -1)
    test_data = test_data.reshape(n_test_samples, -1)

    data_label = list(zip(train_data, train_label))
    random.shuffle(data_label)
    train_data[:], train_label[:] = zip(*data_label)

    data_label = list(zip(test_data, test_label))
    random.shuffle(data_label)
    test_data[:], test_label[:] = zip(*data_label)

    return train_data, train_label, test_data, test_label
