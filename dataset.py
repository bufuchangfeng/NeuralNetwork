import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.utils import shuffle

def load_mnist():
    # 加载数据
    data = loadmat("./data/mnist/mnist_all.mat")

    # print(data.keys())

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for i in range(10):
        temp_df = pd.DataFrame(data["train" + str(i)])
        temp_df['label'] = i
        train_data = train_data.append(temp_df)
        temp_df = pd.DataFrame(data["test" + str(i)])
        temp_df['label'] = i
        test_data = test_data.append(temp_df)

    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

    train_labels = np.array(train_data['label'])
    test_labels = np.array(test_data['label'])


    train_data = train_data.drop('label', axis=1)
    test_data = test_data.drop('label', axis=1)

    train_data = np.array(train_data) / 255
    test_data = np.array(test_data) / 255

    return train_data, test_data, train_labels, test_labels