import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.utils import shuffle

'''
    network architecture
'''


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def relu(x):
    return np.maximum(0, x)


def load_data():
    # 加载数据
    data = loadmat("mnist_all.mat")

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

    # print(train_labels)
    # print(test_labels)
    #
    # print(len(train_labels))
    # print(len(test_labels))

    train_data = train_data.drop('label', axis=1)
    test_data = test_data.drop('label', axis=1)

    train_data = np.array(train_data) / 255
    test_data = np.array(test_data) / 255

    return train_data, test_data, train_labels, test_labels


class NN:
    def __init__(self):
        self.w1 = np.random.rand(784, 400)
        self.w2 = np.random.rand(400, 200)
        self.w3 = np.random.rand(200, 100)
        self.w4 = np.random.rand(100, 10)

    def forward(self, x):
        x = relu(np.dot(x, self.w1))
        x = relu(np.dot(x, self.w2))
        x = relu(np.dot(x, self.w3))
        x = np.dot(x, self.w4)

        return softmax(x)


def main():
    train_data, test_data, train_labels, test_labels = load_data()

    nn = NN()


if __name__ == "__main__":
    main()