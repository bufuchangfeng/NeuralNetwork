import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import copy

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


data = loadmat('ex3data1.mat')

# print(data.keys())
#
# print(data['X'])
#
# print(len(data['X']))
#
# print(data['y'])
#
# print(len(data['y']))

X = data['X']
Y = data['y']

# temp = np.reshape(X[0], (20, 20))
# plt.imshow(temp)
# plt.show()


for i in range(len(Y)):
    if Y[i][0] == 10:
        Y[i] = 0
    else:
        Y[i] = Y[i][0]

labels = []
for label in Y:
     labels.append(label[0])

X_sparse = coo_matrix(X)

X, X_sparse, labels = shuffle(X, X_sparse, labels, random_state=0)

w = np.random.randn(10, 400)
# print(w.shape)
b = np.random.randn(10)
# print(b.shape)

epochs = 10
lr = 0.01

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=0)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0)

# print(len(X_train))
# print(len(X_test))
# print(len(X_valid))

# print(len(w))

best_w = None
best_b = None
best_acc = -1
for epoch in range(epochs):
    loss = 0
    for i in range(len(X_train)):
        data = np.array(X_train[i])
        outputs = np.dot(w, data) + b
        p = softmax(outputs)

        _p = np.log(p + 1e-5)

        # print(outputs)

        true = np.zeros(10)
        true[y_train[i]] = 1

        loss += np.sum(-_p*true)

        for m in range(10):
            for n in range(400):
                w[m][n] -= lr*(p-true)[m]*data[n]
        b -= lr*(p-true)

    print("epoch: {}/{} loss: {}".format(epoch + 1, epochs, loss / len(X_train)))

    valid_correct = 0
    for i in range(len(X_valid)):
        data = np.array(X_valid[i])
        outputs = np.dot(w, data) + b
        p = softmax(outputs)

        if np.argmax(p) == y_valid[i]:
            valid_correct += 1

    acc = valid_correct / len(X_valid)
    print("Valid Acc: {}".format(acc))
    if best_acc < acc:
        best_acc = acc
        best_w = copy.deepcopy(w)
        best_b = copy.deepcopy(b)
        print("get new model!")


test_correct = 0
for i in range(len(X_test)):
    data = np.array(X_test[i])
    outputs = np.dot(best_w, data) + best_b
    p = softmax(outputs)

    if np.argmax(p) == y_test[i]:
        test_correct += 1

print("Test Acc: {}".format(test_correct / len(X_test)))

