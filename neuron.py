import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


'''
    network 
    
    w1
            b
    w2               
'''


def feedforward(w, x, b):
    return sigmoid(np.dot(w, x) + b)


def main():

    w = np.random.random(size=2)
    b = np.random.normal()

    data = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1]
    ])

    y = np.array([
        0,
        0,
        0,
        1
    ])

    epochs = 100000
    learning_rate = 0.005
    for epoch in range(epochs):
        loss = 0
        for i in range(len(y)):
            y_predict = feedforward(w, data[i], b)

            # 反向传播过程
            # dloss/dy_predict * dy_predict/dt * dt/w1
            w[0] -= learning_rate * 2 * (y[i] - y_predict) * (-1) * y_predict * (1 - y_predict) * data[i][0]
            w[1] -= learning_rate * 2 * (y[i] - y_predict) * (-1) * y_predict * (1 - y_predict) * data[i][1]
            b -= learning_rate * 2 * (y[i] - y_predict) * (-1) * y_predict * (1 - y_predict)

            loss += (y[i] - y_predict) * (y[i] - y_predict)

        if epoch % 1000 == 0:
            print("epoch: ", epoch, "loss: ", loss)

    test1 = np.array([1, 0])
    print("test1: ", test1, feedforward(w, test1, b))
    test2 = np.array([0, 0])
    print("test2: ", test2, feedforward(w, test2, b))
    test3 = np.array([0.5, 0.5])
    print("test3: ", test3, feedforward(w, test3, b))
    test4 = np.array([0.2, 0.8])
    print("test4: ", test4, feedforward(w, test4, b))


if __name__ == '__main__':
    main()