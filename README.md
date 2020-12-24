# 2020年12月24日

这个分支上的代码是我初学神经网络时写的，当时急于求得结果，并不能保证正确性。

目前正在新的分支上重写，会更注意代码的规范性和正确性。

感谢阅读！



# 数据

* ex3data1.mat                               从mnist中选择的5000张图片
* mnist_all.mat                               mnist数据集

# 代码



* neuron.py	                                   神经元
* lr.py                                                  logistic regression + softmax
* 4_layer_neural_network.ipynb    全连接神经网络
* cnn.ipynb                                        卷积神经网络
* rnn.ipynb                                        循环神经网络
* lstm.ipynb                                       长短时记忆网络



# TODO

* lstm temp_dc_dxx 是多余的。由于时间原因，不修改了。
* 所有的网络 测试的时候应该使用test_net 而不是net 由于时间原因，不修改了