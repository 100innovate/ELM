"""
作者：陈峻林
学校：中国地质大学（武汉）
"""

from sklearn import svm
import numpy as np
import os
import struct

path = '.\\'


def load_mnist(kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


train_num = 10000
test_num = 1000
x_train, y_train = load_mnist()
x_test, y_test = load_mnist('t10k')

# 获取一个支持向量机模型
predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='poly')
# 把数据丢进去
predictor.fit(x_train[:train_num], y_train[:train_num])
# 预测结果
result = predictor.predict(x_test[:test_num])
resulttrain = predictor.predict(x_train[:train_num])
# 准确率估计
trainaccurancy = np.sum(np.equal(resulttrain, y_train[:train_num])) / train_num
print('train acc',trainaccurancy)
accurancy = np.sum(np.equal(result, y_test[:test_num])) / test_num
print('test acc',accurancy)
