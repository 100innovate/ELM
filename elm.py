"""
作者：陈峻林
学校：中国地质大学（武汉）
"""
import os
import struct
import numpy as np
from sklearn.metrics import log_loss
import time

class ELM:
    path = '.\\'
    nHiddenNeurons = 5000

    def load_mnist(self, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(self.path,
                                   '%s-labels.idx1-ubyte'
                                   % kind)
        images_path = os.path.join(self.path,
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

    def F(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sig(self, X, Iw, Ib):
        H = self.F(X * Iw.T + np.ones((len(X), 1)) * Ib)
        return H

    def loadData(self):
        self.trainData, self.label = system.load_mnist()
        tmp = np.zeros((len(self.label), 10))
        for i in range(len(self.label)):
            tmp[i][int(self.label[i])] = 1
        self.label = tmp
        # self.label = np.genfromtxt('.\\out.csv',
        #                            dtype=int, delimiter=',')

    def train(self):
        label = self.label
        trainData = self.trainData
        nHiddenNeurons = self.nHiddenNeurons

        test, testlabel = self.load_mnist('t10k')
        test = test / 255
        trainData = trainData / 255

        # print(label)

        p0 = np.mat(trainData)  # mat()输入矩阵

        Iw = np.mat(np.random.rand(nHiddenNeurons, len(trainData[0])) * 2 - 1)  # InW
        # print("输入层的权重矩阵为：\n")
        # print(Iw)

        Inb = np.mat(np.random.rand(1, nHiddenNeurons))  # Inb
        # print("b矩阵为：\n")
        # print(Inb)

        H0 = self.sig(p0, Iw, Inb)  # 隐含层矩阵
        # print("隐含层矩阵为:\n")
        # print(H0)

        beta = (H0.T * H0).I * H0.T * label  # β矩阵
        # print("β矩阵为：\n")
        # print(beta)

        # Oi = self.sig(p0, Iw, Inb)
        # Oi = Oi * beta
        Oi = self.sig(p0[0], Iw, Inb)
        Oi = Oi * beta

        # np.savetxt('D:\\data\\Oi.csv', Oi, delimiter=',')

        testOi = self.sig(test, Iw, Inb)
        testOi = testOi * beta

        sum = 0
        loss = 0
        for i in range(60):
            Oi = self.sig(trainData[i * 1000: (i + 1) * 1000], Iw, Inb)
            Oi = Oi * beta
            loss = loss+log_loss(label[i * 1000: (i + 1) * 1000],Oi)
            for x in range(1000):
                if np.argmax(Oi[x]) == np.argmax(label[i * 1000 + x]):
                    sum = sum + 1
        print('训练集loss')
        print(loss/60)

        print('训练集精度:')
        print(sum / len(trainData))


        sum = 0
        for i in range(len(testOi)):
            if np.argmax(testOi[i]) == testlabel[i]:
                sum = sum + 1

        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        testlabel2 = np.zeros((len(testlabel), 10))
        for i in range(len(testlabel)):
            testlabel2[i][testlabel[i]] = 1
        print('测试集loss')
        print(log_loss(testlabel2,testOi))
        print('测试集精度:')
        print(sum / len(testOi))
        print()


if __name__ == '__main__':
    start = time.clock()
    np.set_printoptions(suppress=True)
    system = ELM()
    system.loadData()
    system.train()
    end = time.clock()
    print(end - start)
