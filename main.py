import os
import struct
import numpy as np

class ELM:
    path = 'D:\\data'
    nHiddenNeurons = 1000

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
        self.label = np.genfromtxt('D:\\data\\out.csv',
                               dtype=int, delimiter=',')

    def train(self):
        label = self.label
        trainData = self.trainData
        nHiddenNeurons = self.nHiddenNeurons

        print(label)

        p0 = np.mat(trainData)  # mat()输入矩阵
        p0 = p0 / 255

        Iw = np.mat(np.random.rand(nHiddenNeurons, len(trainData[0])) * 2 - 1)  # InW
        print("输入层的权重矩阵为：\n")
        print(Iw)

        Inb = np.mat(np.random.rand(1, nHiddenNeurons))  # Inb
        print("b矩阵为：\n")
        print(Inb)

        H0 = self.sig(p0, Iw, Inb)  # 隐含层矩阵
        print("隐含层矩阵为:\n")
        print(H0)

        beta = (H0.T * H0).I * H0.T * label  # β矩阵
        print("β矩阵为：\n")
        print(beta)

        #Oi = self.sig(p0, Iw, Inb)
        #Oi = Oi * beta
        Oi = self.sig(p0[0], Iw, Inb)
        Oi = Oi * beta

        #np.savetxt('D:\\data\\Oi.csv', Oi, delimiter=',')

        print("原数据测试结果：")
        print(Oi)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    system = ELM()
    system.loadData()
    system.train()
