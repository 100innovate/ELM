import os
import struct
import numpy as np

class ELM:
    path = 'D:\\data'
    nHiddenNeurons = 6000

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

        test,testlabel=self.load_mnist('t10k')
        test = test / 255
        trainData = trainData /255

        p0 = np.mat(trainData)  # mat()输入矩阵

        p = p0[:10000]

        Iw = np.mat(np.random.rand(nHiddenNeurons, len(trainData[0])) * 2 - 1)  # InW
        print("输入层的权重矩阵为：\n")
        print(Iw)

        Inb = np.mat(np.random.rand(1, nHiddenNeurons))  # Inb
        print("b矩阵为：\n")
        print(Inb)

        H = self.sig(p, Iw, Inb)  # 隐含层矩阵
        print("隐含层矩阵为:\n")
        print(H)

        M = (H.T * H).I
        beta = M * H.T * label[:10000]# np.linalg.pinv(H) * label  (H.T * H).I * H.T * label   β矩阵
        print("β矩阵为：\n")
        print(beta)

        for i in range(21,120):
            H0 = self.sig(p0[i * 500: (i + 1) * 500], Iw, Inb)
            M = M - M * H0.T * (np.eye(1, 1) + H0 * M * H0.T).I * H0 * M
            beta = beta + M * H0.T * (label[i * 500: (i + 1 ) * 500] - H0 * beta)
            print("第",i*500,"个数据")

        testOi = self.sig(test,Iw,Inb)
        testOi = testOi * beta

        sum = 0
        for i in range(60):
            Oi = self.sig(trainData[i * 1000: (i + 1 ) * 1000], Iw, Inb)
            Oi = Oi * beta
            for x in range(1000):
                if np.argmax(Oi[x]) == np.argmax(label[i * 1000 + x]):
                    sum = sum + 1

        print('训练集精度:')
        print(sum/len(trainData))

        sum = 0
        for i in range(len(testOi)):
            if np.argmax(testOi[i]) == testlabel[i]:
                sum = sum + 1

        print('测试集精度:')
        print(sum/len(testOi))

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    system = ELM()
    system.loadData()
    system.train()