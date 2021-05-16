import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def DataSet_Random(random_state):
    # 读数据
    Date = np.genfromtxt('tb-real11_modify.csv', delimiter=',')
    X = Date[:, 1:]
    X = X[1:, ]
    # 标签第一列
    y = Date[:, 0]
    y = y[1:]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return x_train, x_test, y_train, y_test


# 全部的数据
def DataSet_All():
    # 读数据
    Date = np.genfromtxt('tb-real11_modify.csv', delimiter=',')
    # 取所有的行，和第一列之后的数据，因为第一列是标签，后面的是特征
    X = Date[:, 1:]
    X = X[1:, ]
    # 标签第一列
    y = Date[:, 0]
    y = y[1:]

    return X, y


# random.seed(2)
# 生成区间[a, b)内的随机数
def my_rand(a, b):
    return (b - a) * random.random() + a


# 生成大小 I*J 的矩阵，默认零矩阵
def makeMatrix(I, J):
    return np.zeros([I, J], float)


# tanh
def tanh(x):
    return np.tanh(x)


'''
# relu
def tanh(x):
    return np.maximum(0, x)
'''

'''
#sigmoid
def tanh(x):
    return 1/(1 + np.exp(-x))
'''


# tanh
def tanh_backward(y):
    return 1 - np.tanh(y)**2
#1.0 - y ** 2


'''
# relu
def tanh_backward(y):
    return 1 if y>0 else 0
'''

'''
#sigmoid
def tanh_backward(y):
    return 1/((np.exp(y)+1)**2)
'''

class NN:
    ''' 三层反向传播神经网络 '''

    def __init__(self, ni, nh, no):
        # 输入层、隐藏层、输出层的节点（数）
        self.ni = ni + 1  # 增加一个偏差节点
        self.nh = nh + 1  # 增加一个偏差节点
        self.no = no

        # 激活神经网络的所有节点（向量）
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # 建立权重（矩阵）
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        # 设为随机值
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = my_rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = my_rand(-2.0, 2.0)

        # 最后建立Momentum动量因子（矩阵）
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('与输入层节点数不符！')

        # 激活输入层
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # 激活隐藏层
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)

        # 激活输出层
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = tanh(sum)

        return self.ao[:]

    def backPropagate(self, targets, lr, M):
        ''' 反向传播 '''
        if len(targets) != self.no:
            raise ValueError('与输出层节点数不符！')

        # 计算输出层的误差
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = tanh_backward(self.ao[k]) * error

        # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = tanh_backward(self.ah[j]) * error

        # 更新输出层权重
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] - lr * change + M * self.co[j][k]
                self.co[j][k] = change
                # print(N*change, M*self.co[j][k])

        # 更新输入层权重
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] - lr * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # 计算误差
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def weights(self):
        print('输入层权重:')
     
        for i in range(self.ni):
            print(self.wi[i])
            
            print()
        print('输出层权重:')
            
        for j in range(self.nh):
            print(self.wo[j])
            

    def train(self, x, y):                           
        r, dim = x.shape
        theta = np.zeros(dim)  # 参数
        lr = 0.001  # 学习率
        threshold = 0.0001  # 停止迭代的错误阈值
        iterations = 300000  # 迭代次数
        error = 0  # 初始错误为0
        b1 = 0.9  # 建议的默认值
        b2 = 0.999  # 建议的默认值
        e = 1e-8  # 建议的默认值
        m = np.zeros(dim)
        v = np.zeros(dim)
        i = 0
        for i in range(iterations):
            j = i % r
            error = 1 / (2 * r) * np.dot((np.dot(x, theta) - y).T,
                                        (np.dot(x, theta) - y))
            if abs(error) <= threshold:
                break
            lr_t = lr * np.sqrt(1.0 - b2 ** (i + 1)) / (1.0 - b1 ** (i + 1))
            gradient = x[j] * (np.dot(x[j], theta) - y[j])
            m += (1.0 - b1) * (gradient - m)
            v += (1.0 - b2) * (gradient ** 2 - v)
            theta -= lr_t * m / np.sqrt(v + e)
            print('Adam:迭代次数：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)                   
                            

    def test(self, x_test, y_test):
        result = []
        for x in x_test:
            r = self.update(x)
            result.append(r[0])
        print("均方根误差:", self.mean_squared_error(np.array(y_test - np.array(result))))
        

    # 计算均方误差
    def mean_squared_error(self, deviation):
        return (np.sum(np.power(np.array(deviation), 2)) / len(deviation))**0.5
        # 使用一个列表将误差存储下来
        #errhistory.append(mean_squared_error)

def main():
    # 导入数据
    X_train, X_test, Y_train, Y_test = DataSet_Random(3)  # 获取随机拆分后的数据
    #x, y = x[:10], y[:10]
    # 创建一个神经网络：输入层有6个节点，隐藏层节点数为2~12之间，输出层有一个节点
    n = NN(4, 6, 1)  # 输入层/隐藏层/输出层
    # 用一些模式训练它
    n.train(X_train, Y_train)
    # 测试训练的成果
    n.test(X_test, Y_test)
    # 看看训练好的权重（当然可以考虑把训练好的权重持久化）
    n.weights()
     

if __name__ == '__main__':
    X, y = DataSet_All()
    main()
