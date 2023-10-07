目前正在学习吴恩达机器学习和李飞飞计算机视觉的课程，因此这本书读起来较为容易理解

识别手写数字的代码如下：

```python
import numpy
from scipy.special import expit


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设置每层节点的数量
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 随机分配每两层之间的权重，矩阵形式
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 学习率设置
        self.lr = learningrate

        # 激活函数sigmoid()
        self.activation_function = lambda x: expit(x)

        pass

    # 训练函数
    def train(self, inputs_list, targets_list):
        # 将输入转换为二维向量
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算每层的输入输出
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算每层的误(目标值-真实值)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 更新权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # 测试函数
    def query(self, inputs_list):
        # 输入转换为二维向量
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 计算每层的输入输出
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# 设置输入、输出节点数量和学习率大小
input_nodes = 784  # 图片像素大小为28*28
hidden_nodes = 200
output_nodes = 10  # 标签是十个数字
learning_rate = 0.1

# 创建神经网络
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 打开训练集
training_data_file = open("训练集.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 训练神经网络
# 设置迭代次数
epochs = 5

for e in range(epochs):
    # 遍历训练集的每一条数据
    for record in training_data_list:
        all_values = record.split(',')
        # 将输入数据范围归一化到 0.01 到 1.00 的范围
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 每个标签都设置为0.01
        targets = numpy.zeros(output_nodes) + 0.01
        # 只有正确的标签设置为0.99
        targets[int(all_values[0])] = 0.99

        n.train(inputs, targets)
        pass
    pass


# 加载测试集
test_data_file = open("测试集.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 测试神经网络

# 得分表
scorecard = []

# 遍历测试集的每一条数据
for record in test_data_list:
    all_values = record.split(',')
    correct_label = all_values[0]
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    # 询问神经网络
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)

    if label == int(correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass


# 计算分数
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum()/scorecard_array.size)

```

