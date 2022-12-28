'''
任务2——构建语音数字识别神经网络模型
'''
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Conv2D, BatchNorm
from paddle.fluid.layers import softmax_with_cross_entropy, accuracy, reshape


# 定义语音识别网络模型
class AudioCNN(fluid.dygraph.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(num_channels=13, num_filters=16, filter_size=3, stride=1, padding=1)
        self.conv2 = Conv2D(16, 16, (3, 2), (1, 2), (1, 0))
        self.conv3 = Conv2D(16, 32, 3, 1, 1)
        self.conv4 = Conv2D(32, 32, (3, 2), (1, 2), (1, 0))
        self.conv5 = Conv2D(32, 64, 3, 1, 1)
        self.conv6 = Conv2D(64, 64, (3, 2), 2)
        self.fc1 = Linear(input_dim=1 * 8 * 64, output_dim=128, act='relu')
        self.fc2 = Linear(128, 10, act='softmax')

    # 定义前向网络
    def forward(self, inputs, labels=None):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = reshape(out, [-1, 8 * 64])
        out = self.fc1(out)
        out = self.fc2(out)
        if labels is not None:
            loss = softmax_with_cross_entropy(out, labels)
            acc = accuracy(out, labels)
            return loss, acc
        else:
            return out
