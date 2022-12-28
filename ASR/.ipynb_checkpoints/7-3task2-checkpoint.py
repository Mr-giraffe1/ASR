{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "####  任务2——构建语音数字识别神经网络模型"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "#7.3task2\n",
    "'''\n",
    "任务2——构建语音数字识别神经网络模型\n",
    "'''\n",
    "import paddle.fluid as fluid\n",
    "from paddle.fluid.dygraph import Linear, Conv2D, BatchNorm\n",
    "from paddle.fluid.layers import softmax_with_cross_entropy,accuracy,reshape\n",
    "#定义语音识别网络模型\n",
    "class AudioCNN(fluid.dygraph.Layer):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.conv1 = Conv2D(num_channels=13,num_filters=16,filter_size=3,stride=1,padding=1)\n",
    "        self.conv2 = Conv2D(16,16,(3,2),(1,2),(1,0))\n",
    "        self.conv3 = Conv2D(16,32,3,1,1)\n",
    "        self.conv4 = Conv2D(32,32,(3,2),(1,2),(1,0))\n",
    "        self.conv5 = Conv2D(32,64,3,1,1)\n",
    "        self.conv6 = Conv2D(64,64,(3,2),2)\n",
    "        self.fc1 = Linear(input_dim=1*8*64,output_dim=128,act='relu')\n",
    "        self.fc2 = Linear(128,10,act='softmax')\n",
    "\n",
    "    # 定义前向网络\n",
    "    def forward(self, inputs, labels=None):\n",
    "        out = self.conv1(inputs)\n",
    "        out = self.conv2(out)\n",
    "        out = self.conv3(out)\n",
    "        out = self.conv4(out)\n",
    "        out = self.conv5(out)\n",
    "        out = self.conv6(out)\n",
    "        out = reshape(out,[-1,8*64])\n",
    "        out = self.fc1(out)\n",
    "        out = self.fc2(out)\n",
    "        if labels is not None:\n",
    "            loss = softmax_with_cross_entropy(out, labels)\n",
    "            acc = accuracy(out, labels)\n",
    "            return loss, acc\n",
    "        else:\n",
    "            return out"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
