{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "####  任务3——利用训练后的模型来识别语音"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\users\\lenovo\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\scipy\\io\\wavfile.py:273: WavFileWarning: Chunk (non-data) not understood, skipping it.\n",
      "  WavFileWarning)\n"
     ]
    }
   ],
   "source": [
    "#7.3task3\n",
    "'''\n",
    "任务3——利用训练好的模型来识别语音\n",
    "'''\n",
    "#获得模型的语音特征输入数据\n",
    "from VioceFeature import *\n",
    "voicefeature=VioceFeature() \n",
    "audios,samp_rate=voicefeature.vad('data\\\\audio.wav')\n",
    "features = []\n",
    "for audio in audios:\n",
    "    feature = voicefeature.get_mfcc(audio, samp_rate)\n",
    "    features.append(feature)\n",
    "features = np.concatenate(features, 0).astype('float32')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "语音数字的识别结果是： 5 7 9 6\n"
     ]
    }
   ],
   "source": [
    "#识别语音结果\n",
    "import numpy as np\n",
    "import paddle.fluid as fluid\n",
    "from paddle.fluid.dygraph import to_variable, load_dygraph\n",
    "from AudioCNN import AudioCNN\n",
    "with fluid.dygraph.guard(place=fluid.CPUPlace()):\n",
    "    model = AudioCNN()\n",
    "    params_dict, _ = load_dygraph('data/final_model')\n",
    "    model.set_dict(params_dict)\n",
    "    model.eval()\n",
    "    features =to_variable(features)\n",
    "    out = model(features)\n",
    "    result = ' '.join([str(num) for num in np.argmax(out.numpy(),1).tolist()])\n",
    "    print('语音数字的识别结果是：',result)"
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
