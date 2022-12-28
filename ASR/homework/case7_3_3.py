'''
任务3——利用训练好的模型来识别语音
'''
from VioceFeature import *
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable, load_dygraph
from AudioCNN import AudioCNN


# 获得模型的语音特征输入数据
voicefeature = VioceFeature()
audios, samp_rate = voicefeature.vad('data\\audio.wav')
features = []
for audio in audios:
    feature = voicefeature.get_mfcc(audio, samp_rate)
    features.append(feature)
features = np.concatenate(features, 0).astype('float32')


# 识别语音结果
with fluid.dygraph.guard(place=fluid.CPUPlace()):
    model = AudioCNN()
    params_dict, _ = load_dygraph('data/final_model')
    model.set_dict(params_dict)
    model.eval()
    features = to_variable(features)
    out = model(features)
    result = ' '.join([str(num) for num in np.argmax(out.numpy(), 1).tolist()])
    print('语音数字的识别结果是：', result)
