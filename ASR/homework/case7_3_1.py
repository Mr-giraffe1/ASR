'''
任务1——提取音频的语音特征数据
'''
#from VioceFeature import *

from ASR.homework.VioceFeature import *
import numpy as np


voicefeature = VioceFeature()
audios, samp_rate = voicefeature.vad('../data/audio.wav')

features = []
for audio in audios:
    feature = voicefeature.get_mfcc(audio, samp_rate)
    features.append(feature)
features = np.concatenate(features, 0).astype('float32')

features.shape
