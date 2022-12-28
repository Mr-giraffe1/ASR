#!/usr/bin/env python
# coding: utf-8

# #### 音频特征类VioceFeature

# In[3]:


import scipy.io.wavfile as wav
import webrtcvad
import numpy as np
from python_speech_features import mfcc, delta


class VioceFeature():
    # 音频切分
    def vad(self, file_path, mode=3):
        samp_rate, signal_data = wav.read(file_path)
        vad = webrtcvad.Vad(mode=mode)
        signal = np.pad(signal_data, (0, 160 - (signal_data.shape[0] % int(samp_rate * 0.02))), 'constant')
        lens = signal.shape[0]
        signals = np.split(signal, lens // int(samp_rate * 0.02))
        audio = [];
        audios = []
        for signal_item in signals:
            if vad.is_speech(signal_item, samp_rate):
                audio.append(signal_item)
            elif len(audio) > 0 and (not vad.is_speech(signal_item, samp_rate)):
                audios.append(np.concatenate(audio, 0))
                audio = []
        return audios, samp_rate

    # 特征提取
    def get_mfcc(self, data, samp_rate):
        wav_feature = mfcc(data, samp_rate)
        # 对mfcc特征进行一阶差分
        d_mfcc_feat = delta(wav_feature, 1)
        # 对mfcc特征进行二阶差分
        d_mfcc_feat2 = delta(wav_feature, 2)
        # 特征拼接
        feature = np.concatenate(
            [wav_feature.reshape(1, -1, 13), d_mfcc_feat.reshape(1, -1, 13), d_mfcc_feat2.reshape(1, -1, 13)], 0)
        # 对数据进行截取或者填充
        if feature.shape[1] > 64:
            feature = feature[:, :64, :]
        else:
            feature = np.pad(feature, ((0, 0), (0, 64 - feature.shape[1]), (0, 0)), 'constant')
        # 通道转置(HWC->CHW)
        feature = feature.transpose((2, 0, 1))
        # 新建空维度(CHW->NCHW)
        feature = feature[np.newaxis, :]
        return feature
