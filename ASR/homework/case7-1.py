#### 切分音频文件

# case7-1
'''
切分音频文件中有效的语音数据
'''
import scipy.io.wavfile as wav
import webrtcvad
import numpy as np

samp_rate, signal_data = wav.read('../data/1_5.wav')
vad = webrtcvad.Vad(mode=3)
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

signals[0].shape

#### 提取语音特征


