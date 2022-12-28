# case7-2
'''
提取音频信号的特征
'''
from python_speech_features import mfcc, delta

wav_feature = mfcc(audios[0], 8000)
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

feature
