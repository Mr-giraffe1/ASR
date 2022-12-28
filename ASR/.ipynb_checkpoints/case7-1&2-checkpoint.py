{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 切分音频文件"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#case7-1\n",
    "'''\n",
    "切分音频文件中有效的语音数据\n",
    "'''\n",
    "import scipy.io.wavfile as wav\n",
    "import webrtcvad\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "samp_rate, signal_data = wav.read('data/1_5.wav')\n",
    "vad = webrtcvad.Vad(mode=3)\n",
    "signal= np.pad(signal_data,(0,160-(signal_data.shape[0]%int(samp_rate*0.02))),'constant')\n",
    "lens = signal.shape[0]\n",
    "signals = np.split(signal, lens//int(samp_rate*0.02))\n",
    "audio = [];audios = []\n",
    "for signal_item in signals:\n",
    "    if vad.is_speech(signal_item,samp_rate):\n",
    "        audio.append(signal_item)\n",
    "    elif len(audio)>0 and (not vad.is_speech(signal_item,samp_rate)):\n",
    "        audios.append(np.concatenate(audio, 0))\n",
    "        audio= []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[array([ 16, 150, 294, ...,  85,  59,  51], dtype=int16),\n",
       " array([931, 604, 307, ..., -38,  -9,  18], dtype=int16)]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "audios"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 提取语音特征"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "#case7-2\n",
    "'''\n",
    "提取音频信号的特征\n",
    "'''\n",
    "from python_speech_features import mfcc,delta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "wav_feature = mfcc(audios[0],8000)\n",
    "# 对mfcc特征进行一阶差分\n",
    "d_mfcc_feat = delta(wav_feature,1)\n",
    "# 对mfcc特征进行二阶差分\n",
    "d_mfcc_feat2 = delta(wav_feature,2)\n",
    "# 特征拼接\n",
    "feature = np.concatenate([wav_feature.reshape(1,-1,13),d_mfcc_feat.reshape(1,-1,13),d_mfcc_feat2.reshape(1,-1,13)], 0)\n",
    "# 对数据进行截取或者填充\n",
    "if feature.shape[1]>64:\n",
    "    feature = feature[:,:64,:]\n",
    "else:\n",
    "    feature = np.pad(feature,((0,0),(0,64-feature.shape[1]),(0,0)),'constant')\n",
    "# 通道转置(HWC->CHW)\n",
    "feature = feature.transpose((2,0,1))\n",
    "# 新建空维度(CHW->NCHW)\n",
    "feature = feature[np.newaxis,:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[[ 1.50488583e+01,  1.51186821e+01,  1.52201710e+01, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [ 3.49118974e-02,  8.56563109e-02,  1.58480339e-01, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [ 4.12449038e-02,  9.44881568e-02,  1.41938531e-01, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],\n",
       "\n",
       "        [[ 1.72933980e+01,  1.73958691e+01,  1.73137629e+01, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [ 5.12355624e-02,  1.01824666e-02,  2.13479214e-01, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [ 1.43200991e-02,  1.07922404e-01, -1.07565428e-01, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],\n",
       "\n",
       "        [[ 6.81440275e+00,  2.48758464e+00,  4.29244786e+00, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [-2.16340906e+00, -1.26097745e+00,  8.57081266e-01, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [-9.37072791e-01, -7.74726606e-01, -1.61295808e+00, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],\n",
       "\n",
       "        ...,\n",
       "\n",
       "        [[-8.96073994e+00, -7.88636813e+00, -6.40580839e+00, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [ 5.37185906e-01,  1.27746578e+00, -3.45064967e-01, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [ 6.18423493e-01,  3.32341531e-01,  1.00806853e-01, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],\n",
       "\n",
       "        [[ 7.37314187e+00,  8.90517318e+00,  7.03081236e+00, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [ 7.66015658e-01, -1.71164755e-01, -1.59006996e+00, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [ 8.47372297e-02, -3.63854673e-01, -1.05357022e+00, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],\n",
       "\n",
       "        [[-1.30220918e+01, -2.40000422e+01, -9.29902244e+00, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [-5.48897522e+00,  1.86153466e+00,  1.01051916e+01, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],\n",
       "         [-3.53181180e-01,  2.21879350e+00,  2.64683349e+00, ...,\n",
       "           0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]]])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "feature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
