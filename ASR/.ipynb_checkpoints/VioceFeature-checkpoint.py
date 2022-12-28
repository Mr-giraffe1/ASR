{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 音频特征类VioceFeature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import scipy.io.wavfile as wav\n",
    "import webrtcvad\n",
    "import numpy as np\n",
    "from python_speech_features import mfcc,delta\n",
    "class VioceFeature():\n",
    "    #音频切分\n",
    "    def vad(self,file_path,mode=3):\n",
    "        samp_rate, signal_data = wav.read(file_path)\n",
    "        vad = webrtcvad.Vad(mode=mode)\n",
    "        signal= np.pad(signal_data,(0,160-(signal_data.shape[0]%int(samp_rate*0.02))),'constant')\n",
    "        lens = signal.shape[0]\n",
    "        signals = np.split(signal, lens//int(samp_rate*0.02))\n",
    "        audio = [];audios = []\n",
    "        for signal_item in signals:\n",
    "            if vad.is_speech(signal_item,samp_rate):\n",
    "                audio.append(signal_item)\n",
    "            elif len(audio)>0 and (not vad.is_speech(signal_item,samp_rate)):\n",
    "                audios.append(np.concatenate(audio, 0))\n",
    "                audio= []\n",
    "        return audios,samp_rate\n",
    "    #特征提取\n",
    "    def get_mfcc(self,data, samp_rate):\n",
    "        wav_feature = mfcc(data, samp_rate)\n",
    "        # 对mfcc特征进行一阶差分\n",
    "        d_mfcc_feat = delta(wav_feature, 1)\n",
    "        # 对mfcc特征进行二阶差分\n",
    "        d_mfcc_feat2 = delta(wav_feature, 2)\n",
    "        # 特征拼接\n",
    "        feature = np.concatenate([wav_feature.reshape(1, -1, 13), d_mfcc_feat.reshape(1, -1, 13), d_mfcc_feat2.reshape(1, -1, 13)], 0)\n",
    "        # 对数据进行截取或者填充\n",
    "        if feature.shape[1]>64:\n",
    "            feature = feature[:, :64, :]\n",
    "        else:\n",
    "            feature = np.pad(feature, ((0, 0), (0, 64-feature.shape[1]), (0, 0)), 'constant')\n",
    "        # 通道转置(HWC->CHW)\n",
    "        feature = feature.transpose((2, 0, 1))\n",
    "        # 新建空维度(CHW->NCHW)\n",
    "        feature = feature[np.newaxis, :]\n",
    "        return feature    "
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
