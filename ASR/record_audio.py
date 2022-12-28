{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 录制声音"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pyaudio\n",
    "import wave"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 用Pyaudio库录制音频\n",
    "#   out_file:输出音频文件名\n",
    "#   rec_time:音频录制时间(秒)\n",
    "def audio_record(out_file, rec_time):\n",
    "    CHUNK = 1024\n",
    "    FORMAT = pyaudio.paInt16 #16bit编码格式\n",
    "    CHANNELS = 1 #单声道\n",
    "    RATE = 8000 #16000采样频率\n",
    "    pa = pyaudio.PyAudio()\n",
    "    # 创建音频流 \n",
    "    stream = pa.open(format=FORMAT, # 音频流wav格式\n",
    "                    channels=CHANNELS, # 单声道\n",
    "                    rate=RATE, # 采样率8000\n",
    "                    input=True,\n",
    "                    frames_per_buffer=CHUNK)\n",
    "\n",
    "    print(\"Start Recording...\")\n",
    "\n",
    "    frames = [] # 录制的音频流\n",
    "    # 录制音频数据\n",
    "    for i in range(0, int(RATE / CHUNK * rec_time)):\n",
    "        data = stream.read(CHUNK)\n",
    "        frames.append(data)\n",
    "    \n",
    "    # 录制完成\n",
    "    stream.stop_stream()\n",
    "    stream.close()\n",
    "    pa.terminate()\n",
    "\n",
    "    print(\"Recording Done...\")\n",
    "\n",
    "    # 保存音频文件\n",
    "    wf = wave.open(out_file, 'wb')\n",
    "    wf.setnchannels(CHANNELS)\n",
    "    wf.setsampwidth(pa.get_sample_size(FORMAT))\n",
    "    wf.setframerate(RATE)\n",
    "    wf.writeframes(b''.join(frames))\n",
    "    wf.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Start Recording...\n",
      "Recording Done...\n"
     ]
    }
   ],
   "source": [
    "audio_record(r'data\\0.wav',2)"
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
