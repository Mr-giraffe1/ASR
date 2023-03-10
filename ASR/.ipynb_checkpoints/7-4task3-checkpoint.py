{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "####  利用上线的模型进行语音识别"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#7.4task3\n",
    "'''\n",
    "任务3——调用模型进行实时语音识别\n",
    "'''\n",
    "import websocket\n",
    "import threading\n",
    "import time\n",
    "import uuid\n",
    "import json\n",
    "import logging"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "pcm_file = 'realtime_asr/long.pcm'\n",
    "logger = logging.getLogger()\n",
    "\"\"\"\n",
    "连接 ws_app.run_forever()\n",
    "连接成功后发送数据 on_open()\n",
    "发送开始参数帧 send_start_params()\n",
    "发送音频数据帧 send_audio()\n",
    "库接收识别结果 on_message()\n",
    "发送结束帧 send_finish()\n",
    "关闭连接 on_close()\n",
    "库的报错 on_error()\n",
    "\"\"\"\n",
    "def send_start_params(ws):\n",
    "    \"\"\"\n",
    "    开始参数帧\n",
    "    :param websocket.WebSocket ws:\n",
    "    :return:\n",
    "    \"\"\"\n",
    "    req = {\n",
    "        \"type\": \"START\",\n",
    "        \"data\": {\n",
    "            \"appid\": 22834963,  # 网页上的appid\n",
    "            \"appkey\": 'GY7dRpHrj9rYpwrcfTuSg2zp',  # 网页上的appid对应的appkey\n",
    "            \"dev_pid\": 15372,  # 识别模型\n",
    "            \"lm_id\": 11645,  # 自训练平台才有这个参数\n",
    "            \"cuid\": \"yourself_defined_user_id001\",  # 随便填不影响使用。机器的mac或者其它唯一id，百度计算UV用。\n",
    "            \"sample\": 16000,  # 固定参数\n",
    "            \"format\": \"pcm\"  # 固定参数\n",
    "        }\n",
    "    }\n",
    "    body = json.dumps(req)\n",
    "    ws.send(body, websocket.ABNF.OPCODE_TEXT)\n",
    "    logger.info(\"发送带参数的开始帧:\" + body)\n",
    "\n",
    "\n",
    "def send_audio(ws):\n",
    "    \"\"\"\n",
    "    发送二进制音频数据，注意每个帧之间需要有间隔时间\n",
    "    :param  websocket.WebSocket ws:\n",
    "    :return:\n",
    "    \"\"\"\n",
    "    chunk_ms = 160  # 160ms的录音\n",
    "    chunk_len = int(16000 * 2 / 1000 * chunk_ms)\n",
    "    with open(pcm_file, 'rb') as f:\n",
    "        pcm = f.read()\n",
    "    index = 0\n",
    "    total = len(pcm)\n",
    "    logger.info(\"send_audio total={}\".format(total))\n",
    "    while index < total:\n",
    "        end = index + chunk_len\n",
    "        if end >= total:\n",
    "            # 最后一个音频数据帧\n",
    "            end = total\n",
    "        body = pcm[index:end]\n",
    "        ws.send(body, websocket.ABNF.OPCODE_BINARY)\n",
    "        index = end\n",
    "        time.sleep(chunk_ms / 1000.0)  # ws.send 也有点耗时，这里没有计算\n",
    "\n",
    "def send_finish(ws):\n",
    "    \"\"\"\n",
    "    发送结束帧\n",
    "    :param websocket.WebSocket ws:\n",
    "    :return:\n",
    "    \"\"\"\n",
    "    req = {\n",
    "        \"type\": \"FINISH\"\n",
    "    }\n",
    "    body = json.dumps(req)\n",
    "    ws.send(body, websocket.ABNF.OPCODE_TEXT)\n",
    "    logger.info(\"send FINISH frame\")\n",
    "\n",
    "def send_cancel(ws):\n",
    "    \"\"\"\n",
    "    发送取消帧\n",
    "    :param websocket.WebSocket ws:\n",
    "    :return:\n",
    "    \"\"\"\n",
    "    req = {\n",
    "        \"type\": \"CANCEL\"\n",
    "    }\n",
    "    body = json.dumps(req)\n",
    "    ws.send(body, websocket.ABNF.OPCODE_TEXT)\n",
    "    logger.info(\"send Cancel frame\")\n",
    "\n",
    "def on_open(ws):\n",
    "    \"\"\"\n",
    "    连接后发送数据帧\n",
    "    :param  websocket.WebSocket ws:\n",
    "    :return:\n",
    "    \"\"\"\n",
    "\n",
    "    def run(*args):\n",
    "        \"\"\"\n",
    "        发送数据帧\n",
    "        :param args:\n",
    "        :return:\n",
    "        \"\"\"\n",
    "        send_start_params(ws)\n",
    "        send_audio(ws)\n",
    "        send_finish(ws)\n",
    "\n",
    "    threading.Thread(target=run).start()\n",
    "\n",
    "def on_message(ws, message):\n",
    "    \"\"\"\n",
    "    接收服务端返回的消息\n",
    "    :param ws:\n",
    "    :param message: json格式，自行解析\n",
    "    :return:\n",
    "    \"\"\"\n",
    "    data=json.loads(message)\n",
    "    if 'result' in data:\n",
    "        logger.info(\"识别结果: \" +data['result'])\n",
    "\n",
    "def on_error(ws, error):\n",
    "    \"\"\"\n",
    "    库的报错，比如连接超时\n",
    "    :param ws:\n",
    "    :param error: json格式，自行解析\n",
    "    :return:\n",
    "        \"\"\"\n",
    "    logger.error(\"error: \" + str(error))\n",
    "\n",
    "def on_close(ws):\n",
    "    \"\"\"\n",
    "    Websocket关闭\n",
    "    :param websocket.WebSocket ws:\n",
    "    :return:\n",
    "    \"\"\"\n",
    "    logger.info(\"ws close ...\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[2020-12-26 09:46:21,053] [send_start_params()][INFO] 发送带参数的开始帧:{\"type\": \"START\", \"data\": {\"appid\": 22834963, \"appkey\": \"GY7dRpHrj9rYpwrcfTuSg2zp\", \"dev_pid\": 15372, \"lm_id\": 11645, \"cuid\": \"yourself_defined_user_id001\", \"sample\": 16000, \"format\": \"pcm\"}}\n",
      "[2020-12-26 09:46:21,061] [send_audio()][INFO] send_audio total=4621824\n",
      "[2020-12-26 09:46:23,542] [on_message()][INFO] 识别结果: 卫\n",
      "[2020-12-26 09:46:24,029] [on_message()][INFO] 识别结果: 你\n",
      "[2020-12-26 09:46:24,332] [on_message()][INFO] 识别结果: 你好\n",
      "[2020-12-26 09:46:24,540] [on_message()][INFO] 识别结果: 你好我\n",
      "[2020-12-26 09:46:27,050] [on_message()][INFO] 识别结果: 你好我是办公室的\n",
      "[2020-12-26 09:46:28,729] [on_message()][INFO] 识别结果: 你好我是办公室的您身边\n",
      "[2020-12-26 09:46:28,881] [on_message()][INFO] 识别结果: 你好我是办公室的您身边或\n",
      "[2020-12-26 09:46:29,057] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者\n",
      "[2020-12-26 09:46:29,377] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交\n",
      "[2020-12-26 09:46:29,577] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有\n",
      "[2020-12-26 09:46:29,873] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人\n",
      "[2020-12-26 09:46:30,736] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在\n",
      "[2020-12-26 09:46:30,872] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人让我\n",
      "[2020-12-26 09:46:31,064] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们\n",
      "[2020-12-26 09:46:31,272] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家\n",
      "[2020-12-26 09:46:31,911] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家坐着\n",
      "[2020-12-26 09:46:32,223] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家坐着一\n",
      "[2020-12-26 09:46:32,383] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院\n",
      "[2020-12-26 09:46:32,903] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院\n",
      "[2020-12-26 09:46:33,079] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治\n",
      "[2020-12-26 09:46:33,590] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗\n",
      "[2020-12-26 09:46:34,094] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我\n",
      "[2020-12-26 09:46:34,230] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们\n",
      "[2020-12-26 09:46:34,421] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做\n",
      "[2020-12-26 09:46:34,613] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一\n",
      "[2020-12-26 09:46:34,901] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下\n",
      "[2020-12-26 09:46:35,781] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下哪个\n",
      "[2020-12-26 09:46:36,268] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下哪个知道\n",
      "[2020-12-26 09:46:36,612] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查\n",
      "[2020-12-26 09:46:37,148] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查你看\n",
      "[2020-12-26 09:46:37,284] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查你看方\n",
      "[2020-12-26 09:46:37,612] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查您看方便\n",
      "[2020-12-26 09:46:38,619] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查您看方便吗\n",
      "[2020-12-26 09:46:45,213] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查您看方便吗您一生\n",
      "[2020-12-26 09:46:45,524] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查您看方便吗您一生的\n",
      "[2020-12-26 09:46:45,708] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查您看方便吗您一生的服务\n",
      "[2020-12-26 09:46:46,692] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查您看方便吗您一生的服务满不满\n",
      "[2020-12-26 09:46:46,995] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查您看方便吗您一生的服务满不满意\n",
      "[2020-12-26 09:46:48,003] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查您看方便吗您一生的服务满不满意啊\n",
      "[2020-12-26 09:46:52,736] [on_message()][INFO] 识别结果: 你好我是办公室的会议您这边或者交易有人在我们家人民医院住院治疗现在我们做一下这个调查您看方便吗您一生的服务满不满意啊水平是否满意\n",
      "[2020-12-26 09:46:54,855] [on_message()][INFO] 识别结果: 你好，我是办公室的会议，您这边或者交易有人在我们家人民医院住院治疗，现在我们做一下这个调查，您看方便吗？您一生的服务满不满意啊？水平是否满意？\n",
      "[2020-12-26 09:46:55,526] [on_message()][INFO] 识别结果: 这\n",
      "[2020-12-26 09:46:55,894] [on_message()][INFO] 识别结果: 这些\n",
      "[2020-12-26 09:46:56,054] [on_message()][INFO] 识别结果: 接触\n",
      "[2020-12-26 09:46:56,222] [on_message()][INFO] 识别结果: 这些出国\n",
      "[2020-12-26 09:46:56,406] [on_message()][INFO] 识别结果: 这些出国的\n",
      "[2020-12-26 09:46:56,526] [on_message()][INFO] 识别结果: 这些吃过的糊\n",
      "[2020-12-26 09:46:56,686] [on_message()][INFO] 识别结果: 这些出国的护士\n",
      "[2020-12-26 09:46:56,894] [on_message()][INFO] 识别结果: 这接触过的护士的\n",
      "[2020-12-26 09:46:57,205] [on_message()][INFO] 识别结果: 这些出国的护士的服务\n",
      "[2020-12-26 09:46:57,413] [on_message()][INFO] 识别结果: 这些出国的护士的服务太\n",
      "[2020-12-26 09:46:57,701] [on_message()][INFO] 识别结果: 这些出国的护士的服务态度\n",
      "[2020-12-26 09:46:57,901] [on_message()][INFO] 识别结果: 这些出国的护士的服务态度满\n",
      "[2020-12-26 09:46:58,205] [on_message()][INFO] 识别结果: 这些出国的护士的服务态度满意\n",
      "[2020-12-26 09:46:59,036] [on_message()][INFO] 识别结果: 这些出国的护士的服务态度满意吗\n",
      "[2020-12-26 09:47:24,816] [on_message()][INFO] 识别结果: 这些出国的护士的服务态度满意吗？\n",
      "[2020-12-26 09:47:26,255] [on_message()][INFO] 识别结果: 对你\n",
      "[2020-12-26 09:47:26,591] [on_message()][INFO] 识别结果: 对象\n",
      "[2020-12-26 09:47:26,982] [on_message()][INFO] 识别结果: 对影像科\n",
      "[2020-12-26 09:47:27,078] [on_message()][INFO] 识别结果: 对影像科b\n",
      "[2020-12-26 09:47:27,230] [on_message()][INFO] 识别结果: 对影像科b\n",
      "[2020-12-26 09:47:27,446] [on_message()][INFO] 识别结果: 对影像科b超\n",
      "[2020-12-26 09:47:27,550] [on_message()][INFO] 识别结果: 对影像科b超\n",
      "[2020-12-26 09:47:27,726] [on_message()][INFO] 识别结果: 对影像科b超c\n",
      "[2020-12-26 09:47:27,942] [on_message()][INFO] 识别结果: 对影像科b超ct\n",
      "[2020-12-26 09:47:28,054] [on_message()][INFO] 识别结果: 对影像科b超ct\n",
      "[2020-12-26 09:47:28,230] [on_message()][INFO] 识别结果: 对影像科b超ct拍\n",
      "[2020-12-26 09:47:28,430] [on_message()][INFO] 识别结果: 对影像科b超ct拍片\n",
      "[2020-12-26 09:47:28,565] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子\n",
      "[2020-12-26 09:47:28,725] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子\n",
      "[2020-12-26 09:47:28,949] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的\n",
      "[2020-12-26 09:47:29,069] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检\n",
      "[2020-12-26 09:47:29,229] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查\n",
      "[2020-12-26 09:47:29,429] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的\n",
      "[2020-12-26 09:47:29,597] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工\n",
      "[2020-12-26 09:47:29,749] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作\n",
      "[2020-12-26 09:47:30,021] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作\n",
      "[2020-12-26 09:47:30,100] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作\n",
      "[2020-12-26 09:47:30,276] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员\n",
      "[2020-12-26 09:47:30,484] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务\n",
      "[2020-12-26 09:47:30,588] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务\n",
      "[2020-12-26 09:47:30,764] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务太\n",
      "[2020-12-26 09:47:30,956] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度\n",
      "[2020-12-26 09:47:31,093] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度\n",
      "[2020-12-26 09:47:31,253] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度\n",
      "[2020-12-26 09:47:31,452] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度\n",
      "[2020-12-26 09:47:31,588] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度\n",
      "[2020-12-26 09:47:31,764] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度\n",
      "[2020-12-26 09:47:31,964] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度\n",
      "[2020-12-26 09:47:32,092] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度\n",
      "[2020-12-26 09:47:32,268] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度满意吗\n",
      "[2020-12-26 09:47:32,468] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度满意吗\n",
      "[2020-12-26 09:47:32,596] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度满意吗\n",
      "[2020-12-26 09:47:32,755] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度满意吗\n",
      "[2020-12-26 09:47:32,979] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度满意吗\n",
      "[2020-12-26 09:47:33,099] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度满意吗\n",
      "[2020-12-26 09:47:33,227] [on_message()][INFO] 识别结果: 对影像科b超ct拍片子的检查的工作人员服务态度满意吗\n",
      "[2020-12-26 09:47:33,403] [on_message()][INFO] 识别结果: 对影像科B超CT拍片子的检查的工作人员服务态度满意吗？\n",
      "[2020-12-26 09:47:33,963] [on_message()][INFO] 识别结果: 一\n",
      "[2020-12-26 09:47:34,163] [on_message()][INFO] 识别结果: 医院\n",
      "[2020-12-26 09:47:34,459] [on_message()][INFO] 识别结果: 医院的\n",
      "[2020-12-26 09:47:34,666] [on_message()][INFO] 识别结果: 医院的就\n",
      "[2020-12-26 09:47:34,794] [on_message()][INFO] 识别结果: 医院的就医\n",
      "[2020-12-26 09:47:35,154] [on_message()][INFO] 识别结果: 医院的就医环\n",
      "[2020-12-26 09:47:35,290] [on_message()][INFO] 识别结果: 医院的就医环境\n",
      "[2020-12-26 09:47:35,657] [on_message()][INFO] 识别结果: 医院的就医环境还可\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[2020-12-26 09:47:35,964] [on_message()][INFO] 识别结果: 医院的就医环境还可以\n",
      "[2020-12-26 09:47:36,971] [on_message()][INFO] 识别结果: 医院的就医环境还可以吗\n",
      "[2020-12-26 09:47:38,394] [on_message()][INFO] 识别结果: 医院的就医环境还可以吗？\n",
      "[2020-12-26 09:47:39,162] [on_message()][INFO] 识别结果: 一\n",
      "[2020-12-26 09:47:39,642] [on_message()][INFO] 识别结果: 医院\n",
      "[2020-12-26 09:47:39,649] [on_message()][INFO] 识别结果: 医院的\n",
      "[2020-12-26 09:47:39,665] [on_message()][INFO] 识别结果: 医院的就\n",
      "[2020-12-26 09:47:40,033] [on_message()][INFO] 识别结果: 医院的就医\n",
      "[2020-12-26 09:47:40,161] [on_message()][INFO] 识别结果: 医院的就医疗\n",
      "[2020-12-26 09:47:40,353] [on_message()][INFO] 识别结果: 医院的就医流程\n",
      "[2020-12-26 09:47:40,665] [on_message()][INFO] 识别结果: 医院的就医流程是\n",
      "[2020-12-26 09:47:40,857] [on_message()][INFO] 识别结果: 医院的就医流程是否\n",
      "[2020-12-26 09:47:41,169] [on_message()][INFO] 识别结果: 医院的就医流程是否变\n",
      "[2020-12-26 09:47:41,360] [on_message()][INFO] 识别结果: 医院的就医流程是否便捷\n",
      "[2020-12-26 09:47:42,440] [on_message()][INFO] 识别结果: 医院的就医流程是否便捷呢？\n",
      "[2020-12-26 09:47:43,160] [on_message()][INFO] 识别结果: 嗯哪\n",
      "[2020-12-26 09:47:43,543] [on_message()][INFO] 识别结果: 哪方面\n",
      "[2020-12-26 09:47:44,164] [on_message()][INFO] 识别结果: 哪方面的\n",
      "[2020-12-26 09:47:44,847] [on_message()][INFO] 识别结果: 哪方面的坊\n",
      "[2020-12-26 09:47:46,150] [on_message()][INFO] 识别结果: 哪方面的，方便方便。\n",
      "[2020-12-26 09:47:47,224] [on_message()][INFO] 识别结果: 就是\n",
      "[2020-12-26 09:47:47,532] [on_message()][INFO] 识别结果: 就是对\n",
      "[2020-12-26 09:47:47,739] [on_message()][INFO] 识别结果: 就是对医院\n",
      "[2020-12-26 09:47:47,868] [on_message()][INFO] 识别结果: 就是对医院里\n",
      "[2020-12-26 09:47:48,277] [on_message()][INFO] 识别结果: 就是对医院里的就\n",
      "[2020-12-26 09:47:48,399] [on_message()][INFO] 识别结果: 就是对医院里的就医\n",
      "[2020-12-26 09:47:48,739] [on_message()][INFO] 识别结果: 就是对医院里的教育刘\n",
      "[2020-12-26 09:47:49,042] [on_message()][INFO] 识别结果: 就是对医院里的就医流程\n",
      "[2020-12-26 09:47:49,234] [on_message()][INFO] 识别结果: 就是对医院里的就医流程是\n",
      "[2020-12-26 09:47:49,378] [on_message()][INFO] 识别结果: 就是对医院里的就医流程是否\n",
      "[2020-12-26 09:47:49,746] [on_message()][INFO] 识别结果: 就是对医院里的就医流程是否满\n",
      "[2020-12-26 09:47:50,753] [on_message()][INFO] 识别结果: 就是对医院里的就医流程是否满意\n",
      "[2020-12-26 09:47:53,351] [on_message()][INFO] 识别结果: 就是对医院里的就医流程是否满意？\n",
      "[2020-12-26 09:47:55,022] [on_message()][INFO] 识别结果: 把我\n",
      "[2020-12-26 09:47:55,142] [on_message()][INFO] 识别结果: 把我当\n",
      "[2020-12-26 09:47:55,830] [on_message()][INFO] 识别结果: 我没有\n",
      "[2020-12-26 09:47:56,029] [on_message()][INFO] 识别结果: 我没有在我\n",
      "[2020-12-26 09:47:56,310] [on_message()][INFO] 识别结果: 没有在我们\n",
      "[2020-12-26 09:47:56,525] [on_message()][INFO] 识别结果: 我没有在我们市\n",
      "[2020-12-26 09:47:56,637] [on_message()][INFO] 识别结果: 我没有在我们市场\n",
      "[2020-12-26 09:47:56,813] [on_message()][INFO] 识别结果: 我没有在我们市场就\n",
      "[2020-12-26 09:47:57,309] [on_message()][INFO] 识别结果: 我没有在我们食堂就餐\n",
      "[2020-12-26 09:47:58,316] [on_message()][INFO] 识别结果: 我没有在我们食堂就餐的\n",
      "[2020-12-26 09:48:00,539] [on_message()][INFO] 识别结果: 我没有在我们食堂就餐的还可以还可以\n",
      "[2020-12-26 09:48:01,626] [on_message()][INFO] 识别结果: 我没有在我们食堂就餐的，还可以，还可以。\n",
      "[2020-12-26 09:48:02,690] [on_message()][INFO] 识别结果: 嗯敢\n",
      "[2020-12-26 09:48:02,978] [on_message()][INFO] 识别结果: 嗯感觉\n",
      "[2020-12-26 09:48:03,186] [on_message()][INFO] 识别结果: 嗯感觉火\n",
      "[2020-12-26 09:48:03,490] [on_message()][INFO] 识别结果: 嗯感觉伙食\n",
      "[2020-12-26 09:48:03,697] [on_message()][INFO] 识别结果: 嗯感觉伙食怎么\n",
      "[2020-12-26 09:48:04,185] [on_message()][INFO] 识别结果: 嗯感觉伙食怎么样\n",
      "[2020-12-26 09:48:04,993] [on_message()][INFO] 识别结果: 嗯感觉伙食怎么样呢\n",
      "[2020-12-26 09:48:18,551] [on_message()][INFO] 识别结果: 感觉伙食怎么样呢？\n",
      "[2020-12-26 09:48:21,325] [on_message()][INFO] 识别结果: 这次\n",
      "[2020-12-26 09:48:22,341] [on_message()][INFO] 识别结果: 这次一期间有\n",
      "[2020-12-26 09:48:22,525] [on_message()][INFO] 识别结果: 这次一期间有没有\n",
      "[2020-12-26 09:48:23,044] [on_message()][INFO] 识别结果: 这次一期间有没有医院\n",
      "[2020-12-26 09:48:23,548] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作\n",
      "[2020-12-26 09:48:23,828] [on_message()][INFO] 识别结果: 这次期间有没有给医院工作人员\n",
      "[2020-12-26 09:48:24,020] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送\n",
      "[2020-12-26 09:48:24,164] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送货\n",
      "[2020-12-26 09:48:24,529] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红\n",
      "[2020-12-26 09:48:24,665] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红包\n",
      "[2020-12-26 09:48:25,521] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红包呀\n",
      "[2020-12-26 09:48:29,853] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红包呀服务\n",
      "[2020-12-26 09:48:30,364] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红包呀服务器\n",
      "[2020-12-26 09:48:31,068] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红包呀的服务评价是\n",
      "[2020-12-26 09:48:31,372] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红包呀服务总体评价是好\n",
      "[2020-12-26 09:48:31,556] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红包呀服务总体评价是好孩\n",
      "[2020-12-26 09:48:31,691] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红包呀服务总体评价是好还是\n",
      "[2020-12-26 09:48:32,067] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红包呀服务总体评价是好还是不\n",
      "[2020-12-26 09:48:32,195] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红包呀服务总体评价是好还是不好\n",
      "[2020-12-26 09:48:33,065] [on_message()][INFO] 识别结果: 这次一期间有没有医院工作人员送过红包呀服务总体评价是好还是不好呢\n",
      "[2020-12-26 09:48:36,007] [on_message()][INFO] 识别结果: 这次一期间，有没有医院工作人员送过红包呀？服务总体评价是好还是不好呢？\n",
      "[2020-12-26 09:48:37,942] [on_message()][INFO] 识别结果: 你\n",
      "[2020-12-26 09:48:38,749] [on_message()][INFO] 识别结果: 您好\n",
      "[2020-12-26 09:48:38,957] [on_message()][INFO] 识别结果: 您好还\n",
      "[2020-12-26 09:48:39,261] [on_message()][INFO] 识别结果: 您好还在\n",
      "[2020-12-26 09:48:40,468] [on_message()][INFO] 识别结果: 您好还在吗\n",
      "[2020-12-26 09:48:41,843] [on_message()][INFO] 识别结果: 您好，还在吗？\n",
      "[2020-12-26 09:48:45,139] [on_message()][INFO] 识别结果: 你\n",
      "[2020-12-26 09:48:45,770] [on_message()][INFO] 识别结果: 您好\n",
      "[2020-12-26 09:48:46,130] [on_message()][INFO] 识别结果: 您好还\n",
      "[2020-12-26 09:48:46,435] [on_message()][INFO] 识别结果: 您好还在\n",
      "[2020-12-26 09:48:47,266] [on_message()][INFO] 识别结果: 您好还在吗\n",
      "[2020-12-26 09:48:52,571] [send_finish()][INFO] send FINISH frame\n",
      "[2020-12-26 09:48:52,731] [on_message()][INFO] 识别结果: 您好，还在吗？\n",
      "[2020-12-26 09:48:52,731] [on_close()][INFO] ws close ...\n"
     ]
    }
   ],
   "source": [
    "logger.setLevel(logging.INFO)  # 调整为logging.INFO，日志会少一点\n",
    "uri = \"ws://vop.baidu.com/realtime_asr\" + \"?sn=\" + str(uuid.uuid1())\n",
    "ws_app = websocket.WebSocketApp(uri,\n",
    "                                on_open=on_open,  # 连接建立后的回调\n",
    "                                on_message=on_message,  # 接收消息的回调\n",
    "                                on_error=on_error,  # 库遇见错误的回调\n",
    "                                on_close=on_close)  # 关闭后的回调\n",
    "ws_app.run_forever()"
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
