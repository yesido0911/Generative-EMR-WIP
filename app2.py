from flask import Flask, request
import sys
import json
import tempfile
sys.path.append("..")
from common import credential
from asr import flash_recognizer
import openai
from pydub import AudioSegment
app2 = Flask(__name__)

# openai.log = "debug"
#-----------------------------------------------------------
#前置参数设置
APPID = "1318633830"
SECRET_ID = "AKIDxnQp13rUPMhuBICsQozsjbN4asvrdEbl"
SECRET_KEY = "53pOjtqd8OQHwMYbbQ1Shjz7BOQI1c5N"
ENGINE_TYPE = "16k_zh"
# 16k_zh：中文通用；
# 16k_zh-PY：中英粤；
# 16k_zh-TW：中文繁体；
# 16k_zh_edu：中文教育；
# 16k_zh_medical：中文医疗；
# 16k_zh_court：中文法庭；
openai.api_key = "sk-FgiJvZp6HMOQ0kL356vGXluCYyjh6gC7agYVkbmqhL2ngMPa"
openai.api_base = "https://api.chatanywhere.com.cn/v1"


#改变音频采样率#

def wavSample(from_path, to_path, frame_rate=16000, channels=1, startMin=0, endMin=None):
	# 根据文件的类型选择导入方法
    # audio = AudioSegment.from_wav(from_path)
    audio = AudioSegment.from_mp3(from_path)
    # mp3_version = AudioSegment.from_mp3("never_gonna_give_you_up.mp3")
    # ogg_version = AudioSegment.from_ogg("never_gonna_give_you_up.ogg")
    # flv_version = AudioSegment.from_flv("never_gonna_give_you_up.flv")
    startTime = startMin * 60 * 1000  # 单位ms
    endTime = endMin * 60 * 1000 + 1 if endMin else None  # 单位ms
    audio = audio[startTime:endTime]
    mono = audio.set_frame_rate(frame_rate).set_channels(channels)  # 设置声道和采样率
    mono.export(to_path, format='wav', codec='pcm_s16le')  # codec此参数本意是设定16bits pcm编码器


def gpt_35_api_stream(messages: list, medical_record: str):
    """为提供的对话消息创建新的回答 (流式传输)
    Args:
        messages (list): 完整的对话消息
        api_key (str): OpenAI API 密钥
    Returns:
        tuple: (results, error_desc)
    """
    global gpt_message
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            stream=True,
        )
        completion = {'role': '', 'content': ''}
        for event in response:
            if event['choices'][0]['finish_reason'] == 'stop':
                # print(f'gpt的回复: {completion}')
                gpt_message = completion.get("content")
                print(f'gpt的回复:{gpt_message}')
            for delta_k, delta_v in event['choices'][0]['delta'].items():
                # print(f'流响应数据: {delta_k} = {delta_v}')
                completion[delta_k] += delta_v
        messages.append(completion)
        # 直接在传入参数 messages 中追加消息
        with open(medical_record, "w",encoding='utf-8') as f:
            f.write(str(gpt_message))
        return (True, gpt_message)
    except Exception as err:
        return (False, f'OpenAI API 异常: {err}')


@app2.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload MP3 File</title>
    </head>
    <body>
        <h1>欢迎来到病历生成系统，请在下方上传您的MP3文件</h1>
        <form action="/emr/" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    '''
    return html

@app2.route('/emr/', methods=['POST'])
def emr():
    audio = request.files['file']
    audio_name = audio.filename.split(".")[0]
    txt_file = audio_name + ".txt"
    communication_record = audio_name + ".txt"
    asr_result = audio_name + "_new" + ".mp3"
    medical_record = audio_name + "病历" + ".txt"
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_path = temp_audio.name
        audio.save(temp_path)

    # 转换采样率
    wavSample(temp_path, asr_result)

    if APPID == "":
        print("Please set APPID!")
        exit(0)
    if SECRET_ID == "":
        print("Please set SECRET_ID!")
        exit(0)
    if SECRET_KEY == "":
        print("Please set SECRET_KEY!")
        exit(0)

    credential_var = credential.Credential(SECRET_ID, SECRET_KEY)
    # 新建FlashRecognizer，一个recognizer可以执行N次识别请求
    recognizer = flash_recognizer.FlashRecognizer(APPID, credential_var)
    # 新建识别请求
    req = flash_recognizer.FlashRecognitionRequest(ENGINE_TYPE)
    req.set_filter_modal(0)
    req.set_filter_punc(0)
    req.set_filter_dirty(0)
    req.set_voice_format("wav")
    req.set_word_info(0)
    req.set_convert_num_mode(1)

    with open(asr_result, 'rb') as f:
        # 读取音频数据
        data = f.read()
        # 执行识别
        resultData = recognizer.recognize(req, data)
        resp = json.loads(resultData)
        request_id = resp["request_id"]
        code = resp["code"]
        if code != 0:
            print("recognize faild! request_id: ", request_id, " code: ", code, ", message: ", resp["message"])
            exit(0)
        # print("request_id: ", request_id)
        # 一个channl_result对应一个声道的识别结果
        # 大多数音频是单声道，对应一个channl_result
        for channl_result in resp["flash_result"]:
            # print("channel_id: ", channl_result["channel_id"])
            print(f"Asr结果为:\n{channl_result['text']}\n")
    # ---------------------------------------------------------------


    # 打开txt文件，设置写入模式
    with open(txt_file, "w",encoding='utf-8') as f:
        f.write(f"{audio_name}和医生的对话如下：\n")
        for channl_result in resp["flash_result"]:
            f.write(channl_result["text"] + "\n")

    with open(txt_file, "r",encoding='utf-8') as f:
        prompt = f.read() + "\n上述文本是医患的对话记录,请根据上文总结出这位患者的病历，包含但不限于姓名、主诉、检查情况、现病史、既往史、旅居史、呼吸道病史、辅助诊察结果、初步诊断、治疗意见。患者的名字在第一排，不要自己生成以外的信息，注意格式。"
    print("给gpt的prompt为：")
    print(prompt)
    print()
    messages = [{'role': 'user', 'content': prompt}, ]
    emr_result = gpt_35_api_stream(messages, medical_record)
    if emr_result[0]:
        return emr_result[1]
    else:
        return "core error"



if __name__ == '__main__':
    app2.run(debug=True)

