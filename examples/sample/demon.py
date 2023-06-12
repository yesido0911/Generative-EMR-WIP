from flask import Flask, jsonify, request, Response
from datetime import datetime
import json
from typing import List, Union, Tuple
from pydub import AudioSegment
from asr import flash_recognizer
from common import credential
import openai

# 设置OpenAI API密钥和端点
openai.api_key = "sk-FgiJvZp6HMOQ0kL356vGXluCYyjh6gC7agYVkbmqhL2ngMP"
openai.api_base = "https://api.chatanywhere.com.cn/v1"

# 设置腾讯云凭证和ASR参数
APPID = "1318633830"
SECRET_ID = "AKIDxnQp13rUPMhuBICsQozsjbN4asvrdEbl"
SECRET_KEY = "53pOjtqd8OQHwMYbbQ1Shjz7BOQI1c5N"
ENGINE_TYPE = "16k_zh"

# 设置输入和输出文件的路径
audio_path = "武欣.mp3"
asr_result_path = "asr_result.mp3"
txt_file_path = "communication_record.txt"
medical_record_path = "medical_record.txt"

app = Flask(__name__)

# 修改音频采样率
def change_audio_sample_rate(from_path: str, to_path: str, frame_rate=16000, channels=1, start_min=0, end_min=None):
    # 加载音频文件
    audio = AudioSegment.from_mp3(from_path)

    # 设置开始和结束时间（毫秒为单位）
    start_time = start_min * 60 * 1000
    end_time = end_min * 60 * 1000 + 1 if end_min else None

    # 提取音频片段并设置采样率和声道数
    mono = audio[start_time:end_time].set_frame_rate(frame_rate).set_channels(channels)

    # 保存新的音频文件
    mono.export(to_path, format='wav', codec='pcm_s16le')

# 对音频文件进行ASR
def perform_asr(audio_path: str, asr_result_path: str):
    # 检查腾讯云凭证
    if APPID == "":
        print("请设置APPID！")
        exit(0)
    if SECRET_ID == "":
        print("请设置SECRET_ID！")
        exit(0)
    if SECRET_KEY == "":
        print("请设置SECRET_KEY！")
        exit(0)

    # 创建FlashRecognizer
    credential_var = credential.Credential(SECRET_ID, SECRET_KEY)
    recognizer = flash_recognizer.FlashRecognizer(APPID, credential_var)

    # 创建识别请求
    req = flash_recognizer.FlashRecognitionRequest(ENGINE_TYPE)
    req.set_filter_modal(0)
    req.set_filter_punc(0)
    req.set_filter_dirty(0)
    req.set_voice_format("wav")
    req.set_word_info(0)
    req.set_convert_num_mode(1)

    # 对音频文件执行识别
    with open(asr_result_path, 'rb') as f:
        data = f.read()
        result_data = recognizer.recognize(req, data)
        resp = json.loads(result_data)
        request_id = resp["request_id"]
        code = resp["code"]
        if code != 0:
            print("识别失败！请求ID：", request_id, "，错误码：", code, "，错误信息：", resp["message"])
            exit(0)
        for channl_result in resp["flash_result"]:
            print(f"Asr结果为：\n{channl_result['text']}\n")

# 将ASR结果保存到文本文件中
def save_asr_results_to_text_file(asr_result_path: str, txt_file_path: str):
    with open(txt_file_path, "w") as f:
        f.write(f"医患的对话记录如下：\n")
        with open(asr_result_path, 'rb') as f2:
            data = f2.read()
            result_data = json.loads(data.decode())
            for channl_result in result_data["flash_result"]:
                f.write(channl_result["text"] + "\n")

# 使用GPT-3生成病历
def generate_medical_record(txt_file_path: str, medical_record_path: str) -> str:
    with open(txt_file_path, "r") as f:
        prompt = f.read() + "\n上述文本是医患的对话记录，请根据上文总结出这位患者的病历，包含但不限于姓名、主诉、检查情况、现病史、既往史、旅居史、家族史等。"
    # 使用OpenAI生成病历
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5
    )
    # 从OpenAI响应中提取生成的病历
    medical_record = response.choices[0].text.strip()

    # 将生成的病历保存到文件中
    with open(medical_record_path, "w") as f:
        f.write(medical_record)

    return medical_record

# 定义API接口
@app.route('/generate_medical_record', methods=['POST'])
def generate_medical_record_api() -> Union[tuple[Response, int], Response]:
    # 检查请求中是否包含音频文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    # 保存音频文件
    file = request.files['file']
    file.save(audio_path)

    # 修改音频采样率
    change_audio_sample_rate(audio_path, asr_result_path)

    # 对音频文件进行ASR
    perform_asr(audio_path, asr_result_path)

    # 将ASR结果保存到文本文件中
    save_asr_results_to_text_file(asr_result_path, txt_file_path)

    # 使用GPT-3生成病历
    medical_record = generate_medical_record(txt_file_path, medical_record_path)

    # 返回生成的病历
    return jsonify({'medical_record': medical_record})

if __name__ == '__main__':
    app.run(port=5000, debug=True)