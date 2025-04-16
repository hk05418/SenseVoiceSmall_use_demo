SenseVoice专注于高精度多语言语音识别、情感辨识和音频事件检测

多语言识别： 采用超过40万小时数据训练，支持超过50种语言，识别效果上优于Whisper模型。
富文本识别：
具备优秀的情感识别，能够在测试数据上达到和超过目前最佳情感识别模型的效果。
支持声音事件检测能力，支持音乐、掌声、笑声、哭声、咳嗽、喷嚏等多种常见人机交互事件进行检测。
高效推理： SenseVoice-Small模型采用非自回归端到端框架，推理延迟极低，10s音频推理仅耗时70ms，15倍优于Whisper-Large。
微调定制： 具备便捷的微调脚本与策略，方便用户根据业务场景修复长尾样本问题。
服务部署： 具有完整的服务部署链路，支持多并发请求，支持客户端语言有，python、c++、html、java与c#等。



模型下载地址, 阿里魔塔下载地址：
https://modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/model.pt

模型介绍和使用
https://modelscope.cn/models/iic/SenseVoiceSmall


```sh
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
```