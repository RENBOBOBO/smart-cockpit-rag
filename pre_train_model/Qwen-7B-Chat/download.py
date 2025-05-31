#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen-7B-Chat', local_dir="Qwen-7B-Chat")
