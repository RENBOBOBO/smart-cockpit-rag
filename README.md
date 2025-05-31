# 智能座舱 RAG 问答系统

## 项目简介

本项目基于 RAG（Retrieval-Augmented Generation）技术，结合大语言模型和多种检索算法，实现了针对汽车用户手册的智能问答系统。系统支持多轮对话，能够高效、准确地回答用户关于汽车功能、使用方法等相关问题，适用于智能座舱、车载助手等场景。

## 主要功能

- 支持多轮对话的智能问答
- 融合向量检索（Faiss）、关键词检索（BM25）和重排序（ReRank）等多种检索方式
- 支持 PDF 用户手册解析，自动构建知识库
- 基于大语言模型（如 Qwen-7B-Chat）进行答案生成
- 提供 Gradio Web 界面，便于交互体验和演示

## 视频演示

- [B站视频DEMO](https://www.bilibili.com/video/BV1SCj3z6EM4/)

## 目录结构

```
.
├── gradio_app.py         # Gradio Web 应用主入口
├── run.py                # 主流程脚本，数据处理与模型调用
├── vllm_model.py         # 大语言模型推理接口
├── rerank_model.py       # 重排序模型接口
├── faiss_retriever.py    # 向量检索实现
├── bm25_retriever.py     # BM25 检索实现
├── pdf_parse.py          # PDF 用户手册解析
├── requirements.txt      # Python 依赖包列表
├── Dockerfile            # Docker 部署文件
├── run.sh                # 启动脚本
├── build.sh              # 构建脚本
├── data/                 # 数据与知识库目录
├── pre_train_model/      # 预训练模型存放目录
├── useful_code/          # 其他工具代码
└── ...
```

## 环境依赖

- Python 3.8+
- 推荐使用 GPU 环境（CUDA 11+）
- 主要依赖见 `requirements.txt`，包括：
  - gradio
  - langchain
  - transformers
  - faiss
  - PyPDF2
  - pdfplumber
  - numpy
  - pandas
  - tqdm
  - 以及相关大模型和向量模型

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速启动

1. **准备模型和数据**
   - 将所需的预训练大模型（如 Qwen-7B-Chat）、向量模型（如 m3e-large）、重排序模型（如 bge-reranker-large）放入 `pre_train_model/` 目录。
   - 将待解析的 PDF 用户手册放入 `data/` 目录。

2. **启动 Gradio Web 应用**

```bash
python gradio_app.py
```

- 默认在本地 `http://127.0.0.1:7860` 启动 Web 服务。
- 若需局域网访问，可修改 `gradio_app.py` 中的 `demo.launch(server_name="0.0.0.0")`。

3. **命令行运行主流程**

```bash
python run.py
```

## 主要文件说明

- `gradio_app.py`：Web 界面主入口，集成问答主流程和多轮对话逻辑。
- `run.py`：命令行主流程，包含知识库构建、模型加载与问答流程。
- `pdf_parse.py`：PDF 用户手册解析与分块。
- `faiss_retriever.py`、`bm25_retriever.py`：两种检索方式的实现。
- `vllm_model.py`、`rerank_model.py`：大语言模型与重排序模型的推理接口。
- `requirements.txt`：依赖包列表。
- `Dockerfile`：容器化部署支持。

## 致谢

- 本项目由上海对外经贸大学团队开发，创作者：任波、芦子鑫、卢镐楠。
- 感谢开源社区和相关模型的贡献。 