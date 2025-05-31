import gradio as gr
from gradio.themes.base import Base
from vllm_model import ChatLLM
from rerank_model import reRankLLM
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from pdf_parse import DataProcess

# ========== 初始化模型和检索器（只做一次） ==========
# 设置基础路径
base = "."
qwen7 = base + "/pre_train_model/Qwen-7B-Chat"  # 大模型路径
m3e =  base + "/pre_train_model/m3e-large"           # 向量模型路径
bge_reranker_large = base + "/pre_train_model/bge-reranker-large"  # rerank模型路径

# 解析PDF文档，构造知识库数据
# 这里会多次分块和全页解析，适配不同检索需求
# DataProcess为自定义的PDF解析类
# 解析后数据存储在dp.data中
# 你可以根据实际情况调整解析方式和参数

dp = DataProcess(pdf_path = base + "/data/train_a.pdf")
dp.ParseBlock(max_seq = 1024)
dp.ParseBlock(max_seq = 512)
dp.ParseAllPage(max_seq = 256)
dp.ParseAllPage(max_seq = 512)
dp.ParseOnePageWithRule(max_seq = 256)
dp.ParseOnePageWithRule(max_seq = 512)
data = dp.data

# 初始化检索器和大模型
faissretriever = FaissRetriever(m3e, data)  # 向量检索
bm25 = BM25(data)                           # 关键词检索
llm = ChatLLM(qwen7)                        # 大语言模型
rerank = reRankLLM(bge_reranker_large)      # rerank模型

# ========== 问答主流程相关函数 ==========
# 合并faiss和bm25召回的内容，构造prompt
# 返回一个用于大模型推理的输入字符串

def get_emb_bm25_merge(faiss_context, bm25_context, query):
    max_length = 2500
    emb_ans = ""
    cnt = 0
    for doc, score in faiss_context:
        cnt += 1
        if cnt > 6:
            break
        if len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans = emb_ans + doc.page_content
    bm25_ans = ""
    cnt = 0
    for doc in bm25_context:
        cnt += 1
        if len(bm25_ans + doc.page_content) > max_length:
            break
        bm25_ans = bm25_ans + doc.page_content
        if cnt > 6:
            break
    prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。\n如果无法从中得到答案，请说 \"无答案\"或\"无答案\"，不允许在答案中添加编造成分，答案请使用中文。\n已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:\n1: {emb_ans}\n2: {bm25_ans}\n问题:\n{query}"""
    return prompt_template

# 构造只用一种召回内容的prompt

def get_rerank(emb_ans, query):
    prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。\n如果无法从中得到答案，请说 \"无答案\"或\"无答案\" ，不允许在答案中添加编造成分，答案请使用中文。\n已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:\n1: {emb_ans}\n问题:\n{query}"""
    return prompt_template

# rerank召回，融合faiss和bm25的候选，并用rerank模型排序，取top_k

def reRank_func(rerank, top_k, query, bm25_ans, faiss_ans):
    items = []
    max_length = 4000
    for doc, score in faiss_ans:
        items.append(doc)
    items.extend(bm25_ans)
    rerank_ans = rerank.predict(query, items)
    rerank_ans = rerank_ans[:top_k]
    emb_ans = ""
    for doc in rerank_ans:
        if len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans

# ========== 多轮对话主流程 ==========
# Gradio会自动传递历史对话history，格式为[(用户,助手), ...]
# 每次问答会把历史对话拼接进prompt，保证上下文连续

def answer_question(query, history):
    # 拼接历史对话作为上下文
    context = ""
    if history is not None:
        for user, bot in history:
            context += f"用户：{user}\n助手：{bot}\n"
    context += f"用户：{query}\n助手："
    # faiss召回topk
    faiss_context = faissretriever.GetTopK(query, 15)
    emb_ans = ""
    max_length = 4000
    cnt = 0
    for doc, score in faiss_context:
        cnt += 1
        if len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans = emb_ans + doc.page_content
        if cnt > 6:
            break
    # bm25召回topk
    bm25_context = bm25.GetBM25TopK(query, 15)
    bm25_ans = ""
    cnt = 0
    for doc in bm25_context:
        cnt += 1
        if len(bm25_ans + doc.page_content) > max_length:
            break
        bm25_ans = bm25_ans + doc.page_content
        if cnt > 6:
            break
    # 构造不同召回方式的prompt
    emb_bm25_merge_inputs = get_emb_bm25_merge(faiss_context, bm25_context, query)
    bm25_inputs = get_rerank(bm25_ans, query)
    emb_inputs = get_rerank(emb_ans, query)
    rerank_ans = reRank_func(rerank, 6, query, bm25_context, faiss_context)
    rerank_inputs = get_rerank(rerank_ans, query)
    # 批量推理，返回多种答案，这里只取合并召回的结果
    batch_input = [
        emb_bm25_merge_inputs,
        bm25_inputs,
        emb_inputs,
        rerank_inputs
    ]
    batch_output = llm.infer(batch_input)
    answer = batch_output[0]
    if history is None:
        history = []
    history.append((query, answer))
    return history, history

# ========== Gradio多轮对话界面美化 ==========
# 使用自定义主题，主色调为深蓝色，圆角大，中文字体
custom_theme = Base(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
    font=["Source Han Sans SC", "sans-serif"]
)

# 构建Gradio Blocks界面
with gr.Blocks(theme=custom_theme, css="""
#custom-footer {text-align:center; color:#888; font-size:1em; margin-top:24px;}
#submit-btn {background: #1a237e !important; color: #fff !important; border-radius: 1.5em !important; font-weight: bold;}
#clear-btn {border-radius: 1.5em !important;}
#user-input textarea {border-radius: 1.5em !important; border: 2px solid #1a237e !important;}
.svelte-1ipelgc .message.user {background: #e3f2fd !important; color: #0d47a1 !important; border-radius: 1.5em 1.5em 0.2em 1.5em !important;}
.svelte-1ipelgc .message.bot {background: #ede7f6 !important; color: #4a148c !important; border-radius: 1.5em 1.5em 1.5em 0.2em !important;}
""") as demo:
    # 顶部标题和说明
    gr.Markdown(
        '''
        <div style="display:flex;align-items:center;justify-content:center;">
            <span style="font-size:2.2em;font-weight:bold;">智能座舱 RAG 问答系统</span>
        </div>
        <div style="text-align:center;color:gray;">支持多轮对话，输入你的问题，点击提交即可获得答案。</div>
        '''
    )
    # 聊天窗口，支持多轮对话
    chatbot = gr.Chatbot(label="对话", height=400)
    with gr.Row():
        with gr.Column(scale=8):
            # 用户输入框
            user_input = gr.Textbox(show_label=False, placeholder="请输入你的问题", lines=2, elem_id="user-input")
        with gr.Column(scale=2):
            # 提交按钮
            submit_btn = gr.Button("提交", elem_id="submit-btn", variant="primary")
    # 清除对话按钮
    clear_btn = gr.Button("清除对话", elem_id="clear-btn")
    # 用于存储历史对话的状态
    state = gr.State([])

    # 绑定按钮和输入框的事件
    submit_btn.click(answer_question, inputs=[user_input, state], outputs=[chatbot, state])
    user_input.submit(answer_question, inputs=[user_input, state], outputs=[chatbot, state])
    clear_btn.click(lambda: ([], []), None, [chatbot, state], queue=False)

    # 底部署名信息
    gr.Markdown('<div id="custom-footer">上海对外经贸大学——RAG系统demo示例    创作者 任波 芦子鑫 卢镐楠</div>')

# 启动Gradio服务
# 若需手机访问，请将server_name设为"0.0.0.0"
demo.launch()