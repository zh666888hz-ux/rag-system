# -*- coding: utf-8 -*-
"""生成一份中文测试知识文档 PDF，用于 Docker 部署后的 RAG 灌库与问答验证。
内容覆盖 RAG 系统的核心概念，便于验证检索召回与引用溯源。
"""
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm

pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))

CONTENT = [
    ("RAG 检索增强生成系统知识手册", 16),
    ("一、什么是 RAG", 12),
    ("RAG（Retrieval-Augmented Generation，检索增强生成）是一种将信息检索与大语言模型生成相结合的技术架构。"
     "它通过在生成回答之前，先从知识库中检索与用户问题最相关的文本片段，再将这些片段作为上下文提供给大语言模型，"
     "从而显著降低大模型产生幻觉的概率，提高回答的准确性与可追溯性。", 10.5),
    ("二、RAG 的核心流程", 12),
    ("RAG 系统的典型处理流程分为两个阶段：离线索引阶段与在线问答阶段。离线阶段负责将知识文档解析、分块、"
     "向量化后存入向量数据库；在线阶段负责接收用户问题，将其向量化后在向量库中检索相关片段，"
     "并把片段拼接为提示词交给大语言模型生成最终答案。", 10.5),
    ("三、向量检索与混合检索", 12),
    ("传统的向量检索通过计算语义向量之间的余弦相似度来召回最相关的文档片段，能够理解语义层面的相似性。"
     "而混合检索（Hybrid Search）则将向量检索与 BM25 关键词检索相结合：BM25 擅长精确匹配专业术语与专有名词，"
     "向量检索擅长语义理解与同义改写。两者结果通过加权融合后，再用重排序模型（Rerank）对召回结果打分排序，"
     "可以显著提升最终召回质量，降低大模型幻觉。", 10.5),
    ("四、分块策略", 12),
    ("文本分块（Chunking）是 RAG 的关键环节。常见的策略是递归字符分块：设定块大小（chunk_size）与重叠量（overlap），"
     "按段落边界递归切分文本，使每个分块既保持语义完整又不超过向量模型的输入长度限制。"
     "合理的重叠量可以避免语义信息在边界处被截断丢失。", 10.5),
    ("五、会话记忆与多轮对话", 12),
    ("为了支持多轮对话，系统会将每次问答的用户问题、检索到的引用片段编号、最终回答以及时间戳持久化存储到数据库。"
     "当用户发起新一轮提问时，系统自动携带历史会话上下文，使大模型能够理解对话语境，给出连贯的回答。"
     "同时，回答中会附带文档来源片段编号，实现引用溯源，方便用户核对信息来源。", 10.5),
    ("六、模型与向量化", 12),
    ("系统采用本地化的文本嵌入模型（例如 BAAI/bge-small-zh-v1.5）对文本进行向量化，"
     "该模型面向中文优化，向量维度为 512 维，支持离线免费运行。大语言模型部分则通过 OpenAI 兼容接口接入"
     "云端大模型服务（例如 DeepSeek），实现最终的答案生成。", 10.5),
    ("七、部署架构", 12),
    ("整套系统通过 Docker Compose 一键编排部署，包含四个核心服务：MySQL 负责存储用户与会话历史，"
     "Qdrant 负责存储与检索文本向量，FastAPI 提供 RAG 问答与鉴权接口，Streamlit 提供聊天交互前端。"
     "用户通过 JWT 令牌完成登录鉴权，前后端通过 SSE 流式协议传输生成过程。", 10.5),
]

doc = SimpleDocTemplate(
    "docs/测试知识文档.pdf",
    pagesize=A4,
    rightMargin=2 * cm, leftMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
)

styles = {
    'title': ParagraphStyle('title', fontName='STSong-Light', fontSize=16, leading=22, spaceAfter=12),
    'h': ParagraphStyle('h', fontName='STSong-Light', fontSize=12, leading=18, spaceBefore=10, spaceAfter=6),
    'body': ParagraphStyle('body', fontName='STSong-Light', fontSize=10.5, leading=17, wordWrap='CJK'),
}

story = []
for text, size in CONTENT:
    if size == 16:
        story.append(Paragraph(text, styles['title']))
    elif size == 12:
        story.append(Paragraph(text, styles['h']))
    else:
        story.append(Paragraph(text, styles['body']))
        story.append(Spacer(1, 6))

doc.build(story)
print('已生成 docs/测试知识文档.pdf')
