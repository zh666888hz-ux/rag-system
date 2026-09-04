"""
文本分块模块。

核心原理（递归字符切分 RecursiveCharacterTextSplitter）：
1. 目标：把长文本切成「有语义边界」的块，同时让相邻块保留 overlap，
   保证被边界切断的语义在检索时仍能被覆盖；
2. 递归降级切分：维护一组分隔符，优先级从粗到细，例如
   「段落(\\n\\n) → 行(\\n) → 句号(。！？) → 分号 → 逗号 → 空格 → 字符」。
   流程是：先用最粗的分隔符尝试切分；若某个子块仍超过 chunk_size，则换下一级
   更细的分隔符对该子块继续递归切分，直到所有块都满足大小约束。
   这样文本块大概率落在自然语义边界上，而不是被硬生生按字符数截断，检索召回质量更高；
3. overlap：切分时允许相邻块共享 chunk_overlap 个字符，缓解「一句话被切到两块、
   两块各自都不完整」的问题；
4. add_start_index：为每个块记录它在原始文档中的起始偏移，便于溯源回原始文本；
5. 切分后统一注入 chunk_index 元数据（块在全局的序号），方便后续管理与调试。
"""
import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.exceptions import ChunkingError

logger = logging.getLogger(__name__)

# 面向中文的切分分隔符（优先级从高到低）：
# 优先在段落/句子边界切分，中文用全角标点，英文用空格
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", ".", " ", ""]


def build_splitter(
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[list[str]] = None,
) -> RecursiveCharacterTextSplitter:
    """构建分块器，并做参数合法性校验。

    Raises:
        ChunkingError: chunk_overlap >= chunk_size（无法收敛）
    """
    if chunk_overlap >= chunk_size:
        raise ChunkingError(f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or DEFAULT_SEPARATORS,
        length_function=len,  # 以字符数衡量块长度（中文场景字符数更直观）
        add_start_index=True,  # 记录块在原始文本中的起始偏移
    )


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """对一批 Document 进行分块。

    Args:
        documents: 加载后的文档列表
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块重叠字符数

    Returns:
        分块后的 Document 列表；每个块继承原文档的 metadata，并额外带 chunk_index。
    """
    splitter = build_splitter(chunk_size, chunk_overlap)

    try:
        chunks = splitter.split_documents(documents)
    except Exception as exc:  # 极少数异常文本可能导致切分失败，统一归类
        raise ChunkingError(f"文本分块失败: {exc}", cause=exc) from exc

    # 给每个块编上全局序号，便于追踪与调试
    for idx, doc in enumerate(chunks):
        doc.metadata["chunk_index"] = idx

    logger.info(
        "分块完成: %d 篇文档 → %d 个块（chunk_size=%d, overlap=%d）",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
