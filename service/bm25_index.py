"""
BM25 关键词检索索引模块。

核心原理：
1. BM25（Best Matching 25）是经典的概率检索模型，基于「词频 TF 与逆文档频率 IDF」打分：
   - TF 越高、文档包含查询词越多，得分越高；
   - IDF 越高（词越稀缺、越能区分文档），得分越高；常见词被自然压低权重；
   - 加入文档长度归一化，防止长文档因「词多」而无故占优。
2. 与向量检索互补：向量检索理解语义（同义改写也能命中），但会漏掉精确术语；
   BM25 做精确关键词匹配，擅长召回专有名词/编号/缩写，二者混合可显著提升召回覆盖。
3. 为什么用 jieba 分词：BM25 的 term 是「词」，中文没有天然空格分词，
   用 jieba 切词后 BM25 才能对中文关键词有效打分。
4. 持久化策略：BM25 索引是「内存索引」，从 Qdrant（权威存储）全量导出后重建，
   保证重启进程后 query 仍可混合检索。
"""
import logging
from typing import Optional

from jieba import lcut
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.exceptions import RetrieverError

logger = logging.getLogger(__name__)

# 极简中文停用词：这些词没有检索区分度，提前过滤降低噪声
_STOPWORDS = frozenset(
    "的了是在和与就不也都还要以及对于关于可以我们你们他们它们这个那个这些那些"
    "一个一种通过进行使用需要能够什么怎么如何因为所以但是然后如果那么而且或者其"
)


class BM25Index:
    """可重建的 BM25 内存索引。"""

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._corpus: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    # ---------------------------------------------------------------
    # 构建
    # ---------------------------------------------------------------
    def add_documents(self, documents: list[Document]) -> None:
        """追加文档并重建索引（BM25 的 IDF 依赖全库统计，追加后必须重建）。"""
        if not documents:
            return
        self._documents.extend(documents)
        self.rebuild()
        logger.info("BM25 索引已更新: 共 %d 篇", len(self._documents))

    def rebuild(self) -> None:
        """按当前全量文档重建 BM25 模型。"""
        self._corpus = [self._tokenize(d.page_content) for d in self._documents]
        self._bm25 = BM25Okapi(self._corpus)
        logger.info("BM25 索引重建完成: 共 %d 篇文档", len(self._documents))

    def load_from_qdrant(self, store) -> None:
        """从 Qdrant 全量导出文档并重建索引（进程重启后恢复混合检索能力）。"""
        documents = store.get_all_documents()
        self._documents = []
        self.add_documents(documents)
        logger.info("已从 Qdrant 重建 BM25 索引: %d 篇", len(documents))

    # ---------------------------------------------------------------
    # 检索
    # ---------------------------------------------------------------
    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """BM25 检索，返回按分数降序的 (Document, score)。

        Raises:
            RetrieverError: 索引为空（尚未 ingest）
        """
        if not self._bm25 or not self._documents:
            raise RetrieverError("BM25 索引为空，请先执行 ingest 导入文档")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        # 取 Top-K 索引并按分数降序
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = [(self._documents[i], float(scores[i])) for i in top_indices]
        return results

    def size(self) -> int:
        return len(self._documents)

    # ---------------------------------------------------------------
    # 工具
    # ---------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """jieba 分词 + 停用词过滤（保留纯文本 token）。"""
        tokens = []
        for word in lcut(text):
            word = word.strip()
            if word and word not in _STOPWORDS:
                tokens.append(word)
        return tokens
