"""
查询流水线（混合检索 + 重排版）。

流程：问题 → 混合召回（向量+BM25，RRF 融合）→ Rerank 精排 → 高质量片段入 LLM → 答案。

为什么加入 Rerank 能降低幻觉：
1. 混合召回先尽量「别漏」（召回全），Rerank 再「求精」（只留准）；
2. Rerank 用 Cross-Encoder 对「问题-片段」联合编码打分，能识别语义上相似但实际
   不相关/答非所问的片段，把它们过滤掉；
3. 只有高置信片段进入 LLM 上下文，模型无从「引用」错误片段，从源头压制幻觉；
   同时保留阈值可调，宁可少答也不硬答。

设计：
1. Reranker 懒加载：首次问答才加载模型，避免每次启动加载几百 MB；
2. 结构化结果：答案 + 每条片段的来源/页码/分数/摘要，可审计、可溯源；
3. 全程日志：检索、重排、最终上下文逐条记录。
"""
import logging
from typing import Optional

from config.settings import Settings
from service.generator import Generator
from service.reranker import Reranker
from service.retriever import Retriever
from vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# 返回给调用方的片段摘要长度
_SNIPPET_MAX_LEN = 200


class QueryPipeline:
    """问答流水线。"""

    def __init__(self, settings: Settings, store: QdrantStore) -> None:
        self.settings = settings
        self.retriever = Retriever(store, settings)
        self.generator = Generator(settings)
        self._reranker: Optional[Reranker] = None  # 懒加载

    # ---------------------------------------------------------------
    # 主入口
    # ---------------------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        history: Optional[list[dict]] = None,
    ) -> dict:
        """执行一次问答。

        Args:
            question: 用户问题
            top_k: 最终送入生成的片段数（覆盖配置）
            score_threshold: 向量检索相似度阈值（覆盖配置）
            history: 多轮对话历史 [{"role": "user"|"assistant", "content": "..."}]
                     （可选；提供时作为对话记忆一并送入模型）

        Returns:
            {"question", "answer", "sources", "num_sources", "reranked", "refs"}
        """
        final_k = top_k if top_k is not None else self.settings.top_k

        # 1) 混合/向量检索 + 重排
        results, reranked = self._retrieve_and_rerank(question, final_k, score_threshold)

        # 2) 生成（contexts 为空时生成器自带兜底）；history 为多轮记忆
        contexts = [doc for doc, _score in results]
        answer = self.generator.generate(question, contexts, history=history)

        # 4) 引用校验：从答案抽取合法引用编号（越界编号已剔除并告警）
        refs = self.generator.extract_references(answer, max_ref=len(contexts))

        # 5) 组装溯源信息：ref_id 即片段编号（1..N），与答案中 [编号] 一一对应
        sources = [
            {
                "ref_id": idx,  # 引用编号，供前端对齐「答案中的 [n]」
                "source": doc.metadata.get("source", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "page": doc.metadata.get("page", ""),
                "score": round(score, 4),
                "snippet": doc.page_content[:_SNIPPET_MAX_LEN],
            }
            for idx, (doc, score) in enumerate(results, start=1)
        ]

        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
            "reranked": reranked,
            "refs": refs,
        }
        logger.info(
            "问答完成: 命中片段=%d, 已重排=%s, 引用编号=%s, 答案长度=%d 字",
            len(sources),
            reranked,
            refs,
            len(answer),
        )
        return result

    # ---------------------------------------------------------------
    # 流式问答（SSE）
    # ---------------------------------------------------------------
    def ask_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        history: Optional[list[dict]] = None,
    ):
        """流式问答生成器：逐条 yield 事件 dict（供 SSE 协议序列化）。

        事件协议（type 字段区分）：
          meta   检索完成后的元信息（片段数/是否重排）
          source 每个候选片段的溯源信息（生成前即可推送，前端先展示候选）
          delta  答案文本增量（打字机效果）
          refs   答案中实际引用编号（在生成完成后推送）
          done   完整结果（含完整答案与溯源）
          error  中途出错（会话校验错误由路由在开流前处理，这里只兜模型错误）

        设计：检索是阻塞且快的，先生成；生成是慢的，流式推送。
        """
        final_k = top_k if top_k is not None else self.settings.top_k

        # 1) 检索 + 重排（阻塞、快）
        results, reranked = self._retrieve_and_rerank(question, final_k, score_threshold)
        contexts = [doc for doc, _score in results]

        # 2) 先推元信息与候选片段溯源
        yield {"type": "meta", "num_sources": len(contexts), "reranked": reranked}
        for idx, (doc, score) in enumerate(results, start=1):
            yield {
                "type": "source",
                "ref_id": idx,
                "file_name": doc.metadata.get("file_name", doc.metadata.get("source", "")),
                "page": doc.metadata.get("page"),
                "score": round(score, 4),
                "snippet": doc.page_content[:_SNIPPET_MAX_LEN],
            }

        # 3) 无上下文兜底
        if not contexts:
            answer = "根据现有资料无法回答（未检索到相关片段）。"
            yield {"type": "done", "answer": answer, "refs": [], "sources": [], "num_sources": 0}
            return

        # 4) 流式生成 + 引用校验
        parts: list[str] = []
        try:
            for delta in self.generator.stream_generate(question, contexts, history=history):
                parts.append(delta)
                yield {"type": "delta", "text": delta}
        except Exception as exc:
            logger.error("流式生成失败: %s", exc)
            yield {"type": "error", "message": f"大模型调用失败: {exc}"}
            return

        answer = "".join(parts)
        refs = self.generator.extract_references(answer, max_ref=len(contexts))
        sources = [
            {
                "ref_id": idx,
                "source": doc.metadata.get("source", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "page": doc.metadata.get("page", ""),
                "score": round(score, 4),
                "snippet": doc.page_content[:_SNIPPET_MAX_LEN],
            }
            for idx, (doc, score) in enumerate(results, start=1)
        ]
        yield {"type": "refs", "refs": refs}
        yield {
            "type": "done",
            "answer": answer,
            "refs": refs,
            "sources": sources,
            "num_sources": len(sources),
            "reranked": reranked,
        }
        logger.info("流式问答完成: 片段=%d, 引用=%s, 答案=%d 字", len(sources), refs, len(answer))

    # ---------------------------------------------------------------
    # 公共检索 + 重排
    # ---------------------------------------------------------------
    def _retrieve_and_rerank(
        self,
        question: str,
        final_k: int,
        score_threshold: Optional[float],
    ) -> tuple[list, bool]:
        """检索 + 重排，返回 (results, reranked)。同步与流式接口共用。"""
        results = self.retriever.retrieve(question, k=final_k, score_threshold=score_threshold)
        reranked = self.settings.enable_rerank and self.settings.rerank_provider == "local_onnx"
        if reranked and results:
            results = self._apply_rerank(question, results, final_k)
        return results, reranked

    # ---------------------------------------------------------------
    # Rerank
    # ---------------------------------------------------------------
    def _apply_rerank(
        self,
        question: str,
        results: list,
        final_k: int,
    ) -> list:
        """对检索结果做重排打分与筛选，并逐条记录日志。"""
        try:
            reranker = self._get_reranker()
            candidates = [doc for doc, _score in results]
            reranked = reranker.rerank(
                question,
                candidates,
                top_k=final_k,
                threshold=self.settings.rerank_threshold,
            )
        except Exception as exc:
            # 重排是增强项，失败不应阻断问答：降级为「不重排」，仅用检索结果
            logger.warning("Rerank 失败，降级为不重排: %s", exc)
            return results[:final_k]

        # 逐条记录重排结果（可审计）
        for rank, (doc, score) in enumerate(reranked, start=1):
            logger.info(
                "Rerank[%d] 来源=%s 页码=%s 重排分=%.4f 片段=%s...",
                rank,
                doc.metadata.get("file_name", "未知"),
                doc.metadata.get("page", "?"),
                score,
                doc.page_content[:_SNIPPET_MAX_LEN].replace("\n", " "),
            )
        if not reranked:
            logger.warning("重排后无片段通过阈值（threshold=%s），生成器将给出无法回答兜底",
                           self.settings.rerank_threshold)
        return reranked

    def _get_reranker(self) -> Reranker:
        """懒加载重排器：首次使用才加载本地模型。"""
        if self._reranker is None:
            self._reranker = Reranker(self.settings)
        return self._reranker
