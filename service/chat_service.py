"""
多轮对话服务：把「RAG 问答」与「会话/历史持久化」串成完整业务闭环。

职责与流程（API 层只做参数校验与鉴权，业务编排在这里完成）：
1. 会话归属校验：只能操作属于自己的会话；
2. 多轮记忆组装：从 MySQL 取当前会话最近 N 轮历史，随问题一并送入模型，
   使回答理解上文语境（纯上下文窗口式记忆，无需训练/向量化历史）；
3. 调用 QueryPipeline 完成「混合检索 → 重排 → 生成」，拿到带引用编号的答案；
4. 持久化：把本轮「用户问题」与「助手回答（含引用溯源快照）」写入 MySQL；
5. 返回结构化结果：答案 + 引用编号 + 片段溯源明细，前端可直接渲染。

降低幻觉的设计：答案中的引用编号经 Generator.extract_references 校验，
越界编号（模型编造的引用）会被剔除并告警，保证溯源展示真实可信。
"""
import logging
from typing import Optional

from sqlalchemy.orm import Session

from config.settings import Settings
from persistence.repositories import ConversationRepository, MessageRepository
from service.query_pipeline import QueryPipeline
from vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class ChatService:
    """多轮对话编排：记忆组装 + RAG 问答 + 历史持久化。"""

    def __init__(self, settings: Settings, store: QdrantStore) -> None:
        self.settings = settings
        # 复用查询流水线（含检索+重排+生成+引用校验）
        self.pipeline = QueryPipeline(settings, store)

    # ---------------------------------------------------------------
    # 主入口
    # ---------------------------------------------------------------
    def ask(
        self,
        session: Session,
        user_id: int,
        conversation_id: int,
        message: str,
        top_k: Optional[int] = None,
    ) -> dict:
        """在指定会话中执行一次多轮问答并持久化历史。

        Args:
            session: 数据库会话（事务由调用方统一提交/回滚）
            user_id: 当前登录用户
            conversation_id: 目标会话 id
            message: 本轮用户提问
            top_k: 检索片段数（覆盖配置）

        Returns:
            {"conversation_id", "answer", "refs", "sources", "num_sources",
             "reranked", "memory_rounds"}
        """
        # 1) 归属校验：会话必须属于当前用户
        conv = ConversationRepository.get_for_user(session, user_id, conversation_id)

        # 2) 多轮记忆：取最近 N 轮历史（一问一答为一轮）
        history = self._build_history(session, conversation_id)
        logger.info("多轮记忆: 会话=%s 注入历史消息=%d 条（最近 %d 轮）",
                    conversation_id, len(history), self.settings.memory_rounds)

        # 3) RAG 问答（检索→重排→生成→引用校验）
        result = self.pipeline.ask(message, top_k=top_k, history=history or None)

        # 4) 持久化：用户问题 + 助手回答（附引用溯源快照）
        try:
            # 首条消息：把「新会话」标题替换为该问题的核心摘要
            is_first = ConversationRepository.count_messages(session, conversation_id) == 0
            MessageRepository.add(
                session, conversation_id, "user", message, sources=None
            )
            if is_first:
                ConversationRepository.update_title(
                    session, conversation_id, self._summarize_title(message)
                )
            MessageRepository.add(
                session,
                conversation_id,
                "assistant",
                result["answer"],
                sources=self._sources_for_storage(result["sources"], result["refs"]),
            )
            session.flush()  # 触发 INSERT；commit 由调用方 session_scope 统一执行
            logger.info("已持久化本轮对话: conv=%s user=%s assistant 引用=%s",
                        conversation_id, user_id, result["refs"])
        except Exception:
            session.rollback()
            raise

        result["conversation_id"] = conversation_id
        result["memory_rounds"] = self.settings.memory_rounds
        return result

    # ---------------------------------------------------------------
    # 流式问答（SSE）
    # ---------------------------------------------------------------
    def ask_stream(
        self,
        session: Session,
        user_id: int,
        conversation_id: int,
        message: str,
        top_k: Optional[int] = None,
    ):
        """多轮问答的流式版本：逐条 yield 事件 dict（协议见 QueryPipeline.ask_stream）。

        与 ask() 的差异：
        1. 生成阶段流式推送 delta 事件，前端实时渲染；
        2. 收到 done 事件（答案完整）时持久化本轮对话，再转发给客户端——
           保证「客户端看到 done 即历史已落库」，中途断流则丢弃本轮（不写脏数据）。
        """
        # 1) 归属校验（幂等：路由在开流前已校验一次，这里兜底）
        ConversationRepository.get_for_user(session, user_id, conversation_id)

        # 2) 多轮记忆组装
        history = self._build_history(session, conversation_id)
        logger.info("多轮记忆: 会话=%s 注入历史消息=%d 条（最近 %d 轮）",
                    conversation_id, len(history), self.settings.memory_rounds)

        # 3) 流式问答，并在完成时持久化
        for event in self.pipeline.ask_stream(
            message, top_k=top_k, history=history or None
        ):
            if event["type"] == "done":
                # 答案已完整：先落库，再转发 done（含 conversation_id / memory_rounds）
                is_first = ConversationRepository.count_messages(session, conversation_id) == 0
                MessageRepository.add(session, conversation_id, "user", message)
                if is_first:
                    ConversationRepository.update_title(
                        session, conversation_id, self._summarize_title(message)
                    )
                MessageRepository.add(
                    session,
                    conversation_id,
                    "assistant",
                    event["answer"],
                    sources=self._sources_for_storage(event.get("sources", []), event.get("refs", [])),
                )
                session.flush()
                event["conversation_id"] = conversation_id
                event["memory_rounds"] = self.settings.memory_rounds
                logger.info("已持久化本轮流式对话: conv=%s user=%s 引用=%s",
                            conversation_id, user_id, event.get("refs", []))
            yield event

    # ---------------------------------------------------------------
    # 辅助
    # ---------------------------------------------------------------
    def _build_history(self, session: Session, conversation_id: int) -> list[dict]:
        """从 MySQL 组装最近 N 轮历史，转为模型可用的消息字典列表。"""
        messages = MessageRepository.list_recent(
            session, conversation_id, rounds=self.settings.memory_rounds
        )
        history = []
        for msg in messages:
            # 只回放 user/assistant 轮次（system 提示不入库，无需处理）
            history.append({"role": msg.role, "content": msg.content})
        return history

    @staticmethod
    def _summarize_title(message: str, max_len: int = 20) -> str:
        """从首条问题提炼会话标题：合并空白/换行后截断，作为「新会话」的替换名。

        设计：标题取问题原文而非 LLM 摘要，保证即时生成（不额外调用模型）、
        可读且准确反映会话主题。
        """
        text = " ".join(message.split())
        return text[:max_len] if text else "新会话"

    @staticmethod
    def _sources_for_storage(sources: list[dict], refs: list[int] | None = None) -> list[dict]:
        """提取需要持久化的溯源字段，供历史回看完整还原。

        关键点：
        1. 必须保留 snippet（片段原文）——否则切会话/刷新回看历史时，
           前端溯源区只有标题、没有正文内容（历史回看 bug 的根因）；
        2. 附带 cited 标记（该片段是否被答案实际引用），使历史回看也能
           高亮「✅ 引用」徽标（实时对话走 SSE 的 refs 事件，历史走此标记）；
        3. 保留 page / score 用于展示，file_name 回退兼容旧字段 source。
        """
        cited_set = set(refs or [])
        return [
            {
                "ref_id": s.get("ref_id"),
                "file_name": s.get("file_name", s.get("source", "")),
                "page": s.get("page"),
                "score": s.get("score"),
                "snippet": s.get("snippet", ""),  # 片段原文：历史溯源展示的核心内容
                "cited": s.get("ref_id") in cited_set,  # 是否被答案实际引用
            }
            for s in sources
        ]
