"""
数据访问层（Repository）：把 ORM 查询封装为业务语义清晰的方法。

设计原则：
1. Repository 只负责「怎么存」，不关心「业务规则」（校验、鉴权在 Service/API 层）；
2. 所有方法接收已开启的 Session，事务边界由调用方（Service 层 / session_scope）
   统一控制，避免跨方法隐式提交；
3. 底层异常统一收敛为 RAGError 体系的 DatabaseError / UserAlreadyExistsError /
   SessionError，业务层不感知 SQLAlchemy 细节。
"""
import json
import logging
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.exceptions import DatabaseError, SessionError, UserAlreadyExistsError
from persistence.models import Conversation, Message, User

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# 用户
# ---------------------------------------------------------------
class UserRepository:
    """用户表存取。"""

    @staticmethod
    def create(session: Session, username: str, password_hash: str) -> User:
        """创建用户；用户名唯一冲突时抛出 UserAlreadyExistsError。"""
        user = User(username=username, password_hash=password_hash)
        session.add(user)
        try:
            session.flush()  # flush 触发 INSERT 以立即暴露唯一约束冲突
        except IntegrityError as exc:
            session.rollback()
            raise UserAlreadyExistsError(f"用户名已存在: {username}") from exc
        except Exception as exc:
            session.rollback()
            raise DatabaseError(f"创建用户失败: {exc}", cause=exc) from exc
        logger.info("创建用户成功: id=%s username=%s", user.id, username)
        return user

    @staticmethod
    def get_by_username(session: Session, username: str) -> Optional[User]:
        return session.scalar(select(User).where(User.username == username))

    @staticmethod
    def get_by_id(session: Session, user_id: int) -> Optional[User]:
        return session.get(User, user_id)


# ---------------------------------------------------------------
# 会话
# ---------------------------------------------------------------
class ConversationRepository:
    """会话表存取。所有按 id 查询都带 user_id 归属校验，防止越权访问他人会话。"""

    @staticmethod
    def create(session: Session, user_id: int, title: str = "新会话") -> Conversation:
        conv = Conversation(user_id=user_id, title=title or "新会话")
        session.add(conv)
        try:
            session.flush()
        except Exception as exc:
            session.rollback()
            raise DatabaseError(f"创建会话失败: {exc}", cause=exc) from exc
        logger.info("创建会话成功: id=%s user_id=%s", conv.id, user_id)
        return conv

    @staticmethod
    def list_by_user(session: Session, user_id: int, limit: int = 50) -> list[Conversation]:
        """某用户的会话列表，按最近活动倒序。"""
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit)
        )
        return list(session.scalars(stmt).all())

    @staticmethod
    def get_for_user(session: Session, user_id: int, conversation_id: int) -> Conversation:
        """取某用户拥有的会话；不存在或不属于该用户时抛 SessionError。"""
        conv = session.get(Conversation, conversation_id)
        if conv is None:
            raise SessionError(f"会话不存在: id={conversation_id}")
        if conv.user_id != user_id:
            raise SessionError(f"无权访问该会话: id={conversation_id}")
        return conv

    @staticmethod
    def delete_for_user(session: Session, user_id: int, conversation_id: int) -> None:
        """删除某用户的会话（级联删除其消息）。"""
        ConversationRepository.get_for_user(session, user_id, conversation_id)  # 校验归属
        try:
            session.execute(delete(Conversation).where(Conversation.id == conversation_id))
        except Exception as exc:
            session.rollback()
            raise DatabaseError(f"删除会话失败: {exc}", cause=exc) from exc
        logger.info("删除会话成功: id=%s", conversation_id)

    @staticmethod
    def count_by_user(session: Session, user_id: int) -> int:
        return int(session.scalar(select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)) or 0)

    @staticmethod
    def count_messages(session: Session, conversation_id: int) -> int:
        """统计某会话的消息条数（用于判断是否首条消息、是否空会话）。"""
        return int(
            session.scalar(
                select(func.count())
                .select_from(Message)
                .where(Message.conversation_id == conversation_id)
            )
            or 0
        )

    @staticmethod
    def update_title(session: Session, conversation_id: int, title: str) -> None:
        """更新会话标题（用于把「新会话」替换为第一个问题的摘要）。"""
        conv = session.get(Conversation, conversation_id)
        if conv is not None and title:
            conv.title = title
            session.flush()


# ---------------------------------------------------------------
# 消息（聊天历史）
# ---------------------------------------------------------------
class MessageRepository:
    """聊天消息存取。"""

    @staticmethod
    def add(
        session: Session,
        conversation_id: int,
        role: str,
        content: str,
        sources: Optional[list[dict]] = None,
    ) -> Message:
        """写一条消息；sources 为 assistant 消息的引用溯源列表，存为 JSON 快照。"""
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources_json=json.dumps(sources, ensure_ascii=False) if sources else None,
        )
        session.add(msg)
        try:
            session.flush()
        except Exception as exc:
            session.rollback()
            raise DatabaseError(f"保存消息失败: {exc}", cause=exc) from exc
        return msg

    @staticmethod
    def list_recent(session: Session, conversation_id: int, rounds: int) -> list[Message]:
        """取最近 N 轮历史消息（按写入顺序），用于多轮对话记忆。

        轮数换算：rounds 轮 = 2 * rounds 条消息（一问一答两条）。不足则返回全部。
        """
        if rounds <= 0:
            return []
        limit = rounds * 2
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(limit)
        )
        recent = list(session.scalars(stmt).all())
        return list(reversed(recent))  # 恢复时间正序，保证对话语境连贯

    @staticmethod
    def list_all(session: Session, conversation_id: int) -> list[Message]:
        """某会话的全部历史（按写入顺序），用于历史回看接口。"""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.asc())
        )
        return list(session.scalars(stmt).all())

    @staticmethod
    def count_by_conversation_ids(session: Session, conversation_ids: list[int]) -> dict[int, int]:
        """批量统计多个会话的消息条数（一条 GROUP BY 聚合，避免 N+1 查询）。"""
        if not conversation_ids:
            return {}
        stmt = (
            select(Message.conversation_id, func.count(Message.id))
            .where(Message.conversation_id.in_(conversation_ids))
            .group_by(Message.conversation_id)
        )
        return {conv_id: int(cnt) for conv_id, cnt in session.execute(stmt).all()}
