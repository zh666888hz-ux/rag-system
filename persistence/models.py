"""
ORM 数据模型：用户 / 会话 / 聊天消息。

表关系（经典的一对多）：
  users (1) ──< conversations (1) ──< messages

设计要点：
1. 外键加 ON DELETE CASCADE：删除用户时级联删除其会话与消息，避免孤儿数据；
2. messages.role 限定 user/assistant/system 三值，保证历史消息可被安全还原为
   大模型消息序列；
3. messages.sources_json：assistant 消息附带的「引用溯源」快照（JSON 文本），
   前端可复现「答案引用了哪些片段」；存快照而非关联表，避免溯源信息随向量库
   增删而漂移，历史回看始终忠实于当时实际引用；
4. 时间戳由数据库生成（server_default），保证多实例写入时间一致。
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy 2.x 声明式基类。"""


class User(Base):
    """平台用户。"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, comment="登录名")
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False, comment="密码哈希（pbkdf2）")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间"
    )

    # 反向关系：级联删除会话
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User id={self.id} username={self.username}>"


class Conversation(Base):
    """一次独立的多轮对话会话。"""

    __tablename__ = "conversations"
    __table_args__ = (
        # 常用查询：某用户的会话列表按时间倒序
        Index("idx_user_created", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, comment="所属用户"
    )
    title: Mapped[str] = mapped_column(String(255), default="新会话", nullable=False, comment="会话标题")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, comment="最近活动时间"
    )

    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",  # 历史按写入顺序
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Conversation id={self.id} user_id={self.user_id} title={self.title}>"


class Message(Base):
    """一条聊天消息（user 提问 / assistant 回答 / system 提示）。"""

    __tablename__ = "messages"
    __table_args__ = (
        # 常用查询：某会话的最近 N 轮历史
        Index("idx_conv_created", "conversation_id", "id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, comment="所属会话"
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="user/assistant/system")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="消息正文")
    sources_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="assistant 消息的引用溯源快照（JSON）"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, comment="创建时间"
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Message id={self.id} role={self.role} conv={self.conversation_id}>"
