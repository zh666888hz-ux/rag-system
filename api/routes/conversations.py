"""
会话管理路由：会话的增删查 + 历史消息读取。

权限模型：所有会话操作都必须携带登录态，且只能访问「属于当前用户」的会话，
越权访问统一由 ConversationRepository.get_for_user 抛 SessionError（转 404）。
"""
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from api.deps import get_current_user, get_db
from api.schemas import ConversationCreate, ConversationItem, MessageItem
from core.exceptions import SessionError
from persistence.models import Conversation
from persistence.repositories import ConversationRepository, MessageRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["会话"])


def _conv_to_item(conv: Conversation, message_count: int = 0) -> ConversationItem:
    return ConversationItem(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=message_count,
    )


@router.get("", response_model=list[ConversationItem], summary="我的会话列表")
def list_conversations(
    current: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
) -> list[ConversationItem]:
    """当前用户的会话列表，按最近活动倒序。"""
    convs = ConversationRepository.list_by_user(db, current["user_id"], limit=min(max(limit, 1), 200))
    # 一次聚合查询拿到全部会话的消息条数，避免 N+1
    counts = MessageRepository.count_by_conversation_ids(db, [c.id for c in convs])
    return [_conv_to_item(c, counts.get(c.id, 0)) for c in convs]


@router.post("", response_model=ConversationItem, status_code=status.HTTP_201_CREATED, summary="新建会话")
def create_conversation(
    body: ConversationCreate,
    current: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationItem:
    """新建一个空会话，返回会话信息。"""
    conv = ConversationRepository.create(db, current["user_id"], title=body.title)
    return _conv_to_item(conv, 0)


@router.get("/{conversation_id}/messages", response_model=list[MessageItem], summary="会话历史")
def get_messages(
    conversation_id: int,
    current: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[MessageItem]:
    """读取某会话的全部历史消息（归属校验；越权返回 404）。"""
    ConversationRepository.get_for_user(db, current["user_id"], conversation_id)
    messages = MessageRepository.list_all(db, conversation_id)
    items = []
    for msg in messages:
        sources = None
        if msg.sources_json:
            try:
                sources = json.loads(msg.sources_json)
            except json.JSONDecodeError:
                sources = None
        items.append(
            MessageItem(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                sources=sources,
                created_at=msg.created_at,
            )
        )
    return items


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, summary="删除会话")
def delete_conversation(
    conversation_id: int,
    current: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """删除会话（级联删除其全部历史消息）。归属校验：越权返回 404。"""
    ConversationRepository.delete_for_user(db, current["user_id"], conversation_id)
