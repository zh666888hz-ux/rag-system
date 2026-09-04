"""
FastAPI 依赖注入：集中管理跨路由共享的依赖。

核心思路：
1. 应用级单例（Settings / Database / QdrantStore / ChatService）挂在 app.state，
   依赖函数从 request.app.state 取，避免重复创建重量级对象；
2. 数据库会话（Session）为「每个请求独立一个」，请求结束统一关闭——
   这是 FastAPI 与 SQLAlchemy 集成的标准模式，保证请求间无状态串扰；
3. get_current_user 解析 Authorization: Bearer <JWT>，失败抛 AuthError，
   由全局异常处理器统一转 401。
"""
import logging
from typing import Generator, Optional

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from config.settings import Settings
from core.exceptions import AuthError
from core.security import decode_access_token
from persistence.db import Database
from service.chat_service import ChatService
from vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_store(request: Request) -> QdrantStore:
    return request.app.state.store


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_db(db: Database = Depends(get_database)) -> Generator[Session, None, None]:
    """每个请求一个独立数据库会话；请求结束统一提交/回滚并关闭。

    FastAPI + SQLAlchemy 标准模式：路由内只做增删改查（flush），
    请求成功返回后由本依赖统一 commit，异常则 rollback，保证事务边界清晰。
    """
    session = db.new_session()
    try:
        yield session
        session.commit()          # 路由正常结束 → 统一提交
    except Exception:
        session.rollback()        # 路由抛异常 → 回滚，避免脏数据
        raise
    finally:
        session.close()


def _extract_bearer_token(request: Request) -> Optional[str]:
    """从 Authorization 头解析 Bearer token；格式非法时返回 None。"""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return header[len("Bearer "):].strip()


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    """解析 JWT 并返回当前用户信息 {"user_id", "username"}；非法/缺失抛 AuthError。"""
    token = _extract_bearer_token(request)
    if not token:
        raise AuthError("缺少登录凭证，请在 Authorization 头携带 Bearer token")
    settings: Settings = request.app.state.settings
    payload = decode_access_token(settings, token)  # 非法/过期抛 AuthError
    try:
        user_id = int(payload["sub"])
    except (KeyError, ValueError) as exc:
        raise AuthError("无效的登录凭证") from exc
    logger.debug("鉴权通过: user_id=%s username=%s", user_id, payload.get("username"))
    return {"user_id": user_id, "username": payload.get("username", "")}
