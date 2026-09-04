"""
认证路由：注册 / 登录 / 当前用户信息。

流程说明：
- 注册：校验输入 → 哈希密码（PBKDF2）→ 写库（用户名冲突转 409）；
- 登录：校验用户名+密码 → 签发 JWT → 返回 token 与用户信息；
- /me：携带 token 获取当前用户（用于前端校验登录态）。
"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps import get_current_user, get_db, get_settings
from api.schemas import LoginRequest, RegisterRequest, TokenResponse, UserInfo
from config.settings import Settings
from core.exceptions import AuthError
from core.security import create_access_token, hash_password, verify_password
from persistence.models import User
from persistence.repositories import UserRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["认证"])


def _user_to_info(user: User) -> UserInfo:
    return UserInfo(id=user.id, username=user.username, created_at=user.created_at)


@router.post("/register", response_model=UserInfo, status_code=201, summary="注册新用户")
def register(
    body: RegisterRequest,
    db: Session = Depends(get_db),
) -> UserInfo:
    """注册：用户名唯一，密码以 PBKDF2 加盐哈希存储（不存明文）。"""
    password_hash = hash_password(body.password)
    user = UserRepository.create(db, body.username, password_hash)
    return _user_to_info(user)


@router.post("/login", response_model=TokenResponse, summary="登录并获取 JWT")
def login(
    body: LoginRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """登录：校验密码后签发 JWT。任何失败统一为 401，不泄露具体原因。"""
    user = UserRepository.get_by_username(db, body.username)
    if user is None or not verify_password(body.password, user.password_hash):
        logger.warning("登录失败: username=%s（用户不存在或密码错误）", body.username)
        raise AuthError("用户名或密码错误")

    token = create_access_token(settings, user.id, user.username)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.jwt_expire_minutes * 60,
        user=_user_to_info(user),
    )


@router.get("/me", response_model=UserInfo, summary="获取当前登录用户")
def me(current: dict = Depends(get_current_user)) -> UserInfo:
    """校验 token 并返回当前用户信息（token 非法/过期自动 401）。

    直接从 JWT payload 解析，不查库，保证接口轻量实时。
    """
    return UserInfo(
        id=current["user_id"],
        username=current["username"],
    )
