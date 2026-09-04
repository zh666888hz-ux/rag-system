"""
安全模块：密码哈希与 JWT 令牌。

设计决策：
1. 密码不用明文也不直接 SHA256（彩虹表可破解），采用标准库 hashlib.pbkdf2_hmac
   加盐慢哈希。PBKDF2 通过高迭代次数（默认 600_000 次）显著提高暴力破解成本，
   是 OWASP 推荐的密码哈希方案，且零第三方依赖（比 bcrypt 更省安装与兼容问题）；
   哈希格式自带算法/迭代/盐，便于未来平滑升级算法。
2. JWT 用 PyJWT 签发 HS256 签名令牌：payload 仅放用户标识与过期时间（不放敏感
   信息），服务端用 jwt_secret 验签即可确认「是谁、是否过期、是否被篡改」，
   无需在服务端保存会话状态（无状态鉴权，天然适合水平扩展）。
3. 过期时间、算法、密钥全部来自配置；校验失败统一抛 AuthError，由 API 层映射为
   401/403。
"""
import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

from config.settings import Settings
from core.exceptions import AuthError

logger = logging.getLogger(__name__)

# PBKDF2 参数：迭代次数越高越安全但也越慢；600k 为 OWASP 推荐起点
_PBKDF2_ITERATIONS = 600_000
_PBKDF2_ALGO = "sha256"
_HASH_PREFIX = "pbkdf2_sha256"


# ---------------------------------------------------------------
# 密码哈希（PBKDF2-HMAC-SHA256 + 随机盐）
# ---------------------------------------------------------------
def hash_password(password: str) -> str:
    """生成密码哈希，格式：pbkdf2_sha256$迭代次数$盐(hex)$哈希(hex)。"""
    salt = secrets.token_hex(16)  # 128-bit 随机盐，每个用户独立
    digest = hashlib.pbkdf2_hmac(
        _PBKDF2_ALGO, password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return f"{_HASH_PREFIX}${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验密码：从存储串解析参数后重新计算哈希，用 hmac.compare_digest 恒时比较
    防时序侧信道。算法/迭代次数取自存储串，兼容未来调参。"""
    try:
        prefix, iterations, salt_hex, digest_hex = stored.split("$")
        if prefix != _HASH_PREFIX:
            return False
        digest = hashlib.pbkdf2_hmac(
            _PBKDF2_ALGO, password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------
# JWT 签发与校验
# ---------------------------------------------------------------
def create_access_token(settings: Settings, user_id: int, username: str) -> str:
    """签发 JWT：payload 携带用户标识与过期时间（iat/nbf/exp 为标准声明）。"""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": str(user_id),          # subject：用户唯一标识
        "username": username,         # 非敏感信息，便于日志与展示
        "iat": now,                   # issued at：签发时间
        "nbf": now,                   # not before：生效时间（防时钟回拨提前使用）
        "exp": expire,                # expiration：过期时间（服务端强制校验）
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    logger.info("签发 JWT: user_id=%s username=%s 有效期=%d 分钟",
                user_id, username, settings.jwt_expire_minutes)
    return token


def decode_access_token(settings: Settings, token: str) -> dict:
    """校验并解析 JWT，返回 payload；非法/过期抛 AuthError。

    校验维度：签名真实性（防篡改）、过期时间（exp）、生效时间（nbf）。
    对伪造/过期/无效签名给出同一类错误，避免暴露内部细节。
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp"]},  # 强制要求关键声明存在
        )
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("登录已过期，请重新登录") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("无效的登录凭证") from exc
