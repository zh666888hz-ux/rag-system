"""
数据库连接管理模块。

核心原理：
1. SQLAlchemy 2.x 以「声明式 ORM」把 Python 类映射到 MySQL 表，业务代码只操作对象，
   不写裸 SQL；Engine 负责连接池复用与方言差异屏蔽；
2. 连接串走 PyMySQL 驱动（纯 Python，MySQL 8 默认 caching_sha2_password 认证需
   cryptography 库配合）；
3. 数据库初始化采用「启动时幂等建库建表」：数据库不存在则先 CREATE DATABASE，
   表不存在则按模型元数据 CREATE_ALL。生产上可用 Alembic 做版本化迁移，此处用
   轻量自动建表保证开箱即用，同时保留手工执行 schema.sql 的选项。
4. 超时与连接池参数均来自配置：pool_recycle 防止 MySQL wait_timeout 静默断连后
   复用死连接。
"""
import logging
from typing import Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import Settings
from core.exceptions import DatabaseError
from persistence.models import Base

logger = logging.getLogger(__name__)


class Database:
    """MySQL 连接管理门面：负责建库建表与提供会话工厂。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    # ---------------------------------------------------------------
    # 连接管理
    # ---------------------------------------------------------------
    def connect(self) -> "Database":
        """建立连接、确保数据库与表存在，返回 self 便于链式使用。"""
        self._ensure_database()
        self._engine = create_engine(
            self._url(self.settings.mysql_db),
            pool_size=self.settings.mysql_pool_size,
            max_overflow=5,
            pool_recycle=self.settings.mysql_pool_recycle,
            pool_pre_ping=True,               # 取连接前先 ping，规避死连接
            pool_timeout=self.settings.mysql_connect_timeout,
            connect_args={
                "connect_timeout": self.settings.mysql_connect_timeout,
                "charset": "utf8mb4",
            },
            echo=False,
        )
        # sessionmaker 是会话工厂；每个请求独立 Session，用完即关（FastAPI 依赖注入管理）
        self._session_factory = sessionmaker(
            bind=self._engine,
            class_=Session,
            expire_on_commit=False,  # commit 后对象仍可访问，避免懒加载触发额外查询
        )
        self._create_tables()
        logger.info(
            "MySQL 就绪: %s:%s/%s",
            self.settings.mysql_host,
            self.settings.mysql_port,
            self.settings.mysql_db,
        )
        return self

    def _url(self, database: str) -> str:
        """构造 SQLAlchemy 连接串（PyMySQL 驱动）。"""
        return (
            f"mysql+pymysql://{self.settings.mysql_user}:"
            f"{self.settings.mysql_password}@{self.settings.mysql_host}:"
            f"{self.settings.mysql_port}/{database}?charset=utf8mb4"
        )

    def _ensure_database(self) -> None:
        """数据库不存在则创建。需先连到默认库（不带库名）执行 CREATE DATABASE。"""
        try:
            engine = create_engine(
                self._url("mysql"),  # 连到 MySQL 自带的 mysql 系统库（必然存在）
                pool_pre_ping=True,
                connect_args={"connect_timeout": self.settings.mysql_connect_timeout},
            )
            with engine.connect() as conn:
                # 只建不覆盖：已存在则跳过，兼容并发启动
                conn.execute(
                    text(
                        f"CREATE DATABASE IF NOT EXISTS `{self.settings.mysql_db}` "
                        "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                )
                conn.commit()
            engine.dispose()
            logger.info("已确保数据库存在: %s", self.settings.mysql_db)
        except Exception as exc:
            raise DatabaseError(f"连接 MySQL 失败，请检查配置与凭据: {exc}", cause=exc) from exc

    def _create_tables(self) -> None:
        """按 ORM 模型元数据自动建表（幂等，已存在则跳过）。"""
        try:
            Base.metadata.create_all(self._engine)
        except Exception as exc:
            raise DatabaseError(f"初始化数据表失败: {exc}", cause=exc) from exc

    # ---------------------------------------------------------------
    # 会话工厂
    # ---------------------------------------------------------------
    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise DatabaseError("数据库未初始化，请先调用 connect()")
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        if self._session_factory is None:
            raise DatabaseError("数据库未初始化，请先调用 connect()")
        return self._session_factory

    def new_session(self) -> Session:
        """创建独立会话（FastAPI 依赖注入用）。"""
        return self.session_factory()

    def session_scope(self) -> Iterator[Session]:
        """上下文管理器式会话：异常自动回滚，正常自动提交，用完必关。"""
        session = self.new_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        """应用退出时释放连接池。"""
        if self._engine is not None:
            self._engine.dispose()
            logger.info("MySQL 连接池已释放")
