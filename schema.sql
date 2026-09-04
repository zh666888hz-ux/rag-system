-- ============================================================
-- RAG 用户会话 / 聊天历史 表结构
-- 说明：应用启动时（persistence/db.py）会自动执行等价的幂等建表，
--       本文件供手工建库 / DBA 审核 / 数据库迁移参考。
-- 执行示例：
--   mysql -uroot -p < schema.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS `rag_chat`
  DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE `rag_chat`;

-- 用户表
CREATE TABLE IF NOT EXISTS `users` (
  `id`            BIGINT       NOT NULL AUTO_INCREMENT,
  `username`      VARCHAR(64)  NOT NULL COMMENT '登录名',
  `password_hash` VARCHAR(255) NOT NULL COMMENT '密码哈希（pbkdf2_sha256$迭代$盐$哈希）',
  `created_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`)
) ENGINE=InnoDB COMMENT='平台用户';

-- 会话表
CREATE TABLE IF NOT EXISTS `conversations` (
  `id`         BIGINT       NOT NULL AUTO_INCREMENT,
  `user_id`    BIGINT       NOT NULL COMMENT '所属用户',
  `title`      VARCHAR(255) NOT NULL DEFAULT '新会话' COMMENT '会话标题',
  `created_at` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最近活动时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_created` (`user_id`, `created_at`),
  CONSTRAINT `fk_conv_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB COMMENT='多轮对话会话';

-- 消息表
CREATE TABLE IF NOT EXISTS `messages` (
  `id`              BIGINT      NOT NULL AUTO_INCREMENT,
  `conversation_id` BIGINT      NOT NULL COMMENT '所属会话',
  `role`            VARCHAR(16) NOT NULL COMMENT 'user/assistant/system',
  `content`         TEXT        NOT NULL COMMENT '消息正文',
  `sources_json`    TEXT        NULL COMMENT 'assistant 消息的引用溯源快照（JSON）',
  `created_at`      DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_conv_created` (`conversation_id`, `id`),
  CONSTRAINT `fk_msg_conv` FOREIGN KEY (`conversation_id`) REFERENCES `conversations` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB COMMENT='聊天消息';
