# RAG 工程：LangChain + Qdrant + OpenAI 兼容 API

一个分层、模块化、可运行的工程级基础 RAG 系统：加载 PDF → 文本递归分块 → 向量化存入

Qdrant → 用户提问后检索相关片段 → 交给大模型生成带溯源的答案。

## 一、架构与目录结构

### 1. RAG 核心流程（两层三阶段）



```
┌───────────────────────── 索引阶段 Indexing ─────────────────────────┐

│  PDF 文档 ──加载(pypdf)──▶ 逐页 Document ──递归分块──▶ 文本块          │

│                        （保留页码元数据）        （保留语义边界+overlap）│

│                                                                       │

│  文本块 ──Embedding 向量化──▶ 高维向量 ──写入──▶ Qdrant（HNSW 图索引）  │

└──────────────────────────────────────────────────────────────────────┘

┌───────────────────────── 查询阶段 Querying ─────────────────────────┐

│  用户问题                                                                 │

│   ├─▶ 向量检索（dense，语义）──┐                                         │

│   ├─▶ BM25 检索（sparse，关键词）─┼─▶ RRF 融合 ─▶ 候选 Top-K              │

│   │                        （倒数排名融合，去重）            │             │

│   └─▶ Rerank 精排（Cross-Encoder 打分筛选）──▶ 高置信片段                  │

│        ▶ 组装上下文（片段+来源标注）──▶ LLM 生成 ──▶ 带溯源的答案          │

└──────────────────────────────────────────────────────────────────────┘
```



* **为什么分块**：LLM 上下文窗口有限，整篇长文档无法直接喂入；把文档切成有语义边界的块，

  检索时只取最相关的几块，兼顾「覆盖全库」与「喂入精炼」。

* **为什么用向量库**：关键词检索无法理解语义（同义词 / 换说法失效）；向量库把文本映射到

  高维语义空间，用距离度量相似度，再借助 ANN（近似最近邻，Qdrant 用 HNSW 图）实现

  大数据量下的实时检索。

* **为什么混合检索 + 重排**：向量检索懂语义但可能漏精确术语，BM25 做关键词精确匹配，

  两者用 RRF（倒数排名融合）合并，扩大召回覆盖；再经 Cross-Encoder（bge-reranker）

  精排打分，只把高置信片段送入 LLM，从源头过滤噪声、降低幻觉。

### 2. 工程分层



| 层     | 目录                   | 职责                                                                                                                                |
| ----- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 入口层   | `main.py`            | CLI 子命令（含 `serve` 启动 HTTP 服务）、全局异常兜底、退出码约定                                                                                        |
| 配置层   | `config/settings.py` | pydantic-settings 集中配置 + 启动时参数校验                                                                                                  |
| 基础设施层 | `core/`              | 统一异常体系（`exceptions.py`）、日志（`logging.py`）、安全（`security.py`：JWT + 密码哈希）                                                             |
| 数据接入层 | `ingestion/`         | PDF 加载（`loader.py`）、递归分块（`splitter.py`）                                                                                           |
| 向量存储层 | `vectorstore/`       | Embedding 客户端、Qdrant 封装（`qdrant_store.py`）                                                                                        |
| 服务层   | `service/`           | 检索（`retriever.py`）、BM25 索引（`bm25_index.py`）、重排（`reranker.py`）、生成（`generator.py`）、流水线（`query_pipeline.py`）、多轮对话（`chat_service.py`） |
| 持久层   | `persistence/`       | MySQL 连接（`db.py`）、ORM 模型（`models.py`）、仓储（`repositories.py`）                                                                       |
| API 层 | `api/`               | FastAPI 路由（认证 / 会话 / 对话）、请求校验、依赖注入、异常转 HTTP                                                                                       |
| 测试层   | `tests/`             | 离线冒烟测试 + 安全 / 检索单元测试                                                                                                              |



```
rag-langchain-qdrant/

├── api/                        # HTTP 接口层（用户会话/多轮记忆/JWT）

│   ├── app.py                  # FastAPI 应用装配 + 全局异常转 HTTP + 生命周期

│   ├── deps.py                 # 依赖注入：DB 会话 / 当前用户 / 服务单例

│   ├── schemas.py              # 请求/响应 Pydantic 模型（校验）

│   └── routes/

│       ├── auth.py             # 注册 / 登录（签发 JWT）/ me

│       ├── conversations.py    # 会话增删查 + 历史消息

│       └── chat.py             # 多轮 RAG 问答

├── config/

│   └── settings.py             # 配置：环境变量/.env 读取 + 类型与规则校验

├── core/

│   ├── exceptions.py           # RAGError 统一异常体系

│   ├── logging.py              # 控制台 + 滚动文件双通道日志

│   └── security.py             # 密码哈希（PBKDF2）+ JWT 签发/校验

├── ingestion/

│   ├── loader.py               # pypdf 逐页解析，带页码元数据与批量容错

│   └── splitter.py             # RecursiveCharacterTextSplitter 中文分隔符

├── vectorstore/

│   ├── embeddings.py           # OpenAI 兼容/本地 Embedding 客户端

│   └── qdrant\_store.py         # Qdrant 封装：集合管理/写入/检索/统计/全量导出

├── persistence/                # MySQL 持久层

│   ├── db.py                   # 连接池管理 + 自动建库建表（幂等）

│   ├── models.py               # ORM：users / conversations / messages

│   └── repositories.py         # 用户/会话/消息仓储（归属校验）

├── service/

│   ├── retriever.py            # 混合检索器：向量+BM25，RRF 融合，逐条召回日志

│   ├── bm25\_index.py           # BM25 关键词索引（jieba 分词 + rank\_bm25）

│   ├── reranker.py             # Cross-Encoder 精排（本地 ONNX bge-reranker）

│   ├── generator.py            # 生成 + RAG 约束 Prompt + 引用编号溯源

│   ├── ingestion\_pipeline.py   # 索引流水线：load → split → store

│   ├── query\_pipeline.py       # 查询流水线：retrieve → rerank → generate（多轮记忆）

│   └── chat\_service.py         # 多轮对话编排：记忆组装 + 问答 + 历史持久化

├── tests/

│   ├── make\_pdf.py             # 程序化生成最小合法 PDF（测试用）

│   └── test\_smoke.py           # 离线冒烟 + 检索/安全单元测试

├── main.py                     # CLI 入口（ingest/query/stats/drop/serve）

├── schema.sql                  # MySQL 表结构（应用启动也会自动建表）

├── requirements.txt

├── .env.example

└── README.md
```

## 二、依赖清单

见 `requirements.txt`，核心依赖如下：



| 依赖                                  | 用途                                             |
| ----------------------------------- | ---------------------------------------------- |
| `langchain` / `langchain-community` | LangChain 核心框架                                 |
| `langchain-openai`                  | OpenAI 兼容 API 的 LLM / Embedding 封装             |
| `langchain-qdrant`                  | Qdrant 与 LangChain 的集成                         |
| `langchain-text-splitters`          | 递归字符切分器                                        |
| `qdrant-client`                     | Qdrant 客户端（远程 / 本地两种模式）                        |
| `pypdf`                             | PDF 文本层解析                                      |
| `pydantic` / `pydantic-settings`    | 配置模型与校验                                        |
| `python-dotenv`                     | 读取 `.env` 文件                                   |
| `fastembed`                         | 本地 Embedding（bge-small-zh，离线免费）                |
| `rank_bm25` / `jieba`               | BM25 关键词检索与中文分词                                |
| `transformers`                      | 重排模型 tokenizer（ONNX 推理走 onnxruntime，不引入 torch） |
| `pytest`                            | 测试                                             |

## 三、环境说明与启动步骤

### 1. 准备 Python 环境

要求 **Python 3.10+**（代码使用了 `X | None` 类型注解）。



```
cd rag-langchain-qdrant

python -m venv .venv

\# Windows

.venv\Scripts\activate

\# Linux / macOS

source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 准备外部服务

**方式 A：Qdrant 本地文件模式（无需部署，开发推荐）**

不填 `RAG_QDRANT_URL`，数据自动持久化到 `RAG_QDRANT_PATH`。

**方式 B：Qdrant 远程服务（生产推荐）**



```
docker run -d --name qdrant -p 6333:6333 -v \$(pwd)/qdrant\_storage:/qdrant/storage qdrant/qdrant
```

**OpenAI 兼容 API**

本工程通过 `base_url` 对接任意 OpenAI 兼容网关（OpenAI / DeepSeek / 通义 /vLLM/

OneAPI 等），只需在配置中填入网关地址、Key 与模型名。

### 3. 配置



```
cp .env.example .env

\# 编辑 .env：填写 RAG\_OPENAI\_BASE\_URL / RAG\_OPENAI\_API\_KEY，并按需调整其余项

\# 注意：Embedding 模型输出维度必须与 RAG\_VECTOR\_SIZE 一致（如 text-embedding-3-small=1536）
```

**Chat 与 Embedding 可分离（混合架构）**

有些厂商只有对话模型没有 Embedding 接口（如 DeepSeek），生产中也常希望 Chat 与

Embedding 走不同网关。本工程通过 `embedding_provider` 支持解耦：



* `embedding_provider=openai_compatible`：Embedding 走远端兼容网关

  （默认复用 `RAG_OPENAI_BASE_URL`，也可用 `RAG_EMBEDDING_BASE_URL` / `RAG_EMBEDDING_API_KEY` 单独指定）；

* `embedding_provider=local_fastembed`：Embedding 用本地 ONNX 模型（`fastembed`），

  离线免费、无额度限制，适合国内网络 / 开发调试。默认模型 `BAAI/bge-small-zh-v1.5`

  （维度 512），首次运行自动下载（走 hf-mirror 镜像）。

典型组合示例（DeepSeek Chat + 本地 Embedding）：



```
RAG\_OPENAI\_BASE\_URL=https://api.deepseek.com/v1

RAG\_OPENAI\_API\_KEY=sk-xxx

RAG\_CHAT\_MODEL=deepseek-v4-flash

RAG\_EMBEDDING\_PROVIDER=local\_fastembed

RAG\_EMBEDDING\_MODEL=BAAI/bge-small-zh-v1.5

RAG\_VECTOR\_SIZE=512
```

### 4. 运行



```
\# ① 导入文档建索引（支持多个文件或目录）

python main.py ingest ./docs/sample.pdf ./docs/more/

\# ② 提问（默认返回 Top-K=4 条检索片段）

python main.py query "这个项目用什么框架？"

\# ③ 查看向量库统计

python main.py stats

\# ④ 删除集合（危险操作，需输入 yes 二次确认）

python main.py drop

\# ⑤ 启动 HTTP API 服务（用户会话 / 多轮记忆 / JWT 鉴权）

python main.py serve          # 等价于 uvicorn api.app:app

\# 服务启动后访问 http://127.0.0.1:8000/docs 查看自动生成的接口文档
```

### 5. 运行测试（离线，无需真实 API）



```
python -m pytest tests/ -v
```

测试用「本地 Qdrant 文件模式 + 本地假 OpenAI 服务」跑通完整 RAG 流程，可离线重复执行。

## 五、用户会话 / 多轮记忆 / JWT 鉴权

### 1. 能力总览



| 能力     | 说明                                           | 实现                                       |
| ------ | -------------------------------------------- | ---------------------------------------- |
| 用户体系   | 注册 / 登录，密码 PBKDF2 加盐哈希存储                     | `core/security.py` + `users` 表           |
| JWT 鉴权 | 登录签发令牌，后续请求带 `Authorization: Bearer <token>` | `core/security.py` + `api/deps.py`       |
| 会话管理   | 每个用户多个独立会话，多会话隔离，越权访问返回 404                  | `conversations` 表 + 归属校验                 |
| 多轮记忆   | 每次提问把当前会话最近 N 轮历史一并送入模型，理解上文语境               | `service/chat_service.py` + `messages` 表 |
| 引用溯源   | 答案用 `[编号]` 标注引用，编号与片段明细一一对应，越界引用自动剔除         | `service/generator.py`                   |

### 2. 多轮记忆原理

「多轮记忆」= **上下文窗口式记忆**（无需训练 / 无需向量化历史）：



1. 用户提问时，从 MySQL 取当前会话最近 `RAG_MEMORY_ROUNDS` 轮（一问一答为一轮）历史；

2. 历史以 user/assistant 消息序列插入 Prompt（system 约束 → 历史 → 本轮问题）；

3. 模型据此理解上文语境（如「上面提到的那个方案」「第二点再展开」）；

4. 本轮问答结果写入 MySQL，形成可持续增长的记忆链。

### 3. 引用溯源原理



1. 检索到的片段在「参考资料」中编号为 `[片段1]... [片段N]`；

2. 系统提示词强制模型「引用时用 \[编号] 标注，禁止编造不存在的编号」；

3. 答案生成后，`Generator.extract_references` 抽取全部 `[编号]` 并校验是否越界，

   越界引用（幻觉引用）剔除并记录告警日志；

4. 接口返回 `refs`（答案中合法引用编号）与 `sources`（编号→文件名 / 页码 / 分数 / 摘要），

   前端据此渲染可点击的溯源卡片。

### 4. 接口清单（前缀 `/api`）



| 方法     | 路径                             | 鉴权 | 说明                                        |
| ------ | ------------------------------ | -- | ----------------------------------------- |
| POST   | `/auth/register`               | 否  | 注册 `{username, password}`                 |
| POST   | `/auth/login`                  | 否  | 登录，返回 JWT `{access_token}`                |
| GET    | `/auth/me`                     | 是  | 当前用户信息                                    |
| GET    | `/conversations`               | 是  | 我的会话列表                                    |
| POST   | `/conversations`               | 是  | 新建会话 `{title?}`                           |
| GET    | `/conversations/{id}/messages` | 是  | 会话历史（含引用溯源快照）                             |
| DELETE | `/conversations/{id}`          | 是  | 删除会话                                      |
| POST   | `/chat`                        | 是  | 多轮问答 `{conversation_id, message, top_k?}` |
| GET    | `/health`                      | 否  | 健康检查                                      |

### 5. 快速体验（curl 示例）



```
BASE=http://127.0.0.1:8000/api

\# 注册并登录

curl -X POST \$BASE/auth/register -H "Content-Type: application/json" -d '{"username":"alice","password":"secret123"}'

TOKEN=\$(curl -s -X POST \$BASE/auth/login -H "Content-Type: application/json" \\

&#x20; -d '{"username":"alice","password":"secret123"}' | python -c "import sys,json;print(json.load(sys.stdin)\['access\_token'])")

\# 新建会话

CONV=\$(curl -s -X POST \$BASE/conversations -H "Authorization: Bearer \$TOKEN" \\

&#x20; -H "Content-Type: application/json" -d '{}' | python -c "import sys,json;print(json.load(sys.stdin)\['id'])")

\# 多轮问答（回答中含 \[编号] 引用，返回 sources 溯源明细）

curl -X POST \$BASE/chat -H "Authorization: Bearer \$TOKEN" -H "Content-Type: application/json" \\

&#x20; -d "{\\"conversation\_id\\":\$CONV,\\"message\\":\\"RAG 中文本向量化通常用什么模型？\\"}"

\# 第二问（无追问时模型也能结合上文）

curl -X POST \$BASE/chat -H "Authorization: Bearer \$TOKEN" -H "Content-Type: application/json" \\

&#x20; -d "{\\"conversation\_id\\":\$CONV,\\"message\\":\\"那它的中文模型有哪些？\\"}"

\# 查看会话历史

curl -s \$BASE/conversations/\$CONV/messages -H "Authorization: Bearer \$TOKEN"
```

## 六、工程要点说明



1. **分层解耦**：数据接入 / 向量存储 / 检索 / 生成各层只依赖相邻层的抽象输入（Document

   列表），任一层可单独替换（如换成 OCR 加载、语义分块、另一家向量库）。

2. **健壮性**：所有外部调用配置超时 + 自动重试；统一 `RAGError` 异常体系，上层只需捕获

   基类即可全局兜底；单个文件解析失败不阻塞批量导入。

3. **参数校验 fail-fast**：配置在启动时统一校验（如 `chunk_overlap < chunk_size`、

   `base_url` 合法性），检索参数在入口校验，非法值绝不带病运行。

4. **可溯源**：每个文本块携带 `source / page / chunk_index` 元数据，检索结果带相似度分数，

   生成 Prompt 要求模型标注来源。

5. **日志**：控制台 + 滚动文件双通道，第三方库降噪；检索阶段对「向量召回 / BM25 召回 /

   RRF 融合结果 / Rerank 精排结果」**逐条记录**来源、页码、分数、片段摘要，满足召回内容可审计。

## 七、混合检索与重排（新增）

### 1. 混合召回



* 向量检索（`QdrantStore.similarity_search`）：语义相似度召回；

* BM25 检索（`service/bm25_index.py`）：jieba 分词 + rank\_bm25，关键词精确召回；

* RRF 融合（`service/retriever.py`）：`score(doc) = Σ 1/(k + rank)`，只看名次不看原始分数，

  规避两路分数尺度不一致问题；BM25 索引从 Qdrant 全量导出重建，进程重启后仍可用。

### 2. Rerank 精排（降低幻觉）



* `service/reranker.py`：本地 ONNX 版 `bge-reranker-base`（Cross-Encoder），onnxruntime 推理，

  无 torch、离线免费；对「问题 - 片段」联合编码打分并 sigmoid 归一化到 (0,1)；

* 只把重排后高置信的 top\_k 片段送入 LLM，配合可选的 `RAG_RERANK_THRESHOLD` 阈值过滤，

  从源头减少「引用错误片段」导致的幻觉；

* 重排失败自动降级为不重排（增强项不阻断主流程）。

### 3. 关键配置



```
RAG\_RETRIEVAL\_MODE=hybrid        # hybrid=向量+BM25；vector=纯向量

RAG\_BM25\_K=8                     # BM25 召回条数

RAG\_FUSION\_K=60                  # RRF 融合常数

RAG\_ENABLE\_RERANK=true           # 是否重排

RAG\_RERANK\_MODEL=Xenova/bge-reranker-base

RAG\_RERANK\_THRESHOLD=            # 留空=只排序不过滤
```

## 八、已知边界与生产化方向



* **扫描版 PDF**：`pypdf` 只抽取文本层，扫描件需接入 OCR（如 PaddleOCR）。

* **检索质量调优**：`score_threshold`、`top_k`、`chunk_size/overlap` 需在真实语料上

  按召回率 / 命中率实验调参。

* **生产化可扩展**：可叠加混合检索（向量 + BM25）、重排（Reranker）、父子分块、

  增量更新、监控埋点（OpenTelemetry）与鉴权网关。

## 九、SSE 流式接口 / Streamlit 前端 / Docker 一键部署

### 1. SSE 流式问答接口

`POST /api/chat/stream`（需 `Authorization: Bearer <JWT>`），响应 `text/event-stream`。
事件协议（每条帧 `data: {json}\n\n`）：

| 事件 type | 内容 | 说明 |
|---|---|---|
| `meta` | `num_sources` / `reranked` | 检索+重排完成，开始输出候选 |
| `source` | `ref_id` / `file_name` / `page` / `score` / `snippet` | 每个候选一条 |
| `delta` | `text` | 大模型流式文本增量（打字机效果） |
| `refs` | `refs: [编号]` | 最终答案实际引用的来源编号 |
| `done` | `answer` / `refs` / `sources` / `num_sources` / `conversation_id` | 结束；已持久化历史 |
| `error` | `message` | 生成中途异常 |

关键设计：流式路由**手动管理数据库会话**（不使用 `get_db` 依赖，避免 FastAPI 依赖
teardown 在生成器迭代前关闭连接）；`done` 事件时统一落库（user + assistant），客户端
断流则丢弃本轮不落库。同步 `POST /api/chat` 接口保留。

### 2. Streamlit 聊天前端

```
pip install -r webapp/requirements.txt
$env:API_BASE_URL="http://127.0.0.1:8000/api"
streamlit run webapp/app.py            # http://localhost:8501
```

功能：注册/登录（JWT 存 session_state）、侧边栏会话管理（新建/切换/删除/刷新）、
历史消息加载、SSE 流式打字机渲染、回答下方「引用溯源」折叠区（来源文件/页码/相似度/
片段，实际引用编号打 ✅ 标记）。`API_BASE_URL` 支持环境变量注入（容器内指向
`http://api:8000/api`）。

### 3. Docker 一键部署

完整部署教程见 **`DEPLOY.md`**。核心命令：

```
copy .env.example .env        # 填 RAG_OPENAI_API_KEY
docker compose up -d --build  # 拉起 mysql/qdrant/api/webapp 四服务
docker compose exec api python main.py ingest /app/docs/xxx.pdf   # 灌知识库
# 前端 http://localhost:8501，API 文档 http://localhost:8000/docs
```

新增文件：`Dockerfile`（API）、`Dockerfile.webapp`（前端）、`docker-compose.yml`
（四服务编排 + 健康检查 + 命名卷持久化）、`entrypoint.sh`（等 MySQL + 预下载本地
模型 + 起 uvicorn）、`.dockerignore`。本地 Embedding/Rerank 模型缓存于 `model_cache`
卷；`.env` 不入镜像，敏感配置由 compose 注入。