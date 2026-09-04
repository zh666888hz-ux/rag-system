"""
API 集成测试：注册/登录/JWT 鉴权/会话管理/多轮对话/引用溯源。

说明：这些用例依赖真实 MySQL（与 .env 配置一致）与已导入的向量数据，
默认跳过；显式设置环境变量 RAG_TEST_MYSQL=1 时才会运行，用于端到端回归。
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

from config.settings import get_settings

# 默认跳过：集成测试需要真实 MySQL，避免在无数据库环境破坏 CI
pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_TEST_MYSQL") != "1",
    reason="需要真实 MySQL（设置 RAG_TEST_MYSQL=1 启用）",
)

USER = {"username": "apitest", "password": "secret123"}


@pytest.fixture(scope="module")
def client():
    from api.app import create_app

    app = create_app(get_settings())
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def token(client: TestClient) -> str:
    """注册+登录，返回可用 JWT。"""
    client.post("/api/auth/register", json=USER)
    resp = client.post("/api/auth/login", json=USER)
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_health(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_register_duplicate_conflict(client: TestClient):
    """重复注册应返回 409。用独立随机用户名，避免与 token fixture 的用户竞争。"""
    uname = f"dupuser_{os.getpid()}_{id(client)}"  # 保证每次运行唯一
    body = {"username": uname, "password": "secret123"}
    resp1 = client.post("/api/auth/register", json=body)
    assert resp1.status_code == 201, resp1.text
    resp2 = client.post("/api/auth/register", json=body)
    assert resp2.status_code == 409, resp2.text


def test_login_bad_password(client: TestClient):
    """错误密码应返回 401 且不泄露细节。"""
    resp = client.post("/api/auth/login", json={"username": USER["username"], "password": "wrong"})
    assert resp.status_code == 401


def test_me_with_token(client: TestClient, token: str):
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["username"] == USER["username"]


def test_me_without_token_401(client: TestClient):
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_me_with_tampered_token_401(client: TestClient, token: str):
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}x"})
    assert resp.status_code == 401


def test_conversation_crud(client: TestClient, token: str):
    """新建会话 → 列表 → 越权校验 → 删除。"""
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post("/api/conversations", json={}, headers=headers)
    assert resp.status_code == 201, resp.text
    conv_id = resp.json()["id"]

    resp = client.get("/api/conversations", headers=headers)
    assert resp.status_code == 200
    ids = [c["id"] for c in resp.json()]
    assert conv_id in ids

    # 另一用户不能访问该会话 → 404
    other = client.post("/api/auth/register", json={"username": "hacker01", "password": "secret123"})
    other_login = client.post("/api/auth/login", json={"username": "hacker01", "password": "secret123"})
    other_token = other_login.json()["access_token"]
    resp = client.get(f"/api/conversations/{conv_id}/messages", headers={"Authorization": f"Bearer {other_token}"})
    assert resp.status_code == 404

    resp = client.delete(f"/api/conversations/{conv_id}", headers=headers)
    assert resp.status_code == 204


def test_chat_multi_turn_and_traceability(client: TestClient, token: str):
    """多轮问答：验证引用编号溯源 + 历史持久化 + 多轮记忆。"""
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post("/api/conversations", json={}, headers=headers)
    conv_id = resp.json()["id"]

    # 第一问
    resp = client.post(
        "/api/chat",
        json={"conversation_id": conv_id, "message": "什么是 RAG？"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["conversation_id"] == conv_id
    assert data["answer"]
    assert data["num_sources"] >= 1
    assert data["memory_rounds"] >= 0
    # 溯源：sources 每项都有 ref_id；答案中的引用编号不越界
    ref_ids = [s["ref_id"] for s in data["sources"]]
    assert ref_ids == list(range(1, data["num_sources"] + 1))
    for ref in data["refs"]:
        assert ref in ref_ids

    # 第二问（触发多轮记忆）
    resp = client.post(
        "/api/chat",
        json={"conversation_id": conv_id, "message": "继续讲讲向量库的作用"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text

    # 历史持久化：应至少 4 条（2 轮 × user/assistant）
    resp = client.get(f"/api/conversations/{conv_id}/messages", headers=headers)
    assert resp.status_code == 200
    msgs = resp.json()
    assert len(msgs) >= 4
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user" and roles[1] == "assistant"
    # assistant 消息带引用溯源快照
    assistant = msgs[1]
    if assistant.get("sources"):
        assert "ref_id" in assistant["sources"][0]


def test_chat_invalid_conversation_404(client: TestClient, token: str):
    """对不存在的会话提问应返回 404。"""
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post(
        "/api/chat",
        json={"conversation_id": 999999, "message": "hello"},
        headers=headers,
    )
    assert resp.status_code == 404


def test_chat_stream_sse(client: TestClient, token: str):
    """SSE 流式接口：事件序列应包含 meta/source/delta/refs/done，并持久化历史。"""
    headers = {"Authorization": f"Bearer {token}"}
    conv = client.post("/api/conversations", json={}, headers=headers).json()
    cid = conv["id"]

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"conversation_id": cid, "message": "什么是 RAG？"},
        headers=headers,
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = []
        for line in resp.iter_lines():
            if line and line.startswith("data: "):
                events.append(json.loads(line[6:]))

    types = [e["type"] for e in events]
    # meta → source* → delta* → refs → done 的顺序约束
    assert types[0] == "meta"
    assert types.count("meta") == 1
    assert "source" in types
    assert "delta" in types
    assert types[-2] == "refs"
    assert types[-1] == "done"

    done = events[-1]
    assert done["answer"]
    assert done["num_sources"] >= 1
    assert done["conversation_id"] == cid
    ref_ids = [s["ref_id"] for s in done["sources"]]
    assert ref_ids == list(range(1, done["num_sources"] + 1))
    for ref in done.get("refs", []):
        assert ref in ref_ids

    # 流结束后历史已持久化（user + assistant 各一条）
    msgs = client.get(f"/api/conversations/{cid}/messages", headers=headers).json()
    assert len(msgs) >= 2
    assert msgs[-1]["role"] == "assistant"
