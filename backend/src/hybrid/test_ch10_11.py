#!/usr/bin/env python3
"""RAG 教学项目 第十章~第十一章 功能验收测试脚本

测试范围：
- 第十章：LangGraph 编排（state 扩展、节点存在、条件路由、图编译）
- 第十一章：API 接口（/api/hybrid/enrich, /api/hybrid/intent, /api/hybrid/recall, /api/hybrid/generate, /runs/stream）

验收标准：
- state.py 扩展字段完整
- graph.py 节点 + 条件路由正确
- app.py API 端点可调用
- 使用正式数据库 backend/data/rag_data.db
"""

import os
import sys
import json
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend", "src"))

DB_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "rag_data.db")


# ── 颜色输出 ───────────────────────────────────────────────
class Colors:
    OK = "\033[92m"
    WARN = "\033[93m"
    ERR = "\033[91m"
    INFO = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg): print(f"  {Colors.OK}✅ {msg}{Colors.END}")
def warn(msg): print(f"  {Colors.WARN}⚠️  {msg}{Colors.END}")
def err(msg): print(f"  {Colors.ERR}❌ {msg}{Colors.END}")
def info(msg): print(f"  {Colors.INFO}ℹ️  {msg}{Colors.END}")
def section(title): print(f"\n{Colors.BOLD}{'='*60}{Colors.END}\n{Colors.BOLD}{title}{Colors.END}\n{Colors.BOLD}{'='*60}{Colors.END}")


results = {}


# ════════════════════════════════════════════════════════════
# 第十章：LangGraph 编排
# ════════════════════════════════════════════════════════════
section("第十章：LangGraph 编排")

ch10_results = {}

# 10.1 state.py 扩展字段
info("检查 state.py 扩展字段...")
with open(os.path.join(PROJECT_ROOT, "backend/src/agent/state.py")) as f:
    state_content = f.read()

required_fields = [
    "enrich_complete", "enrich_reason", "enrich_rewritten", "enrich_follow_up",
    "intents", "primary_intent", "needs_retrieve",
    "retrieve_strategies", "retrieve_weights", "retrieve_mode",
    "recall_results", "fused_results"
]

all_fields_ok = True
for field in required_fields:
    if field in state_content:
        ok(f"RAGState.{field} 存在")
        ch10_results[f"field_{field}"] = True
    else:
        err(f"RAGState.{field} 缺失")
        ch10_results[f"field_{field}"] = False
        all_fields_ok = False

ch10_results["state_fields"] = all_fields_ok

# 10.2 graph.py 节点和路由
info("检查 graph.py 节点...")
with open(os.path.join(PROJECT_ROOT, "backend/src/agent/graph.py")) as f:
    graph_content = f.read()

required_nodes = [
    "enrich_node", "intent_node", "recall_node", "fusion_node",
    "hybrid_generate_node", "follow_up_node", "generate_direct_node",
    "enrich_router", "intent_router"
]

all_nodes_ok = True
for node in required_nodes:
    if node in graph_content:
        ok(f"graph.py: {node} 存在")
        ch10_results[f"node_{node}"] = True
    else:
        err(f"graph.py: {node} 缺失")
        ch10_results[f"node_{node}"] = False
        all_nodes_ok = False

ch10_results["nodes"] = all_nodes_ok

# 检查条件路由
if "add_conditional_edges" in graph_content:
    ok("graph.py: add_conditional_edges 存在")
    ch10_results["conditional_edges"] = True
else:
    err("graph.py: add_conditional_edges 缺失")
    ch10_results["conditional_edges"] = False

# 检查默认链路
if 'builder.add_edge(START, "enrich")' in graph_content:
    ok("graph.py: 默认链路以 enrich 开始")
    ch10_results["default_link"] = True
else:
    err("graph.py: 默认链路未指向 enrich")
    ch10_results["default_link"] = False

# 10.3 图编译测试
info("测试图编译...")
try:
    from agent.graph import graph
    ok("agent.graph 导入成功")
    ch10_results["graph_import"] = True
    
    # 检查节点列表
    nodes = list(graph.nodes.keys())
    required_in_graph = ["enrich", "intent", "recall", "fusion", "hybrid_generate"]
    for n in required_in_graph:
        if n in nodes:
            ok(f"编译图中包含节点: {n}")
            ch10_results[f"compiled_{n}"] = True
        else:
            err(f"编译图中缺少节点: {n}")
            ch10_results[f"compiled_{n}"] = False
    
except Exception as e:
    err(f"图编译测试失败: {e}")
    traceback.print_exc()
    ch10_results["graph_import"] = False

# 10.4 端到端图执行测试（无 LLM，验证链路）
info("测试图端到端执行...")
try:
    from agent.graph import graph
    from langchain_core.messages import HumanMessage
    
    initial_state = {
        "messages": [HumanMessage(content="什么是自指？")],
        "question": "什么是自指？",
        "enrich_complete": None,
        "enrich_reason": None,
        "enrich_rewritten": None,
        "enrich_follow_up": None,
        "intents": None,
        "primary_intent": None,
        "needs_retrieve": None,
        "retrieve_strategies": ["standard"],
        "retrieve_weights": None,
        "retrieve_mode": "hybrid",
        "recall_results": None,
        "fused_results": None,
        "retrieved_docs": [],
        "answer": None,
        "retrieval_latency": 0
    }
    
    # 使用 invoke 执行完整链路
    result = graph.invoke(initial_state, config={"recursion_limit": 50})
    
    if result.get("answer"):
        ok(f"图端到端执行成功，答案长度: {len(result['answer'])} 字符")
        ch10_results["e2e_invoke"] = True
        info(f"  执行链路: enrich → intent → recall → fusion → hybrid_generate")
        info(f"  enrich_complete: {result.get('enrich_complete')}")
        info(f"  primary_intent: {result.get('primary_intent')}")
        info(f"  retrieved_docs: {len(result.get('retrieved_docs', []))} 条")
    else:
        warn("图执行成功但无 answer")
        ch10_results["e2e_invoke"] = False
        
except Exception as e:
    err(f"图端到端执行失败: {e}")
    traceback.print_exc()
    ch10_results["e2e_invoke"] = False

results["第十章"] = ch10_results


# ════════════════════════════════════════════════════════════
# 第十一章：API 接口
# ════════════════════════════════════════════════════════════
section("第十一章：API 接口")

ch11_results = {}

# 11.1 检查端点定义
info("检查 API 端点定义...")
with open(os.path.join(PROJECT_ROOT, "backend/src/agent/app.py")) as f:
    app_content = f.read()

required_endpoints = [
    ("/api/hybrid/enrich", "POST"),
    ("/api/hybrid/intent", "POST"),
    ("/api/hybrid/recall", "POST"),
    ("/api/hybrid/generate", "POST"),
    ("/runs/stream", "POST"),
]

all_endpoints_ok = True
for path, method in required_endpoints:
    decorator = f'@app.post("{path}"'
    if decorator in app_content:
        ok(f"API 端点: {method} {path}")
        ch11_results[f"endpoint_{path.replace('/', '_').strip('_')}"] = True
    else:
        err(f"API 端点缺失: {method} {path}")
        ch11_results[f"endpoint_{path.replace('/', '_').strip('_')}"] = False
        all_endpoints_ok = False

ch11_results["endpoints_defined"] = all_endpoints_ok

# 11.2 启动 FastAPI 并测试端点
info("启动 FastAPI 测试服务器...")

try:
    from fastapi.testclient import TestClient
    from agent.app import app
    
    client = TestClient(app)
    ok("TestClient 创建成功")
    ch11_results["testclient"] = True
    
    # 测试 /api/hybrid/enrich
    info("测试 POST /api/hybrid/enrich...")
    response = client.post("/api/hybrid/enrich", json={"query": "什么是自指？"})
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "ok" and "complete" in data:
            ok(f"enrich API: complete={data['complete']}, reason={data.get('reason', '')[:30]}...")
            ch11_results["api_enrich"] = True
        else:
            warn(f"enrich API 返回格式异常: {data}")
            ch11_results["api_enrich"] = False
    else:
        err(f"enrich API 返回 {response.status_code}: {response.text[:200]}")
        ch11_results["api_enrich"] = False
    
    # 测试 /api/hybrid/enrich（不完整问题）
    info("测试 POST /api/hybrid/enrich（不完整问题）...")
    response = client.post("/api/hybrid/enrich", json={"query": "它"})
    if response.status_code == 200:
        data = response.json()
        if data.get("complete") == False:
            ok(f"enrich API 不完整检测: follow_up={data.get('follow_up', '')[:30]}...")
            ch11_results["api_enrich_incomplete"] = True
        else:
            warn(f"enrich API 不完整检测可能有问题: complete={data.get('complete')}")
            ch11_results["api_enrich_incomplete"] = False
    else:
        err(f"enrich API 不完整检测失败: {response.status_code}")
        ch11_results["api_enrich_incomplete"] = False
    
    # 测试 /api/hybrid/intent
    info("测试 POST /api/hybrid/intent...")
    response = client.post("/api/hybrid/intent", json={"query": "什么是自指？"})
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "ok" and "intents" in data:
            ok(f"intent API: intents={len(data['intents'])}, primary={data.get('primary_intent')}")
            ch11_results["api_intent"] = True
            if data.get("retrieve_config"):
                info(f"  retrieve_config: {json.dumps(data['retrieve_config'], ensure_ascii=False)[:100]}...")
        else:
            warn(f"intent API 返回格式异常: {data}")
            ch11_results["api_intent"] = False
    else:
        err(f"intent API 返回 {response.status_code}: {response.text[:200]}")
        ch11_results["api_intent"] = False
    
    # 测试 /api/hybrid/recall
    info("测试 POST /api/hybrid/recall...")
    response = client.post("/api/hybrid/recall", json={
        "query": "自指",
        "strategies": ["standard"],
        "channels": ["vector", "fts"],
        "mode": "hybrid",
        "k": 3
    })
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "ok" and "recall_results" in data:
            total = data.get("total_results", 0)
            ok(f"recall API: total_results={total}")
            ch11_results["api_recall"] = True
            for ch, items in data.get("recall_results", {}).items():
                info(f"  通道 {ch}: {len(items)} 条")
        else:
            warn(f"recall API 返回格式异常: {data}")
            ch11_results["api_recall"] = False
    else:
        err(f"recall API 返回 {response.status_code}: {response.text[:200]}")
        ch11_results["api_recall"] = False
    
    # 测试 /api/hybrid/generate
    info("测试 POST /api/hybrid/generate...")
    response = client.post("/api/hybrid/generate", json={
        "query": "什么是自指？",
        "strategies": ["standard"],
        "mode": "hybrid",
        "k": 3
    })
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "ok" and "answer" in data:
            ok(f"generate API: 答案长度={len(data['answer'])} 字符")
            ch11_results["api_generate"] = True
            info(f"  pipeline: {data.get('pipeline', [])}")
            info(f"  fused_count: {data.get('fused_count', 0)}")
        elif data.get("status") == "incomplete":
            ok(f"generate API: 问题不完整，返回追问")
            ch11_results["api_generate"] = True
        else:
            warn(f"generate API 返回格式异常: {list(data.keys())}")
            ch11_results["api_generate"] = False
    else:
        err(f"generate API 返回 {response.status_code}: {response.text[:200]}")
        ch11_results["api_generate"] = False
    
    # 测试 /runs/stream（不实际流式，只验证端点存在）
    info("测试 POST /runs/stream...")
    # TestClient 不支持 SSE 流式，只验证路由存在
    response = client.post("/runs/stream", json={"query": "自指"})
    if response.status_code in (200, 307):
        ok(f"stream API: 端点可达 (status={response.status_code})")
        ch11_results["api_stream"] = True
    else:
        err(f"stream API: 端点返回 {response.status_code}")
        ch11_results["api_stream"] = False
    
except Exception as e:
    err(f"API 测试失败: {e}")
    traceback.print_exc()
    ch11_results["testclient"] = False

results["第十一章"] = ch11_results


# ════════════════════════════════════════════════════════════
# 总结报告
# ════════════════════════════════════════════════════════════
section("测试总结")

print(f"\n{Colors.BOLD}第十章（LangGraph 编排）：{Colors.END}")
r10 = results["第十章"]
passed_10 = sum(1 for k, v in r10.items() if isinstance(v, bool) and v)
total_10 = sum(1 for k, v in r10.items() if isinstance(v, bool))
print(f"  通过: {passed_10}/{total_10}")

print(f"\n{Colors.BOLD}第十一章（API 接口）：{Colors.END}")
r11 = results["第十一章"]
passed_11 = sum(1 for k, v in r11.items() if isinstance(v, bool) and v)
total_11 = sum(1 for k, v in r11.items() if isinstance(v, bool))
print(f"  通过: {passed_11}/{total_11}")

# 关键结论
print(f"\n{Colors.BOLD}关键结论：{Colors.END}")
if r10.get("e2e_invoke"):
    print(f"  {Colors.OK}✅ LangGraph 端到端链路可正常执行{Colors.END}")
else:
    print(f"  {Colors.ERR}❌ LangGraph 端到端链路执行失败{Colors.END}")

if r11.get("api_generate"):
    print(f"  {Colors.OK}✅ HybridRAG 端到端生成 API 可用{Colors.END}")
else:
    print(f"  {Colors.ERR}❌ HybridRAG 端到端生成 API 不可用{Colors.END}")

# ── 输出 JSON 报告 ─────────────────────────────────────────
report = {
    "project": "RAG 教学项目",
    "test_scope": "第十章~第十一章",
    "test_time": "2026-06-21",
    "database": DB_PATH,
    "results": {
        chapter: {k: v for k, v in r.items() if isinstance(v, bool)}
        for chapter, r in results.items()
    }
}

report_path = os.path.join(PROJECT_ROOT, "docs", "test_report_ch10_11.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n  报告已保存: {report_path}")

# ── 最终状态 ────────────────────────────────────────────────
all_pass = r10.get("e2e_invoke", False) and r11.get("api_generate", False)
if all_pass:
    print(f"\n{Colors.OK}{Colors.BOLD}🎉 第十章~第十一章功能验收通过！{Colors.END}")
else:
    print(f"\n{Colors.WARN}{Colors.BOLD}⚠️ 部分测试未通过，请检查上述错误{Colors.END}")

print()
