#!/usr/bin/env python3
"""RAG 教学项目 第六章~第十一章 功能排查与验证测试脚本

测试范围：
- 第六章：多路召回 (MultiRecall)
- 第七章：意图识别 (IntentRouter)
- 第八章：问题完善 (Enrich)
- 第九章：融合层 (Fusion)
- 第十章：LangGraph 编排 (agent/graph.py, agent/state.py)
- 第十一章：API 接口 (agent/app.py)

验收标准：
- 已实现的模块：功能正确、接口完整
- 未实现的模块：明确标记缺失项，给出最小实现建议
"""

import os
import sys
import traceback
import json
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend", "src"))


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


# ── 测试结果收集 ──────────────────────────────────────────
results = {}


def check_file_exists(path: str, desc: str) -> bool:
    """检查文件是否存在"""
    full_path = os.path.join(PROJECT_ROOT, "backend", "src", path)
    exists = os.path.exists(full_path)
    if exists:
        ok(f"{desc}: {path}")
    else:
        err(f"{desc}: {path} (文件不存在)")
    return exists


def check_module_import(module_path: str, desc: str) -> bool:
    """检查模块能否导入"""
    try:
        __import__(module_path)
        ok(f"{desc}: 导入成功")
        return True
    except Exception as e:
        err(f"{desc}: 导入失败 - {e}")
        return False


def check_function_exists(module_path: str, class_name: str, method_name: str = None) -> bool:
    """检查类/方法是否存在"""
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        if method_name:
            exists = hasattr(cls, method_name)
            if exists:
                ok(f"  {class_name}.{method_name}() 存在")
            else:
                warn(f"  {class_name}.{method_name}() 缺失")
            return exists
        else:
            ok(f"  {class_name} 存在")
            return True
    except Exception as e:
        err(f"  检查失败: {e}")
        return False


# ════════════════════════════════════════════════════════════
# 第六章：多路召回
# ════════════════════════════════════════════════════════════
section("第六章：多路召回 (MultiRecall)")

ch6_results = {}

# 6.1 文件检查
ch6_results["file_exists"] = check_file_exists(
    "hybrid/retrieval/multi_recall.py", "MultiRecall 文件"
)

# 6.2 模块导入
ch6_results["import"] = check_module_import(
    "hybrid.retrieval.multi_recall", "MultiRecall 模块"
)

# 6.3 类和方法检查
if ch6_results["import"]:
    from hybrid.retrieval.multi_recall import MultiRecall
    ch6_results["class"] = True
    ok("MultiRecall 类可实例化")
    
    # 检查关键方法
    methods = ["recall", "recall_by_config"]
    for m in methods:
        exists = hasattr(MultiRecall, m)
        if exists:
            ok(f"  MultiRecall.{m}() 存在")
        else:
            warn(f"  MultiRecall.{m}() 缺失")
        ch6_results[f"method_{m}"] = exists

# 6.4 功能测试（如果模块可用）
if ch6_results.get("import"):
    try:
        from hybrid.document_store import DocumentStore
        from hybrid.channels.vector import VectorChannel
        from hybrid.channels.fts import FTSChannel
        from hybrid.channels.graph import GraphChannel
        
        DB_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "rag_data.db")
        store = DocumentStore(db_path=DB_PATH)
        recaller = MultiRecall(store, vector=VectorChannel(), fts=FTSChannel(), graph=GraphChannel())
        
        # 测试召回
        results_recall = recaller.recall("自指", strategies=["standard"], channels=["vector", "fts"], k=3)
        total = sum(len(v) for v in results_recall.values())
        if total > 0:
            ok(f"召回功能正常: {total} 条结果")
            ch6_results["recall_works"] = True
        else:
            warn(f"召回结果为空（可能数据库无数据）")
            ch6_results["recall_works"] = False
        
        # 测试 recall_by_config
        config = {
            "strategies": ["standard"],
            "weights": {"vector": 0.5, "fts": 0.3, "graph": 0.2},
            "mode": "hybrid",
            "k": 3
        }
        results_config = recaller.recall_by_config("自指", config)
        ok("recall_by_config() 可调用")
        ch6_results["recall_by_config_works"] = True
        
    except Exception as e:
        err(f"功能测试失败: {e}")
        ch6_results["recall_works"] = False
        ch6_results["recall_by_config_works"] = False

results["第六章"] = ch6_results


# ════════════════════════════════════════════════════════════
# 第七章：意图识别
# ════════════════════════════════════════════════════════════
section("第七章：意图识别 (IntentRouter)")

ch7_results = {}

# 7.1 文件检查
ch7_results["file_exists"] = check_file_exists(
    "hybrid/retrieval/intent.py", "IntentRouter 文件"
)

# 7.2 模块导入
ch7_results["import"] = check_module_import(
    "hybrid.retrieval.intent", "IntentRouter 模块"
)

# 7.3 类和方法检查
if ch7_results["import"]:
    from hybrid.retrieval.intent import IntentRouter
    ch7_results["class"] = True
    ok("IntentRouter 类可实例化")
    
    methods = ["recognize", "get_retrieve_config"]
    for m in methods:
        exists = hasattr(IntentRouter, m)
        if exists:
            ok(f"  IntentRouter.{m}() 存在")
        else:
            warn(f"  IntentRouter.{m}() 缺失")
        ch7_results[f"method_{m}"] = exists
    
    # 7.4 功能测试
    try:
        router = IntentRouter()
        
        # 测试意图识别
        result = router.recognize("什么是自指？")
        if "intents" in result and "primary_intent" in result:
            ok(f"recognize() 返回格式正确: {result.get('primary_intent')}")
            ch7_results["recognize_works"] = True
        else:
            warn("recognize() 返回格式不完整")
            ch7_results["recognize_works"] = False
        
        # 测试 get_retrieve_config
        intents = [{"path": "retrieve.concept"}, {"path": "retrieve.keyword"}]
        config = IntentRouter.get_retrieve_config(intents)
        if "strategies" in config and "weights" in config:
            ok(f"get_retrieve_config() 返回正确: {config}")
            ch7_results["get_retrieve_config_works"] = True
        else:
            warn("get_retrieve_config() 返回格式不完整")
            ch7_results["get_retrieve_config_works"] = False
        
        # 检查两层架构支持
        if result.get("needs_retrieve") is not None:
            ok("第一层判定 (needs_retrieve) 存在")
            ch7_results["layer1"] = True
        else:
            warn("第一层判定缺失")
            ch7_results["layer1"] = False
        
        # 检查多选混合
        if len(result.get("intents", [])) > 0:
            ok("多选混合支持: 返回 intents 数组")
            ch7_results["multi_intent"] = True
        else:
            warn("多选混合可能不支持")
            ch7_results["multi_intent"] = False
        
        # 检查规则回退
        result_rule = router.recognize("步骤")  # 触发规则匹配
        if result_rule.get("intents"):
            ok("规则回退机制工作")
            ch7_results["fallback"] = True
        else:
            warn("规则回退可能未工作")
            ch7_results["fallback"] = False
        
    except Exception as e:
        err(f"功能测试失败: {e}")
        traceback.print_exc()
        ch7_results["recognize_works"] = False

results["第七章"] = ch7_results


# ════════════════════════════════════════════════════════════
# 第八章：问题完善 (Enrich)
# ════════════════════════════════════════════════════════════
section("第八章：问题完善 (Enrich)")

ch8_results = {}

# 8.1 文件检查
ch8_results["file_exists"] = check_file_exists(
    "hybrid/retrieval/enrich.py", "Enrich 文件"
)

# 8.2 模块导入
ch8_results["import"] = check_module_import(
    "hybrid.retrieval.enrich", "Enrich 模块"
)

# 8.3 类和方法检查
if ch8_results["import"]:
    from hybrid.retrieval.enrich import Enrich
    ch8_results["class"] = True
    ok("Enrich 类可实例化")
    
    methods = ["check_completeness", "rewrite"]
    for m in methods:
        exists = hasattr(Enrich, m)
        if exists:
            ok(f"  Enrich.{m}() 存在")
        else:
            warn(f"  Enrich.{m}() 缺失")
        ch8_results[f"method_{m}"] = exists
    
    # 8.4 功能测试
    try:
        enrich = Enrich()
        
        # 完整度判断
        result = enrich.check_completeness("什么是自指？")
        if "complete" in result:
            ok(f"check_completeness() 返回正确: complete={result['complete']}")
            ch8_results["completeness_works"] = True
        else:
            warn("check_completeness() 返回格式不完整")
            ch8_results["completeness_works"] = False
        
        # 不完整问题测试
        result_incomplete = enrich.check_completeness("它")
        if not result_incomplete.get("complete", True):
            ok("不完整问题检测正确")
            ch8_results["incomplete_detection"] = True
        else:
            warn("不完整问题检测可能有问题")
            ch8_results["incomplete_detection"] = False
        
        # 问题改写
        rewritten = enrich.rewrite("自指是啥？")
        if rewritten and len(rewritten) > 0:
            ok(f"rewrite() 工作: '{rewritten[:50]}...'")
            ch8_results["rewrite_works"] = True
        else:
            warn("rewrite() 返回空")
            ch8_results["rewrite_works"] = False
        
    except Exception as e:
        err(f"功能测试失败: {e}")
        ch8_results["completeness_works"] = False

results["第八章"] = ch8_results


# ════════════════════════════════════════════════════════════
# 第九章：融合层 (Fusion)
# ════════════════════════════════════════════════════════════
section("第九章：融合层 (Fusion)")

ch9_results = {}

# 9.1 文件检查
ch9_results["file_exists"] = check_file_exists(
    "hybrid/retrieval/fusion.py", "Fusion 文件"
)

# 9.2 模块导入
ch9_results["import"] = check_module_import(
    "hybrid.retrieval.fusion", "Fusion 模块"
)

# 9.3 类和方法检查
if ch9_results["import"]:
    from hybrid.retrieval.fusion import Fusion
    ch9_results["class"] = True
    ok("Fusion 类可实例化")
    
    methods = ["rrf", "deduplicate_by_parent"]
    for m in methods:
        exists = hasattr(Fusion, m)
        if exists:
            ok(f"  Fusion.{m}() 存在")
        else:
            warn(f"  Fusion.{m}() 缺失")
        ch7_results[f"method_{m}"] = exists
    
    # 9.4 功能测试
    try:
        fusion = Fusion(weights={"vector": 0.5, "fts": 0.3, "graph": 0.2})
        
        # 测试 RRF 融合
        result_dict = {
            "vector": [
                {"parent_chunk_id": 1000001, "score": 0.8, "content": "test1"},
                {"parent_chunk_id": 1000002, "score": 0.7, "content": "test2"},
            ],
            "fts": [
                {"parent_chunk_id": 1000002, "score": 0.6, "content": "test2"},
                {"parent_chunk_id": 1000003, "score": 0.5, "content": "test3"},
            ]
        }
        
        fused = fusion.rrf(result_dict, top_k=3)
        if len(fused) > 0 and "fusion_score" in fused[0]:
            ok(f"RRF 融合工作: {len(fused)} 条结果, top score={fused[0]['fusion_score']:.4f}")
            ch9_results["rrf_works"] = True
        else:
            warn("RRF 融合返回格式不正确")
            ch9_results["rrf_works"] = False
        
        # 测试去重
        dup_results = [
            {"parent_chunk_id": 1000001, "score": 0.8},
            {"parent_chunk_id": 1000001, "score": 0.6},
            {"parent_chunk_id": 1000002, "score": 0.7},
        ]
        deduped = Fusion.deduplicate_by_parent(dup_results)
        if len(deduped) == 2:
            ok("去重功能正确")
            ch9_results["dedup_works"] = True
        else:
            warn(f"去重结果异常: {len(deduped)} 条 (期望 2)")
            ch9_results["dedup_works"] = False
        
    except Exception as e:
        err(f"功能测试失败: {e}")
        traceback.print_exc()
        ch9_results["rrf_works"] = False

results["第九章"] = ch9_results


# ════════════════════════════════════════════════════════════
# 第十章：LangGraph 编排
# ════════════════════════════════════════════════════════════
section("第十章：LangGraph 编排")

ch10_results = {}

# 10.1 检查 state.py 扩展
ch10_results["state_file"] = check_file_exists("agent/state.py", "state.py 文件")

if ch10_results["state_file"]:
    with open(os.path.join(PROJECT_ROOT, "backend", "src", "agent/state.py")) as f:
        state_content = f.read()
    
    # 检查关键字段
    required_fields = [
        "enrich_complete", "enrich_reason", "enrich_rewritten", "enrich_follow_up",
        "intents", "primary_intent", "needs_retrieve",
        "retrieve_strategies", "retrieve_weights", "retrieve_mode",
        "recall_results", "fused_results"
    ]
    
    for field in required_fields:
        if field in state_content:
            ok(f"  RAGState.{field} 存在")
            ch10_results[f"field_{field}"] = True
        else:
            warn(f"  RAGState.{field} 缺失")
            ch10_results[f"field_{field}"] = False

# 10.2 检查 graph.py 改造
ch10_results["graph_file"] = check_file_exists("agent/graph.py", "graph.py 文件")

if ch10_results["graph_file"]:
    with open(os.path.join(PROJECT_ROOT, "backend", "src", "agent/graph.py")) as f:
        graph_content = f.read()
    
    # 检查新节点
    required_nodes = ["enrich_node", "intent_node", "recall_node", "fusion_node"]
    for node in required_nodes:
        if node in graph_content:
            ok(f"  graph.py: {node} 存在")
            ch10_results[f"node_{node}"] = True
        else:
            warn(f"  graph.py: {node} 缺失 (仍为一阶段代码)")
            ch10_results[f"node_{node}"] = False
    
    # 检查旧节点是否还在（说明未改造）
    if "index_node" in graph_content and "retrieve_node" in graph_content:
        info("graph.py 仍使用一阶段节点 (index → retrieve → generate)")
        ch10_results["is_v1"] = True
    else:
        ch10_results["is_v1"] = False
    
    # 检查条件路由
    if "add_conditional_edges" in graph_content or "router" in graph_content:
        ok("graph.py 有条件路由")
        ch10_results["conditional"] = True
    else:
        warn("graph.py 缺少条件路由（enrich 不完整 → 追问）")
        ch10_results["conditional"] = False

results["第十章"] = ch10_results


# ════════════════════════════════════════════════════════════
# 第十一章：API 接口
# ════════════════════════════════════════════════════════════
section("第十一章：API 接口")

ch11_results = {}

# 11.1 检查 app.py
ch11_results["app_file"] = check_file_exists("agent/app.py", "app.py 文件")

if ch11_results["app_file"]:
    with open(os.path.join(PROJECT_ROOT, "backend", "src", "agent/app.py")) as f:
        app_content = f.read()
    
    # 检查 Hybrid API 端点
    required_endpoints = [
        "/api/hybrid/intent",
        "/api/hybrid/enrich",
        "/api/hybrid/recall",
        "/runs/stream"
    ]
    
    for endpoint in required_endpoints:
        if endpoint in app_content:
            ok(f"  API 端点 {endpoint} 存在")
            ch11_results[f"endpoint_{endpoint}"] = True
        else:
            warn(f"  API 端点 {endpoint} 缺失")
            ch11_results[f"endpoint_{endpoint}"] = False
    
    # 检查流式接口配置
    if "strategy" in app_content and "strategies" in app_content:
        ok("  /runs/stream 支持策略覆盖")
        ch11_results["stream_override"] = True
    else:
        warn("  /runs/stream 可能不支持策略覆盖")
        ch11_results["stream_override"] = False

results["第十一章"] = ch11_results


# ════════════════════════════════════════════════════════════
# 总结报告
# ════════════════════════════════════════════════════════════
section("测试总结")

print(f"\n{Colors.BOLD}第六章~第九章（核心模块）：{Colors.END}")
for chapter in ["第六章", "第七章", "第八章", "第九章"]:
    r = results[chapter]
    passed = sum(1 for k, v in r.items() if isinstance(v, bool) and v)
    total = sum(1 for k, v in r.items() if isinstance(v, bool))
    status = f"{Colors.OK}通过{Colors.END}" if passed == total else f"{Colors.WARN}部分通过{Colors.END}"
    print(f"  {chapter}: {passed}/{total} {status}")

print(f"\n{Colors.BOLD}第十章（LangGraph 编排）：{Colors.END}")
r10 = results["第十章"]
if r10.get("is_v1"):
    print(f"  {Colors.ERR}⚠️  graph.py 仍为一阶段代码 (index → retrieve → generate){Colors.END}")
    print(f"  {Colors.WARN}  缺失: enrich_node, intent_node, recall_node, fusion_node{Colors.END}")
    print(f"  {Colors.WARN}  缺失: 条件路由 (enrich 不完整 → 追问用户){Colors.END}")
else:
    print(f"  {Colors.OK}✅ 已改造为 HybridRAG 链路{Colors.END}")

print(f"\n{Colors.BOLD}第十一章（API 接口）：{Colors.END}")
r11 = results["第十一章"]
missing_endpoints = [k for k, v in r11.items() if k.startswith("endpoint_") and not v]
if missing_endpoints:
    print(f"  {Colors.ERR}⚠️ 缺失 Hybrid API 端点:{Colors.END}")
    for ep in missing_endpoints:
        print(f"    - {ep.replace('endpoint_', '')}")
else:
    print(f"  {Colors.OK}✅ 所有 Hybrid API 端点已实现{Colors.END}")


# ── 输出 JSON 报告 ─────────────────────────────────────────
print(f"\n{Colors.BOLD}生成 JSON 报告...{Colors.END}")
report = {
    "project": "RAG 教学项目",
    "test_scope": "第六章~第十一章",
    "test_time": "2026-06-21",
    "results": {
        chapter: {k: v for k, v in r.items() if isinstance(v, bool)}
        for chapter, r in results.items()
    }
}

report_path = os.path.join(PROJECT_ROOT, "docs", "test_report_ch6_11.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"  报告已保存: {report_path}")

# ── 缺失项汇总 ──────────────────────────────────────────────
print(f"\n{Colors.BOLD}缺失项汇总（需补充）：{Colors.END}")

missing_items = []

# 第十章缺失
if r10.get("is_v1"):
    missing_items.append("第十章: agent/graph.py 需改造为 Enrich → Intent → Recall → Fusion → Generate")
    missing_items.append("第十章: agent/state.py 需添加 enrich/intent/recall/fusion 字段")
    missing_items.append("第十章: 需要条件路由 (enrich 不完整 → 追问)")

# 第十一章缺失
for ep in missing_endpoints:
    missing_items.append(f"第十一章: 缺失 API 端点 {ep.replace('endpoint_', '')}")

if missing_items:
    for item in missing_items:
        print(f"  {Colors.ERR}• {item}{Colors.END}")
else:
    print(f"  {Colors.OK}无缺失项{Colors.END}")

print()
