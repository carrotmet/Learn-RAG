#!/usr/bin/env python3
"""Phase 3 测试：意图识别 + 多路召回 + 融合

验收标准：
- 覆盖 2 个文档（自指学口播文稿_第三版.pdf、神经-心血管机制-20260301.pdf）
- 测试 Enrich（问题完善）
- 测试 IntentRouter（两层意图识别 + 多选混合）
- 测试 MultiRecall（多策略/多通道并行召回）
- 测试 Fusion（RRF 融合）
- 端到端：Intent -> Config -> Recall -> Fusion

运行方式：
    cd RAG教学
    export PYTHONPATH=backend/src
    source backend/venv/bin/activate
    python -m hybrid.test_phase3_intent_recall
"""

import os, sys, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend", "src"))

from hybrid.document_store import DocumentStore
from hybrid.registry import Registry
from hybrid.channels.vector import VectorChannel
from hybrid.channels.fts import FTSChannel
from hybrid.channels.graph import GraphChannel
from hybrid.retrieval.intent import IntentRouter
from hybrid.retrieval.enrich import Enrich
from hybrid.retrieval.multi_recall import MultiRecall
from hybrid.retrieval.fusion import Fusion

DB_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "rag_data.db")


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── 测试 1: Enrich ───────────────────────────────────────
def test_enrich():
    section("测试 1: Enrich（问题完善）")
    enrich = Enrich()

    cases = [
        ("什么是自指？", True),
        ("它", False),
        ("自指与", False),
        ("神经心血管机制详细说明", True),
    ]

    ok = 0
    for q, expected in cases:
        r = enrich.check_completeness(q)
        actual = r.get("complete", True)
        mark = "✅" if actual == expected else "⚠️"
        print(f"  {mark} '{q}' -> complete={actual}")
        if actual == expected:
            ok += 1

    print(f"\n  结果: {ok}/{len(cases)} 通过")
    return ok >= 3  # 完整度判断有主观性，允许 1 个偏差


# ── 测试 2: IntentRouter ─────────────────────────────────
def test_intent():
    section("测试 2: IntentRouter（意图识别）")
    router = IntentRouter()

    cases = [
        ("什么是自指？", "retrieve.concept"),
        ("自指与哥德尔定理的关系是什么？", "retrieve.relation"),
        ("自指在AI中的应用步骤", "retrieve.detail"),
        ("查找心血管系统的神经调节", "retrieve.keyword"),
        ("神经心血管机制是什么？", "retrieve.concept"),
        ("心血管和神经系统的区别", "retrieve.relation"),
    ]

    ok = 0
    for q, expected in cases:
        r = router.recognize(q)
        paths = [i["path"] for i in r.get("intents", [])]
        primary = r.get("primary_intent", "")
        match = expected in paths or expected == primary
        mark = "✅" if match else "⚠️"
        print(f"\n  {mark} '{q}'")
        print(f"      主意图: {primary}, 所有意图: {paths}, 需检索: {r.get('needs_retrieve')}")
        if match:
            ok += 1

    print(f"\n  结果: {ok}/{len(cases)} 通过")
    return ok >= 4  # 允许 1-2 个 LLM 判断偏差


# ── 测试 3: MultiRecall + Fusion ───────────────────────
def test_recall_fusion(store, vector, fts, graph):
    section("测试 3: MultiRecall + Fusion（多路召回 + RRF 融合）")
    recaller = MultiRecall(store, vector=vector, fts=fts, graph=graph)

    queries = ["自指", "心血管", "哥德尔", "神经调节"]
    ok_count = 0

    for q in queries:
        print(f"\n  查询: '{q}'")
        results = recaller.recall(q, strategies=["standard"], channels=["vector", "fts", "graph"], k=5)
        total = sum(len(v) for v in results.values())
        print(f"    召回总数: {total}")
        for ch, items in results.items():
            if items:
                print(f"    [{ch}] {len(items)} 条")

        if total > 0:
            fusion = Fusion(weights={"vector": 0.5, "fts": 0.3, "graph": 0.2})
            fused = fusion.rrf(results, top_k=3)
            print(f"    融合 top-{len(fused)}:")
            for i, r in enumerate(fused):
                src = store.get_chunk(r["parent_chunk_id"])
                print(f"      [{i+1}] score={r['fusion_score']:.4f} source={src.source if src else 'unknown'}")
            ok_count += 1
        else:
            print(f"    无召回结果")

    print(f"\n  结果: {ok_count}/{len(queries)} 有召回")
    return ok_count > 0


# ── 测试 4: 端到端 ─────────────────────────────────────
def test_end_to_end(store, vector, fts, graph):
    section("测试 4: 端到端（Intent -> Config -> Recall -> Fusion）")
    router = IntentRouter()
    recaller = MultiRecall(store, vector=vector, fts=fts, graph=graph)

    queries = ["什么是自指？", "自指与哥德尔的关系", "神经心血管机制"]
    ok = 0

    for q in queries:
        print(f"\n  查询: '{q}'")
        intent_r = router.recognize(q)
        if not intent_r.get("needs_retrieve", False):
            print(f"    -> 不需要检索")
            continue

        config = IntentRouter.get_retrieve_config(intent_r.get("intents", []))
        print(f"    [Intent] {intent_r.get('primary_intent')}")
        print(f"    [Config] 策略={config['strategies']} 权重={config['weights']}")

        recall_r = recaller.recall_by_config(q, config)
        total = sum(len(v) for v in recall_r.values())
        print(f"    [Recall] 总命中: {total}")
        for ch, items in recall_r.items():
            if items:
                print(f"      [{ch}] {len(items)} 条")

        if total > 0:
            fusion = Fusion(weights=config["weights"])
            fused = fusion.rrf(recall_r, top_k=3)
            print(f"    [Fusion] top-{len(fused)}:")
            for i, r in enumerate(fused):
                src = store.get_chunk(r["parent_chunk_id"])
                print(f"      [{i+1}] score={r['fusion_score']:.4f} source={src.source if src else 'unknown'}")
            ok += 1
        else:
            print(f"    [Fusion] 无结果")

    print(f"\n  结果: {ok}/3 通过")
    return ok > 0


# ── 主程序 ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Phase 3: 意图识别 + 多路召回 + 融合")
    print("="*60)
    print(f"\n  DB: {DB_PATH}")
    print(f"  Exists: {os.path.exists(DB_PATH)}")

    store = DocumentStore(db_path=DB_PATH)
    stats = store.get_stats()
    print(f"  Stats: chunks={stats['chunks']}, derivatives={stats['derivatives']}, indexed={stats['indexed']}")

    if stats['chunks'] == 0:
        print("\n  ❌ 数据库为空，请先索引文档")
        return 1

    print(f"\n  已注册策略: {', '.join(Registry.list_strategies())}")

    print("\n  [初始化组件...]")
    vector = VectorChannel()
    fts = FTSChannel(db_path=DB_PATH)
    graph = GraphChannel()
    print("  ✅ 组件就绪")

    results = {}
    results["enrich"] = test_enrich()
    results["intent"] = test_intent()
    results["recall_fusion"] = test_recall_fusion(store, vector, fts, graph)
    results["end_to_end"] = test_end_to_end(store, vector, fts, graph)

    section("测试汇总")
    for name, ok in results.items():
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  总计: {passed}/{total} 通过")

    if passed == total:
        print("\n  🎉 所有测试通过！Phase 3 验收合格。")
        return 0
    else:
        print(f"\n  ⚠️ 部分测试未通过。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
