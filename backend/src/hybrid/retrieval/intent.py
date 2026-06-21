"""意图识别：两层架构 + 纯路由表设计

核心设计：
- 第一层：判定方向（retrieve / skill / tool）
- 第二层：判定子类型（keyword, concept, relation, detail）
- 多选混合：一个查询可同时命中多个子意图
- 纯路由表：只贴标签，不与下游耦合
- 复用一阶段 OpenRouterLLM，通过 INTENT_MODEL 环境变量配置

回退机制：
- LLM 调用失败或超时 → 规则匹配回退
- 规则回退基于关键词正则匹配
"""

import os
import re
import json
from typing import List, Dict, Optional


class IntentRouter:
    """意图识别器：纯路由表设计，只贴标签，不与下游耦合

    复用一阶段 OpenRouterLLM，模型通过 INTENT_MODEL 环境变量配置。
    """

    # 内置意图路由表（YAML 未配置时的默认表）
    DEFAULT_INTENTS = {
        "retrieve.keyword": {
            "description": "基于关键词的精确检索，适合已知实体名、术语、编号",
            "tags": ["retrieve", "search", "exact-match"]
        },
        "retrieve.concept": {
            "description": "概念定义检索，回答'是什么'类问题",
            "tags": ["retrieve", "concept", "definition"]
        },
        "retrieve.relation": {
            "description": "关系图谱检索，查找实体间关联、区别、对比",
            "tags": ["retrieve", "relation", "graph"]
        },
        "retrieve.detail": {
            "description": "细节定位检索，查找具体步骤、方法、精确事实",
            "tags": ["retrieve", "detail", "precision"]
        },
        # 三期预留
        "skill.global": {
            "description": "全局技能调用，跨项目通用能力",
            "tags": ["skill", "global", "cross-project"]
        },
        "skill.sql": {
            "description": "结构化数据查询，通过 SQL 检索",
            "tags": ["skill", "sql", "structured"]
        },
        "tool.report": {
            "description": "报表生成工具",
            "tags": ["tool", "report", "output"]
        },
    }

    # 规则回退关键词映射
    RULE_PATTERNS = {
        "retrieve.relation": re.compile(r"(关系|关联|区别|对比|联系|不同|差异|与.*的关系|和.*的区别)"),
        "retrieve.detail": re.compile(r"(步骤|流程|方法|具体|详细|怎么做|如何|操作|过程|机制|原理)"),
        "retrieve.concept": re.compile(r"(是什么|定义|解释|概念|含义|意思|何为|什么是)"),
        "retrieve.keyword": re.compile(r"(查|找|搜索|查找|列出|列举|所有|全部)"),
    }

    def __init__(self, llm=None):
        self.intents = self.DEFAULT_INTENTS.copy()
        self._llm = llm

    def _get_llm(self):
        """延迟初始化 LLM，避免导入时出错"""
        if self._llm is not None:
            return self._llm

        try:
            from agent.llm import OpenRouterLLM
            intent_model = os.getenv("INTENT_MODEL", os.getenv("DEFAULT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"))
            self._llm = OpenRouterLLM(model_id=intent_model)
        except Exception as e:
            print(f"[IntentRouter] LLM 初始化失败: {e}")
            self._llm = None
        return self._llm

    def recognize(self, query: str, history: list = None) -> Dict:
        """识别用户意图

        Args:
            query: 用户查询字符串
            history: 对话历史（可选）

        Returns:
            {
                "intents": [{"path": "retrieve.concept", "confidence": "high", "reasoning": "..."}, ...],
                "primary_intent": "retrieve.concept",
                "needs_retrieve": True,  # 第一层判定
            }
        """
        # 尝试 LLM 识别
        llm_result = self._llm_recognize(query, history)
        if llm_result is not None:
            result = llm_result
        else:
            # LLM 失败，回退到规则匹配
            result = self._rule_fallback(query)

        # 第一层判定：是否需要检索
        paths = [i["path"] for i in result.get("intents", [])]
        result["needs_retrieve"] = any(p.startswith("retrieve.") for p in paths)

        return result

    def _llm_recognize(self, query: str, history: list = None) -> Optional[Dict]:
        """使用 LLM 识别意图"""
        llm = self._get_llm()
        if llm is None:
            return None

        intent_list = "\n".join(
            f"- {path}: {info['description']} [tags: {', '.join(info['tags'])}]"
            for path, info in self.intents.items()
        )

        prompt = f"""你是意图识别助手。请从以下意图列表中选择最匹配的意图。

可用意图：
{intent_list}

用户问题：{query}

请返回 JSON 格式（不要包含 markdown 代码块标记）：
{{
  "intents": [
    {{"path": "意图路径", "confidence": "high/medium/low", "reasoning": "选择理由"}}
  ],
  "primary_intent": "主意图路径"
}}

注意：
1. 子意图可多选（如同时命中 keyword 和 concept）
2. 只需要识别意图，不需要提取实体或参数
3. confidence 为 high 表示非常确定，medium 表示可能，low 表示不确定"""

        try:
            response = llm.generate(prompt, system="你是一个意图识别专家，只输出 JSON 格式的意图标签。")
            # 清理可能存在的 markdown 代码块
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            result = json.loads(response)
            # 验证意图路径有效性
            valid_intents = []
            for i in result.get("intents", []):
                if i.get("path") in self.intents:
                    valid_intents.append(i)
                else:
                    # 映射到最接近的意图
                    mapped = self._map_to_nearest_intent(i.get("path", ""))
                    if mapped:
                        i["path"] = mapped
                        valid_intents.append(i)

            if valid_intents:
                result["intents"] = valid_intents
                result["primary_intent"] = result.get("primary_intent", valid_intents[0]["path"])
                return result

        except Exception as e:
            print(f"[IntentRouter] LLM 识别失败: {e}")

        return None

    def _rule_fallback(self, query: str) -> Dict:
        """LLM 调用失败时的规则回退"""
        intents = []
        for path, pattern in self.RULE_PATTERNS.items():
            if pattern.search(query):
                intents.append({
                    "path": path,
                    "confidence": "medium",
                    "reasoning": "规则匹配"
                })

        if not intents:
            # 默认回退到 keyword
            intents.append({
                "path": "retrieve.keyword",
                "confidence": "low",
                "reasoning": "默认回退"
            })

        return {
            "intents": intents,
            "primary_intent": intents[0]["path"],
            "fallback": True,
        }

    def _map_to_nearest_intent(self, raw_path: str) -> Optional[str]:
        """将非标准路径映射到最接近的标准意图"""
        if raw_path.startswith("retrieve."):
            for valid in self.intents:
                if valid.startswith("retrieve."):
                    return valid
        if raw_path.startswith("skill."):
            for valid in self.intents:
                if valid.startswith("skill."):
                    return valid
        if raw_path.startswith("tool."):
            for valid in self.intents:
                if valid.startswith("tool."):
                    return valid
        return "retrieve.keyword"  # 最终默认

    @staticmethod
    def get_retrieve_config(intents: List[Dict]) -> Dict:
        """根据检索意图列表，生成召回配置（多意图混合）

        多个检索子意图同时命中时，合并策略和权重。

        Args:
            intents: 意图列表，每个包含 "path" 字段

        Returns:
            {
                "strategies": ["standard", "summary"],  # 多策略并行
                "weights": {"vector": 0.5, "fts": 0.3, "graph": 0.2},
                "mode": "hybrid",
                "k": 5
            }
        """
        # 子意图到配置的映射
        config_map = {
            "retrieve.keyword": {
                "strategies": ["standard"],
                "weights": {"vector": 0.2, "fts": 0.7, "graph": 0.1}
            },
            "retrieve.concept": {
                "strategies": ["summary", "standard"],
                "weights": {"vector": 0.7, "fts": 0.2, "graph": 0.1}
            },
            "retrieve.relation": {
                "strategies": ["standard"],
                "weights": {"vector": 0.3, "fts": 0.1, "graph": 0.6}
            },
            "retrieve.detail": {
                "strategies": ["parent_child", "standard"],
                "weights": {"vector": 0.5, "fts": 0.4, "graph": 0.1}
            },
        }

        # 合并多个意图的配置
        all_strategies = set()
        merged_weights = {"vector": 0, "fts": 0, "graph": 0}

        for intent in intents:
            path = intent.get("path", "")
            if path not in config_map:
                continue
            cfg = config_map[path]
            all_strategies.update(cfg["strategies"])
            for k, v in cfg["weights"].items():
                merged_weights[k] = max(merged_weights[k], v)  # 取最大值合并

        # 如果没有匹配到任何检索意图，使用默认配置
        if not all_strategies:
            return {
                "strategies": ["standard"],
                "weights": {"vector": 0.5, "fts": 0.3, "graph": 0.2},
                "mode": "hybrid",
                "k": 5
            }

        # 归一化权重
        total = sum(merged_weights.values())
        if total > 0:
            merged_weights = {k: round(v / total, 4) for k, v in merged_weights.items()}

        return {
            "strategies": list(all_strategies),
            "weights": merged_weights,
            "mode": "hybrid",
            "k": 5
        }
