"""Enrich：问题完善层

核心功能：
1. 完整度判断：用户问题是否完整？是否需要补充信息？
2. 问题改写：将口语化/模糊问题改写为结构化查询
3. 请求补充信息：问题不完整时，追问用户

定位：HybridRAG 的入口层，也是 HybridRAG 与 AgenticRAG 的桥梁。
"""

import re
from typing import Dict, Optional


class Enrich:
    """问题完善器

    支持 LLM 驱动和规则回退两种模式。
    """

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        """延迟初始化 LLM"""
        if self._llm is not None:
            return self._llm
        try:
            from agent.llm import OpenRouterLLM
            self._llm = OpenRouterLLM()
        except Exception as e:
            print(f"[Enrich] LLM 初始化失败: {e}")
            self._llm = None
        return self._llm

    def check_completeness(self, query: str, history: list = None) -> Dict:
        """判断问题完整度

        Args:
            query: 用户查询
            history: 对话历史（可选）

        Returns:
            {
                "complete": bool,          # 是否完整
                "reason": str,             # 判断理由
                "missing_info": list,      # 缺失信息项
                "rewritten_query": str,    # 改写后的问题（如完整）
                "follow_up_question": str,  # 追问问题（如不完整）
            }
        """
        # 先尝试规则判断（快速路径）
        rule_result = self._rule_completeness(query)
        if rule_result is not None:
            return rule_result

        # 尝试 LLM 判断
        llm_result = self._llm_completeness(query, history)
        if llm_result is not None:
            return llm_result

        # 最终回退：假设完整
        return {
            "complete": True,
            "reason": "默认假设完整（无法判断）",
            "missing_info": [],
            "rewritten_query": query,
            "follow_up_question": "",
        }

    def rewrite(self, query: str, history: list = None) -> str:
        """将口语化/模糊问题改写为结构化查询

        Args:
            query: 原始查询
            history: 对话历史（可选）

        Returns:
            改写后的查询字符串
        """
        llm = self._get_llm()
        if llm is None:
            return query

        prompt = f"""请将以下用户问题改写为更清晰的检索查询，保留核心意图。

原始问题：{query}

改写要求：
- 保留关键实体和概念
- 去除口语化表达
- 输出一条最简洁的检索查询

改写后："""

        try:
            rewritten = llm.generate(prompt, system="你是一个查询改写专家。")
            return rewritten.strip()
        except Exception:
            return query

    def _rule_completeness(self, query: str) -> Optional[Dict]:
        """规则判断问题完整度

        不完整的情况：
        - 问题过短（< 3 字）
        - 以代词开头（它/这个/那个）且无上下文
        - 问题以问号结尾但缺少主语
        - 明显是半截句子
        """
        query = query.strip()

        # 过短
        if len(query) < 3:
            return {
                "complete": False,
                "reason": "问题过短，无法判断意图",
                "missing_info": ["问题内容"],
                "rewritten_query": query,
                "follow_up_question": "您的问题似乎不完整，请提供更多细节。",
            }

        # 以代词开头（无上下文时视为不完整）
        if re.match(r"^(它|这个|那个|这|那|他|她|其)", query):
            return {
                "complete": False,
                "reason": "问题以代词开头，缺少指代对象",
                "missing_info": ["指代对象"],
                "rewritten_query": query,
                "follow_up_question": "您提到的'" + query[:2] + "'是指什么？请具体说明。",
            }

        # 明显是半截句子（以逗号、顿号、或连接词结尾）
        if re.search(r"[，、]$", query) or re.search(r"(以及|还有|并且|而且|或者|比如|例如)$", query):
            return {
                "complete": False,
                "reason": "问题以连接词或标点结尾，似乎是未完成的句子",
                "missing_info": ["问题后半部分"],
                "rewritten_query": query,
                "follow_up_question": "您的问题似乎还没说完，请补充完整。",
            }

        # 无法通过规则判断，返回 None 让 LLM 处理
        return None

    def _llm_completeness(self, query: str, history: list = None) -> Optional[Dict]:
        """使用 LLM 判断完整度"""
        llm = self._get_llm()
        if llm is None:
            return None

        prompt = f"""请判断以下用户问题是否完整，是否需要补充信息才能准确回答。

用户问题：{query}

请按以下格式回答（只输出JSON，不要markdown）：
{{
  "complete": true/false,
  "reason": "判断理由",
  "missing_info": ["缺失项1", "缺失项2"],
  "rewritten_query": "改写后的问题（如完整）",
  "follow_up_question": "追问建议（如不完整）"
}}
"""

        try:
            response = llm.generate(prompt, system="你是一个问题分析助手。")
            # 清理可能的 markdown
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            import json
            result = json.loads(response)
            return {
                "complete": result.get("complete", True),
                "reason": result.get("reason", ""),
                "missing_info": result.get("missing_info", []),
                "rewritten_query": result.get("rewritten_query", query),
                "follow_up_question": result.get("follow_up_question", ""),
            }
        except Exception as e:
            print(f"[Enrich] LLM 完整度判断失败: {e}")
            return None
