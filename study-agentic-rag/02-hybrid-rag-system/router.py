"""
智能路由器
Intelligent Query Router

根据查询类型自动选择传统RAG或Agentic RAG
"""

from openai import OpenAI
from typing import Literal
import json
import config


class QueryRouter:
    """查询路由器 - 智能选择RAG策略"""

    def __init__(self):
        """初始化路由器"""
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL
        )

    def classify_query(self, query: str) -> Literal["simple", "complex"]:
        """
        分类查询类型

        Args:
            query: 用户查询

        Returns:
            "simple" 或 "complex"
        """
        print(f"\n🎯 [路由器] 分析查询类型...")

        # 方法1: 基于关键词的快速判断
        quick_decision = self._quick_classify(query)
        if quick_decision:
            print(f"   快速判断: {quick_decision}")
            return quick_decision

        # 方法2: 使用LLM深度分析
        llm_decision = self._llm_classify(query)
        print(f"   LLM判断: {llm_decision}")
        return llm_decision

    def _quick_classify(self, query: str) -> Literal["simple", "complex", None]:
        """
        基于关键词的快速分类

        Returns:
            "simple" / "complex" / None (不确定)
        """
        query_lower = query.lower()

        # 检查简单查询关键词
        for keyword in config.SIMPLE_QUERY_KEYWORDS:
            if keyword in query_lower:
                return "simple"

        # 检查复杂查询关键词
        for keyword in config.COMPLEX_QUERY_KEYWORDS:
            if keyword in query_lower:
                return "complex"

        # 不确定，需要LLM判断
        return None

    def _llm_classify(self, query: str) -> Literal["simple", "complex"]:
        """
        使用LLM深度分析查询类型

        Returns:
            "simple" 或 "complex"
        """
        prompt = f"""你是一个查询分类专家。请分析以下查询的复杂度。

用户查询: {query}

分类标准:

简单查询 (simple):
- 事实性问题（是什么、谁是、在哪里）
- 直接定义查询
- 单一信息点查询
- 可以从知识库直接检索答案

复杂查询 (complex):
- 需要多步推理（为什么、如何）
- 需要分析、比较、评价
- 需要综合多个信息源
- 需要计算或推导
- 需要最新的外部信息

请以JSON格式回复:
{{
    "complexity": "simple" or "complex",
    "reasoning": "你的判断理由"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的查询分类专家。请始终以JSON格式回复。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 低温度保证分类稳定
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()

            # 提取JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            complexity = result.get("complexity", "complex")
            reasoning = result.get("reasoning", "N/A")

            print(f"   理由: {reasoning}")

            return complexity if complexity in ["simple", "complex"] else "complex"

        except Exception as e:
            print(f"   ⚠️ LLM分类失败: {e}")
            # 默认使用复杂模式（更安全）
            return "complex"

    def route(self, query: str) -> dict:
        """
        路由决策

        Args:
            query: 用户查询

        Returns:
            路由结果: {
                "strategy": "traditional" or "agentic",
                "complexity": "simple" or "complex",
                "reasoning": str
            }
        """
        complexity = self.classify_query(query)

        if complexity == "simple":
            strategy = "traditional"
            reasoning = "查询相对简单，使用传统RAG可以快速准确地回答"
        else:
            strategy = "agentic"
            reasoning = "查询较复杂，使用Agentic RAG进行深度推理"

        result = {
            "strategy": strategy,
            "complexity": complexity,
            "reasoning": reasoning
        }

        print(f"\n{'='*60}")
        print(f"🎯 [路由决策]")
        print(f"   查询: {query}")
        print(f"   复杂度: {complexity}")
        print(f"   策略: {strategy.upper()} RAG")
        print(f"   理由: {reasoning}")
        print(f"{'='*60}")

        return result


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    router = QueryRouter()

    # 测试查询
    test_queries = [
        # 简单查询
        "孙悟空是谁？",
        "唐僧的法号是什么？",
        "西游记的作者是谁？",

        # 复杂查询
        "孙悟空和猪八戒的性格有什么区别？",
        "为什么唐僧要去西天取经？",
        "分析西游记中师徒四人的角色特点",
        "如果唐僧团队每天走50公里，西天路程10万8千里，需要多少天？"
    ]

    for query in test_queries:
        result = router.route(query)
        print()
