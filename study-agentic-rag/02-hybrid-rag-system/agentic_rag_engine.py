"""
Agentic RAG引擎
Agentic RAG Engine with ReAct Framework

用于处理复杂查询，支持多步推理和多工具协作
"""

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from duckduckgo_search import DDGS
from typing import List, Dict, Any
import json
import config


class AgenticRAGEngine:
    """Agentic RAG引擎 - 基于ReAct框架"""

    def __init__(self):
        """初始化Agentic RAG引擎"""
        print("🚀 初始化Agentic RAG引擎...")

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL
        )

        # 初始化ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 获取或创建集合
        try:
            self.collection = self.chroma_client.get_collection(
                name=config.COLLECTION_NAME
            )
            print(f"✅ 加载现有集合: {config.COLLECTION_NAME}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=config.COLLECTION_NAME
            )
            print(f"✨ 创建新集合: {config.COLLECTION_NAME}")

        self.max_iterations = config.MAX_AGENT_ITERATIONS

    def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入"""
        try:
            response = self.client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 嵌入失败: {e}")
            return [0.0] * 1536

    # ========================================
    # 工具定义
    # ========================================

    def tool_vector_search(self, query: str, k: int = config.TOP_K) -> str:
        """
        工具1: 向量检索

        Args:
            query: 搜索查询
            k: 返回文档数

        Returns:
            格式化的检索结果
        """
        print(f"\n🔧 [工具] 向量检索: '{query}'")

        try:
            query_embedding = self.get_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            if not results['documents'] or len(results['documents'][0]) == 0:
                return "未找到相关文档"

            formatted = []
            for i, doc in enumerate(results['documents'][0], 1):
                formatted.append(f"文档{i}: {doc}")

            print(f"   ✅ 找到 {len(results['documents'][0])} 个相关文档")
            return "\n\n".join(formatted)

        except Exception as e:
            print(f"   ❌ 检索失败: {e}")
            return f"检索失败: {str(e)}"

    def tool_web_search(self, query: str, max_results: int = 3) -> str:
        """
        工具2: Web搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            格式化的搜索结果
        """
        if not config.WEB_SEARCH_ENABLED:
            return "Web搜索功能未启用"

        print(f"\n🔧 [工具] Web搜索: '{query}'")

        try:
            results = DDGS().text(query, max_results=max_results)

            if not results:
                return "未找到相关结果"

            formatted = []
            for i, result in enumerate(results, 1):
                formatted.append(
                    f"[{i}] {result['title']}\n{result['body']}\n来源: {result['href']}"
                )

            print(f"   ✅ 找到 {len(results)} 个Web结果")
            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            print(f"   ❌ Web搜索失败: {e}")
            return f"Web搜索失败: {str(e)}"

    def tool_calculator(self, expression: str) -> str:
        """
        工具3: 计算器

        Args:
            expression: 数学表达式

        Returns:
            计算结果
        """
        print(f"\n🔧 [工具] 计算器: '{expression}'")

        try:
            allowed_chars = set('0123456789+-*/() .')
            if not all(c in allowed_chars for c in expression):
                return "错误: 表达式包含不允许的字符"

            result = eval(expression, {"__builtins__": {}}, {})
            print(f"   ✅ 计算结果: {result}")
            return str(result)

        except Exception as e:
            print(f"   ❌ 计算失败: {e}")
            return f"计算错误: {str(e)}"

    def get_available_tools(self) -> Dict[str, Any]:
        """获取可用工具列表"""
        return {
            "vector_search": {
                "description": "从知识库检索相关文档。适用于已知信息、历史数据查询。",
                "function": self.tool_vector_search
            },
            "web_search": {
                "description": "从互联网搜索最新信息。适用于实时数据、新闻、当前事件。",
                "function": self.tool_web_search
            },
            "calculator": {
                "description": "执行数学计算。适用于需要数值运算的问题。",
                "function": self.tool_calculator
            }
        }

    # ========================================
    # ReAct框架
    # ========================================

    def think(self, question: str, context: str) -> Dict[str, Any]:
        """
        步骤1: 思考(Thought)

        Args:
            question: 用户问题
            context: 当前上下文

        Returns:
            决策结果
        """
        print(f"\n💭 [思考] 代理正在分析问题...")

        tools_desc = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.get_available_tools().items()
        ])

        prompt = f"""你是一个智能AI代理，需要帮助用户回答问题。

用户问题: {question}

当前已知信息:
{context if context else "暂无"}

可用工具:
{tools_desc}

请分析问题并决定:
1. 是否需要使用工具获取更多信息?
2. 如果需要，应该使用哪个工具？工具的输入参数是什么?
3. 如果不需要，是否可以直接回答问题?

请以JSON格式回复:
{{
    "need_tool": true/false,
    "action": "工具名称(vector_search, web_search, calculator)",
    "action_input": "工具输入参数",
    "reasoning": "你的推理过程"
}}

如果可以直接回答，设置need_tool=false。"""

        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的AI代理，擅长分析问题并选择合适的工具。请始终以JSON格式回复。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE_REASONING,
                max_tokens=config.MAX_TOKENS_REASONING
            )

            content = response.choices[0].message.content.strip()

            # 提取JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            decision = json.loads(content)
            print(f"   推理: {decision.get('reasoning', 'N/A')}")
            return decision

        except Exception as e:
            print(f"   ⚠️ 决策失败: {e}")
            return {"need_tool": False, "reasoning": "决策错误"}

    def act(self, action: str, action_input: str) -> str:
        """
        步骤2: 行动(Action)

        Args:
            action: 工具名称
            action_input: 工具输入

        Returns:
            工具执行结果
        """
        print(f"\n⚡ [行动] 执行工具: {action}")

        tools = self.get_available_tools()

        if action not in tools:
            return f"错误: 未知工具 '{action}'"

        tool_function = tools[action]["function"]
        result = tool_function(action_input)

        return result

    def observe(self, question: str, observation: str) -> Dict[str, Any]:
        """
        步骤3: 观察(Observation)

        Args:
            question: 用户问题
            observation: 工具返回结果

        Returns:
            评估结果
        """
        print(f"\n👁️  [观察] 评估结果质量...")

        prompt = f"""你是一个信息质量评估专家。请评估检索到的信息是否足以回答用户问题。

用户问题: {question}

检索到的信息:
{observation}

请评估:
1. 这些信息是否与问题相关?
2. 这些信息是否足够回答问题?
3. 是否需要检索更多信息?

请以JSON格式回复:
{{
    "is_sufficient": true/false,
    "reasoning": "你的评估理由"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的信息质量评估专家。请始终以JSON格式回复。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE_REASONING,
                max_tokens=config.MAX_TOKENS_EVALUATION
            )

            content = response.choices[0].message.content.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(content)
            print(f"   评估: {evaluation.get('reasoning', 'N/A')}")
            return evaluation

        except Exception as e:
            print(f"   ⚠️ 评估失败: {e}")
            return {"is_sufficient": True, "reasoning": "评估错误"}

    def generate_final_answer(self, question: str, context: str) -> str:
        """生成最终答案"""
        print(f"\n🤖 [生成] 生成最终答案...")

        prompt = f"""你是一个专业的问答助手。请根据以下信息回答用户的问题。

收集到的信息:
{context}

用户问题: {question}

请提供准确、详细、有条理的答案。如果信息不足，请如实说明。"""

        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的问答助手，擅长综合信息生成高质量答案。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE_GENERATION,
                max_tokens=config.MAX_TOKENS_GENERATION
            )

            answer = response.choices[0].message.content
            print("   ✅ 答案生成完成")
            return answer

        except Exception as e:
            print(f"   ❌ 答案生成失败: {e}")
            return f"抱歉，生成答案时出错: {str(e)}"

    def query(self, question: str) -> str:
        """
        完整的Agentic RAG流程 (ReAct循环)

        Args:
            question: 用户问题

        Returns:
            最终答案
        """
        print(f"\n{'='*60}")
        print(f"📝 [Agentic RAG] 问题: {question}")
        print(f"{'='*60}")

        accumulated_context = ""
        iteration = 0

        # ReAct循环
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔄 迭代 {iteration}/{self.max_iterations}")

            # 步骤1: 思考
            decision = self.think(question, accumulated_context)

            # 如果不需要工具，直接生成答案
            if not decision.get("need_tool", False):
                print("\n✅ 代理决定: 信息充足，无需更多工具")
                break

            # 步骤2: 行动
            action = decision.get("action")
            action_input = decision.get("action_input")

            if not action or not action_input:
                print("\n⚠️ 决策信息不完整，停止迭代")
                break

            observation = self.act(action, action_input)

            # 步骤3: 观察
            evaluation = self.observe(question, observation)

            # 累积上下文
            accumulated_context += f"\n\n[来自 {action}]:\n{observation}"

            # 如果信息充足，停止迭代
            if evaluation.get("is_sufficient", False):
                print("\n✅ 代理评估: 信息充足，停止检索")
                break

            print(f"\n⚠️ 信息不足，继续下一轮...")

        # 生成最终答案
        if not accumulated_context:
            accumulated_context = "无额外信息"

        final_answer = self.generate_final_answer(question, accumulated_context)

        return final_answer


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    # 初始化引擎
    rag = AgenticRAGEngine()

    # 测试查询
    test_questions = [
        "孙悟空有几个师傅？他们分别是谁？有什么区别？",
        "西游记取经团队的成员背景和性格分析",
        "如果唐僧团队每天走50公里，西天路程10万8千里，需要多少天？"
    ]

    for question in test_questions:
        answer = rag.query(question)

        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"💡 答案: {answer}")
        print(f"{'='*60}\n")
