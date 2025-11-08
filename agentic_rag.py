"""
Agentic RAG实现示例
Agentic RAG Implementation with ReAct Framework

工作流程:
1. 用户查询 → AI代理思考(Thought)
2. 决策: 需要什么工具? → 行动(Action)
3. 执行工具(向量搜索/Web搜索/计算等) → 观察(Observation)
4. 评估结果质量 → [循环] 是否需要更多信息?
5. 生成最终答案

特点:
- 智能决策,多工具访问
- 迭代检索,质量验证
- 能处理复杂查询
- 多数据源整合
"""

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from openai import AzureOpenAI
from duckduckgo_search import DDGS
import json
from typing import List, Dict, Any

# 加载环境变量
load_dotenv()

# 配置常量
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_ENDPOINT = os.getenv("ES_ENDPOINT", "localhost")
MODEL_NAME = os.getenv("MODEL_NAME", "text-embedding-ada-002")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_DEPLOYMENT_ID = os.getenv("AZURE_DEPLOYMENT_ID")

ELASTIC_INDEX_NAME = "agentic_rag_index"


class AgenticRAG:
    """Agentic RAG实现 - 基于ReAct框架"""

    def __init__(self):
        """初始化连接、工具和代理配置"""
        # 初始化Elasticsearch连接
        self.es_url = f"https://{ES_USER}:{ES_PASSWORD}@{ES_ENDPOINT}:9200"
        self.es = Elasticsearch(
            self.es_url,
            ca_certs="./http_ca.crt",
            verify_certs=True
        )

        # 初始化Azure OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model=MODEL_NAME,
            azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
            api_key=AZURE_EMBEDDING_API_KEY,
            openai_api_version=AZURE_EMBEDDING_API_VERSION
        )

        # 初始化Azure OpenAI Chat
        self.chat_client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT
        )

        self.docsearch = None

        # 代理记忆 (短期记忆)
        self.memory = []

        # 最大迭代次数(防止无限循环)
        self.max_iterations = 5

    def ingest_documents(self, texts):
        """摄入文档到Elasticsearch向量数据库"""
        print(f"📥 正在摄入 {len(texts)} 个文档到Elasticsearch...")

        if not self.es.indices.exists(index=ELASTIC_INDEX_NAME):
            print(f"✨ 创建新索引: {ELASTIC_INDEX_NAME}")
            self.docsearch = ElasticsearchStore.from_texts(
                texts,
                embedding=self.embeddings,
                es_url=self.es_url,
                es_connection=self.es,
                index_name=ELASTIC_INDEX_NAME,
                es_user=ES_USER,
                es_password=ES_PASSWORD
            )
        else:
            print(f"📂 使用现有索引: {ELASTIC_INDEX_NAME}")
            self.docsearch = ElasticsearchStore(
                es_connection=self.es,
                embedding=self.embeddings,
                es_url=self.es_url,
                index_name=ELASTIC_INDEX_NAME,
                es_user=ES_USER,
                es_password=ES_PASSWORD
            )

        print("✅ 文档摄入完成!")

    # ========================================
    # 工具定义 (Tools)
    # ========================================

    def tool_vector_search(self, query: str, k: int = 3) -> str:
        """
        工具1: 向量搜索

        从向量数据库检索相关文档

        参数:
            query: 搜索查询
            k: 返回TOP-K个文档

        返回:
            格式化的文档内容
        """
        print(f"\n🔧 [工具] 向量搜索: '{query}'")
        docs = self.docsearch.similarity_search(query, k=k)

        if not docs:
            return "未找到相关文档。"

        result = "\n\n---\n\n".join([doc.page_content for doc in docs])
        print(f"   ✅ 找到 {len(docs)} 个相关文档")
        return result

    def tool_web_search(self, query: str, max_results: int = 3) -> str:
        """
        工具2: Web搜索

        使用DuckDuckGo搜索引擎获取实时信息

        参数:
            query: 搜索查询
            max_results: 最大结果数

        返回:
            格式化的搜索结果
        """
        print(f"\n🔧 [工具] Web搜索: '{query}'")
        try:
            results = DDGS().text(query, max_results=max_results)

            if not results:
                return "Web搜索未找到相关结果。"

            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"[{i}] {result['title']}\n{result['body']}\n来源: {result['href']}"
                )

            print(f"   ✅ 找到 {len(results)} 个Web结果")
            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            print(f"   ❌ Web搜索失败: {e}")
            return f"Web搜索失败: {str(e)}"

    def tool_calculator(self, expression: str) -> str:
        """
        工具3: 计算器

        执行数学计算

        参数:
            expression: 数学表达式

        返回:
            计算结果
        """
        print(f"\n🔧 [工具] 计算器: '{expression}'")
        try:
            # 安全的数学计算(仅允许基本运算)
            result = eval(expression, {"__builtins__": {}}, {})
            print(f"   ✅ 计算结果: {result}")
            return str(result)
        except Exception as e:
            print(f"   ❌ 计算失败: {e}")
            return f"计算错误: {str(e)}"

    def get_available_tools(self) -> Dict[str, Any]:
        """
        获取可用工具列表

        返回工具名称和描述的字典,用于代理决策
        """
        return {
            "vector_search": {
                "description": "从内部知识库(向量数据库)检索相关文档。适用于查询已知信息、历史数据、内部文档等。",
                "function": self.tool_vector_search
            },
            "web_search": {
                "description": "从互联网搜索最新信息。适用于需要实时数据、新闻、当前事件、最新发展等。",
                "function": self.tool_web_search
            },
            "calculator": {
                "description": "执行数学计算。适用于需要数值运算的问题。",
                "function": self.tool_calculator
            }
        }

    # ========================================
    # ReAct 框架实现
    # ========================================

    def think(self, question: str, context: str) -> Dict[str, Any]:
        """
        步骤1: 思考(Thought)

        AI代理分析问题和当前上下文,决定下一步行动

        参数:
            question: 用户问题
            context: 当前已知的上下文

        返回:
            决策结果: {action: 工具名称, action_input: 工具参数, reasoning: 推理过程}
        """
        print(f"\n💭 [思考] 代理正在分析问题...")

        # 构建思考提示词
        tools_description = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.get_available_tools().items()
        ])

        prompt = f"""你是一个智能AI代理,需要帮助用户回答问题。

用户问题: {question}

当前已知信息:
{context if context else "暂无"}

可用工具:
{tools_description}

请分析问题并决定:
1. 是否需要使用工具获取更多信息?
2. 如果需要,应该使用哪个工具?工具的输入参数是什么?
3. 如果不需要,是否可以直接回答问题?

请以JSON格式回复,包含以下字段:
{{
    "need_tool": true/false,
    "action": "工具名称(如果need_tool=true)",
    "action_input": "工具输入参数",
    "reasoning": "你的推理过程"
}}

如果可以直接回答,设置need_tool=false。
"""

        # 调用LLM进行推理
        response = self.chat_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_ID,
            messages=[
                {"role": "system", "content": "你是一个专业的AI代理,擅长分析问题并选择合适的工具。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 较低温度保证决策稳定性
            max_tokens=500
        )

        # 解析决策
        try:
            decision = json.loads(response.choices[0].message.content)
            print(f"   推理: {decision.get('reasoning', 'N/A')}")
            return decision
        except json.JSONDecodeError:
            print("   ⚠️ 决策解析失败,默认返回无需工具")
            return {"need_tool": False, "reasoning": "解析错误"}

    def act(self, action: str, action_input: str) -> str:
        """
        步骤2: 行动(Action)

        执行选定的工具

        参数:
            action: 工具名称
            action_input: 工具输入

        返回:
            工具执行结果
        """
        print(f"\n⚡ [行动] 执行工具: {action}")

        tools = self.get_available_tools()

        if action not in tools:
            return f"错误: 未知工具 '{action}'"

        # 执行工具
        tool_function = tools[action]["function"]
        result = tool_function(action_input)

        return result

    def observe(self, question: str, observation: str) -> Dict[str, Any]:
        """
        步骤3: 观察(Observation)

        评估工具执行结果的质量,判断是否需要继续检索

        参数:
            question: 用户问题
            observation: 工具返回的观察结果

        返回:
            评估结果: {is_sufficient: bool, reasoning: str}
        """
        print(f"\n👁️  [观察] 评估结果质量...")

        prompt = f"""你是一个智能AI代理,需要评估检索到的信息是否足以回答用户问题。

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
}}
"""

        response = self.chat_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_ID,
            messages=[
                {"role": "system", "content": "你是一个专业的信息质量评估专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        try:
            evaluation = json.loads(response.choices[0].message.content)
            print(f"   评估: {evaluation.get('reasoning', 'N/A')}")
            return evaluation
        except json.JSONDecodeError:
            print("   ⚠️ 评估解析失败,默认认为信息充足")
            return {"is_sufficient": True, "reasoning": "解析错误"}

    def generate_final_answer(self, question: str, context: str) -> str:
        """
        生成最终答案

        参数:
            question: 用户问题
            context: 所有收集到的上下文

        返回:
            最终答案
        """
        print(f"\n🤖 [生成] 生成最终答案...")

        prompt = f"""你是一个有用的AI助手。请根据以下信息回答用户的问题。

收集到的信息:
{context}

用户问题: {question}

请提供准确、详细、有条理的答案。如果信息不足,请诚实说明。
"""

        response = self.chat_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_ID,
            messages=[
                {"role": "system", "content": "你是一个专业的AI助手,擅长综合信息并生成高质量答案。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        print("   ✅ 答案生成完成!")
        return answer

    # ========================================
    # Agentic RAG 主流程
    # ========================================

    def query(self, question: str) -> str:
        """
        Agentic RAG完整流程 (ReAct循环)

        工作流程:
        1. 思考(Thought): 分析问题,决定是否需要工具
        2. 行动(Action): 执行工具获取信息
        3. 观察(Observation): 评估信息质量
        4. [循环] 直到信息充足或达到最大迭代次数
        5. 生成最终答案

        参数:
            question: 用户问题

        返回:
            最终答案
        """
        print(f"\n{'='*60}")
        print(f"📝 用户问题: {question}")
        print(f"{'='*60}")

        # 初始化上下文和迭代计数
        accumulated_context = ""
        iteration = 0

        # ReAct循环
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔄 迭代 {iteration}/{self.max_iterations}")

            # 步骤1: 思考
            decision = self.think(question, accumulated_context)

            # 如果不需要工具,直接生成答案
            if not decision.get("need_tool", False):
                print("\n✅ 代理决定: 信息充足,无需更多工具")
                break

            # 步骤2: 行动
            action = decision.get("action")
            action_input = decision.get("action_input")

            if not action or not action_input:
                print("\n⚠️ 决策信息不完整,停止迭代")
                break

            observation = self.act(action, action_input)

            # 步骤3: 观察
            evaluation = self.observe(question, observation)

            # 累积上下文
            accumulated_context += f"\n\n[来自 {action}]:\n{observation}"

            # 如果信息充足,停止迭代
            if evaluation.get("is_sufficient", False):
                print("\n✅ 代理评估: 信息充足,停止检索")
                break

            print(f"\n⚠️ 信息不足,继续下一轮...")

        # 生成最终答案
        if not accumulated_context:
            accumulated_context = "无额外信息"

        final_answer = self.generate_final_answer(question, accumulated_context)

        # 记录到内存
        self.memory.append({
            "question": question,
            "context": accumulated_context,
            "answer": final_answer,
            "iterations": iteration
        })

        return final_answer


# ============================================
# 使用示例
# ============================================

def main():
    """主函数 - 演示Agentic RAG的使用"""

    # 初始化Agentic RAG
    agent_rag = AgenticRAG()

    # 示例文档
    sample_documents = [
        """
        人工智能(AI)是计算机科学的一个分支,致力于创建能够执行通常需要人类智能的任务的系统。
        这些任务包括视觉感知、语音识别、决策制定和语言翻译。
        """,
        """
        检索增强生成(RAG)是一种结合信息检索和文本生成的技术。
        它通过从外部知识库检索相关信息来增强语言模型的能力,从而减少幻觉并提高答案的准确性。
        传统RAG使用单一数据源,而Agentic RAG可以智能地选择多个数据源。
        """,
        """
        Agentic RAG使用AI代理来增强检索过程。代理可以访问多种工具,包括向量搜索、Web搜索和计算器。
        通过ReAct框架,代理可以进行推理、行动和观察的循环,直到收集到足够的信息。
        """
    ]

    # 摄入文档
    agent_rag.ingest_documents(sample_documents)

    # 测试查询
    test_questions = [
        "什么是Agentic RAG?它与传统RAG有什么区别?",
        "2024年AI领域有哪些最新进展?",  # 需要Web搜索
        "如果一个RAG系统每天处理1000个查询,每个查询调用LLM 3次,一个月调用多少次?"  # 需要计算器
    ]

    for question in test_questions:
        answer = agent_rag.query(question)

        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"💡 答案: {answer}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


"""
Agentic RAG的优点:
✅ 智能决策 - 代理自主选择工具和检索策略
✅ 多源整合 - 结合向量数据库、Web搜索、API等多个数据源
✅ 质量验证 - 评估检索结果,必要时重新检索
✅ 处理复杂查询 - 多步推理,分解复杂问题
✅ 自适应 - 根据问题类型调整策略
✅ 可扩展 - 易于添加新工具

Agentic RAG的局限:
❌ 实现复杂 - 需要设计代理逻辑、工具集成
❌ 响应时间长 - 多次推理和检索增加延迟
❌ 成本高 - 多次LLM调用增加费用
❌ 调试困难 - 非确定性行为,难以预测
❌ 依赖性强 - 依赖多个外部服务

适用场景:
- 需要多源信息整合的应用
- 需要实时信息的场景
- 复杂推理和分析任务
- 个性化服务(整合用户数据)
- 研究和深度分析
- 不确定性高的查询

核心代码实现差异:
1. 工具系统: Agentic RAG有多个工具,传统RAG只有向量搜索
2. 决策逻辑: Agentic RAG有思考-行动-观察循环,传统RAG是固定流程
3. 质量评估: Agentic RAG评估检索质量,传统RAG不评估
4. 迭代检索: Agentic RAG可多次检索,传统RAG只检索一次
5. LLM调用次数: Agentic RAG多次(思考+评估+生成),传统RAG一次(生成)
"""
