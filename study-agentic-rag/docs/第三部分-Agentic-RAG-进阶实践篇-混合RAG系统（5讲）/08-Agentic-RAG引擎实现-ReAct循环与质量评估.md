# 第8讲：Agentic RAG引擎实现 - ReAct循环与质量评估

在第7讲中，我们构建了传统RAG引擎，用于处理70%的简单查询。但还有30%的复杂查询需要更强大的能力。

这一讲，我们将构建**生产级的Agentic RAG引擎**，利用ReAct框架实现智能推理和多步检索。

---

## 一、Agentic RAG引擎的定位

### 在混合系统中的角色

```
用户查询
   ↓
路由器判断
   ↓
简单查询(70%) → [传统RAG引擎] → 1次检索+生成 → 快速回答（2秒）
复杂查询(30%) → [Agentic RAG引擎] → N次推理循环 → 深度回答（8秒）
```

### 设计目标

| 目标 | 指标 | 为什么重要 |
|------|------|-----------|
| **准确性** | 90%+ | 复杂查询容错率低 |
| **智能性** | 支持多步推理 | 处理"为什么""如何"类问题 |
| **工具能力** | 3+种工具 | 整合多数据源 |
| **可控性** | 最大5次迭代 | 防止无限循环和成本失控 |

### 与第4讲的区别

**第4讲的Agentic RAG**：
- 教学演示版本
- 单文件实现（~620行）
- 完整功能展示

**本讲的Agentic RAG引擎**：
- 生产级封装
- 模块化设计
- 共享向量数据库（与传统RAG）
- 可配置工具开关
- 完善的日志

---

## 二、引擎架构设计

### ReAct循环架构

```
┌────────────────────────────────────────────────┐
│       Agentic RAG引擎 (ReAct Framework)        │
│                                                │
│  查询输入                                       │
│     ↓                                          │
│  ┌──────────── ReAct循环 ──────────────┐       │
│  │                                     │       │
│  │  1️⃣ Think(思考)                      │       │
│  │     分析问题 → 选择工具               │       │
│  │                                     │       │
│  │  2️⃣ Act(行动)                        │       │
│  │     执行工具 → 获取信息               │       │
│  │                                     │       │
│  │  3️⃣ Observe(观察)                    │       │
│  │     评估质量 → 决定是否继续           │       │
│  │                                     │       │
│  │  信息充足? → No → 返回1️⃣ (最多5次)     │       │
│  │           ↓ Yes                     │       │
│  └──────────────────────────────────────┘       │
│                                                │
│  4️⃣ Generate(生成)                              │
│     综合所有信息 → 生成最终答案                  │
│                                                │
└────────────────────────────────────────────────┘
```

### 四大核心模块

```python
class AgenticRAGEngine:
    """Agentic RAG引擎"""

    # 模块1: 初始化
    def __init__(self): ...

    # 模块2: 工具系统
    def tool_vector_search(self, query): ...
    def tool_web_search(self, query): ...
    def tool_calculator(self, expression): ...
    def get_available_tools(self): ...

    # 模块3: ReAct框架
    def think(self, question, context): ...
    def act(self, action, action_input): ...
    def observe(self, question, observation): ...
    def generate_final_answer(self, question, context): ...

    # 模块4: 主循环
    def query(self, question): ...
```

---

## 三、代码实现（分4部分详解）

### 第1部分：初始化与配置

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/agentic_rag_engine.py`

**这部分要做什么？**

建立引擎基础设施，连接LLM和向量数据库，配置Agent参数。

```python
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

        # 初始化ChromaDB（共享向量数据库）
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

        # Agent配置
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
```

**为什么这么写？**

1. **为什么共享ChromaDB路径？**
   - 传统RAG和Agentic RAG使用同一个向量数据库
   - 避免数据重复存储
   - 节省磁盘空间和管理成本

   ```python
   # config.py
   CHROMA_DB_PATH = "./vector_db"  # 两个引擎共享
   COLLECTION_NAME = "hybrid_rag_collection"  # 同一个集合
   ```

2. **为什么用 `max_iterations`？**
   - 防止Agent陷入无限循环
   - 控制成本（每次迭代调用多次LLM）
   - 生产环境通常设为3-5次

3. **为什么Agentic RAG也需要嵌入？**
   - 工具系统中的向量搜索需要嵌入
   - 与传统RAG共享嵌入逻辑
   - 统一的向量化方法

---

### 第2部分：工具系统实现

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/agentic_rag_engine.py`

**这部分要做什么？**

定义Agent可以使用的工具集，包括向量搜索、Web搜索、计算器。

```python
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
```

**为什么这么写？**

1. **为什么工具返回字符串而不是结构化数据？**
   - LLM更容易理解文本格式
   - 便于直接拼接到上下文
   - 格式化输出更易读

   ```python
   # 好的输出格式
   "文档1: 孙悟空是...\n\n文档2: 唐僧是..."

   # 不好的格式（LLM难理解）
   [{"id": 1, "content": "..."}, {"id": 2, "content": "..."}]
   ```

2. **为什么 `WEB_SEARCH_ENABLED` 是可配置的？**
   - 开发环境可能无需Web搜索
   - 减少外部依赖和网络请求
   - 降低成本（部分场景）

3. **为什么计算器要限制字符？**
   - 安全考虑，防止代码注入
   - `eval()` 很危险，只允许数学运算
   - `{"__builtins__": {}}` 禁用内置函数

4. **为什么要 `get_available_tools()`？**
   - 集中管理工具注册
   - 便于动态添加/删除工具
   - 工具描述用于Agent决策

---

### 第3部分：ReAct框架 - Think, Act, Observe

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/agentic_rag_engine.py`

**这部分要做什么？**

实现ReAct框架的三个核心阶段：思考、行动、观察。

```python
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
```

**为什么这么写？**

1. **为什么 `think()` 要输出reasoning？**
   - 可解释性：理解Agent的决策过程
   - 调试：发现错误的推理逻辑
   - 优化：分析决策质量

2. **为什么 `act()` 这么简单？**
   - 单一职责：只负责路由到正确的工具
   - 工具实现在各自的方法里
   - 便于添加新工具（只需注册）

3. **为什么需要 `observe()` 评估？**
   - Agent可能选错工具或参数
   - 检索结果可能不相关
   - 避免低质量信息污染最终答案

4. **为什么所有阶段都用JSON格式？**
   - 结构化输出，便于程序解析
   - 减少自然语言的歧义
   - 统一的接口规范

---

### 第4部分：ReAct主循环

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/agentic_rag_engine.py`

**这部分要做什么？**

整合Think-Act-Observe，实现完整的迭代推理循环。

```python
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
```

**为什么这么写？**

1. **为什么用 `while iteration < max_iterations`？**
   - 允许多次迭代（最多5次）
   - 防止无限循环
   - 每次迭代都是完整的Think-Act-Observe

2. **什么时候停止迭代？**
   - Agent认为不需要工具（`need_tool=false`）
   - Agent评估信息充足（`is_sufficient=true`）
   - 决策信息不完整（防止错误）
   - 达到最大迭代次数（成本控制）

3. **为什么用 `accumulated_context`？**
   - 累积所有迭代获取的信息
   - 第1次可能用向量搜索
   - 第2次可能用Web搜索
   - 最终答案基于所有信息

4. **为什么要 `[来自 {action}]` 标签？**
   - 标识信息来源
   - 便于LLM理解上下文结构
   - 有助于调试和溯源

**ReAct循环示例：**

```
迭代1:
  💭 思考: 需要知识库信息
  ⚡ 行动: vector_search("孙悟空有几个师傅")
  👁️ 观察: 找到相关信息，但不够完整
  → 继续

迭代2:
  💭 思考: 需要更详细的信息
  ⚡ 行动: vector_search("菩提祖师 唐僧")
  👁️ 观察: 信息充足
  → 停止

🤖 生成: 综合所有信息，生成最终答案
```

---

## 四、完整代码总结

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/agentic_rag_engine.py`

完整的 `agentic_rag_engine.py` 文件（约450行）：

<details>
<summary>点击展开完整代码</summary>

```python
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
        """工具1: 向量检索"""
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
        """工具2: Web搜索"""
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
        """工具3: 计算器"""
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
        """步骤1: 思考(Thought)"""
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
        """步骤2: 行动(Action)"""
        print(f"\n⚡ [行动] 执行工具: {action}")

        tools = self.get_available_tools()

        if action not in tools:
            return f"错误: 未知工具 '{action}'"

        tool_function = tools[action]["function"]
        result = tool_function(action_input)

        return result

    def observe(self, question: str, observation: str) -> Dict[str, Any]:
        """步骤3: 观察(Observation)"""
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
        """完整的Agentic RAG流程 (ReAct循环)"""
        print(f"\n{'='*60}")
        print(f"📝 [Agentic RAG] 问题: {question}")
        print(f"{'='*60}")

        accumulated_context = ""
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔄 迭代 {iteration}/{self.max_iterations}")

            decision = self.think(question, accumulated_context)

            if not decision.get("need_tool", False):
                print("\n✅ 代理决定: 信息充足，无需更多工具")
                break

            action = decision.get("action")
            action_input = decision.get("action_input")

            if not action or not action_input:
                print("\n⚠️ 决策信息不完整，停止迭代")
                break

            observation = self.act(action, action_input)
            evaluation = self.observe(question, observation)

            accumulated_context += f"\n\n[来自 {action}]:\n{observation}"

            if evaluation.get("is_sufficient", False):
                print("\n✅ 代理评估: 信息充足，停止检索")
                break

            print(f"\n⚠️ 信息不足，继续下一轮...")

        if not accumulated_context:
            accumulated_context = "无额外信息"

        final_answer = self.generate_final_answer(question, accumulated_context)

        return final_answer


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    rag = AgenticRAGEngine()

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
```

</details>

---

## 五、总结

### 核心要点

1. **Agentic RAG的定位**
   - 处理30%的复杂查询
   - ReAct框架实现智能推理
   - 多工具协作，多数据源整合

2. **四大核心模块**
   - 初始化：连接LLM和向量数据库
   - 工具系统：向量搜索+Web搜索+计算器
   - ReAct框架：Think-Act-Observe循环
   - 主循环：迭代推理至信息充足

3. **生产级特性**
   - ✅ 共享向量数据库（节省资源）
   - ✅ 可配置工具开关（灵活控制）
   - ✅ 迭代次数限制（成本控制）
   - ✅ 完善的错误处理（降级策略）
   - ✅ 详细的日志输出（可观测性）

### 对比总结

| 特性 | 传统RAG引擎 | Agentic RAG引擎 |
|------|------------|----------------|
| **流程** | 固定2步 | 动态N步 |
| **工具** | 1个（向量搜索） | 3+个 |
| **推理** | 无 | ReAct循环 |
| **LLM调用** | 2次/查询 | 6-8次/查询 |
| **响应时间** | 1-2秒 | 8-10秒 |
| **成本** | ¥0.01/次 | ¥0.03/次 |
| **适用场景** | 简单查询 | 复杂查询 |

---

## 下一讲预告

路由器和两个引擎都已完成，现在还缺一个关键环节：**如何组装它们？**

**第9讲：数据处理流水线 - 文档管理与批量导入**

我们将学习如何：
- 批量处理原始文档
- 文档分块策略
- 元数据提取
- 统一管理向量数据库

敬请期待！
