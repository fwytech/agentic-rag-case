# 第4讲：构建Agentic RAG引擎 - ReAct循环与多工具集成

在上一讲中，我们实现了传统RAG系统，体验了最基础的"检索-生成"流程。但面对复杂查询时，传统RAG显得力不从心。

这一讲，我们将构建一个**智能化的Agentic RAG引擎**，让AI不再是被动执行固定流程，而是能够**主动思考**、**选择工具**、**评估质量**、**迭代优化**。

---

## 一、Agentic RAG的智能化工作流程

### 与传统RAG的本质差异

**传统RAG**就像流水线工人：
```
查询 → 向量搜索 → 拼接上下文 → LLM生成 → 返回
```
无论什么问题，都是这一套流程，没有思考空间。

**Agentic RAG**则像专业顾问：
```
查询 → 思考需要什么 → 选择合适工具 → 执行获取信息 → 评估信息质量
     ↑                                                    ↓
     └─────────────── 不够？继续思考 ←──────────────────┘
                              ↓
                         信息充足？生成答案
```

### ReAct框架的三个核心阶段

```
┌─────────────────────────────────────────────────────┐
│                   ReAct 循环                         │
│                                                      │
│  1️⃣ Think (思考)                                      │
│     分析问题，决定需要什么信息，选择哪个工具           │
│     示例："用户问ChromaDB特点，应该用向量搜索"         │
│                                                      │
│  2️⃣ Act (行动)                                        │
│     调用选定的工具，执行具体操作                       │
│     示例：执行 tool_vector_search("ChromaDB特点")     │
│                                                      │
│  3️⃣ Observe (观察)                                    │
│     评估工具返回的信息质量，判断是否充足                │
│     示例："找到3个相关文档，信息充足，可以生成答案"     │
│                                                      │
│  [循环] 如果信息不足，返回步骤1，继续思考下一步        │
│  [终止] 信息充足 or 达到最大迭代次数 → 生成最终答案    │
└─────────────────────────────────────────────────────┘
```

### 多工具系统架构

Agentic RAG的强大之处在于**工具生态**：

```
                    Agent决策中心
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   🔧 工具1          🔧 工具2          🔧 工具3
 向量搜索(内部)    Web搜索(实时)    计算器(运算)
        │                │                │
   ChromaDB        DuckDuckGo         eval()
  (本地知识库)    (互联网搜索)      (数学计算)
```

每个工具都有明确的职责和适用场景，Agent会根据问题智能选择。

---

## 二、环境准备

### 1. 依赖安装

```bash
pip install openai chromadb duckduckgo-search python-dotenv
```

**为什么需要这些库？**
- `openai`：调用阿里云百炼平台（兼容OpenAI接口）
- `chromadb`：本地向量数据库
- `duckduckgo-search`：免费的Web搜索API（无需密钥）
- `python-dotenv`：管理环境变量

### 2. 环境变量配置（.env文件）

```env
API_KEY=你的阿里云百炼API密钥
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_ID=qwen-plus
EMBEDDING_MODEL=text-embedding-v1
```

---

## 三、代码实现（分5部分详解）

### 第1部分：Agent初始化与配置

**代码文件：** `study-agentic-rag/01-rag-comparison/agentic_rag.py`

**这部分要做什么？**

建立Agentic RAG的基础设施：
- 连接LLM（用于思考、评估、生成）
- 连接ChromaDB（向量数据库）
- 初始化Agent配置（记忆、迭代限制）

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from duckduckgo_search import DDGS
import json
from typing import List, Dict, Any

# 加载环境变量
load_dotenv()

# 配置常量
API_KEY = os.getenv("API_KEY", "sk-abe3417c96f6441b83efed38708bcfb6")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")

CHROMA_COLLECTION_NAME = "agentic_rag_collection"


class AgenticRAG:
    """Agentic RAG实现 - 基于ReAct框架"""

    def __init__(self):
        """初始化连接、工具和代理配置"""
        print("🚀 初始化Agentic RAG系统...")

        # 初始化阿里云百炼平台客户端（兼容OpenAI接口）
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

        print(f"✅ 已连接到阿里云百炼平台")
        print(f"   LLM模型: {MODEL_ID}")
        print(f"   嵌入模型: {EMBEDDING_MODEL}")

        # 初始化ChromaDB（本地持久化向量数据库）
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db_agentic",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 获取或创建集合
        try:
            self.collection = self.chroma_client.get_collection(
                name=CHROMA_COLLECTION_NAME
            )
            print(f"📂 使用现有集合: {CHROMA_COLLECTION_NAME}")
            print(f"   当前文档数: {self.collection.count()}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"description": "Agentic RAG collection"}
            )
            print(f"✨ 创建新集合: {CHROMA_COLLECTION_NAME}")

        # Agent配置
        self.memory = []  # 短期记忆（存储问答历史）
        self.max_iterations = 5  # 最大迭代次数（防止无限循环）

    def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 嵌入失败: {e}")
            return [0.0] * 1536  # 返回零向量作为降级

    def ingest_documents(self, texts: List[str]):
        """摄入文档到ChromaDB向量数据库"""
        print(f"\n📥 正在摄入 {len(texts)} 个文档到ChromaDB...")

        # 生成文档ID
        ids = [f"doc_{i}" for i in range(len(texts))]

        # 获取嵌入向量
        print("   🔄 正在生成嵌入向量...")
        embeddings = []
        for i, text in enumerate(texts):
            print(f"      处理文档 {i+1}/{len(texts)}")
            embedding = self.get_embedding(text)
            embeddings.append(embedding)

        # 存储到ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": f"document_{i}"} for i in range(len(texts))]
        )

        print(f"✅ 文档摄入完成!")
        print(f"   集合中共有 {self.collection.count()} 个文档")
```

**为什么这么写？**

1. **为什么用 `self.memory = []`？**
   - Agent需要记住之前的交互历史
   - 实现多轮对话的上下文延续
   - 可用于分析用户行为模式

2. **为什么设置 `max_iterations = 5`？**
   - 防止Agent陷入死循环（一直觉得信息不够）
   - 控制成本（每次迭代都会调用LLM）
   - 在质量和成本间取得平衡

3. **为什么单独定义 `get_embedding` 方法？**
   - 多处需要调用（文档摄入、向量搜索）
   - 统一异常处理（失败时返回零向量降级）
   - 便于后续替换不同的嵌入模型

---

### 第2部分：多工具系统实现

**代码文件：** `study-agentic-rag/01-rag-comparison/agentic_rag.py`

**这部分要做什么？**

定义Agent可以使用的三个核心工具：
- 工具1：向量搜索（查内部知识库）
- 工具2：Web搜索（查实时信息）
- 工具3：计算器（做数学运算）

```python
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

        try:
            # 获取查询的嵌入向量
            query_embedding = self.get_embedding(query)

            # 在ChromaDB中搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            if not results['documents'] or len(results['documents'][0]) == 0:
                return "未找到相关文档。"

            # 格式化结果
            formatted_docs = []
            for i, doc in enumerate(results['documents'][0]):
                formatted_docs.append(f"文档{i+1}: {doc}")

            result = "\n\n".join(formatted_docs)
            print(f"   ✅ 找到 {len(results['documents'][0])} 个相关文档")
            return result

        except Exception as e:
            print(f"   ❌ 向量搜索失败: {e}")
            return f"向量搜索失败: {str(e)}"

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
        """
        获取可用工具列表

        返回工具名称和描述的字典,用于代理决策
        """
        return {
            "vector_search": {
                "description": "从内部知识库(向量数据库)检索相关文档。适用于查询已知信息、历史数据、内部文档等。",
                "function": self.tool_vector_search,
                "parameters": {"query": "搜索查询"}
            },
            "web_search": {
                "description": "从互联网搜索最新信息。适用于需要实时数据、新闻、当前事件、最新发展等。",
                "function": self.tool_web_search,
                "parameters": {"query": "搜索查询"}
            },
            "calculator": {
                "description": "执行数学计算。适用于需要数值运算的问题。",
                "function": self.tool_calculator,
                "parameters": {"expression": "数学表达式"}
            }
        }
```

**为什么这么写？**

1. **为什么用 `get_available_tools()` 统一管理工具？**
   - Agent需要知道"有哪些工具可用"
   - 每个工具的描述会被放入Prompt，供LLM决策
   - 方便扩展新工具（只需在这里注册即可）

2. **为什么计算器要限制字符？**
   - `eval()` 很危险，可以执行任意Python代码
   - 限制为数字和运算符，防止代码注入攻击
   - `{"__builtins__": {}}`禁用内置函数，进一步加固安全

3. **为什么DuckDuckGo搜索不需要API密钥？**
   - 免费开放API，降低使用门槛
   - 适合教学和原型开发
   - 生产环境可替换为Google Custom Search等

4. **工具选择原则？**
   - 向量搜索 → 查"已知的"内部文档
   - Web搜索 → 查"实时的"外部信息
   - 计算器 → 做"精确的"数学运算

---

### 第3部分：ReAct框架 - 思考阶段(Think)

**代码文件：** `study-agentic-rag/01-rag-comparison/agentic_rag.py`

**这部分要做什么？**

实现Agent的"大脑"：分析问题，决定是否需要工具，选择哪个工具。

```python
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
    "action": "工具名称(如果need_tool=true,可选: vector_search, web_search, calculator)",
    "action_input": "工具输入参数",
    "reasoning": "你的推理过程"
}}

如果可以直接回答,设置need_tool=false。
"""

        # 调用LLM进行推理
        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的AI代理,擅长分析问题并选择合适的工具。请始终以JSON格式回复。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 较低温度保证决策稳定性
                max_tokens=500
            )

            # 解析决策
            content = response.choices[0].message.content.strip()

            # 尝试提取JSON（处理可能的markdown代码块）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            decision = json.loads(content)
            print(f"   推理: {decision.get('reasoning', 'N/A')}")
            return decision

        except json.JSONDecodeError as e:
            print(f"   ⚠️ 决策解析失败: {e}")
            print(f"   原始响应: {content}")
            return {"need_tool": False, "reasoning": "解析错误，直接回答"}
        except Exception as e:
            print(f"   ⚠️ LLM调用失败: {e}")
            return {"need_tool": False, "reasoning": "调用错误"}
```

**为什么这么写？**

1. **为什么用JSON格式回复？**
   - 结构化输出，方便程序解析
   - 避免自然语言的歧义
   - 可以明确获取 `need_tool`、`action`、`action_input` 字段

2. **为什么 `temperature=0.3`？**
   - 思考阶段需要稳定、确定性的决策
   - 高温度会导致决策不稳定（有时选A，有时选B）
   - 低温度保证同样问题得到一致的工具选择

3. **为什么要处理markdown代码块？**
   - LLM有时会回复：` ```json\n{...}\n``` `
   - 需要提取其中的JSON部分才能解析
   - 提高鲁棒性，应对不同格式的回复

4. **为什么要异常处理？**
   - LLM输出不可控，可能不符合JSON格式
   - 网络问题可能导致调用失败
   - 降级策略：解析失败时，假设不需要工具，直接回答

---

### 第4部分：ReAct框架 - 行动与观察(Act & Observe)

**代码文件：** `study-agentic-rag/01-rag-comparison/agentic_rag.py`

**这部分要做什么？**

- **Act（行动）**：执行选定的工具
- **Observe（观察）**：评估工具返回的信息质量

```python
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

        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的信息质量评估专家。请始终以JSON格式回复。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            content = response.choices[0].message.content.strip()

            # 提取JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(content)
            print(f"   评估: {evaluation.get('reasoning', 'N/A')}")
            return evaluation

        except json.JSONDecodeError:
            print("   ⚠️ 评估解析失败,默认认为信息充足")
            return {"is_sufficient": True, "reasoning": "解析错误"}
        except Exception as e:
            print(f"   ⚠️ 评估失败: {e}")
            return {"is_sufficient": True, "reasoning": "评估错误"}

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

        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
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

        except Exception as e:
            print(f"   ❌ 答案生成失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"
```

**为什么这么写？**

1. **为什么 `act()` 这么简单？**
   - 只是路由器，根据 `action` 调用对应的工具函数
   - 真正的逻辑在各个工具方法里
   - 单一职责原则：act只负责调用，不负责实现

2. **为什么需要 `observe()` 评估质量？**
   - 传统RAG无法判断检索结果是否真正相关
   - Agent可以"看一眼"结果，决定是否继续找
   - 避免"检索到无关信息"直接生成低质量答案

3. **什么时候 `is_sufficient = true`？**
   - 找到明确回答问题所需的信息
   - 信息量已经足够详细
   - 继续检索不太可能带来新信息

4. **什么时候 `is_sufficient = false`？**
   - 检索结果与问题相关性低
   - 信息不够详细或完整
   - 需要从其他工具获取补充信息

5. **为什么生成答案时 `temperature=0.7`？**
   - 生成阶段需要一定创造性，使答案自然流畅
   - 不需要像决策阶段那么严格
   - 0.7是经验值，平衡准确性和流畅性

---

### 第5部分：完整集成 - ReAct主循环

**代码文件：** `study-agentic-rag/01-rag-comparison/agentic_rag.py`

**这部分要做什么？**

把 Think → Act → Observe 串起来，形成完整的Agentic RAG查询流程。

```python
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

    def reset_collection(self):
        """重置集合（清空所有文档）"""
        try:
            self.chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"description": "Agentic RAG collection"}
            )
            print("✅ 集合已重置")
        except Exception as e:
            print(f"❌ 重置失败: {e}")
```

**为什么这么写？**

1. **为什么用 `while iteration < self.max_iterations`？**
   - 允许多次迭代（Think-Act-Observe循环）
   - 防止无限循环（最多5次）
   - 每次迭代都是一轮完整的"思考-行动-观察"

2. **什么时候会停止迭代？**
   - Agent认为不需要工具（`need_tool=false`）
   - Agent评估信息充足（`is_sufficient=true`）
   - 决策信息不完整（action或action_input缺失）
   - 达到最大迭代次数（防止成本失控）

3. **为什么用 `accumulated_context += ...` 累积上下文？**
   - 第1次迭代可能用向量搜索
   - 第2次迭代可能用Web搜索
   - 需要把所有信息汇总，供最终生成答案使用

4. **为什么要 `self.memory.append(...)`？**
   - 记录每次查询的完整过程
   - 可用于调试分析（为什么用了3次迭代？）
   - 未来可扩展为长期记忆（记住用户偏好）

**ReAct循环流程图：**

```
开始查询
   ↓
初始化 context = ""
iteration = 0
   ↓
[迭代开始] iteration++
   ↓
💭 思考: need_tool?
   ├─ No → 跳出循环 → 生成答案
   └─ Yes
      ↓
   ⚡ 行动: 执行工具
      ↓
   👁️ 观察: is_sufficient?
      ├─ Yes → 跳出循环 → 生成答案
      └─ No → 累积context → 继续迭代
                 ↓
            iteration < 5?
              ├─ Yes → 返回[迭代开始]
              └─ No → 跳出循环 → 生成答案
                         ↓
                      返回答案
```

---

## 四、完整代码总结

**代码文件：** `study-agentic-rag/01-rag-comparison/agentic_rag.py`

为了确保代码完整性，这里给出完整的 `agentic_rag.py` 文件：

<details>
<summary>点击展开完整代码（约620行）</summary>

```python
"""
Agentic RAG实现示例 - 使用阿里云百炼平台和ChromaDB
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
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from duckduckgo_search import DDGS
import json
from typing import List, Dict, Any

# 加载环境变量
load_dotenv()

# 配置常量
API_KEY = os.getenv("API_KEY", "sk-abe3417c96f6441b83efed38708bcfb6")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")

CHROMA_COLLECTION_NAME = "agentic_rag_collection"


class AgenticRAG:
    """Agentic RAG实现 - 基于ReAct框架，使用阿里云百炼平台和ChromaDB"""

    def __init__(self):
        """初始化连接、工具和代理配置"""
        print("🚀 初始化Agentic RAG系统...")

        # 初始化阿里云百炼平台客户端（兼容OpenAI接口）
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

        print(f"✅ 已连接到阿里云百炼平台")
        print(f"   LLM模型: {MODEL_ID}")
        print(f"   嵌入模型: {EMBEDDING_MODEL}")

        # 初始化ChromaDB（本地持久化向量数据库）
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db_agentic",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 获取或创建集合
        try:
            self.collection = self.chroma_client.get_collection(
                name=CHROMA_COLLECTION_NAME
            )
            print(f"📂 使用现有集合: {CHROMA_COLLECTION_NAME}")
            print(f"   当前文档数: {self.collection.count()}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"description": "Agentic RAG collection"}
            )
            print(f"✨ 创建新集合: {CHROMA_COLLECTION_NAME}")

        # 代理记忆 (短期记忆)
        self.memory = []

        # 最大迭代次数(防止无限循环)
        self.max_iterations = 5

    def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 嵌入失败: {e}")
            return [0.0] * 1536

    def ingest_documents(self, texts: List[str]):
        """摄入文档到ChromaDB向量数据库"""
        print(f"\n📥 正在摄入 {len(texts)} 个文档到ChromaDB...")

        # 生成文档ID
        ids = [f"doc_{i}" for i in range(len(texts))]

        # 获取嵌入向量
        print("   🔄 正在生成嵌入向量...")
        embeddings = []
        for i, text in enumerate(texts):
            print(f"      处理文档 {i+1}/{len(texts)}")
            embedding = self.get_embedding(text)
            embeddings.append(embedding)

        # 存储到ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": f"document_{i}"} for i in range(len(texts))]
        )

        print(f"✅ 文档摄入完成!")
        print(f"   集合中共有 {self.collection.count()} 个文档")

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

        try:
            # 获取查询的嵌入向量
            query_embedding = self.get_embedding(query)

            # 在ChromaDB中搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            if not results['documents'] or len(results['documents'][0]) == 0:
                return "未找到相关文档。"

            # 格式化结果
            formatted_docs = []
            for i, doc in enumerate(results['documents'][0]):
                formatted_docs.append(f"文档{i+1}: {doc}")

            result = "\n\n".join(formatted_docs)
            print(f"   ✅ 找到 {len(results['documents'][0])} 个相关文档")
            return result

        except Exception as e:
            print(f"   ❌ 向量搜索失败: {e}")
            return f"向量搜索失败: {str(e)}"

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
        """
        获取可用工具列表

        返回工具名称和描述的字典,用于代理决策
        """
        return {
            "vector_search": {
                "description": "从内部知识库(向量数据库)检索相关文档。适用于查询已知信息、历史数据、内部文档等。",
                "function": self.tool_vector_search,
                "parameters": {"query": "搜索查询"}
            },
            "web_search": {
                "description": "从互联网搜索最新信息。适用于需要实时数据、新闻、当前事件、最新发展等。",
                "function": self.tool_web_search,
                "parameters": {"query": "搜索查询"}
            },
            "calculator": {
                "description": "执行数学计算。适用于需要数值运算的问题。",
                "function": self.tool_calculator,
                "parameters": {"expression": "数学表达式"}
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
    "action": "工具名称(如果need_tool=true,可选: vector_search, web_search, calculator)",
    "action_input": "工具输入参数",
    "reasoning": "你的推理过程"
}}

如果可以直接回答,设置need_tool=false。
"""

        # 调用LLM进行推理
        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的AI代理,擅长分析问题并选择合适的工具。请始终以JSON格式回复。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 较低温度保证决策稳定性
                max_tokens=500
            )

            # 解析决策
            content = response.choices[0].message.content.strip()

            # 尝试提取JSON（处理可能的markdown代码块）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            decision = json.loads(content)
            print(f"   推理: {decision.get('reasoning', 'N/A')}")
            return decision

        except json.JSONDecodeError as e:
            print(f"   ⚠️ 决策解析失败: {e}")
            print(f"   原始响应: {content}")
            return {"need_tool": False, "reasoning": "解析错误，直接回答"}
        except Exception as e:
            print(f"   ⚠️ LLM调用失败: {e}")
            return {"need_tool": False, "reasoning": "调用错误"}

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

        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的信息质量评估专家。请始终以JSON格式回复。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            content = response.choices[0].message.content.strip()

            # 提取JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(content)
            print(f"   评估: {evaluation.get('reasoning', 'N/A')}")
            return evaluation

        except json.JSONDecodeError:
            print("   ⚠️ 评估解析失败,默认认为信息充足")
            return {"is_sufficient": True, "reasoning": "解析错误"}
        except Exception as e:
            print(f"   ⚠️ 评估失败: {e}")
            return {"is_sufficient": True, "reasoning": "评估错误"}

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

        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
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

        except Exception as e:
            print(f"   ❌ 答案生成失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"

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

    def reset_collection(self):
        """重置集合（清空所有文档）"""
        try:
            self.chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"description": "Agentic RAG collection"}
            )
            print("✅ 集合已重置")
        except Exception as e:
            print(f"❌ 重置失败: {e}")


# ============================================
# 使用示例
# ============================================

def main():
    """主函数 - 演示Agentic RAG的使用"""

    print("="*60)
    print("🎯 Agentic RAG 演示")
    print("   使用阿里云百炼平台 (Qwen) + ChromaDB + ReAct框架")
    print("="*60)

    # 初始化Agentic RAG
    agent_rag = AgenticRAG()

    # 示例文档
    sample_documents = [
        """
        人工智能(AI)是计算机科学的一个分支,致力于创建能够执行通常需要人类智能的任务的系统。
        这些任务包括视觉感知、语音识别、决策制定和语言翻译。AI技术在近年来取得了显著进展，
        特别是在深度学习和神经网络领域。
        """,
        """
        检索增强生成(RAG)是一种结合信息检索和文本生成的技术。
        它通过从外部知识库检索相关信息来增强语言模型的能力,从而减少幻觉并提高答案的准确性。
        传统RAG使用单一数据源,而Agentic RAG可以智能地选择多个数据源。
        RAG系统通常包括向量数据库、嵌入模型和大型语言模型三个核心组件。
        """,
        """
        Agentic RAG使用AI代理来增强检索过程。代理可以访问多种工具,包括向量搜索、Web搜索和计算器。
        通过ReAct框架,代理可以进行推理、行动和观察的循环,直到收集到足够的信息。
        这种方法显著提高了RAG系统处理复杂查询的能力。
        """,
        """
        ChromaDB是一个轻量级的本地向量数据库，非常适合开发和小规模应用。
        它支持本地持久化存储，不需要额外的服务器部署，使用简单方便。
        ChromaDB特别适合快速原型开发和教学演示。
        """
    ]

    # 摄入文档
    agent_rag.ingest_documents(sample_documents)

    # 测试查询
    test_questions = [
        "什么是Agentic RAG?它与传统RAG有什么区别?",
        "ChromaDB有什么特点和优势?",
        "如果一个RAG系统每天处理1000个查询,每个查询调用LLM 3次,一个月调用多少次?"  # 需要计算器
    ]

    for question in test_questions:
        answer = agent_rag.query(question)

        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"💡 答案:\n{answer}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
```

</details>

---

## 五、运行测试

### 1. 运行程序

```bash
python agentic_rag.py
```

### 2. 预期输出（示例）

**问题1：什么是Agentic RAG?它与传统RAG有什么区别?**

```
🔄 迭代 1/5

💭 [思考] 代理正在分析问题...
   推理: 用户询问Agentic RAG的定义和与传统RAG的区别，应该从内部知识库检索相关信息

⚡ [行动] 执行工具: vector_search

🔧 [工具] 向量搜索: 'Agentic RAG'
   ✅ 找到 3 个相关文档

👁️  [观察] 评估结果质量...
   评估: 检索到的信息包含Agentic RAG的定义和工作原理，以及与传统RAG的对比，信息充足

✅ 代理评估: 信息充足,停止检索

🤖 [生成] 生成最终答案...
   ✅ 答案生成完成!

💡 答案:
Agentic RAG是一种使用AI代理来增强检索过程的技术。它的核心特点是通过ReAct框架，
让代理能够进行"推理-行动-观察"的循环，直到收集到足够的信息。

与传统RAG的主要区别：
1. **数据源**: 传统RAG使用单一数据源，而Agentic RAG可以智能选择多个数据源
2. **工具能力**: Agentic RAG代理可以访问多种工具（向量搜索、Web搜索、计算器等）
3. **处理复杂度**: Agentic RAG显著提高了处理复杂查询的能力
4. **工作流程**: 传统RAG是固定流程，Agentic RAG可以根据情况动态调整
```

**问题3：如果一个RAG系统每天处理1000个查询,每个查询调用LLM 3次,一个月调用多少次?**

```
🔄 迭代 1/5

💭 [思考] 代理正在分析问题...
   推理: 这是一个数学计算问题，需要使用计算器工具

⚡ [行动] 执行工具: calculator

🔧 [工具] 计算器: '1000 * 3 * 30'
   ✅ 计算结果: 90000

👁️  [观察] 评估结果质量...
   评估: 计算结果准确，信息充足

✅ 代理评估: 信息充足,停止检索

🤖 [生成] 生成最终答案...
   ✅ 答案生成完成!

💡 答案:
如果一个RAG系统每天处理1000个查询，每个查询调用LLM 3次，
一个月（按30天计算）调用LLM的总次数为：

1000（查询/天）× 3（调用/查询）× 30（天）= 90,000次

因此，一个月总共会调用LLM **90,000次**。
```

---

## 六、核心差异总结

**传统RAG vs Agentic RAG 实现差异：**

| 维度 | 传统RAG | Agentic RAG |
|------|---------|-------------|
| **代码行数** | ~200行 | ~620行 |
| **工具数量** | 1个（向量搜索） | 3+个（可扩展） |
| **LLM调用** | 1次（生成） | 3-5次（思考+评估+生成） |
| **决策能力** | 无（固定流程） | 有（智能选择工具） |
| **质量评估** | 无 | 有（observe阶段） |
| **迭代能力** | 无 | 有（ReAct循环） |
| **适用场景** | 简单查询 | 复杂查询 |
| **响应时间** | 1-2秒 | 5-10秒 |
| **成本** | 低 | 中高 |

---

## 下一讲预告

我们已经实现了传统RAG和Agentic RAG两个系统，但到底哪个更好呢？

**第5讲：RAG技术对比 - 准确性成本与性能分析**

我们将：
- 设计对比实验
- 测试准确率、响应时间、成本
- 分析各自的优势和劣势
- 给出技术选型建议

敬请期待！
