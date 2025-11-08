# 第7讲：传统RAG引擎封装 - 快速检索与答案生成

在第6讲中，我们实现了智能路由器，能够自动识别查询复杂度。现在路由器会把70%的简单查询分配给传统RAG引擎。

这一讲，我们将构建一个**生产级的传统RAG引擎**，专注于快速、准确地处理简单查询。

---

## 一、传统RAG引擎的定位

### 在混合系统中的角色

```
用户查询
   ↓
路由器判断
   ↓
简单查询(70%) → [传统RAG引擎] → 快速回答（1-2秒）
复杂查询(30%) → [Agentic RAG引擎] → 深度推理（8-10秒）
```

### 设计目标

| 目标 | 指标 | 为什么重要 |
|------|------|-----------|
| **速度** | <2秒响应 | 简单查询要求快速反馈 |
| **成本** | 单次¥0.01 | 高频使用，成本敏感 |
| **准确性** | 85%+ | 简单查询有参考答案 |
| **稳定性** | 99%可用 | 处理大多数流量 |

### 与第3讲的区别

**第3讲的传统RAG**：
- 教学演示版本
- 单文件实现（~200行）
- 最小功能集

**本讲的传统RAG引擎**：
- 生产级封装
- 模块化设计
- 完善的错误处理
- 可配置参数
- 批量文档处理
- 性能优化

---

## 二、引擎架构设计

### 核心流程

```
┌────────────────────────────────────────────────┐
│          传统RAG引擎工作流程                      │
│                                                │
│  1️⃣ 初始化阶段                                   │
│     连接LLM → 连接ChromaDB → 加载集合            │
│                                                │
│  2️⃣ 文档管理阶段                                 │
│     添加文档 → 生成嵌入 → 存储向量               │
│                                                │
│  3️⃣ 查询阶段                                     │
│     检索(Retrieve) → 生成(Generate) → 返回答案   │
│                                                │
└────────────────────────────────────────────────┘
```

### 三大核心模块

```python
class TraditionalRAGEngine:
    """传统RAG引擎"""

    # 模块1: 初始化与配置
    def __init__(self): ...
    def get_embedding(self, text: str): ...

    # 模块2: 文档管理
    def add_documents(self, texts, metadatas): ...

    # 模块3: 查询处理
    def retrieve(self, query: str, k: int): ...
    def generate_answer(self, query: str, docs: List): ...
    def query(self, question: str): ...
```

---

## 三、代码实现（分3部分详解）

### 第1部分：初始化与配置

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/traditional_rag_engine.py`

**这部分要做什么？**

建立引擎的基础设施：连接LLM、连接ChromaDB、加载或创建向量集合。

```python
"""
传统RAG引擎
Traditional RAG Engine

用于处理简单的事实性查询
"""

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from typing import List, Tuple
import config


class TraditionalRAGEngine:
    """传统RAG引擎 - 一次性检索+生成"""

    def __init__(self):
        """初始化传统RAG引擎"""
        print("🚀 初始化Traditional RAG引擎...")

        # 初始化OpenAI客户端（阿里云百炼）
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
            print(f"   文档数: {self.collection.count()}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=config.COLLECTION_NAME
            )
            print(f"✨ 创建新集合: {config.COLLECTION_NAME}")

    def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 嵌入失败: {e}")
            return [0.0] * 1536  # 返回零向量作为降级
```

**为什么这么写？**

1. **为什么用 `PersistentClient` 而不是 `Client`？**
   - `PersistentClient`：数据持久化到磁盘
   - 重启程序后数据不丢失
   - 生产环境必须持久化

   ```python
   # 对比
   Client()  # 内存数据库（重启丢失）
   PersistentClient(path="./db")  # 持久化（重启保留）
   ```

2. **为什么 `anonymized_telemetry=False`？**
   - ChromaDB默认会发送匿名使用数据
   - 企业环境需要关闭（数据隐私）
   - 避免网络请求影响性能

3. **为什么用 `try-except` 处理集合？**
   - 首次运行时集合不存在（异常）
   - 后续运行时集合已存在（正常）
   - 优雅处理两种情况

4. **为什么嵌入失败时返回零向量？**
   - 降级策略，避免程序崩溃
   - 零向量会导致检索结果差，但不会报错
   - 生产环境可以记录到日志，触发告警

**初始化流程图：**

```
启动引擎
   ↓
连接LLM
   ↓
连接ChromaDB
   ↓
尝试加载集合
   ├─ 成功 → 输出文档数
   └─ 失败 → 创建新集合
   ↓
初始化完成
```

---

### 第2部分：文档管理

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/traditional_rag_engine.py`

**这部分要做什么？**

批量添加文档到向量数据库，包括生成嵌入和存储。

```python
    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """
        添加文档到向量数据库

        Args:
            texts: 文本列表
            metadatas: 元数据列表（可选）
        """
        print(f"\n📥 正在添加 {len(texts)} 个文档...")

        # 如果没有提供元数据，自动生成
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]

        # 生成嵌入
        embeddings = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"   处理进度: {i+1}/{len(texts)}")
            embedding = self.get_embedding(text)
            embeddings.append(embedding)

        # 添加到ChromaDB
        ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"✅ 文档添加完成！")
```

**为什么这么写？**

1. **为什么元数据是可选的？**
   - 简单场景不需要复杂元数据
   - 自动生成默认值（source字段）
   - 保持API灵活性

   ```python
   # 用法示例
   engine.add_documents(["文档1", "文档2"])  # 自动生成元数据
   engine.add_documents(
       ["文档1", "文档2"],
       [{"type": "FAQ"}, {"type": "Manual"}]  # 自定义元数据
   )
   ```

2. **为什么显示处理进度？**
   - 大批量文档可能需要几分钟
   - 进度提示改善用户体验
   - 便于判断是否卡住

3. **为什么ID用 `doc_{i}` 格式？**
   - 简单递增，避免冲突
   - 便于调试和查询
   - 生产环境可用UUID或hash

4. **为什么嵌入生成是串行的？**
   - 阿里云API有限流限制
   - 并行可能导致429错误
   - 简单场景串行足够（可优化为批量）

**优化建议（生产环境）：**

```python
def add_documents_batch(self, texts: List[str], batch_size=10):
    """批量处理文档（优化版）"""
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # 批量生成嵌入
        embeddings = self.get_embeddings_batch(batch)
        # 批量添加
        self.collection.add(...)
```

---

### 第3部分：查询处理

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/traditional_rag_engine.py`

**这部分要做什么？**

实现完整的RAG流程：检索相关文档 → 生成答案。

```python
    def retrieve(self, query: str, k: int = config.TOP_K) -> List[dict]:
        """
        检索相关文档

        Args:
            query: 查询文本
            k: 返回文档数量

        Returns:
            文档列表
        """
        print(f"\n🔍 检索: '{query}'")

        # 获取查询嵌入
        query_embedding = self.get_embedding(query)

        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # 格式化结果
        docs = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0] if results['metadatas'] else [{}] * k,
                results['distances'][0] if results['distances'] else [0] * k
            )):
                docs.append({
                    'content': doc,
                    'metadata': metadata,
                    'distance': distance
                })

        print(f"✅ 找到 {len(docs)} 个相关文档")
        return docs

    def generate_answer(self, query: str, docs: List[dict]) -> str:
        """
        生成答案

        Args:
            query: 用户问题
            docs: 检索到的文档

        Returns:
            生成的答案
        """
        print(f"\n🤖 生成答案...")

        # 构建上下文
        context = "\n\n".join([
            f"参考资料{i+1}:\n{doc['content']}"
            for i, doc in enumerate(docs)
        ])

        # 构建提示词
        prompt = f"""你是一个专业的问答助手。请根据以下参考资料回答用户的问题。

参考资料:
{context}

用户问题: {query}

要求:
1. 仅使用参考资料中的信息回答
2. 回答要准确、简洁、相关
3. 如果参考资料不足以回答问题，请如实说明

回答:"""

        # 调用LLM
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的问答助手，擅长从参考资料中提取信息回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE_GENERATION,
                max_tokens=config.MAX_TOKENS_GENERATION
            )

            answer = response.choices[0].message.content
            print("✅ 答案生成完成")
            return answer

        except Exception as e:
            print(f"❌ 答案生成失败: {e}")
            return f"抱歉，生成答案时出错: {str(e)}"

    def query(self, question: str) -> Tuple[str, List[dict]]:
        """
        完整的传统RAG流程

        Args:
            question: 用户问题

        Returns:
            (答案, 检索到的文档)
        """
        print(f"\n{'='*60}")
        print(f"📝 [传统RAG] 问题: {question}")
        print(f"{'='*60}")

        # 步骤1: 检索
        docs = self.retrieve(question)

        # 步骤2: 生成答案
        answer = self.generate_answer(question, docs)

        return answer, docs
```

**为什么这么写？**

1. **为什么 `retrieve()` 返回dict列表？**
   - 包含多个信息：内容、元数据、距离
   - 便于后续分析和调试
   - 灵活的数据结构

   ```python
   # 返回格式
   [
       {
           'content': "文档内容...",
           'metadata': {"source": "doc_0"},
           'distance': 0.234  # 越小越相似
       },
       ...
   ]
   ```

2. **为什么用 `distance` 而不是 `similarity`？**
   - ChromaDB默认返回distance（欧氏距离）
   - distance = 0 表示完全相同
   - distance越小，相似度越高

3. **为什么Prompt要求"仅使用参考资料"？**
   - 避免LLM幻觉
   - 确保答案可追溯
   - 符合RAG的设计初衷

4. **为什么 `temperature=0.7`？**
   - 生成阶段需要一定创造性
   - 使答案自然流畅
   - 不需要像分类那么严格（0.1）

5. **为什么返回 `(answer, docs)` 元组？**
   - 答案是主要输出
   - docs用于溯源和调试
   - 便于统计和分析

**查询流程图：**

```
query(question)
   ↓
retrieve(question)
   ├─ 查询嵌入
   ├─ 向量检索
   └─ 返回TOP-K文档
   ↓
generate_answer(question, docs)
   ├─ 构建上下文
   ├─ 调用LLM
   └─ 返回答案
   ↓
返回(answer, docs)
```

---

## 四、完整代码总结

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/traditional_rag_engine.py`

完整的 `traditional_rag_engine.py` 文件（约250行）：

<details>
<summary>点击展开完整代码</summary>

```python
"""
传统RAG引擎
Traditional RAG Engine

用于处理简单的事实性查询
"""

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from typing import List, Tuple
import config


class TraditionalRAGEngine:
    """传统RAG引擎 - 一次性检索+生成"""

    def __init__(self):
        """初始化传统RAG引擎"""
        print("🚀 初始化Traditional RAG引擎...")

        # 初始化OpenAI客户端（阿里云百炼）
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
            print(f"   文档数: {self.collection.count()}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=config.COLLECTION_NAME
            )
            print(f"✨ 创建新集合: {config.COLLECTION_NAME}")

    def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 嵌入失败: {e}")
            return [0.0] * 1536

    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """
        添加文档到向量数据库

        Args:
            texts: 文本列表
            metadatas: 元数据列表
        """
        print(f"\n📥 正在添加 {len(texts)} 个文档...")

        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]

        # 生成嵌入
        embeddings = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"   处理进度: {i+1}/{len(texts)}")
            embedding = self.get_embedding(text)
            embeddings.append(embedding)

        # 添加到ChromaDB
        ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"✅ 文档添加完成！")

    def retrieve(self, query: str, k: int = config.TOP_K) -> List[dict]:
        """
        检索相关文档

        Args:
            query: 查询文本
            k: 返回文档数量

        Returns:
            文档列表
        """
        print(f"\n🔍 检索: '{query}'")

        # 获取查询嵌入
        query_embedding = self.get_embedding(query)

        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # 格式化结果
        docs = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0] if results['metadatas'] else [{}] * k,
                results['distances'][0] if results['distances'] else [0] * k
            )):
                docs.append({
                    'content': doc,
                    'metadata': metadata,
                    'distance': distance
                })

        print(f"✅ 找到 {len(docs)} 个相关文档")
        return docs

    def generate_answer(self, query: str, docs: List[dict]) -> str:
        """
        生成答案

        Args:
            query: 用户问题
            docs: 检索到的文档

        Returns:
            生成的答案
        """
        print(f"\n🤖 生成答案...")

        # 构建上下文
        context = "\n\n".join([
            f"参考资料{i+1}:\n{doc['content']}"
            for i, doc in enumerate(docs)
        ])

        # 构建提示词
        prompt = f"""你是一个专业的问答助手。请根据以下参考资料回答用户的问题。

参考资料:
{context}

用户问题: {query}

要求:
1. 仅使用参考资料中的信息回答
2. 回答要准确、简洁、相关
3. 如果参考资料不足以回答问题，请如实说明

回答:"""

        # 调用LLM
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的问答助手，擅长从参考资料中提取信息回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE_GENERATION,
                max_tokens=config.MAX_TOKENS_GENERATION
            )

            answer = response.choices[0].message.content
            print("✅ 答案生成完成")
            return answer

        except Exception as e:
            print(f"❌ 答案生成失败: {e}")
            return f"抱歉，生成答案时出错: {str(e)}"

    def query(self, question: str) -> Tuple[str, List[dict]]:
        """
        完整的传统RAG流程

        Args:
            question: 用户问题

        Returns:
            (答案, 检索到的文档)
        """
        print(f"\n{'='*60}")
        print(f"📝 [传统RAG] 问题: {question}")
        print(f"{'='*60}")

        # 步骤1: 检索
        docs = self.retrieve(question)

        # 步骤2: 生成答案
        answer = self.generate_answer(question, docs)

        return answer, docs


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    # 初始化引擎
    rag = TraditionalRAGEngine()

    # 示例文档
    sample_docs = [
        "孙悟空原本是花果山上的一块仙石孕育而生的石猴，后来拜菩提祖师为师，学得七十二变和筋斗云等神通。",
        "唐僧俗姓陈，法号玄奘，是如来佛祖的二弟子金蝉子转世。他奉唐太宗之命前往西天取经。",
        "猪八戒原是天蓬元帅，因调戏嫦娥被贬下凡，错投猪胎。后被观音菩萨点化，等待取经人。",
        "沙悟净原是天庭的卷帘大将，因失手打碎琉璃盏被贬下界，在流沙河为妖。后被观音点化，保护唐僧取经。",
        "孙悟空的两个师傅分别是菩提祖师和唐僧。菩提祖师教他法术神通，唐僧则是他取经路上的师父。"
    ]

    # 添加文档
    rag.add_documents(sample_docs)

    # 测试查询
    test_questions = [
        "孙悟空的师傅是谁？",
        "唐僧是谁转世？",
        "猪八戒为什么被贬下凡？"
    ]

    for question in test_questions:
        answer, docs = rag.query(question)

        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"💡 答案: {answer}")
        print(f"\n📚 参考文档:")
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] (距离: {doc['distance']:.4f})")
            print(f"      {doc['content'][:80]}...")
        print(f"{'='*60}\n")
```

</details>

---

## 五、运行测试

### 1. 配置文件（config.py）

```python
import os
from dotenv import load_dotenv

load_dotenv()

# LLM配置
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_ID = "qwen-plus"
EMBEDDING_MODEL = "text-embedding-v1"

# ChromaDB配置
CHROMA_DB_PATH = "./vector_db"
COLLECTION_NAME = "hybrid_rag_collection"

# RAG参数
TOP_K = 3
TEMPERATURE_GENERATION = 0.7
MAX_TOKENS_GENERATION = 1000
```

### 2. 运行引擎

```bash
python traditional_rag_engine.py
```

### 3. 预期输出

```
🚀 初始化Traditional RAG引擎...
✨ 创建新集合: hybrid_rag_collection

📥 正在添加 5 个文档...
✅ 文档添加完成！

============================================================
📝 [传统RAG] 问题: 孙悟空的师傅是谁？
============================================================

🔍 检索: '孙悟空的师傅是谁？'
✅ 找到 3 个相关文档

🤖 生成答案...
✅ 答案生成完成

============================================================
❓ 问题: 孙悟空的师傅是谁？
💡 答案: 孙悟空有两个师傅：菩提祖师和唐僧。菩提祖师教他法术神通，唐僧是他取经路上的师父。

📚 参考文档:
  [1] (距离: 0.1234)
      孙悟空的两个师傅分别是菩提祖师和唐僧。菩提祖师教他法术神通，唐僧则是他取经路上的师父。...
  [2] (距离: 0.2456)
      孙悟空原本是花果山上的一块仙石孕育而生的石猴，后来拜菩提祖师为师，学得七十二变和筋斗云等神通...
  [3] (距离: 0.3567)
      唐僧俗姓陈，法号玄奘，是如来佛祖的二弟子金蝉子转世。他奉唐太宗之命前往西天取经...
============================================================
```

---

## 六、性能优化

### 优化1：嵌入缓存

```python
class TraditionalRAGEngine:
    def __init__(self):
        ...
        self.embedding_cache = {}  # 嵌入缓存

    def get_embedding(self, text: str):
        # 检查缓存
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        # 调用API
        embedding = self._call_embedding_api(text)

        # 保存缓存
        self.embedding_cache[text] = embedding
        return embedding
```

**价值：**
- 相同查询直接返回缓存
- 节省API调用成本
- 减少响应时间

### 优化2：批量嵌入

```python
def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
    """批量生成嵌入（更高效）"""
    try:
        response = self.client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=texts  # 批量输入
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"❌ 批量嵌入失败: {e}")
        return [[0.0] * 1536] * len(texts)
```

**价值：**
- 单次API调用处理多个文本
- 减少网络开销
- 提升吞吐量

### 优化3：检索结果过滤

```python
def retrieve(self, query: str, k: int = config.TOP_K):
    ...
    # 过滤距离过大的文档
    docs = [
        doc for doc in docs
        if doc['distance'] < config.SIMILARITY_THRESHOLD
    ]

    if not docs:
        print("⚠️ 未找到足够相似的文档")

    return docs
```

**价值：**
- 避免不相关文档污染上下文
- 提升答案质量
- 防止误导性回答

---

## 七、总结

### 核心要点

1. **传统RAG引擎的定位**
   - 处理70%的简单查询
   - 追求速度和成本效率
   - 固定两步流程（检索→生成）

2. **三大核心模块**
   - 初始化：连接LLM和ChromaDB
   - 文档管理：批量添加文档
   - 查询处理：检索+生成

3. **生产级特性**
   - ✅ 持久化存储（PersistentClient）
   - ✅ 错误处理（try-except）
   - ✅ 进度提示（用户体验）
   - ✅ 灵活配置（config文件）
   - ✅ 元数据支持（便于溯源）

### 传统RAG vs 教学版本

| 特性 | 教学版本（第3讲） | 生产版本（本讲） |
|------|----------------|----------------|
| **持久化** | 可选 | 必须 |
| **错误处理** | 基础 | 完善 |
| **配置管理** | 硬编码 | config文件 |
| **元数据** | 无 | 支持 |
| **进度提示** | 无 | 有 |
| **代码行数** | ~200行 | ~250行 |

---

## 下一讲预告

传统RAG引擎完成了，下一步是构建Agentic RAG引擎。

**第8讲：Agentic RAG引擎实现 - ReAct循环与质量评估**

我们将学习如何封装**生产级的Agentic RAG引擎**：
- ReAct框架完整实现
- 多工具系统集成
- 智能决策与质量评估
- 可配置的迭代策略
- 完善的日志和监控

敬请期待！
