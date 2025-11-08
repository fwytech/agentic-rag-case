# 第3讲：从零搭建传统RAG - 文档检索与答案生成

## 本讲目标

- 掌握 ChromaDB 向量数据库的使用
- 理解文档向量化的完整流程
- 实现相似度检索算法
- 完成 LLM 答案生成
- 运行一个完整的传统 RAG 系统

## 前置知识

- 已学习第1-2讲的理论知识
- 了解 Python 基础语法
- 知道什么是 API 和环境变量

---

## 一、传统 RAG 工作流程回顾

在开始编码前，我们先回顾一下传统 RAG 的完整流程：

```
┌─────────────────────────────────────────────────┐
│          传统 RAG 完整工作流程                    │
└─────────────────────────────────────────────────┘

【离线阶段 - 构建知识库】
┌──────────────┐
│ 1. 文档准备  │  准备要索引的文档
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 2. 文本分块  │  将长文档切分成小段
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 3. 向量化    │  调用Embedding API，文本→向量
│   (Embedding)│  "机器学习" → [0.12, -0.34, 0.89, ...]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 4. 存储到向量库│  ChromaDB存储：文本 + 向量
└──────────────┘

【在线阶段 - 回答问题】
┌──────────────┐
│ 5. 用户提问  │  "什么是机器学习？"
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 6. 查询向量化│  问题 → 向量
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 7. 相似度搜索│  在向量库中找最相似的文档
│   (TOP-K)    │  向量距离：0.15, 0.28, 0.45...
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 8. 拼接上下文│  将找到的文档组合成上下文
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 9. LLM生成   │  基于上下文生成答案
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 10. 返回答案 │
└──────────────┘
```

---

## 二、环境准备

### 2.1 安装依赖

创建 `requirements.txt`：

```txt
# LLM和API客户端
openai>=1.0.0           # 兼容阿里云百炼
python-dotenv>=1.0.0    # 环境变量管理

# 向量数据库
chromadb>=0.4.22        # 本地向量数据库

# 文档处理（可选）
PyPDF2>=3.0.0           # PDF处理
```

**安装命令：**

```bash
# 使用 uv（推荐）
cd study-agentic-rag/01-rag-comparison
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 2.2 配置环境变量

创建 `.env` 文件：

```bash
# 阿里云百炼配置
API_KEY=sk-abe3417c96f6441b83efed38708bcfb6
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_ID=qwen-plus
EMBEDDING_MODEL=text-embedding-v1
```

### 2.3 为什么选择这些技术？

**为什么用阿里云百炼？**
- 兼容 OpenAI API，代码通用
- 国内访问稳定快速
- Qwen 模型中文效果好

**为什么用 ChromaDB？**
- 本地化部署，无需服务器
- 轻量级，适合学习和小项目
- 自动持久化，重启不丢数据
- Python原生支持，易于使用

---

## 三、代码实战：逐步实现

### 第一部分：初始化和配置（建立连接）

**代码文件：** `study-agentic-rag/01-rag-comparison/traditional_rag.py`

**这部分要做什么？**
1. 加载环境变量
2. 连接到阿里云百炼 LLM
3. 初始化 ChromaDB 向量数据库

**代码实现：**

```python
"""
traditional_rag.py - 传统RAG完整实现
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from typing import List

# 1. 加载环境变量（从.env文件读取配置）
load_dotenv()

# 2. 配置常量
# 为什么从环境变量读取？方便不同环境切换配置，不用改代码
API_KEY = os.getenv("API_KEY", "默认值")  # 第二个参数是默认值
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")

# ChromaDB集合名称（类似数据库表名）
CHROMA_COLLECTION_NAME = "traditional_rag_collection"


class TraditionalRAG:
    """传统RAG实现类"""

    def __init__(self):
        """初始化：建立LLM和向量数据库连接"""
        print("🚀 初始化Traditional RAG系统...")

        # 3. 初始化OpenAI客户端（兼容阿里云百炼）
        # 为什么用OpenAI库？因为阿里云百炼兼容OpenAI接口格式
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL  # 改成阿里云的地址
        )

        print(f"✅ 已连接到阿里云百炼平台")
        print(f"   模型: {MODEL_ID}")
        print(f"   嵌入模型: {EMBEDDING_MODEL}")

        # 4. 初始化ChromaDB（本地向量数据库）
        # PersistentClient: 数据会持久化保存到磁盘
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",  # 数据存储路径
            settings=Settings(
                anonymized_telemetry=False,  # 关闭遥测
                allow_reset=True  # 允许重置
            )
        )

        # 5. 获取或创建集合（类似数据库的表）
        try:
            # 尝试获取已存在的集合
            self.collection = self.chroma_client.get_collection(
                name=CHROMA_COLLECTION_NAME
            )
            print(f"📂 使用现有集合: {CHROMA_COLLECTION_NAME}")
            print(f"   当前文档数: {self.collection.count()}")
        except Exception:
            # 如果不存在，创建新集合
            self.collection = self.chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"description": "Traditional RAG collection"}
            )
            print(f"✨ 创建新集合: {CHROMA_COLLECTION_NAME}")
```

**代码解释：**

**为什么要用类（class）？**
- 把相关的数据和方法组织在一起
- 方便维护和复用
- `self.client` 和 `self.collection` 可以在所有方法中使用

**为什么要 try-except？**
- 第一次运行时集合不存在，`get_collection` 会报错
- 用 try-except 处理：存在就用，不存在就创建

**PersistentClient vs Client 的区别？**
- `Client`：数据只在内存中，程序关闭就没了
- `PersistentClient`：数据保存到磁盘，下次启动还在

---

### 第二部分：文档嵌入（文本变成数字）

**代码文件：** `study-agentic-rag/01-rag-comparison/traditional_rag.py`

**这部分要做什么？**
1. 将文本转换为向量（调用 Embedding API）
2. 将文档批量导入向量数据库

**代码实现：**

```python
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量

        参数:
            text: 要转换的文本

        返回:
            一个浮点数列表（向量），长度通常是1536
        """
        try:
            # 调用阿里云的嵌入API
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            # 返回第一个结果的embedding
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 嵌入失败: {e}")
            # 失败时返回零向量（实际项目应该抛出异常）
            return [0.0] * 1536

    def ingest_documents(self, texts: List[str]):
        """
        将文档导入向量数据库

        流程：
        1. 为每个文档生成ID
        2. 调用Embedding API获取向量
        3. 存储到ChromaDB
        """
        print(f"\n📥 正在摄入 {len(texts)} 个文档...")

        # 步骤1：生成文档ID
        # 为什么需要ID？ChromaDB需要唯一标识每个文档
        ids = [f"doc_{i}" for i in range(len(texts))]

        # 步骤2：批量生成嵌入向量
        print("   🔄 正在生成嵌入向量...")
        embeddings = []
        for i, text in enumerate(texts):
            print(f"      处理文档 {i+1}/{len(texts)}")
            embedding = self.get_embedding(text)
            embeddings.append(embedding)

        # 步骤3：存储到ChromaDB
        # 为什么分开存储ids、embeddings、documents？
        # ChromaDB需要分别指定：ID列表、向量列表、原文列表
        self.collection.add(
            ids=ids,                    # 文档ID列表
            embeddings=embeddings,       # 向量列表
            documents=texts,             # 原文列表
            metadatas=[{"source": f"document_{i}"} for i in range(len(texts))]  # 元数据
        )

        print(f"✅ 文档摄入完成!")
        print(f"   集合中共有 {self.collection.count()} 个文档")
```

**代码解释：**

**什么是 Embedding（嵌入向量）？**
```
文本："机器学习是AI的一个分支"
↓ (调用Embedding API)
向量：[0.12, -0.34, 0.89, 0.45, ..., -0.23]  # 1536个数字
```

**为什么要转成向量？**
- 计算机只认识数字，不认识文字
- 相似的文本会有相似的向量
- 可以用向量距离衡量文本相似度

**为什么循环调用 API 而不是一次性？**
- 有些 Embedding API 单次只能处理一定长度
- 分批处理更稳定，出错了可以只重试失败的部分
- 可以显示进度

**metadata 是什么？**
- 元数据，存储文档的附加信息
- 例如：来源、作者、时间等
- 搜索时可以用来过滤

---

### 第三部分：向量搜索（找到最相关的文档）

**代码文件：** `study-agentic-rag/01-rag-comparison/traditional_rag.py`

**这部分要做什么？**
1. 将用户问题转成向量
2. 在向量库中找最相似的文档
3. 返回 Top-K 个结果

**代码实现：**

```python
    def search(self, query: str, k: int = 3) -> List[dict]:
        """
        执行向量相似度搜索

        参数:
            query: 用户查询
            k: 返回TOP-K个最相似的文档（默认3个）

        返回:
            文档列表，每个文档包含：content（内容）、metadata（元数据）、distance（距离）
        """
        print(f"\n🔍 执行向量搜索: '{query}'")
        print(f"   检索TOP-{k}个相关文档...")

        # 步骤1：将查询转成向量
        # 为什么？要和文档向量进行比较，必须在同一个向量空间
        query_embedding = self.get_embedding(query)

        # 步骤2：在ChromaDB中搜索
        # query: 查询向量 → 返回最相似的n_results个文档
        results = self.collection.query(
            query_embeddings=[query_embedding],  # 查询向量（列表形式）
            n_results=k  # 返回TOP-K
        )

        # 步骤3：格式化结果
        docs = []
        if results['documents'] and len(results['documents']) > 0:
            # ChromaDB返回的结果是嵌套列表，需要展开
            for i, doc in enumerate(results['documents'][0]):
                docs.append({
                    'content': doc,  # 文档原文
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })

        print(f"✅ 找到 {len(docs)} 个相关文档")
        return docs
```

**代码解释：**

**相似度是怎么计算的？**
```
用户查询向量：   [0.12, -0.34, 0.89, ...]
文档1向量：      [0.15, -0.30, 0.92, ...]  → 距离: 0.15（很相似✓）
文档2向量：      [0.80,  0.65, -0.12, ...] → 距离: 1.85（不相似✗）
```

常用距离度量：
- **欧氏距离**：直线距离
- **余弦距离**：角度距离
- **点积**：向量乘积

ChromaDB 默认使用欧氏距离。

**为什么返回 Top-K 而不是所有匹配的？**
- LLM 上下文有长度限制（如4K、8K tokens）
- 太多文档会引入噪音，降低准确性
- Top-3 或 Top-5 通常就够用了

**distance 越小越好还是越大越好？**
- **距离越小越相似**
- distance=0 表示完全相同
- distance>1.0 通常表示不太相关

---

### 第四部分：LLM 答案生成（基于上下文回答）

**代码文件：** `study-agentic-rag/01-rag-comparison/traditional_rag.py`

**这部分要做什么？**
1. 将检索到的文档拼接成上下文
2. 构造 Prompt（提示词）
3. 调用 LLM 生成答案

**代码实现：**

```python
    def format_context(self, docs: List[dict]) -> str:
        """
        格式化检索到的文档为上下文字符串

        为什么要格式化？
        - 让文档之间有明确分隔
        - 方便LLM理解这是多个文档片段
        """
        # 用 "---" 分隔每个文档
        context = "\n\n---\n\n".join([doc['content'] for doc in docs])
        return context

    def generate_answer(self, query: str, context: str) -> str:
        """
        使用LLM生成答案

        参数:
            query: 用户问题
            context: 检索到的上下文

        返回:
            LLM生成的答案
        """
        print(f"\n🤖 正在生成答案...")

        # 步骤1：构建Prompt
        # 为什么这样设计Prompt？
        # - 明确告诉LLM它的角色
        # - 提供上下文信息
        # - 说明如果不知道要诚实回答
        prompt = f"""你是一个有用的AI助手。请根据以下上下文回答用户的问题。

上下文:
{context}

问题: {query}

请提供准确、详细的答案。如果上下文中没有足够的信息，请诚实地说明。"""

        # 步骤2：调用阿里云百炼LLM
        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    # System消息：定义LLM的角色
                    {"role": "system", "content": "你是一个专业的AI助手，擅长根据提供的上下文回答问题。"},
                    # User消息：用户的实际提问
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,      # 创造性程度（0-1，越高越随机）
                max_tokens=1000       # 最大生成长度
            )

            # 步骤3：提取答案
            answer = response.choices[0].message.content
            print("✅ 答案生成完成!")
            return answer

        except Exception as e:
            print(f"❌ LLM调用失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"
```

**代码解释：**

**temperature 参数是什么？**
```
temperature = 0.0  → 答案固定，每次都一样（适合事实性问题）
temperature = 0.7  → 稍有变化，比较平衡（推荐）
temperature = 1.0  → 非常随机，很有创意（适合创作）
```

**为什么要分 system 和 user 消息？**
- `system`：设定 LLM 的行为规则（不会直接显示）
- `user`：实际的用户提问
- `assistant`：LLM 的回复（在多轮对话中使用）

**为什么要在 Prompt 中说"诚实回答"？**
- 防止 LLM 编造答案（幻觉问题）
- 如果上下文中真的没有答案，应该说"不知道"
- 这是 Prompt 工程的最佳实践

---

### 第五部分：完整流程整合（一气呵成）

**代码文件：** `study-agentic-rag/01-rag-comparison/traditional_rag.py`

**这部分要做什么？**
将上面的步骤串起来，形成完整的 RAG 查询流程。

**代码实现：**

```python
    def query(self, question: str, k: int = 3) -> tuple:
        """
        传统RAG完整流程：搜索 → 拼接 → 生成

        参数:
            question: 用户问题
            k: 检索文档数量

        返回:
            (答案, 检索到的文档列表)
        """
        print(f"\n{'='*60}")
        print(f"📝 用户问题: {question}")
        print(f"{'='*60}")

        # 步骤1：向量搜索（包含了查询向量化）
        docs = self.search(question, k=k)

        # 步骤2：检查是否找到文档
        if not docs:
            return "抱歉，没有找到相关信息。", []

        # 步骤3：格式化上下文
        context = self.format_context(docs)
        print(f"\n📄 上下文长度: {len(context)} 字符")

        # 步骤4：LLM生成答案
        answer = self.generate_answer(question, context)

        # 步骤5：返回答案和文档
        # 为什么也返回docs？方便查看引用了哪些文档
        return answer, docs
```

**代码解释：**

**为什么返回 tuple（元组）？**
- 同时返回两个信息：答案 + 使用的文档
- 用户可以查看答案的来源（可追溯性）
- `answer, docs = rag.query("问题")` 可以同时接收

**为什么要检查 docs 是否为空？**
- 如果向量库是空的，或者查询太奇怪
- 避免把空上下文传给 LLM
- 提前返回友好的错误消息

---

## 四、完整代码汇总

为了方便你复制运行，这里给出完整的可运行代码：

**代码文件：** `study-agentic-rag/01-rag-comparison/traditional_rag.py`

**`traditional_rag.py` 完整代码：**

```python
"""
传统RAG完整实现 - 使用阿里云百炼平台和ChromaDB
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import PyPDF2
from typing import List

# 加载环境变量
load_dotenv()

# 配置常量
API_KEY = os.getenv("API_KEY", "sk-abe3417c96f6441b83efed38708bcfb6")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
CHROMA_COLLECTION_NAME = "traditional_rag_collection"


class TraditionalRAG:
    """传统RAG实现类"""

    def __init__(self):
        """初始化：建立LLM和向量数据库连接"""
        print("🚀 初始化Traditional RAG系统...")

        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        print(f"✅ 已连接到阿里云百炼平台")

        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        try:
            self.collection = self.chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
            print(f"📂 使用现有集合，当前文档数: {self.collection.count()}")
        except:
            self.collection = self.chroma_client.create_collection(name=CHROMA_COLLECTION_NAME)
            print(f"✨ 创建新集合")

    def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        try:
            response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 嵌入失败: {e}")
            return [0.0] * 1536

    def load_pdf(self, pdf_path: str) -> List[str]:
        """加载PDF文档"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            texts = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    texts.append(text)
        return texts

    def ingest_documents(self, texts: List[str]):
        """将文档导入向量数据库"""
        print(f"\n📥 正在摄入 {len(texts)} 个文档...")

        ids = [f"doc_{i}" for i in range(len(texts))]
        embeddings = []

        for i, text in enumerate(texts):
            print(f"      处理文档 {i+1}/{len(texts)}")
            embeddings.append(self.get_embedding(text))

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": f"document_{i}"} for i in range(len(texts))]
        )

        print(f"✅ 文档摄入完成! 集合中共有 {self.collection.count()} 个文档")

    def search(self, query: str, k: int = 3) -> List[dict]:
        """执行向量相似度搜索"""
        print(f"\n🔍 执行向量搜索: '{query}'")

        query_embedding = self.get_embedding(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                docs.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })

        print(f"✅ 找到 {len(docs)} 个相关文档")
        return docs

    def format_context(self, docs: List[dict]) -> str:
        """格式化检索到的文档为上下文"""
        return "\n\n---\n\n".join([doc['content'] for doc in docs])

    def generate_answer(self, query: str, context: str) -> str:
        """使用LLM生成答案"""
        print(f"\n🤖 正在生成答案...")

        prompt = f"""你是一个有用的AI助手。请根据以下上下文回答用户的问题。

上下文:
{context}

问题: {query}

请提供准确、详细的答案。如果上下文中没有足够的信息，请诚实地说明。"""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的AI助手，擅长根据提供的上下文回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            print("✅ 答案生成完成!")
            return answer
        except Exception as e:
            print(f"❌ LLM调用失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"

    def query(self, question: str, k: int = 3) -> tuple:
        """传统RAG完整流程"""
        print(f"\n{'='*60}")
        print(f"📝 用户问题: {question}")
        print(f"{'='*60}")

        docs = self.search(question, k=k)

        if not docs:
            return "抱歉，没有找到相关信息。", []

        context = self.format_context(docs)
        answer = self.generate_answer(question, context)

        return answer, docs

    def reset_collection(self):
        """重置集合（清空所有文档）"""
        try:
            self.chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"description": "Traditional RAG collection"}
            )
            print("✅ 集合已重置")
        except Exception as e:
            print(f"❌ 重置失败: {e}")


# ============================================
# 使用示例
# ============================================

def main():
    """主函数：演示Traditional RAG的使用"""
    print("="*60)
    print("🎯 Traditional RAG 演示")
    print("="*60)

    # 初始化
    rag = TraditionalRAG()

    # 示例文档
    sample_documents = [
        "人工智能(AI)是计算机科学的一个分支,致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是AI的一个子集,专注于开发能够从数据中学习和改进的算法。",
        "检索增强生成(RAG)是一种结合信息检索和文本生成的技术，能够减少幻觉并提高准确性。",
        "ChromaDB是一个轻量级的本地向量数据库，非常适合开发和小规模应用。"
    ]

    # 摄入文档
    rag.ingest_documents(sample_documents)

    # 测试查询
    question = "什么是RAG?"
    answer, docs = rag.query(question)

    print(f"\n{'='*60}")
    print(f"❓ 问题: {question}")
    print(f"{'='*60}")
    print(f"💡 答案:\n{answer}")
    print(f"\n📚 使用的文档:")
    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] (距离: {doc['distance']:.4f})")
        print(f"{doc['content'][:100]}...")


if __name__ == "__main__":
    main()


"""
传统RAG的优点:
✅ 实现简单,易于理解和维护
✅ 响应速度快(单次检索 + 单次LLM调用)
✅ 成本低(LLM调用次数少)
✅ 行为可预测,易于调试
✅ 适合高并发场景
✅ 本地化部署(使用ChromaDB)

传统RAG的局限:
❌ 单一数据源,知识覆盖有限
❌ 无法获取实时信息
❌ 一次性检索,无法根据结果调整
❌ 不验证检索质量
❌ 难以处理复杂的多步推理查询
❌ 无法使用外部工具(计算器、API等)

适用场景:
- 企业内部文档查询
- 产品手册/技术文档问答
- 简单的FAQ系统
- 客服机器人(快速响应)
- 成本敏感的应用
- 高并发场景
- 不需要实时外部信息的应用

技术栈:
- LLM: 阿里云百炼平台 Qwen-Plus
- 嵌入: 阿里云 text-embedding-v1
- 向量数据库: ChromaDB (本地持久化)
- 兼容: OpenAI API格式
"""
```

---

## 五、运行测试

### 5.1 创建测试文件

将上面的完整代码保存为 `traditional_rag.py`。

### 5.2 运行

```bash
# 使用 uv
uv run python traditional_rag.py

# 或使用 python
python traditional_rag.py
```

### 5.3 预期输出

```
🚀 初始化Traditional RAG系统...
✅ 已连接到阿里云百炼平台
✨ 创建新集合

📥 正在摄入 4 个文档...
      处理文档 1/4
      处理文档 2/4
      处理文档 3/4
      处理文档 4/4
✅ 文档摄入完成! 集合中共有 4 个文档

============================================================
📝 用户问题: 什么是RAG?
============================================================

🔍 执行向量搜索: '什么是RAG?'
✅ 找到 3 个相关文档

📄 上下文长度: 245 字符

🤖 正在生成答案...
✅ 答案生成完成!

============================================================
❓ 问题: 什么是RAG?
============================================================
💡 答案:
RAG（检索增强生成）是一种结合了信息检索和文本生成的技术。
它通过从外部知识库检索相关信息来增强语言模型的能力，
从而减少幻觉并提高答案的准确性。

📚 使用的文档:
[1] (距离: 0.1234)
检索增强生成(RAG)是一种结合信息检索和文本生成的技术...
```

---

## 本讲总结

**核心知识点：**

1. **ChromaDB 使用**：PersistentClient、collection、add/query
2. **Embedding API**：文本向量化，相似度搜索基础
3. **向量搜索**：query_embeddings、n_results、距离度量
4. **Prompt 设计**：明确角色、提供上下文、要求诚实
5. **完整流程**：摄入→查询→检索→拼接→生成

**关键代码模式：**

```python
# 1. 初始化
rag = TraditionalRAG()

# 2. 导入文档
rag.ingest_documents(documents)

# 3. 查询
answer, docs = rag.query("问题")
```

**传统 RAG 优缺点：**

✅ 简单、快速、成本低、可预测
❌ 单一数据源、无法验证、不能处理复杂查询

---

## 下节预告

下一讲，我们将实现 **Agentic RAG 引擎**：

- ReAct 循环的完整实现
- 多工具系统集成（向量搜索 + Web 搜索 + 计算器）
- 智能决策和迭代检索
- 质量验证机制

对比传统 RAG，你将看到 Agentic RAG 如何通过"思考-行动-观察"循环解决复杂问题！

准备好迎接更强大的 RAG 系统了吗？下一讲见！🚀
