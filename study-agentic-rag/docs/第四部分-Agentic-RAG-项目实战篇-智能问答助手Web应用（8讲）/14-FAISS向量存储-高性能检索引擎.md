# 第14讲：FAISS向量存储 - 高性能检索引擎

> **本讲目标**：掌握FAISS向量数据库的使用，构建高性能的文档检索系统

## 一、为什么从ChromaDB切换到FAISS？

在前面的课程中，我们使用ChromaDB作为向量数据库。但在生产环境中，我们选择了FAISS。为什么？

**ChromaDB vs FAISS对比**：

| 维度 | ChromaDB | FAISS | 适用场景 |
|------|---------|-------|---------|
| **开发难度** | ⭐ 简单（API友好） | ⭐⭐ 中等 | ChromaDB适合快速原型 |
| **检索速度** | 中等（千级文档） | 极快（百万级文档） | FAISS适合大规模数据 |
| **内存占用** | 较高 | 可控 | FAISS可调优 |
| **持久化** | 自动持久化 | 手动保存 | ChromaDB更便捷 |
| **索引类型** | 固定 | 丰富（Flat、IVF、HNSW） | FAISS可定制 |
| **依赖** | 需要服务端 | 纯Python库 | FAISS部署简单 |

**为什么在生产环境选择FAISS？**

1. **性能**：Facebook开源，专为大规模向量搜索优化
2. **轻量**：不需要运行独立服务（ChromaDB需要）
3. **成熟**：在工业界大量使用，稳定可靠
4. **灵活**：支持多种索引类型，可根据场景调优

**学习路径**：
```
ChromaDB（教学） → FAISS（生产）
  ↓                    ↓
快速上手            性能优化
```

## 二、FAISS索引类型选择

FAISS提供多种索引类型，各有优劣：

| 索引类型 | 检索精度 | 检索速度 | 内存占用 | 适用场景 |
|---------|---------|---------|---------|---------|
| **Flat** | 100% 精确 | 慢（暴力搜索） | 高 | <1万向量 |
| **IVF** | ~95% 近似 | 快 | 中 | 1万-100万向量 |
| **HNSW** | ~99% 近似 | 极快 | 高 | 需要极速响应 |
| **PQ** | ~90% 近似 | 极快 | 极低 | 百万级+内存受限 |

**我们的选择：Flat索引**

LangChain的`FAISS.from_documents()`默认使用Flat索引（精确搜索）：
- 文档量通常在千级-万级（企业知识库规模）
- 精度优先（RAG需要准确的检索）
- 内存不是瓶颈（现代服务器）

如果未来文档量超过10万，可以切换到IVF索引：
```python
# 高级用法（本项目暂未使用）
import faiss
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
```

## 三、向量存储服务架构

我们的`vector_store.py`（310行）提供完整的向量存储服务：

```mermaid
graph TD
    A[文档输入] --> B[split_documents 文档分割]
    B --> C[add_documents 添加向量]
    C --> D[create_vector_store 创建索引]
    D --> E[save_index 持久化]
    E --> F[磁盘存储]
    F --> G[load_index 加载]
    G --> H[search 检索]
    H --> I[相似度搜索 / MMR搜索]
    I --> J[返回结果]
```

**核心功能**：
1. **创建与加载**：从文档创建索引，或加载已有索引
2. **检索**：相似度搜索、MMR搜索、阈值过滤
3. **文档管理**：添加、删除、清空
4. **持久化**：保存索引、文档、元数据
5. **统计**：文档数、向量数、维度信息

## 四、代码实现详解

我们将310行代码拆分成5个部分讲解。

### 第一部分：初始化和创建向量存储（1-51行）

这部分初始化服务，并提供从文档创建向量存储的方法。

<details>
<summary>点击展开代码</summary>

```python
import os
import json
import pickle
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import faiss
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import Settings
from services.llm_client import UnifiedEmbeddingClient

logger = logging.getLogger(__name__)

class VectorStoreService:
    """向量存储服务类 - 支持 Ollama 和在线 API embedding"""

    def __init__(self):
        self.settings = Settings()
        # 使用统一的嵌入客户端
        self.embedding_client = UnifiedEmbeddingClient()
        self.embeddings = self.embedding_client.get_embeddings()
        self.vector_store = None
        self.documents = []
        self.index_path = None

        logger.info(f"向量存储服务初始化成功 - 提供商: {self.settings.LLM_PROVIDER}, 嵌入模型: {self.settings.get_embedding_model()}")

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """创建向量存储"""
        try:
            logger.info(f"创建向量存储，文档数量: {len(documents)}")

            # 创建向量存储
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            self.vector_store = vector_store
            self.documents = documents

            logger.info("向量存储创建成功")
            return vector_store

        except Exception as e:
            logger.error(f"创建向量存储失败: {str(e)}")
            raise
```

</details>

**为什么这么写？**

1. **为什么单独保存`self.documents`？**
   ```python
   self.vector_store = vector_store  # FAISS索引
   self.documents = documents        # 原始文档
   ```
   - **FAISS只存储向量**，不存储原始文档
   - 我们需要原始文档来显示检索结果
   - 保存`documents`用于持久化和删除操作

2. **为什么用`UnifiedEmbeddingClient`？**
   ```python
   self.embedding_client = UnifiedEmbeddingClient()
   self.embeddings = self.embedding_client.get_embeddings()
   ```
   - 支持Ollama和在线API双模式
   - 统一接口，配置驱动切换
   - 回顾第13讲的统一客户端设计

3. **为什么用`FAISS.from_documents()`？**
   ```python
   vector_store = FAISS.from_documents(
       documents=documents,
       embedding=self.embeddings
   )
   ```
   - LangChain封装的便捷方法
   - 自动处理：文档 → 嵌入 → 创建索引
   - 相比原生FAISS API，减少90%代码

### 第二部分：索引加载和保存（53-118行）

这部分实现向量存储的持久化功能。

<details>
<summary>点击展开代码</summary>

```python
    def load_index(self, index_path: str) -> bool:
        """加载向量存储索引"""
        try:
            logger.info(f"加载向量存储索引: {index_path}")

            if not os.path.exists(index_path):
                logger.warning(f"索引路径不存在: {index_path}")
                return False

            # 加载FAISS索引
            self.vector_store = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # 加载文档（如果存在）
            docs_path = f"{index_path}_docs.pkl"
            if os.path.exists(docs_path):
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)

            self.index_path = index_path
            logger.info("向量存储索引加载成功")
            return True

        except Exception as e:
            logger.error(f"加载向量存储索引失败: {str(e)}")
            return False

    def save_index(self, index_path: str) -> bool:
        """保存向量存储索引"""
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化，无法保存")
                return False

            logger.info(f"保存向量存储索引: {index_path}")

            # 保存FAISS索引
            self.vector_store.save_local(index_path)

            # 保存文档
            docs_path = f"{index_path}_docs.pkl"
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)

            # 保存元数据
            metadata = {
                "created_at": datetime.now().isoformat(),
                "documents_count": len(self.documents),
                "embedding_model": self.embedding_model_name,
                "vector_dimension": self.settings.VECTOR_DIMENSION
            }

            metadata_path = f"{index_path}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.index_path = index_path
            logger.info("向量存储索引保存成功")
            return True

        except Exception as e:
            logger.error(f"保存向量存储索引失败: {str(e)}")
            return False
```

</details>

**为什么这么写？**

1. **为什么用`allow_dangerous_deserialization=True`？**
   ```python
   self.vector_store = FAISS.load_local(
       index_path,
       self.embeddings,
       allow_dangerous_deserialization=True
   )
   ```
   - FAISS使用pickle序列化（可能被恶意利用）
   - LangChain要求显式确认风险
   - **生产建议**：只加载可信来源的索引

2. **为什么保存三个文件？**
   ```python
   index_path/                    # FAISS索引文件（.faiss, .pkl）
   index_path_docs.pkl            # 原始文档
   index_path_metadata.json       # 元数据
   ```
   - **FAISS索引**：向量+索引结构（`save_local`自动生成）
   - **文档文件**：原始文档对象（用于显示检索结果）
   - **元数据文件**：时间、模型、维度（便于管理和调试）

3. **为什么用pickle保存文档？**
   ```python
   with open(docs_path, 'wb') as f:
       pickle.dump(self.documents, f)
   ```
   - Document对象包含复杂的元数据
   - pickle可以完整序列化Python对象
   - JSON只能存储简单类型（需要手动转换）

### 第三部分：搜索功能（120-166行）

这部分实现两种检索策略：相似度搜索和MMR搜索。

<details>
<summary>点击展开代码</summary>

```python
    def search(
        self,
        query: str,
        top_k: int = 3,
        search_type: str = "similarity",
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """搜索向量存储"""
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []

            logger.info(f"搜索查询: {query}, top_k: {top_k}, search_type: {search_type}")

            if search_type == "similarity":
                # 相似度搜索
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k
                )
            elif search_type == "mmr":
                # MMR搜索（最大边际相关性）
                results = self.vector_store.max_marginal_relevance_search_with_score(
                    query=query,
                    k=top_k,
                    fetch_k=top_k * 2
                )
            else:
                raise ValueError(f"不支持的搜索类型: {search_type}")

            # 格式化结果
            formatted_results = []
            for doc, score in results:
                if score >= score_threshold:  # 过滤低分结果
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    })

            logger.info(f"搜索完成，找到 {len(formatted_results)} 个结果")
            return formatted_results

        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []
```

</details>

**为什么这么写？**

1. **为什么支持两种搜索类型？**

   **相似度搜索（Similarity）**：
   ```python
   results = self.vector_store.similarity_search_with_score(query, k=top_k)
   ```
   - 返回最相似的top_k个文档
   - 问题：可能返回内容重复的文档
   - 适用场景：需要最相关的结果

   **MMR搜索（Maximal Marginal Relevance）**：
   ```python
   results = self.vector_store.max_marginal_relevance_search_with_score(
       query=query,
       k=top_k,
       fetch_k=top_k * 2  # 先取2倍候选，再去重
   )
   ```
   - 平衡相关性和多样性
   - 避免返回相似的重复内容
   - 适用场景：需要多角度信息

2. **为什么用`fetch_k=top_k * 2`？**
   ```python
   fetch_k=top_k * 2
   ```
   - MMR分两步：先取`fetch_k`个候选 → 再选`k`个多样化的
   - `fetch_k`太小：多样性不足
   - `fetch_k`太大：计算开销大
   - 2倍是工程经验值

3. **为什么过滤`score_threshold`？**
   ```python
   if score >= score_threshold:
       formatted_results.append(...)
   ```
   - 相似度低的文档可能是噪音
   - 提高答案质量（宁缺毋滥）
   - 用户可配置阈值（默认0.5）

### 第四部分：文档管理（168-213行）

这部分实现文档的增删改功能。

<details>
<summary>点击展开代码</summary>

```python
    def add_documents(self, documents: List[Document]) -> bool:
        """添加文档到向量存储"""
        try:
            logger.info(f"添加文档到向量存储，数量: {len(documents)}")

            if not self.vector_store:
                # 如果向量存储不存在，创建新的
                self.create_vector_store(documents)
            else:
                # 添加到现有向量存储
                self.vector_store.add_documents(documents)
                self.documents.extend(documents)

            logger.info("文档添加成功")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        """从向量存储中删除文档"""
        try:
            logger.info(f"删除文档: {doc_id}")

            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return False

            # FAISS不直接支持删除操作，需要重新创建索引
            remaining_docs = [
                doc for doc in self.documents
                if doc.metadata.get("id") != doc_id
            ]

            if len(remaining_docs) < len(self.documents):
                self.create_vector_store(remaining_docs)
                logger.info(f"文档删除成功: {doc_id}")
                return True
            else:
                logger.warning(f"未找到文档: {doc_id}")
                return False

        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False

    def clear(self):
        """清空向量存储"""
        try:
            logger.info("清空向量存储")

            self.vector_store = None
            self.documents = []
            self.index_path = None

            logger.info("向量存储已清空")

        except Exception as e:
            logger.error(f"清空向量存储失败: {str(e)}")
```

</details>

**为什么这么写？**

1. **为什么添加文档时判断是否存在？**
   ```python
   if not self.vector_store:
       self.create_vector_store(documents)  # 不存在，创建新的
   else:
       self.vector_store.add_documents(documents)  # 存在，增量添加
       self.documents.extend(documents)
   ```
   - 第一次上传文档：创建索引
   - 后续上传：增量添加（无需重建）
   - 提升性能，减少计算

2. **为什么删除文档要重建索引？**
   ```python
   # FAISS不直接支持删除操作，需要重新创建索引
   remaining_docs = [
       doc for doc in self.documents
       if doc.metadata.get("id") != doc_id
   ]
   self.create_vector_store(remaining_docs)
   ```
   - **FAISS的限制**：索引是不可变的（immutable）
   - 删除 = 过滤文档 + 重建索引
   - 频繁删除会影响性能（不推荐）

3. **为什么清空时设为`None`而不是空列表？**
   ```python
   self.vector_store = None   # 而不是 []
   self.documents = []
   ```
   - `None`明确表示"未初始化"
   - 其他方法通过`if not self.vector_store`判断状态
   - 空列表会导致判断失效

### 第五部分：辅助功能（215-310行）

这部分提供统计、文档分割等辅助功能。

<details>
<summary>点击展开代码</summary>

```python
    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        try:
            stats = {
                "documents_count": len(self.documents),
                "vector_store_initialized": self.vector_store is not None,
                "embedding_model": self.embedding_model_name,
                "index_path": self.index_path
            }

            if self.vector_store:
                # 获取索引信息
                index = self.vector_store.index
                stats.update({
                    "total_vectors": index.ntotal if hasattr(index, 'ntotal') else 0,
                    "dimension": index.d if hasattr(index, 'd') else 0
                })

            return stats

        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {"error": str(e)}

    def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Document]:
        """分割文档"""
        try:
            chunk_size = chunk_size or self.settings.CHUNK_SIZE
            chunk_overlap = chunk_overlap or self.settings.CHUNK_OVERLAP

            logger.info(f"分割文档，chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )

            split_docs = text_splitter.split_documents(documents)

            # 添加元数据
            for i, doc in enumerate(split_docs):
                if "chunk_id" not in doc.metadata:
                    doc.metadata["chunk_id"] = i
                if "chunk_size" not in doc.metadata:
                    doc.metadata["chunk_size"] = len(doc.page_content)

            logger.info(f"文档分割完成，片段数量: {len(split_docs)}")
            return split_docs

        except Exception as e:
            logger.error(f"文档分割失败: {str(e)}")
            return documents

    def similarity_search_with_threshold(
        self,
        query: str,
        threshold: float = 0.7,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """带阈值的相似度搜索"""
        try:
            results = self.search(query, top_k * 2, "similarity")  # 获取更多结果用于过滤

            # 按阈值过滤
            filtered_results = [
                result for result in results
                if result["score"] >= threshold
            ]

            # 返回前top_k个结果
            return filtered_results[:top_k]

        except Exception as e:
            logger.error(f"阈值搜索失败: {str(e)}")
            return []
```

</details>

**为什么这么写？**

1. **为什么用`index.ntotal`和`index.d`？**
   ```python
   index = self.vector_store.index  # 获取原生FAISS索引
   stats.update({
       "total_vectors": index.ntotal,  # 总向量数
       "dimension": index.d            # 向量维度
   })
   ```
   - 访问FAISS原生索引（不是LangChain封装）
   - `ntotal`：索引中的向量总数
   - `d`：向量维度（如768维）
   - 用于监控和调试

2. **为什么文档分割用中文分隔符？**
   ```python
   separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
   ```
   - **优先级从高到低**：先按段落分 → 再按句子分 → 最后按字符分
   - 中文标点符号：`。！？，`
   - 保持语义完整性（不会在句子中间切断）

3. **为什么添加`chunk_id`和`chunk_size`元数据？**
   ```python
   doc.metadata["chunk_id"] = i
   doc.metadata["chunk_size"] = len(doc.page_content)
   ```
   - `chunk_id`：追踪文档片段顺序
   - `chunk_size`：监控分割质量（是否符合预期大小）
   - 便于调试和分析

4. **为什么`similarity_search_with_threshold`取2倍结果？**
   ```python
   results = self.search(query, top_k * 2, "similarity")
   filtered_results = [r for r in results if r["score"] >= threshold]
   return filtered_results[:top_k]
   ```
   - 先取2倍候选 → 过滤低分 → 返回top_k
   - 确保有足够的高质量结果
   - 如果直接取top_k，可能全部被过滤掉

## 五、完整代码总结

上面的5个部分组成了完整的`vector_store.py`（310行）：

1. **初始化和创建**（51行）：集成嵌入客户端，创建FAISS索引
2. **加载和保存**（66行）：三文件持久化（索引、文档、元数据）
3. **搜索功能**（47行）：相似度搜索、MMR搜索、阈值过滤
4. **文档管理**（46行）：增量添加、重建删除、清空
5. **辅助功能**（96行）：统计信息、文档分割、高级搜索

**核心设计模式**：

| 模式 | 应用场景 | 代码位置 |
|------|---------|---------|
| **服务层模式** | 封装向量存储逻辑 | `VectorStoreService`类 |
| **工厂模式** | 创建或加载索引 | `create_vector_store` / `load_index` |
| **策略模式** | 切换搜索算法 | `search_type` 参数 |
| **模板方法** | 统一异常处理 | `try-except-logger` |

**持久化文件结构**：
```
data/vector_store/
├── faiss_index.faiss           # FAISS索引（向量+索引结构）
├── faiss_index.pkl             # LangChain元数据
├── faiss_index_docs.pkl        # 原始文档（自定义）
└── faiss_index_metadata.json  # 元数据（自定义）
```

## 六、实际使用示例

### 示例1：创建向量存储

```python
from services.vector_store import VectorStoreService
from langchain.schema import Document

# 创建服务
vs = VectorStoreService()

# 准备文档
documents = [
    Document(page_content="Python是一门编程语言", metadata={"source": "doc1"}),
    Document(page_content="RAG是检索增强生成技术", metadata={"source": "doc2"})
]

# 分割文档
split_docs = vs.split_documents(documents, chunk_size=500, chunk_overlap=50)

# 创建向量存储
vs.create_vector_store(split_docs)

# 保存
vs.save_index("data/vector_store/my_index")
```

### 示例2：加载和搜索

```python
# 加载已有索引
vs = VectorStoreService()
vs.load_index("data/vector_store/my_index")

# 相似度搜索
results = vs.search("什么是RAG？", top_k=3, search_type="similarity")
for result in results:
    print(f"内容: {result['content']}")
    print(f"得分: {result['score']}")
    print(f"元数据: {result['metadata']}")
    print("---")

# MMR搜索（多样化结果）
results_mmr = vs.search("编程语言", top_k=3, search_type="mmr")
```

### 示例3：增量添加文档

```python
# 加载现有索引
vs = VectorStoreService()
vs.load_index("data/vector_store/my_index")

# 新增文档
new_docs = [
    Document(page_content="LangChain是LLM开发框架", metadata={"source": "doc3"})
]
vs.add_documents(new_docs)

# 重新保存
vs.save_index("data/vector_store/my_index")
```

### 示例4：获取统计信息

```python
vs = VectorStoreService()
vs.load_index("data/vector_store/my_index")

stats = vs.get_stats()
print(stats)
# 输出：
# {
#     "documents_count": 5,
#     "vector_store_initialized": True,
#     "embedding_model": "nomic-embed-text:latest",
#     "total_vectors": 5,
#     "dimension": 768,
#     "index_path": "data/vector_store/my_index"
# }
```

## 七、ChromaDB迁移到FAISS

如果你在早期用ChromaDB开发，迁移到FAISS只需3步：

### 步骤1：导出ChromaDB文档

```python
import chromadb
from langchain.schema import Document

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("my_collection")

# 获取所有文档
results = collection.get()
documents = [
    Document(
        page_content=content,
        metadata=metadata
    )
    for content, metadata in zip(results['documents'], results['metadatas'])
]
```

### 步骤2：创建FAISS索引

```python
from services.vector_store import VectorStoreService

vs = VectorStoreService()
vs.create_vector_store(documents)
vs.save_index("data/vector_store/migrated_index")
```

### 步骤3：更新应用代码

```python
# 替换 ChromaDB 调用
# 旧代码：
# collection.query(query_texts=["问题"], n_results=3)

# 新代码：
vs = VectorStoreService()
vs.load_index("data/vector_store/migrated_index")
results = vs.search("问题", top_k=3)
```

## 八、本讲总结

我们完成了基于FAISS的向量存储服务：

1. **技术选型**：FAISS vs ChromaDB，生产环境选FAISS
2. **索引类型**：Flat（精确）vs IVF（近似）vs HNSW（极速）
3. **持久化设计**：三文件模式（索引、文档、元数据）
4. **搜索策略**：相似度搜索、MMR搜索、阈值过滤
5. **文档管理**：增量添加、重建删除、智能分割

**关键技术点**：
- FAISS不存储原始文档，需要手动管理
- 删除操作需要重建索引（FAISS限制）
- MMR搜索平衡相关性和多样性
- 中文分隔符保持语义完整
- 原生FAISS索引暴露底层统计信息

**性能优化建议**：
- 文档量<1万：使用Flat索引
- 文档量>10万：切换IVF索引
- 嵌入维度：768维（Ollama）或1536维（OpenAI）
- 分块大小：500-1000字符，重叠10-20%

---

**下一讲预告**

第15讲：Agent工具系统 - ReAct框架与LangChain集成

我们将学习如何构建完整的Agent工具系统：
- LangChain Agent架构
- Tool定义和注册
- ReAct思考-行动-观察循环
- 自定义工具开发
- 完整的agent.py实现（约230行代码详解）
