# 第10讲：混合RAG主控系统 - 整合与部署

经过前面9讲的学习，我们已经拥有了所有核心组件：智能路由器、双引擎、数据流水线。

这一讲，我们将把它们整合成一个**完整的混合RAG系统**，实现生产级的问答服务。

---

## 一、系统架构总览

### 完整的技术栈

```
┌─────────────────────────────────────────────────┐
│            混合RAG系统架构                         │
│                                                 │
│  用户查询                                         │
│     ↓                                           │
│  ┌────────────────┐                             │
│  │  HybridRAG     │ ← 主控类                      │
│  │  (Main Class)  │                             │
│  └───────┬────────┘                             │
│          │                                      │
│    ┌─────┴─────┐                                │
│    ↓           ↓                                │
│  路由器     统计模块                              │
│    │                                            │
│    ├── simple → Traditional RAG引擎              │
│    │              ├─ ChromaDB                   │
│    │              └─ LLM (生成)                  │
│    │                                            │
│    └── complex → Agentic RAG引擎                 │
│                    ├─ ChromaDB                  │
│                    ├─ Web Search               │
│                    ├─ Calculator               │
│                    └─ LLM (推理+生成)            │
│                                                 │
│  数据管理                                         │
│    ↓                                            │
│  DataPipeline                                   │
│    ├─ 文档加载                                    │
│    ├─ 文本清洗                                    │
│    ├─ 智能分块                                    │
│    └─ 批量导入                                    │
└─────────────────────────────────────────────────┘
```

### 核心价值

| 特性 | 实现方式 | 价值 |
|------|---------|------|
| **智能路由** | 自动识别复杂度 | 节省47%成本 |
| **双引擎协作** | 传统+Agentic | 平衡速度和质量 |
| **数据管理** | 统一流水线 | 批量处理文档 |
| **性能监控** | 实时统计 | 数据驱动优化 |

---

## 二、HybridRAG主控类实现

### 主控类设计

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/hybrid_rag.py`

```python
"""
混合RAG系统
Hybrid RAG System

自动选择传统RAG或Agentic RAG，提供最佳问答体验
"""

from traditional_rag_engine import TraditionalRAGEngine
from agentic_rag_engine import AgenticRAGEngine
from router import QueryRouter
from typing import Tuple, Dict
import time


class HybridRAG:
    """
    混合RAG系统

    根据查询复杂度自动选择:
    - 简单查询 → 传统RAG (快速、经济)
    - 复杂查询 → Agentic RAG (智能、全面)
    """

    def __init__(self):
        """初始化混合RAG系统"""
        print("\n" + "="*60)
        print("🚀 初始化混合RAG系统")
        print("="*60)

        # 初始化路由器
        print("\n1️⃣ 初始化智能路由器...")
        self.router = QueryRouter()

        # 初始化传统RAG引擎
        print("\n2️⃣ 初始化传统RAG引擎...")
        self.traditional_rag = TraditionalRAGEngine()

        # 初始化Agentic RAG引擎
        print("\n3️⃣ 初始化Agentic RAG引擎...")
        self.agentic_rag = AgenticRAGEngine()

        print("\n" + "="*60)
        print("✅ 混合RAG系统初始化完成！")
        print("="*60)

        # 统计信息
        self.stats = {
            "total_queries": 0,
            "traditional_count": 0,
            "agentic_count": 0,
            "traditional_time": 0,
            "agentic_time": 0
        }
```

**为什么这么写？**

1. **为什么按顺序初始化？**
   - 路由器最轻量，先初始化
   - 传统RAG次之
   - Agentic RAG最重，最后初始化
   - 便于排查初始化错误

2. **为什么记录统计信息？**
   - 分析路由准确性
   - 监控系统性能
   - 优化成本分配
   - 数据驱动决策

3. **为什么不初始化DataPipeline？**
   - 数据处理通常是离线批量操作
   - 在线查询不需要
   - 按需创建即可

---

### 文档管理方法

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/hybrid_rag.py`

```python
    def add_documents(self, texts: list, metadatas: list = None):
        """
        添加文档到知识库

        Args:
            texts: 文本列表
            metadatas: 元数据列表
        """
        print("\n📚 添加文档到知识库...")

        # 只需添加到传统RAG（Agentic RAG会共享同一个向量数据库）
        self.traditional_rag.add_documents(texts, metadatas)

        print("✅ 文档添加完成！")
```

**为什么这么写？**

1. **为什么只调用 `traditional_rag.add_documents()`？**
   - 两个引擎共享同一个ChromaDB实例
   - `config.CHROMA_DB_PATH` 和 `config.COLLECTION_NAME` 相同
   - 避免重复存储，节省空间

2. **为什么不提供删除文档方法？**
   - ChromaDB支持 `collection.delete(ids=...)`
   - 生产环境可扩展
   - 本教程聚焦核心功能

---

### 核心查询方法

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/hybrid_rag.py`

```python
    def query(self, question: str, force_strategy: str = None) -> Dict:
        """
        查询混合RAG系统

        Args:
            question: 用户问题
            force_strategy: 强制使用策略 ("traditional" 或 "agentic")，None为自动路由

        Returns:
            {
                "question": str,
                "answer": str,
                "strategy": str,
                "complexity": str,
                "response_time": float,
                "metadata": dict
            }
        """
        start_time = time.time()

        print("\n" + "="*60)
        print(f"📝 [混合RAG] 用户问题: {question}")
        print("="*60)

        # 路由决策
        if force_strategy:
            route_result = {
                "strategy": force_strategy,
                "complexity": "unknown",
                "reasoning": f"强制使用{force_strategy} RAG"
            }
            print(f"\n🎯 [强制模式] 使用 {force_strategy.upper()} RAG")
        else:
            route_result = self.router.route(question)

        strategy = route_result["strategy"]
        complexity = route_result["complexity"]

        # 执行查询
        if strategy == "traditional":
            answer, docs = self.traditional_rag.query(question)
            metadata = {"retrieved_docs": docs}
            self.stats["traditional_count"] += 1
        else:  # agentic
            answer = self.agentic_rag.query(question)
            metadata = {}
            self.stats["agentic_count"] += 1

        # 计算响应时间
        response_time = time.time() - start_time

        # 更新统计
        self.stats["total_queries"] += 1
        if strategy == "traditional":
            self.stats["traditional_time"] += response_time
        else:
            self.stats["agentic_time"] += response_time

        # 构建结果
        result = {
            "question": question,
            "answer": answer,
            "strategy": strategy,
            "complexity": complexity,
            "response_time": response_time,
            "metadata": metadata
        }

        # 打印结果
        print("\n" + "="*60)
        print(f"✅ [回答完成]")
        print(f"   策略: {strategy.upper()} RAG")
        print(f"   复杂度: {complexity}")
        print(f"   响应时间: {response_time:.2f}秒")
        print(f"="*60)
        print(f"\n💡 答案:\n{answer}")
        print("="*60)

        return result
```

**为什么这么写？**

1. **为什么提供 `force_strategy`？**
   - 调试时强制使用某个引擎
   - 对比两个引擎的效果
   - 特殊场景手动控制

   ```python
   # 示例
   result1 = hybrid_rag.query("孙悟空是谁？", force_strategy="traditional")
   result2 = hybrid_rag.query("孙悟空是谁？", force_strategy="agentic")
   # 对比两个结果
   ```

2. **为什么返回dict而不是字符串？**
   - 包含丰富的元信息
   - 便于后续分析和日志
   - 可扩展性强

3. **为什么记录每次查询的时间？**
   - 分析平均响应时间
   - 识别性能瓶颈
   - 监控系统健康度

4. **为什么metadata不同？**
   - 传统RAG返回检索文档（溯源）
   - Agentic RAG工具调用信息已在日志中
   - 灵活的元数据设计

---

### 统计分析方法

**代码文件：** `study-agentic-rag/02-hybrid-rag-system/hybrid_rag.py`

```python
    def get_stats(self) -> Dict:
        """
        获取系统统计信息

        Returns:
            统计信息字典
        """
        stats = self.stats.copy()

        if stats["traditional_count"] > 0:
            stats["avg_traditional_time"] = stats["traditional_time"] / stats["traditional_count"]
        else:
            stats["avg_traditional_time"] = 0

        if stats["agentic_count"] > 0:
            stats["avg_agentic_time"] = stats["agentic_time"] / stats["agentic_count"]
        else:
            stats["avg_agentic_time"] = 0

        return stats

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("📊 系统统计信息")
        print("="*60)
        print(f"总查询数: {stats['total_queries']}")
        print(f"\n传统RAG:")
        print(f"  - 使用次数: {stats['traditional_count']}")
        print(f"  - 平均响应时间: {stats['avg_traditional_time']:.2f}秒")
        print(f"\nAgentic RAG:")
        print(f"  - 使用次数: {stats['agentic_count']}")
        print(f"  - 平均响应时间: {stats['avg_agentic_time']:.2f}秒")

        # 计算使用比例
        if stats['total_queries'] > 0:
            trad_ratio = stats['traditional_count'] / stats['total_queries'] * 100
            agent_ratio = stats['agentic_count'] / stats['total_queries'] * 100
            print(f"\n使用分布:")
            print(f"  - 传统RAG: {trad_ratio:.1f}%")
            print(f"  - Agentic RAG: {agent_ratio:.1f}%")

        print("="*60)
```

**为什么这么写？**

1. **为什么计算平均时间？**
   - 比总时间更有意义
   - 便于横向对比
   - 识别性能回归

2. **为什么显示使用分布？**
   - 验证路由器效果（预期70%传统，30%Agentic）
   - 如果分布偏离，需调整路由策略
   - 数据驱动优化

3. **为什么用 `get_stats()` 和 `print_stats()` 分离？**
   - `get_stats()`：获取原始数据（API）
   - `print_stats()`：人类可读展示（CLI）
   - 职责分离

---

## 三、完整使用示例

### 示例1：基础使用

```python
"""
混合RAG系统使用示例
"""

from hybrid_rag import HybridRAG

# 初始化系统
hybrid_rag = HybridRAG()

# 添加示例文档
sample_docs = [
    "《西游记》是中国古代第一部浪漫主义章回体长篇神魔小说。作者是明代吴承恩。",
    "孙悟空原本是花果山上的一块仙石孕育而生的石猴，后来拜菩提祖师为师，学得七十二变和筋斗云等神通。",
    "唐僧俗姓陈，法号玄奘，是如来佛祖的二弟子金蝉子转世。他奉唐太宗之命前往西天取经。",
    "猪八戒原是天蓬元帅，因调戏嫦娥被贬下凡，错投猪胎。后被观音菩萨点化，等待取经人。",
    "沙悟净原是天庭的卷帘大将，因失手打碎琉璃盏被贬下界，在流沙河为妖。",
    "孙悟空有两个师傅：菩提祖师和唐僧。菩提祖师教他法术神通，唐僧是他取经路上的师父。",
    "师徒四人历经九九八十一难，最终到达西天，取得真经。孙悟空被封为斗战胜佛。",
]

hybrid_rag.add_documents(sample_docs)

# 测试查询
test_queries = [
    # 简单查询 (应该使用传统RAG)
    "西游记的作者是谁？",
    "孙悟空的师傅是谁？",
    "唐僧的法号是什么？",

    # 复杂查询 (应该使用Agentic RAG)
    "孙悟空有几个师傅？他们分别教了他什么？",
    "分析师徒四人的来历和背景",
    "为什么说《西游记》是浪漫主义小说？",
]

print("\n" + "="*60)
print("🎯 开始测试混合RAG系统")
print("="*60)

results = []
for query in test_queries:
    result = hybrid_rag.query(query)
    results.append(result)
    print("\n" + "-"*60 + "\n")

# 打印统计信息
hybrid_rag.print_stats()
```

### 示例2：批量文档导入

```python
from hybrid_rag import HybridRAG
from data_processor import DataPipeline

# 1. 初始化系统
hybrid_rag = HybridRAG()
pipeline = DataPipeline()

# 2. 批量处理文档
chunks = pipeline.process_directory(
    input_dir="./data/documents",
    use_semantic=True
)

# 3. 导入系统
hybrid_rag.add_documents(chunks)

# 4. 开始查询
result = hybrid_rag.query("产品A的主要功能是什么？")
print(result["answer"])
```

### 示例3：性能对比

```python
# 对比传统RAG和Agentic RAG的效果
question = "孙悟空有几个师傅？分别是谁？"

# 强制使用传统RAG
result_trad = hybrid_rag.query(question, force_strategy="traditional")

# 强制使用Agentic RAG
result_agent = hybrid_rag.query(question, force_strategy="agentic")

# 对比
print("\n对比结果:")
print(f"\n传统RAG:")
print(f"  答案: {result_trad['answer'][:100]}...")
print(f"  响应时间: {result_trad['response_time']:.2f}秒")

print(f"\nAgentic RAG:")
print(f"  答案: {result_agent['answer'][:100]}...")
print(f"  响应时间: {result_agent['response_time']:.2f}秒")
```

---

## 四、生产部署建议

### 1. Docker容器化

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制代码
COPY . .

# 暴露端口（如果有Web接口）
EXPOSE 8000

# 启动命令
CMD ["python", "app.py"]
```

**requirements.txt:**
```
openai==1.12.0
chromadb==0.4.22
duckduckgo-search==4.1.0
python-dotenv==1.0.0
```

### 2. 配置管理

```python
# config/production.py

import os

# LLM配置
API_KEY = os.getenv("API_KEY")  # 从环境变量读取
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_ID = "qwen-plus"

# ChromaDB配置（生产环境）
CHROMA_DB_PATH = "/data/vector_db"  # 持久化卷
COLLECTION_NAME = "prod_rag_collection"

# 路由器配置
SIMPLE_QUERY_KEYWORDS = [...]  # 根据实际数据优化
COMPLEX_QUERY_KEYWORDS = [...]

# 性能配置
TOP_K = 5  # 生产环境可能需要更多文档
MAX_AGENT_ITERATIONS = 3  # 降低成本
```

### 3. 监控和日志

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_rag.log'),
        logging.StreamHandler()
    ]
)

class HybridRAG:
    def query(self, question: str):
        logging.info(f"Query: {question}")

        result = ...

        logging.info(f"Strategy: {result['strategy']}, Time: {result['response_time']:.2f}s")

        return result
```

### 4. 错误处理

```python
class HybridRAG:
    def query(self, question: str) -> Dict:
        try:
            # 正常流程
            ...
        except Exception as e:
            logging.error(f"Query failed: {e}")

            # 降级策略：返回默认回答
            return {
                "question": question,
                "answer": "抱歉，系统暂时无法处理您的问题，请稍后重试。",
                "strategy": "fallback",
                "complexity": "unknown",
                "response_time": 0,
                "metadata": {"error": str(e)}
            }
```

---

## 五、系统优化方向

### 优化1：缓存机制

```python
from functools import lru_cache

class HybridRAG:
    def __init__(self):
        ...
        self.query_cache = {}

    def query(self, question: str):
        # 检查缓存
        if question in self.query_cache:
            print("✅ 命中缓存")
            return self.query_cache[question]

        # 执行查询
        result = self._do_query(question)

        # 保存缓存
        self.query_cache[question] = result

        return result
```

**价值：**
- 相同问题零成本返回
- 提升响应速度
- 适合高频重复查询

### 优化2：异步处理

```python
import asyncio

class HybridRAG:
    async def query_async(self, question: str):
        """异步查询"""
        # 异步调用LLM API
        result = await self._async_query_logic(question)
        return result

# 使用
async def main():
    hybrid_rag = HybridRAG()

    # 并发处理多个查询
    questions = ["问题1", "问题2", "问题3"]
    tasks = [hybrid_rag.query_async(q) for q in questions]
    results = await asyncio.gather(*tasks)

    return results
```

**价值：**
- 提升并发能力
- 更好的资源利用
- 适合高QPS场景

### 优化3：路由器自我学习

```python
class AdaptiveRouter:
    """自适应路由器"""

    def __init__(self):
        self.router = QueryRouter()
        self.feedback_data = []

    def route_with_feedback(self, question: str, user_satisfaction: int):
        """带反馈的路由"""

        # 路由决策
        result = self.router.route(question)

        # 记录反馈
        self.feedback_data.append({
            "question": question,
            "strategy": result["strategy"],
            "satisfaction": user_satisfaction  # 1-5分
        })

        # 定期分析和优化
        if len(self.feedback_data) >= 100:
            self._optimize_routing_rules()

        return result

    def _optimize_routing_rules(self):
        """基于反馈数据优化路由规则"""
        # 分析哪些问题被误判
        # 更新关键词列表
        # 调整LLM Prompt
        pass
```

---

## 六、完整代码总结

完整的 `hybrid_rag.py` 文件（约250行）见混合RAG项目。

---

## 七、总结

### 第三部分学习回顾

通过第6-10讲，我们完整实现了混合RAG系统：

| 讲次 | 主题 | 核心内容 |
|------|------|---------|
| **第6讲** | 智能路由器 | 两层架构、关键词+LLM分类 |
| **第7讲** | 传统RAG引擎 | 快速检索、生成答案 |
| **第8讲** | Agentic RAG引擎 | ReAct循环、多工具协作 |
| **第9讲** | 数据处理流水线 | 清洗、分块、批量导入 |
| **第10讲** | 混合RAG主控 | 系统整合、统计监控 |

### 混合RAG系统的价值

| 维度 | 纯传统RAG | 纯Agentic RAG | **混合RAG** |
|------|----------|--------------|-----------|
| **成本** | ¥3,000/月 | ¥9,000/月 | **¥4,800/月** |
| **速度** | 1.8秒 | 8.5秒 | **3.2秒** |
| **准确率** | 65% | 85% | **71%** |
| **推荐度** | ⭐⭐⭐ | ⭐⭐⭐⭐ | **⭐⭐⭐⭐⭐** |

**混合方案是最优选择，平衡了成本、速度和质量。**

---

## 下一步学习

恭喜完成第三部分！你已经掌握了构建生产级混合RAG系统的所有核心技术。

**第四部分：Agentic RAG 生产实践篇 - 智能问答Web应用**

我们将学习如何把混合RAG系统封装成Web应用：
- 第11讲：Web应用架构设计
- 第12讲：Streamlit界面开发
- 第13讲：API服务封装
- 第14讲：用户会话管理
- 第15讲：多模型支持（Ollama+百炼）
- 第16讲：性能优化与监控
- 第17讲：部署与运维

敬请期待！

---

## 附录：混合RAG系统检查清单

### 开发阶段
- [ ] 实现智能路由器（准确率>90%）
- [ ] 实现传统RAG引擎（响应<2秒）
- [ ] 实现Agentic RAG引擎（准确率>85%）
- [ ] 实现数据处理流水线
- [ ] 整合混合RAG主控
- [ ] 单元测试覆盖率>80%

### 上线前
- [ ] 性能测试（QPS、响应时间）
- [ ] 成本评估（月度预算）
- [ ] 监控配置（日志、告警）
- [ ] 错误处理（降级策略）
- [ ] 文档完善（API文档、运维手册）
- [ ] 备份恢复方案

### 上线后
- [ ] 监控路由准确率
- [ ] 收集用户反馈
- [ ] 定期优化关键词
- [ ] 扩展工具集（根据需求）
- [ ] 成本优化（缓存、批处理）
- [ ] A/B测试新策略

祝你构建出色的RAG系统！
