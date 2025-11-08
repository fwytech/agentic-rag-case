# 项目1：RAG原理对比实现

## 项目简介

本项目是 **Agentic RAG 实战教程** 第二部分（第3-5讲）的配套代码。

通过实现传统 RAG 和 Agentic RAG 两个版本，帮助你深入理解两种技术方案的差异和各自的适用场景。

## 对应教程

- **第3讲**：从零搭建传统RAG - 文档检索与答案生成
- **第4讲**：构建Agentic RAG引擎 - ReAct循环与多工具集成
- **第5讲**：RAG技术对比 - 准确性成本与性能分析

## 项目结构

```
01-rag-comparison/
├── traditional_rag.py      # 传统RAG完整实现
├── agentic_rag.py          # Agentic RAG完整实现
├── requirements.txt        # 项目依赖
├── pyproject.toml          # uv 包管理配置
├── .env.example            # 环境变量模板
└── README.md               # 本文档
```

## 核心功能

### traditional_rag.py

- ChromaDB 向量数据库初始化
- 文档向量化和索引
- 相似度检索
- LLM 答案生成
- 简单直接的流水线架构

### agentic_rag.py

- ReAct 框架实现
- 多工具系统（向量搜索、Web搜索、计算器）
- 智能决策和迭代检索
- 质量验证机制

## 快速开始

### 1. 安装依赖

#### 使用 uv（推荐）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 进入项目目录
cd 01-rag-comparison

# 同步依赖
uv sync
```

#### 使用 pip

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
nano .env
```

### 3. 运行示例

#### 使用 uv

```bash
# 运行传统RAG
uv run python traditional_rag.py

# 运行Agentic RAG
uv run python agentic_rag.py
```

#### 使用 pip

```bash
# 运行传统RAG
python traditional_rag.py

# 运行Agentic RAG
python agentic_rag.py
```

## 学习重点

### 传统RAG学习要点

1. **向量数据库使用**：如何初始化和使用 ChromaDB
2. **嵌入向量生成**：如何将文本转换为向量
3. **相似度检索**：如何找到最相关的文档
4. **Prompt 设计**：如何组织上下文和问题

### Agentic RAG学习要点

1. **ReAct 框架**：Think-Act-Observe 循环
2. **工具系统**：如何定义和调用多个工具
3. **智能决策**：如何让 Agent 自主选择策略
4. **迭代优化**：如何评估结果并调整

## 对比分析

| 特性 | Traditional RAG | Agentic RAG |
|------|----------------|-------------|
| **代码行数** | ~200行 | ~400行 |
| **复杂度** | 简单 | 中等 |
| **响应时间** | 1-2秒 | 5-10秒 |
| **准确率** | 70-80% | 85-95% |
| **适用场景** | 简单查询 | 复杂查询 |

## 下一步

完成本项目后，建议继续学习：

- **项目2**：混合RAG系统（智能路由 + 成本优化）
- **项目3**：生产级智能问答Web应用

## 技术栈

- **LLM**：阿里云百炼平台（兼容 OpenAI API）
- **向量数据库**：ChromaDB
- **Web搜索**：DuckDuckGo
- **Python**：3.8+

## 常见问题

**Q: ChromaDB 数据存储在哪里？**
A: 默认存储在 `./chroma_db_traditional/` 和 `./chroma_db_agentic/` 目录

**Q: 可以使用其他 LLM 吗？**
A: 可以，修改 `.env` 文件中的 BASE_URL 和 API_KEY

**Q: 如何清空向量数据库？**
A: 删除 `chroma_db_*` 目录即可

## 参考资料

- [教程导航](../docs/README.md)
- [第3讲：传统RAG实现](../docs/part2-Agentic-RAG-基础实践篇-RAG原理对比实现/03-传统RAG实现-文档检索与答案生成.md)
- [第4讲：Agentic RAG引擎](../docs/part2-Agentic-RAG-基础实践篇-RAG原理对比实现/04-Agentic-RAG引擎-ReAct循环与多工具集成.md)
