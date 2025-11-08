# 项目2：混合RAG系统

## 项目简介

本项目是 **Agentic RAG 实战教程** 第三部分（第6-10讲）的配套代码。

通过构建智能路由 + 双引擎架构的混合 RAG 系统，实现成本优化和性能提升的最佳平衡。

## 对应教程

- **第6讲**：智能路由器设计 - 自动识别查询复杂度
- **第7讲**：传统RAG引擎封装 - 快速响应的检索系统
- **第8讲**：Agentic RAG引擎实现 - 智能推理的检索系统
- **第9讲**：数据处理流水线 - 文档分块与语义优化
- **第10讲**：混合RAG主控系统 - 整合路由与双引擎

## 项目结构

```
02-hybrid-rag-system/
├── config.py                      # 配置文件
├── router.py                      # 智能路由器
├── traditional_rag_engine.py      # 传统RAG引擎
├── agentic_rag_engine.py         # Agentic RAG引擎
├── data_processor.py              # 数据处理流水线
├── hybrid_rag.py                  # 混合RAG主控系统
├── requirements.txt               # 项目依赖
├── pyproject.toml                 # uv 包管理配置
├── .env.example                   # 环境变量模板
└── README.md                      # 本文档
```

## 核心功能

### 智能路由器（router.py）

- 双层判断机制（关键词 + LLM）
- 自动识别查询复杂度
- 动态选择最佳 RAG 策略

### 双引擎架构

**传统引擎（traditional_rag_engine.py）**
- 快速响应（1-2秒）
- 低成本（单次 LLM 调用）
- 适合简单查询

**Agentic引擎（agentic_rag_engine.py）**
- 智能推理（ReAct框架）
- 多工具集成
- 适合复杂查询

### 数据处理（data_processor.py）

- 多格式文档加载
- 语义分块优化
- 批量处理流程

### 主控系统（hybrid_rag.py）

- 路由 + 双引擎集成
- 性能统计分析
- 成本追踪

## 系统架构

```
用户查询
    ↓
┌─────────────┐
│ 智能路由器  │ 判断查询复杂度
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
简单 │       │ 复杂
   │       │
   ▼       ▼
┌─────┐ ┌─────┐
│传统 │ │Agentic│
│引擎 │ │引擎  │
└──┬──┘ └──┬──┘
   │       │
   └───┬───┘
       │
       ▼
   统一返回
```

## 快速开始

### 1. 安装依赖

#### 使用 uv（推荐）

```bash
cd 02-hybrid-rag-system
uv sync
```

#### 使用 pip

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
nano .env
```

### 3. 准备数据

```bash
# 创建数据目录
mkdir -p data/documents

# 将文本文件放入该目录
cp your_documents/*.txt data/documents/
```

### 4. 运行系统

#### 使用 uv

```bash
uv run python hybrid_rag.py
```

#### 使用 pip

```bash
python hybrid_rag.py
```

## 核心优势

### 1. 成本优化

```
传统策略（全部用Agentic RAG）：
1000次查询 × $0.030 = $30

混合策略：
- 700次简单查询（Traditional）：$3.50
- 300次复杂查询（Agentic）：$9.00
总计：$12.50

节省：56.7% 💰
```

### 2. 性能提升

```
- 70%的简单查询：1-2秒快速响应 ⚡
- 30%的复杂查询：8-12秒准确答案 🎯
- 综合响应时间：降低40%
```

### 3. 准确性保证

```
- 简单查询准确率：80%（Traditional RAG）
- 复杂查询准确率：90%（Agentic RAG）
- 综合准确率：83%（vs 纯Traditional的80%）
```

## 学习重点

### 第6讲：智能路由器

- 如何设计路由决策逻辑
- 关键词匹配 vs LLM判断
- 路由准确性评估

### 第7讲：传统引擎封装

- 引擎接口设计
- 配置管理
- 可复用组件

### 第8讲：Agentic引擎实现

- ReAct框架封装
- 工具注册机制
- 扩展新工具

### 第9讲：数据处理

- 文档分块策略
- 语义分块 vs 固定分块
- 批量处理优化

### 第10讲：主控系统

- 组件集成
- 错误处理
- 性能监控

## 性能监控

系统提供详细的性能统计：

```python
{
    "total_queries": 100,
    "traditional_count": 68,
    "agentic_count": 32,
    "traditional_ratio": "68.0%",
    "avg_traditional_time": "1.8s",
    "avg_agentic_time": "9.2s",
    "total_cost": "$1.84",
    "avg_accuracy": "83%"
}
```

## 下一步

完成本项目后，建议继续学习：

- **项目3**：智能问答Web应用（Streamlit + LangChain）

## 技术栈

- **LLM**：阿里云百炼平台
- **向量数据库**：ChromaDB
- **Web搜索**：DuckDuckGo
- **Python**：3.8+

## 常见问题

**Q: 如何调整路由策略？**
A: 修改 `config.py` 中的 `SIMPLE_QUERY_KEYWORDS` 和 `COMPLEX_QUERY_KEYWORDS`

**Q: 如何添加新工具？**
A: 在 `agentic_rag_engine.py` 中定义新工具方法，并注册到工具列表

**Q: 如何查看路由决策过程？**
A: 运行时会打印路由器的判断逻辑和选择结果

## 参考资料

- [教程导航](../docs/README.md)
- [第6讲：智能路由器设计](../docs/part3-Agentic-RAG-进阶实践篇-混合RAG系统/)
- [项目1：RAG原理对比](../01-rag-comparison/)
