# 项目3：智能问答Web应用

## 项目简介

本项目是 **Agentic RAG 实战教程** 第四部分（第11-17讲）的配套代码。

这是一个生产级的智能问答系统，集成了 Web UI、双模式 LLM 支持、文档管理等完整功能，可直接部署使用。

## 对应教程

- **第11讲**：项目架构设计 - 分层架构与模块划分
- **第12讲**：配置管理系统 - 双模式LLM支持
- **第13讲**：统一LLM客户端 - 兼容本地与在线模型
- **第14讲**：向量存储服务 - FAISS高效检索实现
- **第15讲**：Agent核心引擎 - ReAct框架与LangChain集成
- **第16讲**：文档处理与工具集成 - 多格式支持与API调用
- **第17讲**：Streamlit Web界面 - 打造用户友好的问答系统

## 项目结构

```
03-smart-qa-application/
├── config/
│   └── settings.py              # 配置管理
├── models/
│   └── agent.py                 # Agent核心引擎
├── services/
│   ├── llm_client.py            # 统一LLM客户端
│   ├── vector_store.py          # 向量存储服务
│   └── weather_tools.py         # 工具集成示例
├── utils/
│   ├── document_processor.py    # 文档处理
│   ├── chat_history.py          # 聊天历史管理
│   ├── ui_components.py         # UI组件
│   └── decorators.py            # 工具装饰器
├── app.py                       # Streamlit主应用
├── requirements.txt             # 项目依赖
├── pyproject.toml               # uv包管理配置
├── .env.example                 # 环境变量模板
└── README.md                    # 本文档
```

## 核心功能

### 1. Web用户界面（Streamlit）

- 现代化的聊天界面
- 实时流式响应
- 聊天历史管理
- 文档上传功能
- 模型参数配置

### 2. 双模式LLM支持

**本地模式（Ollama）**
- 完全本地化部署
- 数据隐私保护
- 无API费用

**在线模式（阿里云百炼）**
- 无需本地GPU
- 快速部署
- 稳定可靠

### 3. 智能Agent引擎

- LangChain Agent 集成
- ReAct 框架实现
- 多工具协作
- 可扩展工具系统

### 4. 文档管理

- PDF、Word、TXT 支持
- 智能文档分块
- 向量化索引
- 文档缓存机制

### 5. 工具集成

- 文档检索工具
- 天气查询工具（示例）
- 自定义工具框架

## 系统架构

```
┌─────────────────────────────────────┐
│         Streamlit UI Layer          │
│      (用户界面 + 交互逻辑)            │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│         Application Layer           │
│         (app.py)                    │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│ Config │ │ Models │ │Services│
│  Layer │ │  Layer │ │  Layer │
└────────┘ └────────┘ └────────┘
    │          │          │
    │          │          │
    ▼          ▼          ▼
┌──────────────────────────────────┐
│        Utils & Tools Layer       │
│  (文档处理、UI组件、工具装饰器)    │
└──────────────────────────────────┘
```

## 快速开始

### 方式一：使用本地Ollama

#### 1. 安装Ollama

```bash
curl https://ollama.ai/install.sh | sh
```

#### 2. 下载模型

```bash
ollama pull qwen:7b
ollama pull nomic-embed-text
```

#### 3. 启动Ollama服务

```bash
ollama serve
```

#### 4. 安装依赖并运行

使用 uv（推荐）：
```bash
cd 03-smart-qa-application
uv sync
uv run streamlit run app.py
```

使用 pip：
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 方式二：使用在线API（阿里云百炼）

#### 1. 配置环境变量

```bash
cp .env.example .env
nano .env
```

修改：
```
LLM_PROVIDER=online
ONLINE_API_KEY=你的阿里云百炼API密钥
```

#### 2. 安装依赖并运行

使用 uv：
```bash
uv sync
uv run streamlit run app.py
```

使用 pip：
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 主要特性

### 1. 分层架构设计

**优势：**
- 模块职责清晰
- 易于维护和扩展
- 便于测试
- 代码复用性高

**分层说明：**
- `config/`：配置管理，环境变量
- `models/`：核心业务逻辑（Agent）
- `services/`：服务层（LLM、向量存储、工具）
- `utils/`：工具函数和辅助类

### 2. 统一LLM客户端

```python
# 自动适配本地和在线模型
client = UnifiedLLMClient(model_name="qwen:7b")

# 无需关心底层实现
response = client.generate(prompt)
```

### 3. 可扩展工具系统

添加新工具只需三步：

```python
# 1. 定义工具函数
def my_custom_tool(query: str) -> str:
    return "result"

# 2. 添加装饰器
@tool_decorator(
    name="my_tool",
    description="工具描述"
)
def my_custom_tool(query: str) -> str:
    return "result"

# 3. 注册到Agent
agent.register_tool(my_custom_tool)
```

### 4. 完整的错误处理

- 网络错误自动重试
- 优雅的降级策略
- 详细的日志记录
- 用户友好的错误提示

## 使用示例

### 1. 基本问答

```
用户: 什么是RAG？

系统: RAG（检索增强生成）是一种结合了信息检索和
大语言模型的技术...
```

### 2. 上传文档并提问

```
1. 点击侧边栏"上传文档"
2. 选择PDF/Word/TXT文件
3. 等待文档处理完成
4. 提问："请总结这份文档的主要内容"
```

### 3. 工具调用

```
用户: 今天北京天气怎么样？

系统: [调用天气工具]
今天北京晴，温度15-25度，适合出行。
```

### 4. 复杂推理

```
用户: 根据文档分析，我们产品的主要优势是什么？

系统: [多步推理]
1. 检索产品相关文档
2. 分析功能特点
3. 对比竞品
4. 总结优势
```

## 学习重点

### 第11讲：架构设计

- 为什么要分层
- 如何划分模块职责
- 目录结构设计原则

### 第12讲：配置管理

- 环境变量管理
- 配置分类和组织
- 双模式切换实现

### 第13讲：统一客户端

- 接口设计原则
- 适配器模式应用
- 兼容性处理

### 第14讲：向量存储

- FAISS vs ChromaDB
- 向量存储优化
- 检索性能调优

### 第15讲：Agent引擎

- LangChain Agent 使用
- ReAct 框架集成
- 工具定义规范

### 第16讲：文档处理

- 多格式文档支持
- 智能分块策略
- 工具扩展框架

### 第17讲：Web界面

- Streamlit 核心概念
- 布局和组件
- 状态管理
- 用户体验优化

## 性能指标

| 指标 | 本地模式 | 在线模式 |
|------|---------|---------|
| 响应时间 | 3-8秒 | 2-5秒 |
| 准确率 | 85%+ | 90%+ |
| 成本 | 免费 | ~$0.02/查询 |
| 部署难度 | 中等 | 简单 |

## 下一步

完成本项目后，建议学习：

- **第18讲**：项目部署与性能优化
- 探索更多工具集成可能
- 尝试部署到生产环境

## 技术栈

- **前端**：Streamlit
- **后端**：Python 3.8+
- **LLM**：Ollama / 阿里云百炼
- **向量数据库**：FAISS
- **框架**：LangChain
- **文档处理**：PyPDF2, python-docx

## 常见问题

**Q: 如何切换LLM模式？**
A: 修改 `.env` 文件中的 `LLM_PROVIDER` 参数

**Q: 支持哪些文档格式？**
A: 目前支持 PDF、Word（.docx）、TXT

**Q: 如何添加新的工具？**
A: 参考 `services/weather_tools.py` 的实现

**Q: 聊天历史存储在哪里？**
A: 存储在 `chat_history/` 目录下的 JSON 文件中

**Q: 如何自定义UI？**
A: 修改 `app.py` 和 `utils/ui_components.py`

## 部署建议

### 本地开发

```bash
streamlit run app.py
```

### 生产部署

```bash
# 使用Docker
docker build -t smart-qa-app .
docker run -p 8501:8501 smart-qa-app

# 或使用Streamlit Cloud
# 参考：https://streamlit.io/cloud
```

## 参考资料

- [教程导航](../docs/README.md)
- [第11-17讲教程](../docs/part4-Agentic-RAG-生产实践篇-智能问答Web应用/)
- [Streamlit 文档](https://docs.streamlit.io/)
- [LangChain 文档](https://python.langchain.com/)
