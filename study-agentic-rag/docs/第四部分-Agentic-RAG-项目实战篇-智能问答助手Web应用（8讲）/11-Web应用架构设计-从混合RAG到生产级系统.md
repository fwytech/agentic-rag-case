# 第11讲：Web应用架构设计 - 从混合RAG到生产级系统

在前面10讲中，我们构建了完整的混合RAG系统，但它只是命令行工具。真实场景中，用户需要一个**友好的Web界面**。

这一讲，我们将设计一个**生产级智能问答Web应用**，让非技术用户也能轻松使用Agentic RAG。

---

## 一、从CLI到Web应用的演进

### CLI版本的局限

**混合RAG系统（第10讲）：**

```python
# 使用方式（命令行）
hybrid_rag = HybridRAG()
hybrid_rag.add_documents(docs)
result = hybrid_rag.query("孙悟空是谁？")
print(result["answer"])
```

**问题：**
- ❌ 需要编程知识才能使用
- ❌ 无法方便地上传文档
- ❌ 没有历史对话记录
- ❌ 无法保存和分享会话
- ❌ 不支持多用户

### Web应用的优势

```
┌─────────────────────────────────────────────────┐
│         生产级智能问答Web应用                      │
│                                                 │
│  用户界面                                         │
│  ├─ 聊天对话框 →  轻松提问                         │
│  ├─ 文档上传   →  拖拽上传PDF/Word/TXT             │
│  ├─ 历史记录   →  查看和导出                       │
│  ├─ 设置面板   →  可视化配置                       │
│  └─ 实时反馈   →  进度条、状态提示                  │
│                                                 │
│  技术特性                                         │
│  ├─ 多用户支持                                    │
│  ├─ 会话管理                                      │
│  ├─ 权限控制                                      │
│  ├─ 数据持久化                                    │
│  └─ 性能监控                                      │
└─────────────────────────────────────────────────┘
```

---

## 二、技术选型

### Web框架选择

**为什么选择Streamlit？**

| 框架 | 优点 | 缺点 | 适合场景 |
|------|------|------|---------|
| **Streamlit** | 快速开发、纯Python、自动刷新 | 定制化有限 | **内部工具、MVP** |
| Flask | 灵活、轻量 | 需要前端知识 | API服务 |
| FastAPI | 高性能、自动文档 | 需要前端分离 | 微服务 |
| Django | 功能全面 | 学习曲线陡 | 大型应用 |

**Streamlit的核心优势：**

```python
# 只需几行代码就能创建UI
import streamlit as st

st.title("智能问答系统")
question = st.text_input("请输入问题")
if st.button("提问"):
    answer = generate_answer(question)
    st.write(answer)
```

- ✅ 纯Python开发，无需HTML/CSS/JavaScript
- ✅ 自动响应式设计
- ✅ 内置组件丰富（文件上传、图表、表格）
- ✅ 适合数据科学和AI应用
- ✅ 快速原型到生产（几天即可上线）

### 向量数据库选择

**为什么选择FAISS？**

| 向量库 | 优点 | 缺点 | 适合场景 |
|-------|------|------|---------|
| **FAISS** | 速度快、本地部署、免费 | 无分布式 | **中小规模** |
| ChromaDB | 简单易用、持久化 | 性能一般 | 教学演示 |
| Pinecone | 云原生、分布式 | 收费 | 大规模生产 |
| Milvus | 功能强大、分布式 | 复杂 | 企业级 |

**FAISS特点：**
- Facebook开发的高性能向量搜索库
- 支持百万级向量快速检索（<100ms）
- 本地部署，无需外部服务
- 内存高效，支持量化压缩

### LLM提供商：双模式支持

**创新设计：Ollama + 阿里云百炼双模式**

```
┌─────────────────────────────────────────────────┐
│            双模式LLM架构                          │
│                                                 │
│  模式1：Ollama（本地）                            │
│  ├─ 适用场景：开发测试、隐私要求高                │
│  ├─ 优势：免费、离线、数据隐私                    │
│  └─ 劣势：需要GPU、模型能力有限                   │
│                                                 │
│  模式2：阿里云百炼（在线）                         │
│  ├─ 适用场景：生产环境、高质量要求                │
│  ├─ 优势：模型强大、无需GPU、稳定可靠             │
│  └─ 劣势：按调用付费、需要网络                   │
│                                                 │
│  统一接口：UnifiedLLMClient                      │
│  └─ 根据配置自动切换提供商                       │
└─────────────────────────────────────────────────┘
```

**为什么双模式？**

1. **开发阶段**：使用Ollama本地测试（免费、快速迭代）
2. **生产环境**：切换到百炼在线API（高质量、稳定）
3. **私有部署**：使用Ollama保护数据隐私
4. **成本优化**：根据流量灵活切换

---

## 三、系统架构设计

### 整体架构图

```
┌──────────────────────────────────────────────────────┐
│                 用户浏览器                            │
└────────────────┬─────────────────────────────────────┘
                 │
                 │ HTTP
                 ↓
┌──────────────────────────────────────────────────────┐
│              Streamlit Web服务                        │
│                                                      │
│  app.py (主应用入口)                                  │
│     ├─ 页面路由                                       │
│     ├─ UI渲染                                         │
│     └─ 会话管理                                       │
└────────────────┬─────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ↓                 ↓
┌──────────────┐   ┌──────────────┐
│  Config层     │   │  Models层    │
│              │   │              │
│  settings.py │   │  agent.py    │
│  ├─双模式配置 │   │  └─Agent代理  │
│  ├─参数管理   │   │              │
│  └─路径配置   │   └──────────────┘
└──────────────┘
        │
        ↓
┌──────────────────────────────────────────────┐
│              Services层（服务层）              │
│                                              │
│  ┌──────────────┐  ┌──────────────┐         │
│  │ llm_client   │  │vector_store  │         │
│  │ ├─Ollama     │  │ ├─FAISS      │         │
│  │ └─百炼API    │  │ └─嵌入模型   │         │
│  └──────────────┘  └──────────────┘         │
│                                              │
│  ┌──────────────┐                            │
│  │weather_tools │  (可扩展更多工具)          │
│  │ └─天气查询   │                            │
│  └──────────────┘                            │
└──────────────────────────────────────────────┘
        │
        ↓
┌──────────────────────────────────────────────┐
│              Utils层（工具层）                 │
│                                              │
│  ┌─────────────────┐  ┌──────────────┐      │
│  │document_processor│  │chat_history  │      │
│  │ ├─文件解析       │  │ ├─记录保存    │      │
│  │ ├─文本分块       │  │ └─导出CSV     │      │
│  │ └─元数据提取     │  │              │      │
│  └─────────────────┘  └──────────────┘      │
│                                              │
│  ┌──────────────┐  ┌──────────────┐         │
│  │ui_components │  │decorators    │         │
│  │ └─UI辅助     │  │ └─错误处理   │         │
│  └──────────────┘  └──────────────┘         │
└──────────────────────────────────────────────┘
```

### 分层设计原则

**为什么要分层？**

1. **职责分离**：每一层只负责特定功能
2. **易于测试**：可以独立测试每一层
3. **便于扩展**：添加新功能不影响其他层
4. **团队协作**：不同层可由不同人开发

**各层职责：**

| 层级 | 职责 | 示例 |
|------|------|------|
| **App层** | 页面渲染、用户交互 | `app.py` |
| **Config层** | 配置管理、参数定义 | `settings.py` |
| **Models层** | 核心业务逻辑 | `agent.py` |
| **Services层** | 外部服务调用 | `llm_client.py`, `vector_store.py` |
| **Utils层** | 通用工具函数 | `document_processor.py` |

---

## 四、目录结构

### 完整的项目结构

```
agentic_rag_smart_qa_project/
├── app.py                          # 主应用入口（约365行）
│
├── config/                         # 配置层
│   ├── __init__.py
│   └── settings.py                 # 系统配置（约211行）
│       ├─ LLM提供商配置（Ollama + 百炼）
│       ├─ 模型参数配置
│       ├─ 路径配置
│       └─ 工具开关
│
├── models/                         # 模型层
│   ├── __init__.py
│   └── agent.py                    # Agent代理（约230行）
│       ├─ LangChain集成
│       ├─ ReAct框架
│       ├─ 工具编排
│       └─ 记忆管理
│
├── services/                       # 服务层
│   ├── __init__.py
│   ├── llm_client.py              # LLM客户端（统一接口）
│   ├── vector_store.py            # 向量存储服务（约310行）
│   └── weather_tools.py           # 天气工具
│
├── utils/                          # 工具层
│   ├── __init__.py
│   ├── document_processor.py      # 文档处理
│   ├── chat_history.py            # 聊天历史
│   ├── ui_components.py           # UI组件
│   └── decorators.py              # 装饰器（错误处理、日志）
│
├── data/                           # 数据目录
│   └── documents/                 # 上传的文档
│
├── vector_store/                   # 向量存储
│   └── faiss_index/               # FAISS索引文件
│
├── chat_history/                   # 聊天记录
│   └── chat_history.json          # 历史记录文件
│
├── logs/                           # 日志目录
│   └── app.log                    # 应用日志
│
├── requirements.txt                # 依赖列表
├── .env                           # 环境变量（不提交到Git）
└── README.md                      # 项目说明
```

### 为什么这么组织？

1. **扁平化设计**
   - 不超过2层嵌套
   - 便于快速定位文件
   - 符合Python习惯

2. **按职责分组**
   - config：所有配置集中管理
   - models：业务逻辑
   - services：外部调用
   - utils：通用工具

3. **数据分离**
   - 代码和数据分开
   - data/、vector_store/、chat_history/独立
   - 便于备份和迁移

---

## 五、核心配置设计

### settings.py核心内容

**代码文件：** `study-agentic-rag/03-smart-qa-application/config/settings.py`

```python
import os
from pathlib import Path

class Settings:
    """系统配置类"""

    # ==================== 基础路径 ====================
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    VECTOR_STORE_DIR = BASE_DIR / "vector_store"
    CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
    LOG_DIR = BASE_DIR / "logs"

    # ==================== LLM 提供商配置 ====================
    # 可选: "ollama" 或 "online"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # 默认使用 Ollama

    # ==================== Ollama 配置 ====================
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODELS = [
        "qwen:7b",
        "qwen:14b",
        "llama2:7b",
        "mistral:7b",
    ]
    OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

    # ==================== 在线 API 配置（阿里云百炼） ====================
    ONLINE_API_KEY = os.getenv("ONLINE_API_KEY", "sk-xxx")
    ONLINE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ONLINE_MODELS = [
        "qwen-plus",
        "qwen-turbo",
        "qwen-max",
    ]
    ONLINE_EMBEDDING_MODEL = "text-embedding-v1"

    # ==================== 通用模型配置 ====================
    @classmethod
    def get_available_models(cls) -> List[str]:
        """根据 LLM 提供商返回可用模型列表"""
        if cls.LLM_PROVIDER == "ollama":
            return cls.OLLAMA_MODELS
        else:  # online
            return cls.ONLINE_MODELS

    @classmethod
    def get_default_model(cls) -> str:
        """获取默认模型"""
        if cls.LLM_PROVIDER == "ollama":
            return "qwen:7b"
        else:  # online
            return "qwen-plus"

    # RAG参数
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TOP_K = 3
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # 工具开关
    ENABLE_WEATHER_TOOL = True
    ENABLE_DOCUMENT_TOOL = True
```

**为什么这么写？**

1. **为什么用 `Path` 而不是字符串？**
   - `Path` 自动处理跨平台路径分隔符（Windows用`\`，Linux用`/`）
   - 支持链式操作：`BASE_DIR / "data" / "docs"`
   - 更安全，自动转义特殊字符

2. **为什么用 `os.getenv()` 读取环境变量？**
   - 安全：敏感信息（API密钥）不写在代码里
   - 灵活：不同环境（开发/测试/生产）用不同配置
   - 标准：符合12要素应用（12-Factor App）原则

3. **为什么用 `@classmethod`？**
   - 无需实例化，直接调用：`Settings.get_default_model()`
   - 节省内存，全局共享配置
   - 便于在任何地方访问配置

4. **为什么要工具开关？**
   - 开发环境可能没有天气API密钥
   - 按需启用工具，降低复杂度
   - 便于调试和测试

### 环境变量配置（.env文件）

```bash
# .env 文件（不提交到Git，添加到.gitignore）

# LLM提供商选择
LLM_PROVIDER=ollama  # 或 online

# Ollama配置（本地部署）
OLLAMA_BASE_URL=http://localhost:11434

# 阿里云百炼配置（在线API）
ONLINE_API_KEY=<your api key>
ONLINE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 天气API密钥（可选）
WEATHER_API_KEY=你的高德地图API密钥

# 日志级别
LOG_LEVEL=INFO
```

**使用方式：**

```python
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 现在可以用os.getenv()读取
api_key = os.getenv("ONLINE_API_KEY")
```

---

## 六、数据流转流程

### 用户提问的完整流程

```
用户在Web界面输入问题
   ↓
app.py: 捕获用户输入
   ↓
app.py: 调用 generate_response(query)
   ↓
app.py: 创建Agent（如果未创建）
   ├─ 从settings读取配置
   ├─ 创建llm_client（根据LLM_PROVIDER选择Ollama或百炼）
   ├─ 创建vector_store（FAISS）
   └─ 注册工具（文档搜索、天气查询）
   ↓
models/agent.py: agent.generate_response(query)
   ├─ LangChain ReAct循环
   ├─ 决定是否使用工具
   │  ├─ 使用文档搜索 → services/vector_store.py
   │  └─ 使用天气查询 → services/weather_tools.py
   ├─ LLM推理和生成 → services/llm_client.py
   └─ 返回答案
   ↓
app.py: 显示答案到界面
   ↓
app.py: 保存到聊天历史 → utils/chat_history.py
   ↓
用户看到回答
```

### 文档上传的完整流程

```
用户上传PDF文件
   ↓
app.py: 捕获文件
   ↓
app.py: process_uploaded_files(files)
   ↓
utils/document_processor.py: 解析文件
   ├─ PDF → 提取文本
   ├─ Word → 提取文本
   ├─ TXT/MD → 直接读取
   ├─ 文本清洗
   └─ 文本分块（chunk_size=1000, overlap=200）
   ↓
services/vector_store.py: add_documents(chunks)
   ├─ 生成嵌入向量（使用llm_client的嵌入模型）
   ├─ 构建FAISS索引
   └─ 保存索引到磁盘
   ↓
app.py: 更新向量存储状态
   ↓
用户看到"向量存储已准备"
```

---

## 七、关键设计决策

### 决策1：为什么用LangChain？

**LangChain的价值：**

```python
# 不用LangChain（需要自己实现ReAct）
def my_react_loop(query):
    for i in range(max_iterations):
        # 自己写Think逻辑
        decision = think(query, context)
        # 自己写Act逻辑
        observation = act(decision)
        # 自己写Observe逻辑
        is_done = observe(observation)
        if is_done:
            break
    return generate(context)

# 用LangChain（框架已实现）
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent, tools, max_iterations=5)
result = executor.invoke({"input": query})
```

**优势：**
- ✅ 成熟的ReAct实现（经过大量测试）
- ✅ 工具集成简单（`Tool`类）
- ✅ 记忆管理（`ConversationBufferMemory`）
- ✅ 丰富的集成（FAISS、OpenAI、Ollama）
- ✅ 社区活跃，问题好解决

### 决策2：为什么用会话状态管理？

**Streamlit的特点：**

```python
# Streamlit每次交互都会重新运行整个脚本
st.title("智能问答")

# 问题：每次都会重置为空列表
chat_history = []  # ❌ 用户刷新页面，历史丢失

# 解决：使用session_state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 添加消息
st.session_state.chat_history.append({"role": "user", "content": query})

# 历史会保留在整个会话中 ✅
```

**为什么需要？**
- Streamlit无状态架构，每次交互重新运行
- `session_state`保持会话数据
- 聊天历史、向量存储状态都依赖它

### 决策3：为什么要错误处理装饰器？

```python
# utils/decorators.py
def error_handler(func):
    """错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} 出错: {str(e)}")
            st.error(f"操作失败: {str(e)}")
            return None
    return wrapper

# 使用
@error_handler
def process_uploaded_files(self, files):
    # 文件处理逻辑
    ...
```

**价值：**
- 统一错误处理逻辑
- 自动记录日志
- 友好的用户错误提示
- 代码更简洁

---

## 八、总结

### 核心架构特点

1. **分层设计**
   - 4层架构：App → Config/Models → Services → Utils
   - 职责清晰，易于维护

2. **双模式LLM**
   - Ollama（本地）+ 阿里云百炼（在线）
   - 统一接口，灵活切换

3. **技术选型**
   - Streamlit：快速构建Web UI
   - FAISS：高性能向量检索
   - LangChain：成熟的Agent框架

4. **生产就绪**
   - 配置管理（环境变量）
   - 错误处理（装饰器）
   - 日志记录
   - 数据持久化

### 架构优势

| 特性 | 实现方式 | 价值 |
|------|---------|------|
| **快速开发** | Streamlit纯Python | 几天上线MVP |
| **灵活部署** | 双模式LLM | 本地+云端 |
| **易于扩展** | 分层+工具系统 | 添加新功能简单 |
| **用户友好** | Web界面 | 非技术用户可用 |
| **成本优化** | Ollama免费 | 降低开发成本 |

---

## 下一讲预告

架构设计完成了，下一步是具体实现。

**第12讲：Streamlit界面开发 - 打造友好的用户体验**

我们将学习如何用Streamlit构建完整的Web界面：
- 聊天对话界面设计
- 文档上传组件
- 侧边栏配置面板
- 实时状态反馈
- 响应式布局
- 完整的app.py实现（365行代码详解）

敬请期待！
