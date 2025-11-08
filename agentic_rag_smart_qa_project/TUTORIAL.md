# Agentic RAG 智能问答系统 - 完整使用教程

## 📚 目录

1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [环境要求](#环境要求)
4. [安装配置](#安装配置)
5. [快速开始](#快速开始)
6. [功能详解](#功能详解)
7. [高级配置](#高级配置)
8. [故障排除](#故障排除)
9. [扩展开发](#扩展开发)
10. [最佳实践](#最佳实践)

---

## 📖 项目概述

### 什么是 Agentic RAG?

**Agentic RAG (Retrieval-Augmented Generation with Agents)** 是一种结合了检索增强生成 (RAG) 和智能代理 (Agent) 技术的高级 AI 系统。与传统 RAG 相比,Agentic RAG 具有以下优势:

| 特性 | 传统 RAG | Agentic RAG |
|------|---------|------------|
| **工作流程** | 单向：检索→生成 | 循环：思考→行动→观察→生成 |
| **工具使用** | 仅限文档检索 | 多工具协作（文档+天气+计算等） |
| **决策能力** | 被动响应 | 主动决策和规划 |
| **上下文理解** | 静态上下文 | 动态对话记忆 |

### 本项目特点

✨ **核心功能**
- 🤖 基于 LangChain ReAct Agent 的智能问答
- 📚 支持多种文档格式（PDF、TXT、MD、DOCX）
- 🔍 FAISS 向量检索（支持相似度和 MMR 搜索）
- 🌤️ 集成天气查询工具（高德地图 API）
- 💬 完整的对话历史管理
- 📊 实时性能统计和监控

✨ **技术亮点**
- **本地优先**: 使用 Ollama 本地模型，数据隐私有保障
- **零配置向量库**: FAISS 持久化存储，无需额外数据库
- **模块化设计**: 服务分离，易于扩展和维护
- **完善的错误处理**: 装饰器模式实现统一异常管理
- **现代化 UI**: Streamlit 构建的友好交互界面

---

## 🏗️ 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    用户界面层 (Streamlit)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 文档上传    │  │ 参数配置    │  │ 聊天界面    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    应用层 (app.py)                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         AgenticRAGSystem (主应用类)                   │  │
│  │  • 初始化系统组件                                     │  │
│  │  • 管理会话状态                                       │  │
│  │  • 协调各个服务                                       │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼────────┐ ┌───▼──────┐ ┌─────▼────────┐
│   Agent 层      │ │ 服务层    │ │   工具层     │
│                 │ │           │ │              │
│ AgenticRAGAgent │ │ Vector    │ │ Document     │
│  (ReAct Agent)  │ │ Store     │ │ Processor    │
│                 │ │           │ │              │
│ • Think决策     │ │ • FAISS   │ │ • PDF解析    │
│ • Act执行       │ │ • 向量化  │ │ • 文本分块   │
│ • Observe评估   │ │ • 检索    │ │ • 缓存管理   │
│                 │ │           │ │              │
│ Tools:          │ │ Weather   │ │ Chat         │
│ ├─ doc_search   │ │ Tools     │ │ History      │
│ └─ weather      │ │           │ │              │
│                 │ │ • 高德API │ │ • 历史保存   │
│                 │ │ • 天气查询│ │ • 导出功能   │
└─────────────────┘ └───────────┘ └──────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    基础设施层                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Ollama   │  │  FAISS   │  │  Cache   │  │  Logs    │  │
│  │ (LLM)    │  │ (Vector) │  │  (Disk)  │  │  (File)  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件说明

#### 1. **AgenticRAGAgent** (models/agent.py)
智能代理核心,实现 ReAct 框架:

```python
# ReAct 循环流程
while not finished:
    # Think: 分析当前状态,决定下一步行动
    decision = agent.think(query, context)

    # Act: 执行选定的工具
    if decision.need_tool:
        observation = agent.act(tool_name, tool_input)

    # Observe: 评估结果,决定是否继续
    evaluation = agent.observe(observation)

    if evaluation.is_sufficient:
        break

# Generate: 生成最终回答
answer = agent.generate_final_answer(query, accumulated_context)
```

**关键特性:**
- 🧠 基于 LangChain 的 create_react_agent
- 🔧 动态工具注册和管理
- 💾 对话记忆（ConversationBufferMemory）
- 🔄 最多 5 次迭代保证质量

#### 2. **VectorStoreService** (services/vector_store.py)
向量存储和检索服务:

```python
# 创建向量存储
vector_store = FAISS.from_documents(
    documents=documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text")
)

# 两种搜索模式
# 1. 相似度搜索 - 基于余弦相似度
results = vector_store.similarity_search_with_score(query, k=3)

# 2. MMR搜索 - 平衡相关性和多样性
results = vector_store.max_marginal_relevance_search(query, k=3)
```

**关键特性:**
- 📦 FAISS 索引持久化
- 🔢 向量维度: 768 (nomic-embed-text)
- 🔍 支持元数据过滤
- 💾 自动保存文档和元数据

#### 3. **DocumentProcessor** (utils/document_processor.py)
文档处理和缓存管理:

```python
# 处理流程
file → 哈希计算 → 缓存检查 → 文件解析 → 文本分块 → 返回Document[]

# 支持的文件类型
• PDF    → PyPDFLoader
• TXT    → TextLoader
• MD     → TextLoader
• DOCX   → UnstructuredWordDocumentLoader

# 分块策略
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
```

**关键特性:**
- 🗂️ MD5 哈希缓存，避免重复处理
- ⏱️ 缓存过期时间: 3600秒（1小时）
- 📏 智能分块，保留语义完整性
- 🔖 自动添加元数据（source, page, chunk_id等）

#### 4. **WeatherTools** (services/weather_tools.py)
天气查询工具集成:

```python
# 获取当前天气
weather = weather_tools.get_current_weather("北京")

# 获取天气预报（1-7天）
forecast = weather_tools.get_weather_forecast("上海", days=3)

# 返回格式化的天气信息
🏙️ **北京市** 当前天气
🌤️ **天气状况**: 晴
🌡️ **气温**: 25°C
💨 **风向风力**: 南风 3级
💧 **湿度**: 45%
```

**关键特性:**
- 🔑 高德地图 API 集成
- 🗺️ 城市代码缓存
- 💡 智能天气建议
- 📅 支持 1-7 天预报

---

## 💻 环境要求

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Linux/macOS/Windows | Linux/macOS |
| **Python** | 3.8+ | 3.10+ |
| **内存** | 8GB | 16GB+ |
| **磁盘** | 10GB | 20GB+ (用于模型存储) |
| **CPU** | 4核 | 8核+ |
| **GPU** | 可选 | NVIDIA GPU (用于加速) |

### 必需软件

#### 1. **Python 3.8+**

```bash
# 检查 Python 版本
python --version
# 或
python3 --version

# 推荐使用 pyenv 管理 Python 版本
pyenv install 3.10.0
pyenv local 3.10.0
```

#### 2. **Ollama** (本地 LLM 引擎)

**什么是 Ollama?**
Ollama 是一个本地运行大语言模型的工具，支持 Llama2、Mistral、Qwen 等多种开源模型。

**安装 Ollama:**

```bash
# macOS / Linux
curl https://ollama.ai/install.sh | sh

# Windows
# 访问 https://ollama.ai/download 下载安装包

# 验证安装
ollama --version
```

**下载模型:**

```bash
# 下载 Qwen 7B 模型（推荐）
ollama pull qwen:7b

# 下载 Llama2 7B 模型
ollama pull llama2:7b

# 下载嵌入模型（必需）
ollama pull nomic-embed-text

# 列出已下载的模型
ollama list
```

**启动 Ollama 服务:**

```bash
# 启动服务（默认端口 11434）
ollama serve

# 测试模型
ollama run qwen:7b "你好"
```

#### 3. **高德地图 API 密钥** (可选)

如果需要使用天气查询功能，需要申请高德地图 API Key:

1. 访问 [高德开放平台](https://lbs.amap.com/)
2. 注册账号并登录
3. 进入"控制台" → "应用管理" → "创建新应用"
4. 添加 Key，服务类型选择"Web服务"
5. 复制 API Key

---

## ⚙️ 安装配置

### 方式一：使用虚拟环境（推荐）

```bash
# 1. 克隆项目（如果还没有）
cd agentic_rag_smart_qa_project

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# 4. 升级 pip
pip install --upgrade pip

# 5. 安装依赖
pip install -r requirements.txt
```

### 方式二：使用 Conda

```bash
# 1. 创建 Conda 环境
conda create -n agentic_rag python=3.10

# 2. 激活环境
conda activate agentic_rag

# 3. 安装依赖
pip install -r requirements.txt
```

### 依赖说明

项目主要依赖（requirements.txt）:

```
streamlit==1.28.1          # Web UI 框架
langchain==0.0.335         # AI Agent 框架
langchain-community        # LangChain 社区组件
faiss-cpu==1.7.4          # 向量数据库
ollama==0.1.7             # Ollama Python SDK
pypdf==3.17.0             # PDF 解析
python-docx==1.1.0        # Word 文档处理
requests==2.31.0          # HTTP 请求
numpy==1.24.3             # 数值计算
pandas==2.1.3             # 数据处理
plotly==5.18.0            # 数据可视化
scikit-learn==1.3.2       # 机器学习工具
sentence-transformers     # 句子嵌入
```

### 配置文件设置

编辑 `config/settings.py` 中的关键配置:

```python
# 天气 API 配置（如果使用天气功能）
WEATHER_API_KEY = "你的高德地图API密钥"

# 模型配置（根据你下载的模型调整）
AVAILABLE_MODELS = [
    "qwen:7b",      # 推荐：中文表现好
    "llama2:7b",    # 英文表现好
    "mistral:7b",   # 速度快
]

DEFAULT_MODEL = "qwen:7b"

# 向量存储配置
CHUNK_SIZE = 1000          # 文本分块大小
CHUNK_OVERLAP = 200        # 分块重叠
VECTOR_DIMENSION = 768     # 向量维度（nomic-embed-text）

# 缓存配置
CACHE_ENABLED = True
CACHE_EXPIRE_TIME = 3600   # 1小时

# 日志配置
LOG_LEVEL = "INFO"         # DEBUG/INFO/WARNING/ERROR
```

### 目录结构初始化

```bash
# 项目会自动创建以下目录，也可手动创建
mkdir -p data/document_cache
mkdir -p data/temp
mkdir -p vector_store
mkdir -p chat_history
mkdir -p logs
```

---

## 🚀 快速开始

### 1. 启动 Ollama 服务

```bash
# 在新终端窗口启动 Ollama
ollama serve

# 保持此窗口运行
```

### 2. 启动应用

```bash
# 确保在项目目录下
cd agentic_rag_smart_qa_project

# 激活虚拟环境（如果使用）
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 启动 Streamlit 应用
streamlit run app.py

# 看到以下输出表示成功:
#   You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
```

### 3. 访问应用

浏览器自动打开 http://localhost:8501

如果没有自动打开，手动访问该地址。

### 4. 第一次使用

#### 步骤 1: 配置模型

1. 在左侧边栏 "⚙️ 系统配置" 中选择模型
2. 推荐首次使用 `qwen:7b`
3. 调整温度系数（0.7 为平衡值）

#### 步骤 2: 上传文档（可选）

1. 在左侧边栏 "📄 文档上传" 区域
2. 点击 "Browse files" 上传 PDF/TXT/MD/DOCX
3. 点击 "🔄 处理文档"
4. 等待处理完成，看到 "✅ 向量存储已准备"

#### 步骤 3: 开始对话

1. 在主界面输入框输入问题
2. 示例问题:
   ```
   # 如果上传了文档
   文档中讲了什么内容？

   # 如果配置了天气 API
   北京今天天气怎么样？

   # 通用问题
   请介绍一下人工智能的发展历史
   ```
3. 按 Enter 或点击发送
4. 等待 Agent 思考和回答

### 5. 查看 Agent 工作过程

在终端窗口可以看到 Agent 的思考过程:

```
开始执行函数: generate_response
> Entering new AgentExecutor chain...
[Agent] 思考: 用户问了关于文档的问题，我需要使用文档搜索工具
[Agent] 行动: tool_0
[Agent] 行动输入: 文档主要内容
[Tool] 观察: 【文档1】内容: ...
[Agent] 思考: 我已经获得了足够的信息，可以回答了
[Agent] 最终回答: 根据文档内容，...
> Finished chain.
```

---

## 🔧 功能详解

### 功能 1: 文档问答

#### 1.1 上传和处理文档

**支持的文件格式:**

| 格式 | 扩展名 | 说明 | 最佳实践 |
|------|--------|------|----------|
| PDF | .pdf | 最常用 | 确保 PDF 可选择文字（非扫描版） |
| 文本 | .txt | 纯文本 | UTF-8 编码 |
| Markdown | .md | 结构化文本 | 适合技术文档 |
| Word | .docx | Office 文档 | .doc 不支持，需转换为 .docx |

**上传步骤:**

```bash
1. 点击左侧边栏 "📄 文档上传"
2. 选择文件（可多选）
3. 查看文件列表确认
4. 点击 "🔄 处理文档"
5. 观察处理进度条
6. 看到 "✅ 成功处理 N 个文档片段"
```

**文档处理流程:**

```
文件上传
  ↓
MD5 哈希计算
  ↓
检查缓存 → 命中 → 直接返回
  ↓ 未命中
文件解析
  ↓
文本提取
  ↓
智能分块 (chunk_size=1000, overlap=200)
  ↓
向量化 (nomic-embed-text)
  ↓
存入 FAISS 索引
  ↓
保存缓存
  ↓
完成
```

#### 1.2 查询文档

**示例问题:**

```markdown
# 简单问题（直接检索）
Q: 文档的作者是谁？
Q: 第三章讲了什么？

# 复杂问题（Agent 多步推理）
Q: 总结文档的核心观点
Q: 文档中提到的方法有哪些优缺点？
Q: 对比文档中的两种技术方案

# 结合外部知识
Q: 文档中的技术在实际中如何应用？
```

**查询参数调优:**

在左侧边栏 "🔍 RAG设置" 中:

| 参数 | 说明 | 推荐值 | 使用场景 |
|------|------|--------|----------|
| **Top-K** | 检索文档数量 | 3-5 | 简单问题用3,复杂问题用5-10 |
| **搜索类型** | similarity/mmr | similarity | 一般问题用similarity,需要多样性用mmr |

**相似度 vs MMR:**

```python
# Similarity: 纯相似度搜索
# 返回: [相似度0.95, 相似度0.94, 相似度0.93]
# 优点: 相关性高
# 缺点: 可能重复

# MMR: 最大边际相关性
# 返回: [相似度0.95, 相似度0.85, 相似度0.75]
# 优点: 内容多样
# 缺点: 可能不够精准

# 公式: MMR = λ * Similarity - (1-λ) * Redundancy
```

#### 1.3 向量存储管理

**保存索引:**
```bash
# 自动保存到 vector_store/faiss_index
# 包含以下文件:
vector_store/
  ├── faiss_index/index.faiss          # FAISS 索引
  ├── faiss_index/index.pkl            # 索引元数据
  ├── faiss_index_docs.pkl             # 文档内容
  └── faiss_index_metadata.json        # 统计信息
```

**加载已有索引:**
```bash
1. 确保 vector_store/faiss_index 目录存在
2. 点击左侧边栏 "📂 加载已有向量存储"
3. 看到 "✅ 向量存储加载成功"
```

**清空索引:**
```bash
1. 点击左侧边栏 "🗑️ 清空向量存储"
2. 确认操作
3. 向量存储被清空，需重新上传文档
```

### 功能 2: 天气查询

#### 2.1 配置天气 API

编辑 `config/settings.py`:

```python
# 高德地图 API 配置
WEATHER_API_KEY = "你的API密钥"  # 替换为实际密钥
WEATHER_API_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
WEATHER_CITY_URL = "https://restapi.amap.com/v3/config/district"
```

#### 2.2 查询天气

**当前天气:**
```markdown
Q: 北京今天天气怎么样？
Q: 上海现在的气温是多少？
Q: 深圳的天气如何？

# Agent 会自动调用天气工具
[Agent] 行动: weather_query
[Agent] 行动输入: {"city": "北京", "forecast_days": 1}
```

**天气预报:**
```markdown
Q: 上海未来3天的天气预报
Q: 北京这周的天气情况
Q: 广州明天会下雨吗？
```

**天气返回格式:**
```
🏙️ **北京市** 当前天气

🌤️ **天气状况**: 晴
🌡️ **气温**: 25°C
💨 **风向风力**: 南风 3级
💧 **湿度**: 45%
📅 **发布时间**: 2024-01-15 14:00:00

💡 **温馨提示**:
• 天气舒适，适合外出。
• 湿度较低，注意补水。
```

### 功能 3: 智能对话

#### 3.1 对话记忆

系统自动保存对话上下文:

```python
# 示例对话
用户: 我想了解人工智能
助手: 人工智能是...

用户: 它有哪些应用？  # Agent 知道"它"指人工智能
助手: 人工智能的应用包括...

用户: 第一个应用能详细说说吗？  # 记得之前提到的应用
助手: 关于第一个应用...
```

**对话记忆配置:**

```python
# models/agent.py
ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

# 记忆策略
• ConversationBufferMemory: 保存完整历史
• 最大上下文: 受模型限制（通常 4096 tokens）
• 超出处理: 自动截断最早的消息
```

#### 3.2 多轮对话示例

**场景 1: 文档问答**
```
用户: 上传的文档讲了什么？
助手: 文档主要讲解了...共分为3个部分

用户: 第一部分详细说说
助手: 第一部分重点介绍了...

用户: 有什么实际应用案例吗？
助手: 根据文档，实际应用包括...
```

**场景 2: 天气查询**
```
用户: 北京今天天气怎么样？
助手: 北京今天晴，温度25°C...

用户: 那上海呢？
助手: 上海今天多云，温度23°C...

用户: 明天会下雨吗？
助手: 明天（上海）有小雨...
```

**场景 3: 混合查询**
```
用户: 文档中提到的项目在哪里实施？
助手: 根据文档，项目在北京实施

用户: 那边现在天气如何？
助手: [调用天气工具] 北京当前...
```

### 功能 4: 聊天历史管理

#### 4.1 查看聊天统计

右侧栏显示:
```
📊 聊天统计
├─ 总消息数: 20
├─ 用户消息: 10
└─ 助手消息: 10

最近消息
👤: 北京今天天气怎么样？
🤖: 北京今天晴，温度...
👤: 文档讲了什么？
🤖: 文档主要介绍...
```

#### 4.2 导出聊天记录

**导出为 CSV:**
```bash
1. 点击左侧边栏 "📥 导出聊天记录"
2. 选择 "下载CSV文件"
3. 文件格式:
   role,content,timestamp,id
   user,"你好",2024-01-15T14:00:00,uuid-1
   assistant,"你好！",2024-01-15T14:00:01,uuid-2
```

**导出为 JSON:**
```json
[
  {
    "role": "user",
    "content": "你好",
    "timestamp": "2024-01-15T14:00:00",
    "id": "uuid-1"
  },
  {
    "role": "assistant",
    "content": "你好！有什么我可以帮助你的吗？",
    "timestamp": "2024-01-15T14:00:01",
    "id": "uuid-2"
  }
]
```

#### 4.3 清空聊天记录

```bash
# 方式 1: UI 操作
左侧边栏 → "🗑️ 清空聊天记录" → 确认

# 方式 2: 手动删除
rm -f chat_history/chat_history.json

# 清空后:
• 聊天界面清空
• 对话记忆重置
• 历史文件被删除
```

### 功能 5: 参数调优

#### 5.1 模型参数

**温度系数 (Temperature):**

```python
# 影响回答的随机性
temperature = 0.0    # 完全确定性，适合数学、代码
temperature = 0.3    # 低随机性，适合事实问答
temperature = 0.7    # 平衡创造性，适合一般对话 ✓ 推荐
temperature = 1.0    # 高随机性，适合创意写作
```

**最大 Token 数 (Max Tokens):**

```python
# 控制回答长度
max_tokens = 100     # 简短回答
max_tokens = 512     # 中等长度 ✓ 推荐快速响应
max_tokens = 2048    # 详细回答 ✓ 推荐完整答案
max_tokens = 4000    # 长文回答
```

#### 5.2 RAG 参数

**Top-K:**

```python
# 检索文档数量
top_k = 1            # 最相关的1个文档
top_k = 3            # 推荐值，平衡质量和速度 ✓
top_k = 5            # 复杂问题
top_k = 10           # 需要综合多个文档
```

**搜索类型:**

```python
# Similarity - 相似度搜索
适用场景:
✓ 精确问答
✓ 事实查询
✓ 单一主题

# MMR - 最大边际相关性
适用场景:
✓ 需要多样性
✓ 综合分析
✓ 对比问题
```

---

## 🎓 高级配置

### 1. 自定义模型配置

编辑 `config/settings.py`:

```python
# 添加新模型
AVAILABLE_MODELS = [
    "qwen:7b",
    "qwen:14b",        # 更大的模型，更好的性能
    "llama2:7b",
    "llama2:13b",
    "codellama:7b",    # 代码专用模型
    "mistral:7b",      # 快速模型
]

# 为不同模型设置不同参数
def get_model_config(cls, model_name: str) -> Dict[str, Any]:
    model_configs = {
        "qwen:14b": {
            "temperature": 0.7,
            "max_tokens": 4096,      # 更大的上下文
            "top_p": 0.9,
        },
        "codellama:7b": {
            "temperature": 0.2,      # 代码生成用低温度
            "max_tokens": 2048,
        }
    }
    return model_configs.get(model_name, {...})
```

### 2. 自定义文本分块策略

编辑 `services/vector_store.py`:

```python
# 调整分块参数
CHUNK_SIZE = 1000           # 每块字符数
CHUNK_OVERLAP = 200         # 重叠字符数

# 优化分隔符（中文文档）
separators = [
    "\n\n",        # 段落
    "\n",          # 行
    "。",          # 句号
    "！",          # 感叹号
    "？",          # 问号
    "；",          # 分号
    "，",          # 逗号
    " ",           # 空格
    ""             # 字符
]

# 英文文档分隔符
separators = [
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    ""
]
```

### 3. 添加自定义工具

#### 步骤 1: 创建工具函数

在 `services/` 目录创建新文件 `custom_tools.py`:

```python
import logging

logger = logging.getLogger(__name__)

class Calculator:
    """计算器工具"""

    @staticmethod
    def calculate(expression: str) -> str:
        """执行数学计算"""
        try:
            # 安全评估表达式
            result = eval(expression, {"__builtins__": {}})
            return f"计算结果: {result}"
        except Exception as e:
            logger.error(f"计算错误: {str(e)}")
            return f"计算失败: {str(e)}"

class WebSearch:
    """网络搜索工具"""

    @staticmethod
    def search(query: str, num_results: int = 3) -> str:
        """搜索互联网"""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))

            formatted = []
            for i, result in enumerate(results, 1):
                formatted.append(
                    f"{i}. {result['title']}\n"
                    f"   {result['body']}\n"
                    f"   {result['href']}"
                )

            return "\n\n".join(formatted)
        except Exception as e:
            return f"搜索失败: {str(e)}"
```

#### 步骤 2: 注册工具到 Agent

编辑 `app.py`:

```python
from services.custom_tools import Calculator, WebSearch

class AgenticRAGSystem:
    def _create_agent(self):
        tools = []

        # 现有工具
        if st.session_state.vector_store_ready:
            tools.append(self._create_document_search_tool())
        tools.append(self._create_weather_tool())

        # 添加新工具
        tools.append(self._create_calculator_tool())
        tools.append(self._create_web_search_tool())

        self.agent = AgenticRAGAgent(
            model_name=st.session_state.current_model,
            tools=tools
        )

    def _create_calculator_tool(self):
        """创建计算器工具"""
        def calculator(expression: str) -> str:
            """执行数学计算，支持加减乘除、乘方等"""
            return Calculator.calculate(expression)
        return calculator

    def _create_web_search_tool(self):
        """创建网络搜索工具"""
        def web_search(query: str) -> str:
            """搜索互联网获取最新信息"""
            return WebSearch.search(query, num_results=3)
        return web_search
```

#### 步骤 3: 更新系统提示词

编辑 `models/agent.py`:

```python
def _get_default_system_prompt(self) -> str:
    return """你是一个智能问答助手，具备以下能力：

1. 文档问答：基于上传的文档回答问题
2. 天气查询：查询实时天气和预报
3. 数学计算：执行数学运算
4. 网络搜索：搜索最新信息

工具使用说明：
- document_search: 搜索文档内容
- weather_query: 查询天气信息
- calculator: 执行数学计算
- web_search: 搜索互联网信息

根据用户问题智能选择合适的工具。
"""
```

#### 步骤 4: 测试新工具

```markdown
# 测试计算器
Q: 计算 25 * 4 + 10
A: [Agent调用calculator] 计算结果: 110

# 测试网络搜索
Q: 最新的AI技术进展
A: [Agent调用web_search] 根据搜索结果，最新的AI技术包括...
```

### 4. 配置日志系统

编辑 `config/settings.py`:

```python
# 日志级别
LOG_LEVEL = "DEBUG"    # 开发环境
LOG_LEVEL = "INFO"     # 生产环境 ✓
LOG_LEVEL = "WARNING"  # 仅警告和错误
LOG_LEVEL = "ERROR"    # 仅错误

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 日志文件配置
LOG_FILE = LOG_DIR / "app.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5               # 保留5个备份

# 启用日志轮转
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT
)
handler.setFormatter(logging.Formatter(LOG_FORMAT))

logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)
logger.addHandler(handler)
```

**查看日志:**

```bash
# 实时查看日志
tail -f logs/app.log

# 查看最近100行
tail -n 100 logs/app.log

# 搜索错误
grep ERROR logs/app.log

# 查看某个时间段的日志
grep "2024-01-15 14:" logs/app.log
```

### 5. 性能优化

#### 5.1 向量存储优化

```python
# 使用GPU加速（如果可用）
import faiss

# CPU索引
index = faiss.IndexFlatL2(dimension)

# GPU索引（需要faiss-gpu）
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

# 使用更快的索引类型
# IVF索引 - 近似搜索，速度快
nlist = 100  # 聚类中心数
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# 训练索引（首次需要）
index.train(training_vectors)
index.add(vectors)

# HNSW索引 - 高精度近似搜索
M = 32  # 连接数
index = faiss.IndexHNSWFlat(dimension, M)
```

#### 5.2 缓存优化

```python
# 启用文档缓存
CACHE_ENABLED = True
CACHE_EXPIRE_TIME = 3600  # 1小时

# 使用Redis缓存（高级）
import redis

class RedisCache:
    def __init__(self):
        self.redis = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )

    def set(self, key: str, value: Any, expire: int = 3600):
        import json
        self.redis.setex(key, expire, json.dumps(value))

    def get(self, key: str) -> Any:
        import json
        data = self.redis.get(key)
        return json.loads(data) if data else None
```

#### 5.3 并发处理

```python
# 异步文档处理
import asyncio

async def process_documents_async(files):
    tasks = [
        asyncio.create_task(process_file(file))
        for file in files
    ]
    results = await asyncio.gather(*tasks)
    return results

# 并行向量化
from concurrent.futures import ThreadPoolExecutor

def vectorize_parallel(texts, batch_size=100):
    with ThreadPoolExecutor(max_workers=4) as executor:
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        futures = [executor.submit(embeddings.embed_documents, batch) for batch in batches]
        results = [future.result() for future in futures]
    return [vec for batch in results for vec in batch]
```

---

## 🐛 故障排除

### 常见问题

#### 问题 1: Ollama 连接失败

**错误信息:**
```
Error: Failed to connect to Ollama service
ConnectionError: HTTPConnectionPool(host='localhost', port=11434)
```

**解决方案:**

```bash
# 1. 检查 Ollama 是否运行
ps aux | grep ollama

# 2. 启动 Ollama 服务
ollama serve

# 3. 检查端口是否被占用
lsof -i :11434

# 4. 测试连接
curl http://localhost:11434/api/tags

# 5. 如果端口不同，修改配置
# 编辑 models/agent.py
Ollama(base_url="http://localhost:自定义端口")
```

#### 问题 2: 模型未找到

**错误信息:**
```
Error: model 'qwen:7b' not found
```

**解决方案:**

```bash
# 1. 列出已安装的模型
ollama list

# 2. 下载缺失的模型
ollama pull qwen:7b

# 3. 验证模型
ollama run qwen:7b "测试"

# 4. 如果下载慢，使用镜像
export OLLAMA_MIRRORS=https://registry.cn-hangzhou.aliyuncs.com
ollama pull qwen:7b
```

#### 问题 3: FAISS 索引损坏

**错误信息:**
```
Error: Failed to load FAISS index
RuntimeError: Invalid index file
```

**解决方案:**

```bash
# 1. 删除损坏的索引
rm -rf vector_store/faiss_index*

# 2. 在 UI 中重新上传文档
# 或手动重建索引

# 3. 备份索引（预防）
cp -r vector_store vector_store_backup_$(date +%Y%m%d)
```

#### 问题 4: 内存不足

**错误信息:**
```
MemoryError: Unable to allocate array
RuntimeError: CUDA out of memory
```

**解决方案:**

```python
# 1. 减小批处理大小
CHUNK_SIZE = 500  # 从1000降到500

# 2. 限制Top-K
top_k = 3  # 不要超过5

# 3. 使用更小的模型
ollama pull qwen:7b  # 而不是 qwen:14b

# 4. 释放GPU内存
import torch
torch.cuda.empty_cache()

# 5. 使用CPU模式
# requirements.txt
faiss-cpu  # 而不是 faiss-gpu
```

#### 问题 5: 文档上传失败

**错误信息:**
```
Error: File size exceeds limit
ValueError: Unsupported file type
```

**解决方案:**

```python
# 1. 检查文件大小限制
# config/settings.py
MAX_FILE_SIZE = 50 * 1024 * 1024  # 增加到50MB

# 2. 检查文件类型
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md", ".docx"]

# 3. PDF 转换问题
# 安装 poppler
# Ubuntu/Debian:
sudo apt-get install poppler-utils

# macOS:
brew install poppler

# Windows:
# 下载 https://github.com/oschwartz10612/poppler-windows/releases/

# 4. 检查文件编码（文本文件）
# 确保使用 UTF-8 编码
iconv -f GB2312 -t UTF-8 input.txt > output.txt
```

#### 问题 6: 天气查询失败

**错误信息:**
```
Error: 获取天气信息失败
API key invalid
```

**解决方案:**

```bash
# 1. 检查API密钥配置
# config/settings.py
WEATHER_API_KEY = "你的正确密钥"

# 2. 测试API密钥
curl "https://restapi.amap.com/v3/config/district?keywords=北京&key=你的密钥"

# 3. 检查API配额
# 登录高德开放平台查看使用情况

# 4. 禁用天气功能（临时）
# config/settings.py
ENABLE_WEATHER_TOOL = False
```

#### 问题 7: Streamlit 端口冲突

**错误信息:**
```
OSError: [Errno 48] Address already in use
```

**解决方案:**

```bash
# 1. 查找占用端口的进程
lsof -i :8501

# 2. 杀死进程
kill -9 <PID>

# 3. 使用不同端口
streamlit run app.py --server.port 8502

# 4. 配置默认端口
# .streamlit/config.toml
[server]
port = 8502
```

### 调试技巧

#### 1. 启用详细日志

```python
# config/settings.py
LOG_LEVEL = "DEBUG"

# 查看Agent思考过程
AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 启用详细输出
    return_intermediate_steps=True
)
```

#### 2. 使用Python调试器

```python
# 在代码中添加断点
import pdb; pdb.set_trace()

# 或使用IPython
import IPython; IPython.embed()

# 运行时检查变量
(Pdb) print(documents)
(Pdb) print(vector_store.get_stats())
```

#### 3. 性能分析

```python
# 使用装饰器
from utils.decorators import performance_monitor

@performance_monitor(warning_threshold=1.0, error_threshold=5.0)
def slow_function():
    # 你的代码
    pass

# 查看性能日志
grep "性能" logs/app.log
```

---

## 🔨 扩展开发

### 1. 添加新的 UI 页面

创建 `pages/analytics.py`:

```python
import streamlit as st
import pandas as pd
from utils.chat_history import ChatHistoryManager

def main():
    st.set_page_config(page_title="分析面板", page_icon="📊")

    st.title("📊 聊天分析面板")

    # 加载聊天历史
    chat_manager = ChatHistoryManager()
    history = chat_manager.get_history()

    if not history:
        st.info("暂无聊天记录")
        return

    # 统计分析
    df = pd.DataFrame(history)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总消息数", len(df))
    with col2:
        user_msgs = len(df[df['role'] == 'user'])
        st.metric("用户消息", user_msgs)
    with col3:
        asst_msgs = len(df[df['role'] == 'assistant'])
        st.metric("助手消息", asst_msgs)

    # 时间分析
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour

        st.subheader("📈 每小时消息分布")
        hourly = df.groupby('hour').size()
        st.bar_chart(hourly)

    # 消息长度分析
    st.subheader("📏 消息长度分析")
    df['length'] = df['content'].str.len()
    st.line_chart(df.groupby('role')['length'].mean())

if __name__ == "__main__":
    main()
```

运行新页面:
```bash
streamlit run pages/analytics.py
```

### 2. 实现自定义嵌入模型

创建 `services/custom_embeddings.py`:

```python
from typing import List
from langchain.embeddings.base import Embeddings
import requests

class CustomEmbeddings(Embeddings):
    """自定义嵌入模型"""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"texts": texts}
        )
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        return self.embed_documents([text])[0]

# 使用自定义嵌入
from services.custom_embeddings import CustomEmbeddings

embeddings = CustomEmbeddings(
    api_url="https://api.example.com/embeddings",
    api_key="your-api-key"
)

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)
```

### 3. 集成外部数据库

#### PostgreSQL + pgvector

```python
# 安装依赖
pip install psycopg2-binary pgvector

# services/postgres_vector_store.py
import psycopg2
from pgvector.psycopg2 import register_vector

class PostgresVectorStore:
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
        register_vector(self.conn)
        self._create_tables()

    def _create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    embedding vector(768),
                    metadata JSONB
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embedding_idx
                ON documents USING ivfflat (embedding vector_cosine_ops)
            """)
            self.conn.commit()

    def add_documents(self, documents, embeddings):
        with self.conn.cursor() as cur:
            for doc, emb in zip(documents, embeddings):
                cur.execute(
                    "INSERT INTO documents (content, embedding, metadata) VALUES (%s, %s, %s)",
                    (doc.page_content, emb, doc.metadata)
                )
            self.conn.commit()

    def search(self, query_embedding, top_k=3):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT content, metadata, embedding <-> %s as distance
                FROM documents
                ORDER BY distance
                LIMIT %s
            """, (query_embedding, top_k))
            return cur.fetchall()

# 使用
vector_store = PostgresVectorStore(
    "postgresql://user:password@localhost/dbname"
)
```

### 4. 实现用户认证

创建 `utils/auth.py`:

```python
import streamlit as st
import hashlib
import json
from pathlib import Path

class UserAuth:
    def __init__(self, users_file="data/users.json"):
        self.users_file = Path(users_file)
        self.users = self._load_users()

    def _load_users(self):
        if self.users_file.exists():
            with open(self.users_file) as f:
                return json.load(f)
        return {}

    def _save_users(self):
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)

    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, username: str, password: str) -> bool:
        if username in self.users:
            return False
        self.users[username] = self.hash_password(password)
        self._save_users()
        return True

    def login(self, username: str, password: str) -> bool:
        if username not in self.users:
            return False
        return self.users[username] == self.hash_password(password)

    def require_login(self):
        """装饰器：要求登录"""
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False

        if not st.session_state.logged_in:
            st.title("🔐 登录")

            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("登录"):
                    if self.login(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")

            with col2:
                if st.button("注册"):
                    if self.register(username, password):
                        st.success("注册成功！请登录")
                    else:
                        st.error("用户名已存在")

            st.stop()

# 在 app.py 中使用
auth = UserAuth()
auth.require_login()

st.write(f"欢迎, {st.session_state.username}!")
```

### 5. 添加 API 接口

创建 `api.py`:

```python
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from services.vector_store import VectorStoreService
from models.agent import AgenticRAGAgent

app = FastAPI(title="Agentic RAG API")

# 初始化服务
vector_store = VectorStoreService()
agent = AgenticRAGAgent(model_name="qwen:7b")

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """查询接口"""
    try:
        # 检索相关文档
        docs = vector_store.search(request.question, top_k=request.top_k)

        # 生成回答
        answer = agent.generate_response(request.question)

        return QueryResponse(
            answer=answer,
            sources=docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """文档上传接口"""
    try:
        # 处理文档
        content = await file.read()
        # ... 文档处理逻辑

        return {"message": "文档上传成功", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

# 运行API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

运行API:
```bash
# 安装依赖
pip install fastapi uvicorn

# 启动API服务
python api.py

# 测试API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "你好"}'
```

---

## 💡 最佳实践

### 1. 文档准备

**文档质量检查清单:**

- [ ] 文本可选择（PDF非扫描版）
- [ ] 编码正确（UTF-8）
- [ ] 内容结构清晰（有标题、段落）
- [ ] 无大量重复内容
- [ ] 文件大小适中（< 50MB）

**文档预处理建议:**

```bash
# PDF OCR识别（扫描版PDF）
pip install pytesseract
tesseract input.pdf output -l chi_sim

# 文本清洗
# 移除多余空行、统一换行符
sed 's/\r$//' input.txt > output.txt

# 编码转换
iconv -f GB2312 -t UTF-8 input.txt > output.txt
```

### 2. 提示词优化

**好的问题示例:**

```markdown
✓ 文档中第三章的核心观点是什么？
✓ 总结作者提出的三个主要论点
✓ 对比文中提到的两种方法的优缺点
✓ 北京今天的天气如何？需要带伞吗？
✓ 计算一下如果温度从25°C降到15°C，下降了多少？
```

**不好的问题示例:**

```markdown
✗ 这是什么？（过于模糊）
✗ 告诉我一切（范围太大）
✗ 好不好？（需要更多上下文）
✗ 文档（不完整的问题）
```

**提示词技巧:**

```markdown
# 1. 明确指定范围
Q: 根据文档第5页的内容，...

# 2. 分步骤提问
Q: 首先，文档讲了几个主要概念？
Q: 然后，能详细解释第一个概念吗？

# 3. 要求特定格式
Q: 请用列表形式总结文档的要点

# 4. 结合多个信息源
Q: 根据文档中提到的城市，查询它们今天的天气
```

### 3. 性能优化建议

**向量检索优化:**

```python
# 1. 合理设置Top-K
简单问题: top_k = 1-3
复杂问题: top_k = 5-10

# 2. 使用MMR减少重复
search_type = "mmr"  # 当结果重复时

# 3. 添加相似度阈值
results = [r for r in results if r['score'] > 0.7]

# 4. 缓存热门查询
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query):
    return vector_store.search(query)
```

**模型推理优化:**

```python
# 1. 批量处理
queries = ["问题1", "问题2", "问题3"]
answers = agent.batch_generate(queries)

# 2. 流式输出
for chunk in agent.stream_generate(query):
    print(chunk, end="")

# 3. 降低温度提高速度
temperature = 0.3  # 更确定，更快

# 4. 限制最大token数
max_tokens = 512  # 对简单问题
```

### 4. 安全建议

**数据安全:**

```python
# 1. API密钥不要硬编码
import os
API_KEY = os.getenv("WEATHER_API_KEY")

# 2. 使用 .gitignore
echo "config/secrets.py" >> .gitignore
echo ".env" >> .gitignore
echo "data/" >> .gitignore

# 3. 文档访问控制
def check_document_permission(user, doc_id):
    # 检查用户权限
    pass

# 4. 输入验证
def sanitize_input(user_input):
    # 移除恶意内容
    return user_input.strip()
```

**代码安全:**

```python
# 1. 不要用eval执行用户输入
# 错误:
result = eval(user_input)

# 正确:
import ast
result = ast.literal_eval(user_input)

# 2. SQL注入防护（如果用数据库）
cursor.execute("SELECT * FROM docs WHERE id = %s", (doc_id,))

# 3. 文件路径验证
def safe_path(user_path, base_dir):
    full_path = Path(base_dir) / user_path
    if not str(full_path).startswith(str(base_dir)):
        raise ValueError("Invalid path")
    return full_path
```

### 5. 监控和维护

**健康检查:**

```python
# utils/health_check.py
def system_health():
    checks = {
        "ollama": check_ollama_connection(),
        "vector_store": check_vector_store(),
        "disk_space": check_disk_space(),
        "memory": check_memory_usage(),
    }
    return checks

def check_ollama_connection():
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

def check_disk_space():
    import shutil
    stat = shutil.disk_usage("/")
    free_gb = stat.free / (1024**3)
    return free_gb > 5  # 至少5GB空闲

# 定期运行
import schedule

schedule.every(5).minutes.do(system_health)
```

**性能监控:**

```python
# 记录关键指标
import time

metrics = {
    "query_count": 0,
    "avg_response_time": 0,
    "total_documents": 0,
    "cache_hit_rate": 0,
}

def log_metrics():
    logger.info(f"Metrics: {metrics}")

schedule.every(1).hours.do(log_metrics)
```

### 6. 备份策略

**自动备份脚本:**

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$DATE"

mkdir -p $BACKUP_DIR

# 备份向量存储
cp -r vector_store $BACKUP_DIR/

# 备份聊天记录
cp -r chat_history $BACKUP_DIR/

# 备份配置
cp config/settings.py $BACKUP_DIR/

# 压缩
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

# 只保留最近7天的备份
find backups/ -type f -name "*.tar.gz" -mtime +7 -delete

echo "备份完成: $BACKUP_DIR.tar.gz"
```

**定时备份:**

```bash
# 添加到crontab
crontab -e

# 每天凌晨2点备份
0 2 * * * /path/to/backup.sh
```

---

## 📝 总结

这份教程涵盖了 Agentic RAG 智能问答系统的：

✅ **基础使用**
- 环境配置和安装
- 快速开始指南
- 核心功能使用

✅ **进阶功能**
- 自定义工具开发
- 参数调优技巧
- 性能优化方案

✅ **生产部署**
- 故障排除方法
- 安全最佳实践
- 监控和维护

✅ **扩展开发**
- API接口开发
- 自定义嵌入模型
- 数据库集成

### 下一步建议

1. **初学者**:
   - 完成快速开始
   - 尝试上传文档和提问
   - 探索不同的参数设置

2. **进阶用户**:
   - 添加自定义工具
   - 优化向量检索
   - 实现API接口

3. **开发者**:
   - 研究源代码
   - 贡献新功能
   - 分享使用经验

### 获取帮助

- 📖 阅读源代码注释
- 💬 查看聊天历史示例
- 🐛 提交 Issue 报告问题
- 🤝 参与社区讨论

---

**祝使用愉快！**

如有问题，欢迎联系: fwytech@126.com
