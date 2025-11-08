# 项目重构说明 - 双模式 LLM 支持

## 🎯 重构目标

本次重构为 Agentic RAG 智能问答系统添加了**双模式 LLM 支持**，用户可以灵活选择：
1. **本地模式** - 使用 Ollama 运行本地大模型
2. **在线模式** - 使用阿里云百炼平台的在线 API

## ✨ 新增功能

### 1. 双 LLM 提供商支持

| 模式 | 提供商 | LLM 模型 | 嵌入模型 | 优势 |
|------|--------|---------|---------|------|
| **本地** | Ollama | qwen:7b, llama2:7b, mistral:7b 等 | nomic-embed-text | 数据隐私、无网络依赖 |
| **在线** | 阿里云百炼 | qwen-plus, qwen-turbo, qwen-max | text-embedding-v1 | 性能强大、无需本地资源 |

### 2. 统一的 LLM 客户端

新增 `services/llm_client.py`，提供统一接口：
- **UnifiedLLMClient** - 统一的 LLM 调用接口
- **UnifiedEmbeddingClient** - 统一的嵌入模型接口
- 自动根据配置选择本地或在线模式
- 兼容 LangChain 框架

### 3. 环境变量配置

通过 `.env` 文件轻松切换模式：

```bash
# 本地模式
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434

# 在线模式
LLM_PROVIDER=online
ONLINE_API_KEY=sk-abe3417c96f6441b83efed38708bcfb6
ONLINE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 4. UI 改进

- ✅ 侧边栏显示当前 LLM 提供商信息
- ✅ 根据提供商动态显示可用模型列表
- ✅ 显示服务地址和嵌入模型信息

## 📁 修改的文件

### 核心文件

1. **config/settings.py**
   - 新增 `LLM_PROVIDER` 配置
   - 分别配置 Ollama 和在线 API 参数
   - 新增 `get_available_models()` 等辅助方法

2. **services/llm_client.py** ⭐ 新文件
   - `UnifiedLLMClient` - LLM 统一客户端
   - `UnifiedEmbeddingClient` - Embedding 统一客户端

3. **models/agent.py**
   - 使用 `UnifiedLLMClient` 替代直接调用 Ollama
   - 支持本地和在线两种模式
   - 在日志中显示提供商信息

4. **services/vector_store.py**
   - 使用 `UnifiedEmbeddingClient` 替代直接调用 OllamaEmbeddings
   - 支持本地和在线两种嵌入模型

5. **app.py**
   - 侧边栏显示 LLM 提供商信息
   - 动态加载可用模型列表
   - 初始化时设置默认模型

### 配置文件

6. **requirements.txt**
   - 新增 `openai>=1.0.0` (兼容阿里云百炼)
   - 新增 `python-dotenv` (环境变量管理)
   - 优化依赖结构和注释

7. **.env.example** ⭐ 新文件
   - 完整的环境变量模板
   - 详细的配置说明
   - 使用指南

## 🚀 快速开始

### 方式一：使用本地 Ollama (默认)

```bash
# 1. 安装 Ollama
curl https://ollama.ai/install.sh | sh

# 2. 下载模型
ollama pull qwen:7b
ollama pull nomic-embed-text

# 3. 启动 Ollama 服务
ollama serve

# 4. 配置环境变量 (可选，默认即为本地模式)
cp .env.example .env
# LLM_PROVIDER=ollama  # 默认

# 5. 安装依赖
pip install -r requirements.txt

# 6. 启动应用
streamlit run app.py
```

### 方式二：使用在线 API (阿里云百炼)

```bash
# 1. 配置环境变量
cp .env.example .env

# 2. 编辑 .env 文件
nano .env

# 修改以下内容:
LLM_PROVIDER=online
ONLINE_API_KEY=你的阿里云百炼API密钥  # 或使用默认测试密钥

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动应用 (无需 Ollama)
streamlit run app.py
```

## 📊 对比：本地 vs 在线

| 特性 | Ollama 本地 | 阿里云百炼在线 |
|------|------------|--------------|
| **网络依赖** | ❌ 无需联网 | ✅ 需要联网 |
| **硬件要求** | 高 (8GB+ RAM) | 低 (仅需网络) |
| **模型性能** | 中等 (7B-13B) | 强大 (Qwen-Plus/Max) |
| **响应速度** | 取决于硬件 | 快速稳定 |
| **数据隐私** | ✅ 完全本地 | ⚠️ 需信任服务商 |
| **成本** | ✅ 免费 | 💰 按调用付费 |
| **适用场景** | 开发测试、隐私要求高 | 生产环境、性能要求高 |

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_PROVIDER` | LLM 提供商 (ollama/online) | `ollama` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `ONLINE_API_KEY` | 在线 API 密钥 | `sk-abe3417c96f6441b83efed38708bcfb6` |
| `ONLINE_BASE_URL` | 在线 API 地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `WEATHER_API_KEY` | 高德地图 API (可选) | - |
| `LOG_LEVEL` | 日志级别 | `INFO` |

### 可用模型

**Ollama 本地模型:**
- qwen:7b, qwen:14b, qwen:32b
- llama2:7b, llama2:13b, llama2:70b
- mistral:7b, codellama:7b, vicuna:7b

**阿里云百炼在线模型:**
- qwen-plus (推荐)
- qwen-turbo (快速)
- qwen-max (最强)
- qwen-max-longcontext (长文本)

## 🐛 故障排除

### 问题 1: 在线模式连接失败

**错误**: `API connection failed`

**解决**:
```bash
# 检查网络连接
ping dashscope.aliyuncs.com

# 检查 API Key 是否正确
echo $ONLINE_API_KEY

# 测试 API
curl -H "Authorization: Bearer $ONLINE_API_KEY" \
  https://dashscope.aliyuncs.com/compatible-mode/v1/models
```

### 问题 2: 本地模式 Ollama 未运行

**错误**: `Ollama service not available`

**解决**:
```bash
# 检查 Ollama 是否运行
ps aux | grep ollama

# 启动 Ollama
ollama serve

# 检查端口
lsof -i :11434
```

### 问题 3: 模型未找到

**本地模式**:
```bash
# 列出已安装模型
ollama list

# 下载缺失模型
ollama pull qwen:7b
```

**在线模式**:
- 检查 API Key 是否有效
- 确认模型名称正确 (qwen-plus, qwen-turbo, qwen-max)

## 📖 API 使用示例

### Python 代码示例

```python
from services.llm_client import UnifiedLLMClient, UnifiedEmbeddingClient

# 创建 LLM 客户端 (自动根据环境变量选择模式)
llm_client = UnifiedLLMClient(
    model_name="qwen-plus",  # 或 "qwen:7b"
    temperature=0.7,
    max_tokens=2048
)

# 调用 LLM
response = llm_client.invoke("你好，介绍一下人工智能")
print(response)

# 创建 Embedding 客户端
embedding_client = UnifiedEmbeddingClient()

# 嵌入查询
query_vector = embedding_client.embed_query("这是一个测试查询")
print(f"向量维度: {len(query_vector)}")  # 768

# 嵌入多个文档
docs = ["文档1", "文档2", "文档3"]
doc_vectors = embedding_client.embed_documents(docs)
print(f"文档数量: {len(doc_vectors)}")  # 3
```

## 🎓 技术细节

### 架构设计

```
┌─────────────────────────────────────┐
│          应用层 (app.py)             │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│     Agent 层 (models/agent.py)      │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  LLM 客户端 (llm_client.py)         │
│  ┌──────────────┐  ┌──────────────┐│
│  │UnifiedLLM    │  │UnifiedEmbed  ││
│  │Client        │  │dingClient    ││
│  └──────┬───────┘  └──────┬───────┘│
│         │                 │        │
│    ┌────▼─────┐    ┌──────▼────┐  │
│    │if ollama │    │if ollama  │  │
│    │  Ollama  │    │  Ollama   │  │
│    │else      │    │  Embed    │  │
│    │ ChatOpenAI   │else OpenAI│  │
│    └──────────┘    │ Embed     │  │
│                    └───────────┘  │
└───────────────────────────────────┘
```

### 兼容性说明

1. **OpenAI SDK 兼容**
   - 阿里云百炼使用 OpenAI 兼容接口
   - 通过 `openai_api_base` 和 `openai_api_key` 配置
   - 无需修改业务逻辑代码

2. **LangChain 集成**
   - UnifiedLLMClient 返回 LangChain 兼容的 LLM 对象
   - 可直接用于 Agent、Chain 等组件
   - 完全支持 ReAct 框架

3. **向量维度一致**
   - Ollama nomic-embed-text: 768 维
   - 阿里云 text-embedding-v1: 768 维
   - 两种模式的向量数据可互换使用

## 📝 更新日志

### v2.0.0 (2024-01-XX)

**新增功能**
- ✅ 支持 Ollama 本地模型和阿里云百炼在线 API 双模式
- ✅ 统一的 LLM 和 Embedding 客户端
- ✅ 环境变量配置支持
- ✅ UI 显示当前提供商信息

**优化改进**
- 🔧 重构 agent.py 使用统一客户端
- 🔧 重构 vector_store.py 支持多种嵌入模型
- 🔧 优化 requirements.txt 依赖管理
- 📝 完善文档和配置说明

**向后兼容**
- ✅ 保持原有 API 不变
- ✅ 默认使用 Ollama 本地模式
- ✅ 现有代码无需修改

## 🤝 贡献

如果您有建议或发现问题，欢迎：
- 提交 Issue
- 发起 Pull Request
- 联系维护者: fwytech@126.com

## 📄 许可证

本项目采用 MIT 许可证。

---

**祝使用愉快！** 🎉

如有问题，请查看 [TUTORIAL.md](TUTORIAL.md) 获取详细使用教程。
