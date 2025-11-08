# Agentic RAG 智能问答系统

## 项目简介

这是一个基于 Agentic RAG（智能体驱动的检索增强生成）技术的智能问答系统项目。项目结合了传统 RAG 技术与 AI 智能体的自主决策能力，实现了更智能、更准确的问答体验。

## 核心特性

- **Agentic RAG 架构**: 引入自主 AI 智能体，实现动态检索策略和上下文优化
- **多智能体协作**: 支持多个智能体协同工作，提高问答准确性
- **实时信息检索**: 结合外部知识库，提供最新、最相关的信息
- **智能上下文理解**: 通过迭代优化机制，提升回答质量
- **用户友好界面**: 基于 Streamlit 的现代化 Web 界面

## 项目结构

```
agentic_rag_smart_qa_project/
├── app.py                    # 主应用文件
├── requirements.txt          # 项目依赖
├── config/
│   └── settings.py          # 配置文件
├── models/
│   └── agent.py             # 智能体模型定义
├── services/
│   ├── vector_store.py      # 向量存储服务
│   └── weather_tools.py     # 天气工具服务
└── utils/
    ├── document_processor.py # 文档处理工具
    ├── ui_components.py      # UI组件
    ├── chat_history.py       # 聊天记录管理
    └── decorators.py         # 装饰器工具

Agentic RAG 项目理论部分.md  # 项目理论文档
```

## 技术栈

- **后端**: Python, Streamlit
- **AI/ML**: LangChain, OpenAI API, 向量数据库
- **数据处理**: 文档解析、向量化处理
- **工具集成**: 天气查询、网络搜索等外部工具

## 主要功能

1. **智能问答**: 基于 Agentic RAG 的智能问答系统
2. **文档处理**: 支持多种文档格式的解析和处理
3. **向量检索**: 高效的向量相似度搜索
4. **聊天历史**: 完整的对话记录和管理功能
5. **工具调用**: 集成外部工具扩展功能

## 快速开始

1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量:
   ```bash
   # 设置 OpenAI API 密钥
   export OPENAI_API_KEY="your-api-key"
   ```

3. 运行应用:
   ```bash
   streamlit run app.py
   ```

## 理论文档

详细的项目理论说明请参考: [Agentic RAG 项目理论部分.md](agentic_rag_smart_qa_project/Agentic%20RAG%20%E9%A1%B9%E7%9B%AE%E7%90%86%E8%AE%BA%E9%83%A8%E5%88%86.md)

## 核心概念

### Agentic RAG vs 传统 RAG

Agentic RAG 通过引入智能体概念，解决了传统 RAG 的以下局限：

- **静态检索策略**: 传统 RAG 使用固定的检索方法
- **缺乏上下文优化**: 无法根据对话历史调整检索策略
- **单一工具使用**: 无法灵活选择和组合多个工具
- **缺乏自我反思**: 无法评估和改进回答质量

### 智能体设计模式

1. **反思模式**: 自我评估和优化输出
2. **规划模式**: 任务分解和动态调整
3. **工具使用**: 外部工具集成和调用
4. **多智能体协作**: 智能体间协同工作

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过 GitHub Issue 联系我们。