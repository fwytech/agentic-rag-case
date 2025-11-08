# Agentic RAG 实战教程

## 📚 教程简介

这是一套完整的 Agentic RAG 技术学习教程，从理论到实践，从简单到复杂，帮助你全面掌握如何构建生产级智能问答系统。

**教程特色：**
- 🎯 **理论与实践结合**：2讲理论 + 16讲实战
- 💻 **三个完整项目**：从对比实现到生产应用
- 📖 **详细代码解释**：每一行代码都有注释说明
- 🚀 **可直接运行**：所有代码都经过测试验证

## 🗂️ 项目结构

```
study-agentic-rag/
├── docs/                               # 教程文档（18讲）
│   ├── README.md                       # 教程导航
│   ├── part1-Agentic-RAG-理论基础篇/
│   │   ├── 01-RAG技术演进-从检索增强到智能体驱动.md
│   │   └── 02-ReAct框架解析-核心技术与工具编排.md
│   ├── part2-Agentic-RAG-基础实践篇-RAG原理对比实现/
│   ├── part3-Agentic-RAG-进阶实践篇-混合RAG系统/
│   ├── part4-Agentic-RAG-生产实践篇-智能问答Web应用/
│   └── part5-Agentic-RAG-部署与优化篇/
│
├── 01-rag-comparison/                  # 项目1：RAG原理对比实现
│   ├── traditional_rag.py
│   ├── agentic_rag.py
│   └── README.md
│
├── 02-hybrid-rag-system/               # 项目2：混合RAG系统
│   ├── router.py
│   ├── traditional_rag_engine.py
│   ├── agentic_rag_engine.py
│   ├── hybrid_rag.py
│   └── README.md
│
├── 03-smart-qa-application/            # 项目3：智能问答Web应用
│   ├── config/
│   ├── models/
│   ├── services/
│   ├── utils/
│   ├── app.py
│   └── README.md
│
└── README.md                           # 本文档
```

## 📖 课程大纲

### 第一部分：理论基础（2讲）

建立扎实的理论基础，理解 RAG 技术演进和核心原理

| 讲次 | 标题 | 时长 |
|------|------|------|
| 第1讲 | [RAG技术演进 - 从检索增强到智能体驱动](./docs/part1-Agentic-RAG-理论基础篇/01-RAG技术演进-从检索增强到智能体驱动.md) | 1小时 |
| 第2讲 | [ReAct框架解析 - 核心技术与工具编排](./docs/part1-Agentic-RAG-理论基础篇/02-ReAct框架解析-核心技术与工具编排.md) | 1小时 |

### 第二部分：基础实践（3讲）

通过实践理解传统RAG和Agentic RAG的差异

| 讲次 | 标题 | 时长 |
|------|------|------|
| 第3讲 | 从零搭建传统RAG - 文档检索与答案生成 | 2小时 |
| 第4讲 | 构建Agentic RAG引擎 - ReAct循环与多工具集成 | 2.5小时 |
| 第5讲 | RAG技术对比 - 准确性成本与性能分析 | 1.5小时 |

**配套项目：** [01-rag-comparison](./01-rag-comparison/)

### 第三部分：进阶实践（5讲）

构建成本优化的混合RAG系统

| 讲次 | 标题 | 时长 |
|------|------|------|
| 第6讲 | 智能路由器设计 - 自动识别查询复杂度 | 2小时 |
| 第7讲 | 传统RAG引擎封装 - 快速响应的检索系统 | 1.5小时 |
| 第8讲 | Agentic RAG引擎实现 - 智能推理的检索系统 | 2.5小时 |
| 第9讲 | 数据处理流水线 - 文档分块与语义优化 | 1.5小时 |
| 第10讲 | 混合RAG主控系统 - 整合路由与双引擎 | 2小时 |

**配套项目：** [02-hybrid-rag-system](./02-hybrid-rag-system/)

### 第四部分：生产实践（7讲）

开发生产级智能问答系统

| 讲次 | 标题 | 时长 |
|------|------|------|
| 第11讲 | 项目架构设计 - 分层架构与模块划分 | 1.5小时 |
| 第12讲 | 配置管理系统 - 双模式LLM支持 | 1.5小时 |
| 第13讲 | 统一LLM客户端 - 兼容本地与在线模型 | 2小时 |
| 第14讲 | 向量存储服务 - FAISS高效检索实现 | 2小时 |
| 第15讲 | Agent核心引擎 - ReAct框架与LangChain集成 | 2.5小时 |
| 第16讲 | 文档处理与工具集成 - 多格式支持与API调用 | 2小时 |
| 第17讲 | Streamlit Web界面 - 打造用户友好的问答系统 | 2.5小时 |

**配套项目：** [03-smart-qa-application](./03-smart-qa-application/)

### 第五部分：部署优化（1讲）

掌握部署、优化和故障排查

| 讲次 | 标题 | 时长 |
|------|------|------|
| 第18讲 | 项目部署与性能优化 - 从本地到生产环境 | 2小时 |

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/fwytech/agentic-rag-case.git
cd agentic-rag-case/study-agentic-rag
```

### 2. 选择学习路径

#### 路径A：完整学习（推荐新手）

```
第1-2讲（理论）→ 第3-5讲（基础）→ 第6-10讲（进阶）→ 第11-17讲（生产）→ 第18讲（部署）
```

#### 路径B：快速上手（有基础）

```
第1-2讲（浏览）→ 第3-5讲（实践）→ 第11-17讲（生产项目）
```

#### 路径C：生产应用（工程师）

```
第1-2讲（概念）→ 第11-18讲（生产+部署）
```

### 3. 开始学习

**[📖 点击进入教程导航](./docs/README.md)**

---

## 💡 学习建议

### 1. 理论先行

务必认真学习第1-2讲的理论基础，理解：
- 为什么需要 RAG
- ReAct 框架的工作原理
- 传统 RAG 和 Agentic RAG 的本质区别

### 2. 动手实践

每讲的代码都要亲自敲一遍：
- ✅ 不要复制粘贴
- ✅ 理解每一行代码的作用
- ✅ 尝试修改参数看效果
- ✅ 调试并解决问题

### 3. 对比理解

重点理解不同方案的差异：
- 传统 RAG vs Agentic RAG
- 简单查询 vs 复杂查询
- 本地模型 vs 在线模型

### 4. 循序渐进

不要跳跃学习：
- 基础不牢，地动山摇
- 每个概念都要理解透彻
- 遇到问题及时复习

### 5. 举一反三

学以致用：
- 思考如何应用到自己的项目
- 尝试添加新的工具
- 优化系统性能

---

## 🎯 学习成果

完成本教程后，你将能够：

✅ 深入理解 RAG 技术原理和演进过程
✅ 掌握 ReAct 框架的实现方法
✅ 从零构建传统 RAG 系统
✅ 实现完整的 Agentic RAG 系统
✅ 开发生产级智能问答应用
✅ 掌握性能优化和成本控制
✅ 具备独立开发和部署能力

---

## 📦 三个项目概览

| 项目 | 复杂度 | 代码量 | 适用场景 | 亮点 |
|------|-------|--------|---------|------|
| [RAG原理对比](./01-rag-comparison/) | ⭐ | 400行 | 学习研究 | 理解RAG本质差异 |
| [混合RAG系统](./02-hybrid-rag-system/) | ⭐⭐ | 1500行 | 成本优化 | 智能路由+双引擎 |
| [智能问答应用](./03-smart-qa-application/) | ⭐⭐⭐ | 3000行 | 生产部署 | 完整Web应用 |

---

## 🛠️ 技术栈

- **Python**：3.8+
- **LLM**：
  - 本地：Ollama（qwen、llama等）
  - 在线：阿里云百炼（qwen-plus）
- **向量数据库**：ChromaDB、FAISS
- **框架**：LangChain、Streamlit
- **工具**：DuckDuckGo、自定义API

---

## 📚 参考资料

### 官方文档

- [LangChain 文档](https://python.langchain.com/)
- [Streamlit 文档](https://docs.streamlit.io/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [阿里云百炼](https://bailian.console.aliyun.com/)

### 相关论文

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 如何贡献

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 开启 Pull Request

### 贡献方向

- 🐛 修复教程中的错误
- 📝 改进文档说明
- 💡 添加新的示例
- 🔧 优化代码实现
- 🌐 翻译成其他语言

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](../LICENSE) 文件

---

## 🙏 致谢

感谢所有为 Agentic RAG 技术发展做出贡献的研究者和开发者！

---

## 📞 联系方式

- GitHub Issues: [提交问题](https://github.com/fwytech/agentic-rag-case/issues)
- Email: fwytech@126.com

---

**准备好开始学习了吗？**

**[👉 点击进入教程](./docs/README.md)**

祝学习愉快！🎉
