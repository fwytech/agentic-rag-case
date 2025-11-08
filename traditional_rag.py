"""
传统RAG实现示例
Traditional RAG Implementation

工作流程：
1. 用户查询 → 向量化
2. 向量相似度搜索 → 检索TOP-K文档
3. 拼接上下文
4. LLM生成答案

特点：
- 简单、直接、可预测
- 单一数据源（向量数据库）
- 一次性检索，无验证
- 响应快速
"""

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from openai import AzureOpenAI
import PyPDF2

# 加载环境变量
load_dotenv()

# 配置常量
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_ENDPOINT = os.getenv("ES_ENDPOINT", "localhost")
MODEL_NAME = os.getenv("MODEL_NAME", "text-embedding-ada-002")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_DEPLOYMENT_ID = os.getenv("AZURE_DEPLOYMENT_ID")

ELASTIC_INDEX_NAME = "traditional_rag_index"


class TraditionalRAG:
    """传统RAG实现"""

    def __init__(self):
        """初始化连接和配置"""
        # 初始化Elasticsearch连接
        self.es_url = f"https://{ES_USER}:{ES_PASSWORD}@{ES_ENDPOINT}:9200"
        self.es = Elasticsearch(
            self.es_url,
            ca_certs="./http_ca.crt",
            verify_certs=True
        )

        # 初始化Azure OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model=MODEL_NAME,
            azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
            api_key=AZURE_EMBEDDING_API_KEY,
            openai_api_version=AZURE_EMBEDDING_API_VERSION
        )

        # 初始化Azure OpenAI Chat
        self.chat_client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT
        )

        self.docsearch = None

    def load_pdf(self, pdf_path):
        """加载PDF文档"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            texts = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    texts.append(text)
        return texts

    def ingest_documents(self, texts):
        """
        摄入文档到Elasticsearch向量数据库

        步骤：
        1. 检查索引是否存在
        2. 如果不存在，创建新索引并添加文档
        3. 如果存在，使用现有索引
        """
        print(f"📥 正在摄入 {len(texts)} 个文档到Elasticsearch...")

        if not self.es.indices.exists(index=ELASTIC_INDEX_NAME):
            print(f"✨ 创建新索引: {ELASTIC_INDEX_NAME}")
            self.docsearch = ElasticsearchStore.from_texts(
                texts,
                embedding=self.embeddings,
                es_url=self.es_url,
                es_connection=self.es,
                index_name=ELASTIC_INDEX_NAME,
                es_user=ES_USER,
                es_password=ES_PASSWORD
            )
        else:
            print(f"📂 使用现有索引: {ELASTIC_INDEX_NAME}")
            self.docsearch = ElasticsearchStore(
                es_connection=self.es,
                embedding=self.embeddings,
                es_url=self.es_url,
                index_name=ELASTIC_INDEX_NAME,
                es_user=ES_USER,
                es_password=ES_PASSWORD
            )

        print("✅ 文档摄入完成!")

    def search(self, query, k=3):
        """
        执行向量相似度搜索

        参数:
            query: 用户查询
            k: 返回TOP-K个最相似的文档

        返回:
            相关文档列表
        """
        print(f"\n🔍 执行向量搜索: '{query}'")
        print(f"   检索TOP-{k}个相关文档...")

        docs = self.docsearch.similarity_search(query, k=k)

        print(f"✅ 找到 {len(docs)} 个相关文档")
        return docs

    def format_context(self, docs):
        """格式化检索到的文档为上下文字符串"""
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return context

    def generate_answer(self, query, context):
        """
        使用LLM生成答案

        参数:
            query: 用户问题
            context: 检索到的上下文

        返回:
            LLM生成的答案
        """
        print(f"\n🤖 正在生成答案...")

        # 构建提示词
        prompt = f"""你是一个有用的AI助手。请根据以下上下文回答用户的问题。

上下文:
{context}

问题: {query}

请提供准确、详细的答案。如果上下文中没有足够的信息，请诚实地说明。"""

        # 调用LLM
        response = self.chat_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_ID,
            messages=[
                {"role": "system", "content": "你是一个专业的AI助手，擅长根据提供的上下文回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        print("✅ 答案生成完成!")
        return answer

    def query(self, question, k=3):
        """
        传统RAG完整流程

        工作流程:
        1. 向量化用户查询
        2. 相似度搜索检索文档
        3. 格式化上下文
        4. LLM生成答案
        5. 返回答案

        参数:
            question: 用户问题
            k: 检索文档数量

        返回:
            LLM生成的答案
        """
        print(f"\n{'='*60}")
        print(f"📝 用户问题: {question}")
        print(f"{'='*60}")

        # 步骤1 & 2: 向量搜索
        docs = self.search(question, k=k)

        # 步骤3: 格式化上下文
        context = self.format_context(docs)
        print(f"\n📄 上下文长度: {len(context)} 字符")

        # 步骤4: 生成答案
        answer = self.generate_answer(question, context)

        # 步骤5: 返回
        return answer, docs


# ============================================
# 使用示例
# ============================================

def main():
    """主函数 - 演示传统RAG的使用"""

    # 初始化传统RAG
    rag = TraditionalRAG()

    # 示例文档（实际使用时可以从PDF或其他源加载）
    sample_documents = [
        """
        人工智能(AI)是计算机科学的一个分支,致力于创建能够执行通常需要人类智能的任务的系统。
        这些任务包括视觉感知、语音识别、决策制定和语言翻译。
        """,
        """
        机器学习是AI的一个子集,专注于开发能够从数据中学习和改进的算法,而无需明确编程。
        深度学习是机器学习的一个子领域,使用类似于人脑的神经网络。
        """,
        """
        自然语言处理(NLP)是AI的一个领域,专注于计算机与人类语言之间的交互。
        NLP技术使计算机能够理解、解释和生成人类语言。
        """,
        """
        检索增强生成(RAG)是一种结合信息检索和文本生成的技术。
        它通过从外部知识库检索相关信息来增强语言模型的能力,从而减少幻觉并提高答案的准确性。
        """
    ]

    # 摄入文档
    rag.ingest_documents(sample_documents)

    # 测试查询
    test_questions = [
        "什么是机器学习?",
        "RAG是如何工作的?",
        "NLP的应用有哪些?"
    ]

    for question in test_questions:
        answer, docs = rag.query(question)

        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"💡 答案: {answer}")
        print(f"\n📚 使用的文档片段:")
        for i, doc in enumerate(docs, 1):
            print(f"\n  [{i}] {doc.page_content[:100]}...")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


"""
传统RAG的优点:
✅ 实现简单,易于理解和维护
✅ 响应速度快(单次检索 + 单次LLM调用)
✅ 成本低(LLM调用次数少)
✅ 行为可预测,易于调试
✅ 适合高并发场景

传统RAG的局限:
❌ 单一数据源,知识覆盖有限
❌ 无法获取实时信息
❌ 一次性检索,无法根据结果调整
❌ 不验证检索质量
❌ 难以处理复杂的多步推理查询
❌ 无法使用外部工具(计算器、API等)

适用场景:
- 企业内部文档查询
- 产品手册/技术文档问答
- 简单的FAQ系统
- 客服机器人(快速响应)
- 成本敏感的应用
- 高并发场景
"""
