"""
传统RAG实现示例 - 使用阿里云百炼平台和ChromaDB
Traditional RAG Implementation with Alibaba Qwen and ChromaDB

工作流程:
1. 用户查询 → 向量化
2. 向量相似度搜索 → 检索TOP-K文档
3. 拼接上下文
4. LLM生成答案

特点:
- 简单、直接、可预测
- 单一数据源（ChromaDB本地向量数据库）
- 一次性检索，无验证
- 响应快速
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import PyPDF2
from typing import List

# 加载环境变量
load_dotenv()

# 配置常量
API_KEY = os.getenv("API_KEY", "<your api key>")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")

CHROMA_COLLECTION_NAME = "traditional_rag_collection"


class TraditionalRAG:
    """传统RAG实现 - 使用阿里云百炼平台和ChromaDB"""

    def __init__(self):
        """初始化连接和配置"""
        print("🚀 初始化Traditional RAG系统...")

        # 初始化阿里云百炼平台客户端（兼容OpenAI接口）
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

        print(f"✅ 已连接到阿里云百炼平台")
        print(f"   模型: {MODEL_ID}")
        print(f"   嵌入模型: {EMBEDDING_MODEL}")

        # 初始化ChromaDB（本地持久化向量数据库）
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 获取或创建集合
        try:
            self.collection = self.chroma_client.get_collection(
                name=CHROMA_COLLECTION_NAME
            )
            print(f"📂 使用现有集合: {CHROMA_COLLECTION_NAME}")
            print(f"   当前文档数: {self.collection.count()}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"description": "Traditional RAG collection"}
            )
            print(f"✨ 创建新集合: {CHROMA_COLLECTION_NAME}")

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量

        参数:
            text: 要嵌入的文本

        返回:
            嵌入向量列表
        """
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 嵌入失败: {e}")
            # 返回一个默认向量（实际应用中应该处理错误）
            return [0.0] * 1536

    def load_pdf(self, pdf_path: str) -> List[str]:
        """加载PDF文档"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            texts = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    texts.append(text)
        return texts

    def ingest_documents(self, texts: List[str]):
        """
        摄入文档到ChromaDB向量数据库

        步骤：
        1. 为每个文档生成嵌入向量
        2. 存储到ChromaDB
        """
        print(f"\n📥 正在摄入 {len(texts)} 个文档到ChromaDB...")

        # 生成文档ID
        ids = [f"doc_{i}" for i in range(len(texts))]

        # 获取嵌入向量
        print("   🔄 正在生成嵌入向量...")
        embeddings = []
        for i, text in enumerate(texts):
            print(f"      处理文档 {i+1}/{len(texts)}")
            embedding = self.get_embedding(text)
            embeddings.append(embedding)

        # 存储到ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": f"document_{i}"} for i in range(len(texts))]
        )

        print(f"✅ 文档摄入完成!")
        print(f"   集合中共有 {self.collection.count()} 个文档")

    def search(self, query: str, k: int = 3) -> List[dict]:
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

        # 获取查询的嵌入向量
        query_embedding = self.get_embedding(query)

        # 在ChromaDB中搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # 格式化结果
        docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                docs.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })

        print(f"✅ 找到 {len(docs)} 个相关文档")
        return docs

    def format_context(self, docs: List[dict]) -> str:
        """格式化检索到的文档为上下文字符串"""
        context = "\n\n---\n\n".join([doc['content'] for doc in docs])
        return context

    def generate_answer(self, query: str, context: str) -> str:
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

        # 调用阿里云百炼平台LLM
        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
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

        except Exception as e:
            print(f"❌ LLM调用失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"

    def query(self, question: str, k: int = 3) -> tuple:
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
            (答案, 文档列表)
        """
        print(f"\n{'='*60}")
        print(f"📝 用户问题: {question}")
        print(f"{'='*60}")

        # 步骤1 & 2: 向量搜索
        docs = self.search(question, k=k)

        if not docs:
            return "抱歉，没有找到相关信息。", []

        # 步骤3: 格式化上下文
        context = self.format_context(docs)
        print(f"\n📄 上下文长度: {len(context)} 字符")

        # 步骤4: 生成答案
        answer = self.generate_answer(question, context)

        # 步骤5: 返回
        return answer, docs

    def reset_collection(self):
        """重置集合（清空所有文档）"""
        try:
            self.chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"description": "Traditional RAG collection"}
            )
            print("✅ 集合已重置")
        except Exception as e:
            print(f"❌ 重置失败: {e}")


# ============================================
# 使用示例
# ============================================

def main():
    """主函数 - 演示传统RAG的使用"""

    print("="*60)
    print("🎯 Traditional RAG 演示")
    print("   使用阿里云百炼平台 (Qwen) + ChromaDB")
    print("="*60)

    # 初始化传统RAG
    rag = TraditionalRAG()

    # 示例文档（实际使用时可以从PDF或其他源加载）
    sample_documents = [
        """
        人工智能(AI)是计算机科学的一个分支,致力于创建能够执行通常需要人类智能的任务的系统。
        这些任务包括视觉感知、语音识别、决策制定和语言翻译。AI技术在近年来取得了显著进展，
        特别是在深度学习和神经网络领域。
        """,
        """
        机器学习是AI的一个子集,专注于开发能够从数据中学习和改进的算法,而无需明确编程。
        深度学习是机器学习的一个子领域,使用类似于人脑的神经网络。常见的机器学习方法包括
        监督学习、无监督学习和强化学习。
        """,
        """
        自然语言处理(NLP)是AI的一个领域,专注于计算机与人类语言之间的交互。
        NLP技术使计算机能够理解、解释和生成人类语言。主要应用包括机器翻译、情感分析、
        文本摘要和问答系统。
        """,
        """
        检索增强生成(RAG)是一种结合信息检索和文本生成的技术。
        它通过从外部知识库检索相关信息来增强语言模型的能力,从而减少幻觉并提高答案的准确性。
        RAG系统通常包括向量数据库、嵌入模型和大型语言模型三个核心组件。
        """,
        """
        向量数据库是专门用于存储和检索向量嵌入的数据库系统。它们支持高效的相似度搜索，
        这对于RAG系统至关重要。流行的向量数据库包括ChromaDB、Pinecone、Weaviate和Milvus。
        ChromaDB是一个轻量级的本地向量数据库，非常适合开发和小规模应用。
        """
    ]

    # 摄入文档
    rag.ingest_documents(sample_documents)

    # 测试查询
    test_questions = [
        "什么是机器学习?",
        "RAG是如何工作的?",
        "ChromaDB有什么特点?"
    ]

    for question in test_questions:
        answer, docs = rag.query(question)

        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"{'='*60}")
        print(f"💡 答案:\n{answer}")
        print(f"\n📚 使用的文档片段:")
        for i, doc in enumerate(docs, 1):
            print(f"\n  [{i}] (距离: {doc.get('distance', 'N/A'):.4f})")
            print(f"  {doc['content'][:150]}...")
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
✅ 本地化部署(使用ChromaDB)

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
- 不需要实时外部信息的应用

技术栈:
- LLM: 阿里云百炼平台 Qwen-Plus
- 嵌入: 阿里云 text-embedding-v1
- 向量数据库: ChromaDB (本地持久化)
- 兼容: OpenAI API格式
"""
