"""
传统RAG引擎
Traditional RAG Engine

用于处理简单的事实性查询
"""

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from typing import List, Tuple
import config


class TraditionalRAGEngine:
    """传统RAG引擎 - 一次性检索+生成"""

    def __init__(self):
        """初始化传统RAG引擎"""
        print("🚀 初始化Traditional RAG引擎...")

        # 初始化OpenAI客户端（阿里云百炼）
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL
        )

        # 初始化ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 获取或创建集合
        try:
            self.collection = self.chroma_client.get_collection(
                name=config.COLLECTION_NAME
            )
            print(f"✅ 加载现有集合: {config.COLLECTION_NAME}")
            print(f"   文档数: {self.collection.count()}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=config.COLLECTION_NAME
            )
            print(f"✨ 创建新集合: {config.COLLECTION_NAME}")

    def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 嵌入失败: {e}")
            return [0.0] * 1536

    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """
        添加文档到向量数据库

        Args:
            texts: 文本列表
            metadatas: 元数据列表
        """
        print(f"\n📥 正在添加 {len(texts)} 个文档...")

        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]

        # 生成嵌入
        embeddings = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"   处理进度: {i+1}/{len(texts)}")
            embedding = self.get_embedding(text)
            embeddings.append(embedding)

        # 添加到ChromaDB
        ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"✅ 文档添加完成！")

    def retrieve(self, query: str, k: int = config.TOP_K) -> List[dict]:
        """
        检索相关文档

        Args:
            query: 查询文本
            k: 返回文档数量

        Returns:
            文档列表
        """
        print(f"\n🔍 检索: '{query}'")

        # 获取查询嵌入
        query_embedding = self.get_embedding(query)

        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # 格式化结果
        docs = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0] if results['metadatas'] else [{}] * k,
                results['distances'][0] if results['distances'] else [0] * k
            )):
                docs.append({
                    'content': doc,
                    'metadata': metadata,
                    'distance': distance
                })

        print(f"✅ 找到 {len(docs)} 个相关文档")
        return docs

    def generate_answer(self, query: str, docs: List[dict]) -> str:
        """
        生成答案

        Args:
            query: 用户问题
            docs: 检索到的文档

        Returns:
            生成的答案
        """
        print(f"\n🤖 生成答案...")

        # 构建上下文
        context = "\n\n".join([
            f"参考资料{i+1}:\n{doc['content']}"
            for i, doc in enumerate(docs)
        ])

        # 构建提示词
        prompt = f"""你是一个专业的问答助手。请根据以下参考资料回答用户的问题。

参考资料:
{context}

用户问题: {query}

要求:
1. 仅使用参考资料中的信息回答
2. 回答要准确、简洁、相关
3. 如果参考资料不足以回答问题，请如实说明

回答:"""

        # 调用LLM
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[
                    {"role": "system", "content": "你是一个专业的问答助手，擅长从参考资料中提取信息回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE_GENERATION,
                max_tokens=config.MAX_TOKENS_GENERATION
            )

            answer = response.choices[0].message.content
            print("✅ 答案生成完成")
            return answer

        except Exception as e:
            print(f"❌ 答案生成失败: {e}")
            return f"抱歉，生成答案时出错: {str(e)}"

    def query(self, question: str) -> Tuple[str, List[dict]]:
        """
        完整的传统RAG流程

        Args:
            question: 用户问题

        Returns:
            (答案, 检索到的文档)
        """
        print(f"\n{'='*60}")
        print(f"📝 [传统RAG] 问题: {question}")
        print(f"{'='*60}")

        # 步骤1: 检索
        docs = self.retrieve(question)

        # 步骤2: 生成答案
        answer = self.generate_answer(question, docs)

        return answer, docs


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    # 初始化引擎
    rag = TraditionalRAGEngine()

    # 示例文档
    sample_docs = [
        "孙悟空原本是花果山上的一块仙石孕育而生的石猴，后来拜菩提祖师为师，学得七十二变和筋斗云等神通。",
        "唐僧俗姓陈，法号玄奘，是如来佛祖的二弟子金蝉子转世。他奉唐太宗之命前往西天取经。",
        "猪八戒原是天蓬元帅，因调戏嫦娥被贬下凡，错投猪胎。后被观音菩萨点化，等待取经人。",
        "沙悟净原是天庭的卷帘大将，因失手打碎琉璃盏被贬下界，在流沙河为妖。后被观音点化，保护唐僧取经。",
        "孙悟空的两个师傅分别是菩提祖师和唐僧。菩提祖师教他法术神通，唐僧则是他取经路上的师父。"
    ]

    # 添加文档
    rag.add_documents(sample_docs)

    # 测试查询
    test_questions = [
        "孙悟空的师傅是谁？",
        "唐僧是谁转世？",
        "猪八戒为什么被贬下凡？"
    ]

    for question in test_questions:
        answer, docs = rag.query(question)

        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"💡 答案: {answer}")
        print(f"\n📚 参考文档:")
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] (距离: {doc['distance']:.4f})")
            print(f"      {doc['content'][:80]}...")
        print(f"{'='*60}\n")
