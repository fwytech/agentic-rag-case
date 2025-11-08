"""
混合RAG系统
Hybrid RAG System

自动选择传统RAG或Agentic RAG，提供最佳问答体验
"""

from traditional_rag_engine import TraditionalRAGEngine
from agentic_rag_engine import AgenticRAGEngine
from router import QueryRouter
from typing import Tuple, Dict
import time


class HybridRAG:
    """
    混合RAG系统

    根据查询复杂度自动选择:
    - 简单查询 → 传统RAG (快速、经济)
    - 复杂查询 → Agentic RAG (智能、全面)
    """

    def __init__(self):
        """初始化混合RAG系统"""
        print("\n" + "="*60)
        print("🚀 初始化混合RAG系统")
        print("="*60)

        # 初始化路由器
        print("\n1️⃣ 初始化智能路由器...")
        self.router = QueryRouter()

        # 初始化传统RAG引擎
        print("\n2️⃣ 初始化传统RAG引擎...")
        self.traditional_rag = TraditionalRAGEngine()

        # 初始化Agentic RAG引擎
        print("\n3️⃣ 初始化Agentic RAG引擎...")
        self.agentic_rag = AgenticRAGEngine()

        print("\n" + "="*60)
        print("✅ 混合RAG系统初始化完成！")
        print("="*60)

        # 统计信息
        self.stats = {
            "total_queries": 0,
            "traditional_count": 0,
            "agentic_count": 0,
            "traditional_time": 0,
            "agentic_time": 0
        }

    def add_documents(self, texts: list, metadatas: list = None):
        """
        添加文档到知识库

        Args:
            texts: 文本列表
            metadatas: 元数据列表
        """
        print("\n📚 添加文档到知识库...")
        self.traditional_rag.add_documents(texts, metadatas)
        print("✅ 文档添加完成！")

    def query(self, question: str, force_strategy: str = None) -> Dict:
        """
        查询混合RAG系统

        Args:
            question: 用户问题
            force_strategy: 强制使用策略 ("traditional" 或 "agentic")，None为自动路由

        Returns:
            {
                "question": str,
                "answer": str,
                "strategy": str,
                "complexity": str,
                "response_time": float,
                "metadata": dict
            }
        """
        start_time = time.time()

        print("\n" + "="*60)
        print(f"📝 [混合RAG] 用户问题: {question}")
        print("="*60)

        # 路由决策
        if force_strategy:
            route_result = {
                "strategy": force_strategy,
                "complexity": "unknown",
                "reasoning": f"强制使用{force_strategy} RAG"
            }
            print(f"\n🎯 [强制模式] 使用 {force_strategy.upper()} RAG")
        else:
            route_result = self.router.route(question)

        strategy = route_result["strategy"]
        complexity = route_result["complexity"]

        # 执行查询
        if strategy == "traditional":
            answer, docs = self.traditional_rag.query(question)
            metadata = {"retrieved_docs": docs}
            self.stats["traditional_count"] += 1
        else:  # agentic
            answer = self.agentic_rag.query(question)
            metadata = {}
            self.stats["agentic_count"] += 1

        # 计算响应时间
        response_time = time.time() - start_time

        # 更新统计
        self.stats["total_queries"] += 1
        if strategy == "traditional":
            self.stats["traditional_time"] += response_time
        else:
            self.stats["agentic_time"] += response_time

        # 构建结果
        result = {
            "question": question,
            "answer": answer,
            "strategy": strategy,
            "complexity": complexity,
            "response_time": response_time,
            "metadata": metadata
        }

        # 打印结果
        print("\n" + "="*60)
        print(f"✅ [回答完成]")
        print(f"   策略: {strategy.upper()} RAG")
        print(f"   复杂度: {complexity}")
        print(f"   响应时间: {response_time:.2f}秒")
        print(f"="*60)
        print(f"\n💡 答案:\n{answer}")
        print("="*60)

        return result

    def get_stats(self) -> Dict:
        """
        获取系统统计信息

        Returns:
            统计信息字典
        """
        stats = self.stats.copy()

        if stats["traditional_count"] > 0:
            stats["avg_traditional_time"] = stats["traditional_time"] / stats["traditional_count"]
        else:
            stats["avg_traditional_time"] = 0

        if stats["agentic_count"] > 0:
            stats["avg_agentic_time"] = stats["agentic_time"] / stats["agentic_count"]
        else:
            stats["avg_agentic_time"] = 0

        return stats

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("📊 系统统计信息")
        print("="*60)
        print(f"总查询数: {stats['total_queries']}")
        print(f"\n传统RAG:")
        print(f"  - 使用次数: {stats['traditional_count']}")
        print(f"  - 平均响应时间: {stats['avg_traditional_time']:.2f}秒")
        print(f"\nAgentic RAG:")
        print(f"  - 使用次数: {stats['agentic_count']}")
        print(f"  - 平均响应时间: {stats['avg_agentic_time']:.2f}秒")
        print("="*60)


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    # 初始化混合RAG系统
    hybrid_rag = HybridRAG()

    # 添加示例文档
    sample_docs = [
        "《西游记》是中国古代第一部浪漫主义章回体长篇神魔小说。作者是明代吴承恩。",
        "孙悟空原本是花果山上的一块仙石孕育而生的石猴，后来拜菩提祖师为师，学得七十二变和筋斗云等神通。",
        "唐僧俗姓陈，法号玄奘，是如来佛祖的二弟子金蝉子转世。他奉唐太宗之命前往西天取经。",
        "猪八戒原是天蓬元帅，因调戏嫦娥被贬下凡，错投猪胎。后被观音菩萨点化，等待取经人。",
        "沙悟净原是天庭的卷帘大将，因失手打碎琉璃盏被贬下界，在流沙河为妖。",
        "孙悟空有两个师傅：菩提祖师和唐僧。菩提祖师教他法术神通，唐僧是他取经路上的师父。",
        "师徒四人历经九九八十一难，最终到达西天，取得真经。孙悟空被封为斗战胜佛。",
    ]

    hybrid_rag.add_documents(sample_docs)

    # 测试查询
    test_queries = [
        # 简单查询 (应该使用传统RAG)
        "西游记的作者是谁？",
        "孙悟空的师傅是谁？",
        "唐僧的法号是什么？",

        # 复杂查询 (应该使用Agentic RAG)
        "孙悟空有几个师傅？他们分别教了他什么？",
        "分析师徒四人的来历和背景",
        "为什么说《西游记》是浪漫主义小说？",
    ]

    print("\n" + "="*60)
    print("🎯 开始测试混合RAG系统")
    print("="*60)

    results = []
    for query in test_queries:
        result = hybrid_rag.query(query)
        results.append(result)
        print("\n" + "-"*60 + "\n")
        time.sleep(1)  # 避免API限流

    # 打印统计信息
    hybrid_rag.print_stats()

    # 打印汇总
    print("\n" + "="*60)
    print("📋 查询结果汇总")
    print("="*60)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['question']}")
        print(f"   策略: {result['strategy'].upper()}")
        print(f"   复杂度: {result['complexity']}")
        print(f"   响应时间: {result['response_time']:.2f}秒")
        print(f"   答案: {result['answer'][:100]}...")

    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)
