"""
混合RAG系统完整演示
Complete Hybrid RAG Demo

展示传统RAG、Agentic RAG和智能路由的功能
"""

import time
from hybrid_rag import HybridRAG
from data_processor import DataPipeline


def print_header(title):
    """打印标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_section(title):
    """打印小节"""
    print("\n" + "-"*70)
    print(f"  {title}")
    print("-"*70)


def demo_data_processing():
    """演示数据处理"""
    print_header("📚 演示1: 数据处理流程")

    pipeline = DataPipeline()

    # 处理示例文档
    print("\n正在处理示例文档...")
    chunks = pipeline.process_directory(
        input_dir="data/documents",
        output_dir="data/processed",
        use_semantic=True
    )

    print(f"\n✅ 处理完成！")
    print(f"   生成文本块数: {len(chunks)}")
    print(f"\n前3个文本块示例:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n  块{i}:")
        print(f"  {chunk[:100]}...")

    return chunks


def demo_hybrid_rag(chunks):
    """演示混合RAG系统"""
    print_header("🚀 演示2: 混合RAG系统")

    # 初始化系统
    print("\n正在初始化混合RAG系统...")
    rag = HybridRAG()

    # 添加文档
    print("\n正在添加文档到知识库...")
    rag.add_documents(chunks)

    # 测试查询
    print_section("测试查询集")

    test_cases = [
        {
            "category": "简单查询（事实性）",
            "questions": [
                "《西游记》的作者是谁？",
                "孙悟空是谁？",
                "唐僧的法号是什么？"
            ]
        },
        {
            "category": "复杂查询（分析推理）",
            "questions": [
                "孙悟空有几个师傅？他们分别是谁？有什么不同？",
                "分析唐僧师徒四人的性格特点",
                "为什么说《西游记》是浪漫主义小说？"
            ]
        },
        {
            "category": "复杂查询（计算）",
            "questions": [
                "师徒四人经历了多少难？如果每个难平均需要5天解决，总共需要多少天？"
            ]
        }
    ]

    all_results = []

    for test_case in test_cases:
        print_section(f"🔍 {test_case['category']}")

        for question in test_case['questions']:
            print(f"\n❓ 问题: {question}")

            # 查询
            result = rag.query(question)

            # 显示结果
            print(f"\n策略: {result['strategy'].upper()} RAG")
            print(f"复杂度: {result['complexity']}")
            print(f"响应时间: {result['response_time']:.2f}秒")
            print(f"\n💡 答案:\n{result['answer']}")
            print("\n" + "-"*70)

            all_results.append(result)
            time.sleep(1)  # 避免API限流

    return rag, all_results


def demo_force_strategy(rag):
    """演示强制使用特定策略"""
    print_header("🎯 演示3: 强制使用特定策略")

    question = "孙悟空有几个师傅？"

    print(f"\n测试问题: {question}")

    # 强制使用传统RAG
    print_section("强制使用传统RAG")
    result_trad = rag.query(question, force_strategy="traditional")
    print(f"响应时间: {result_trad['response_time']:.2f}秒")
    print(f"答案: {result_trad['answer'][:200]}...")

    time.sleep(1)

    # 强制使用Agentic RAG
    print_section("强制使用Agentic RAG")
    result_agen = rag.query(question, force_strategy="agentic")
    print(f"响应时间: {result_agen['response_time']:.2f}秒")
    print(f"答案: {result_agen['answer'][:200]}...")

    # 对比
    print_section("性能对比")
    print(f"传统RAG响应时间: {result_trad['response_time']:.2f}秒")
    print(f"Agentic RAG响应时间: {result_agen['response_time']:.2f}秒")
    print(f"时间差: {result_agen['response_time'] - result_trad['response_time']:.2f}秒")
    print(f"速度提升: {result_trad['response_time'] / result_agen['response_time'] * 100:.1f}%")


def demo_statistics(rag):
    """演示统计信息"""
    print_header("📊 演示4: 系统统计信息")

    rag.print_stats()

    stats = rag.get_stats()

    print_section("详细分析")

    # 使用率分析
    if stats['total_queries'] > 0:
        trad_rate = stats['traditional_count'] / stats['total_queries'] * 100
        agen_rate = stats['agentic_count'] / stats['total_queries'] * 100

        print(f"\n策略使用率:")
        print(f"  传统RAG: {trad_rate:.1f}%")
        print(f"  Agentic RAG: {agen_rate:.1f}%")

    # 性能对比
    print(f"\n性能对比:")
    print(f"  传统RAG平均响应: {stats['avg_traditional_time']:.2f}秒")
    print(f"  Agentic RAG平均响应: {stats['avg_agentic_time']:.2f}秒")
    if stats['avg_traditional_time'] > 0:
        speedup = stats['avg_agentic_time'] / stats['avg_traditional_time']
        print(f"  速度比: {speedup:.2f}x (Agentic是Traditional的{speedup:.2f}倍时间)")


def demo_router_analysis():
    """演示路由器分析"""
    print_header("🎯 演示5: 智能路由器分析")

    from router import QueryRouter
    router = QueryRouter()

    test_queries = [
        ("西游记的作者是谁？", "simple"),
        ("孙悟空是什么？", "simple"),
        ("孙悟空和猪八戒有什么区别？", "complex"),
        ("分析师徒四人的关系", "complex"),
        ("为什么唐僧要去西天取经？", "complex"),
    ]

    print("\n路由决策测试:")
    correct = 0
    total = len(test_queries)

    for query, expected in test_queries:
        result = router.route(query)
        is_correct = result['complexity'] == expected

        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        print(f"\n{status} 查询: {query}")
        print(f"   预期: {expected}")
        print(f"   实际: {result['complexity']}")
        print(f"   理由: {result['reasoning']}")

    accuracy = correct / total * 100
    print(f"\n路由准确率: {accuracy:.1f}% ({correct}/{total})")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("  🎯 混合RAG系统完整演示")
    print("  Hybrid RAG Complete Demo")
    print("="*70)

    try:
        # 1. 数据处理
        chunks = demo_data_processing()

        # 2. 混合RAG系统
        rag, results = demo_hybrid_rag(chunks)

        # 3. 强制策略
        demo_force_strategy(rag)

        # 4. 统计信息
        demo_statistics(rag)

        # 5. 路由器分析
        demo_router_analysis()

        # 总结
        print_header("✅ 演示完成！")
        print("\n核心要点:")
        print("  1. 数据处理: 支持批量处理和语义分块")
        print("  2. 智能路由: 自动选择最佳RAG策略")
        print("  3. 传统RAG: 快速响应简单查询 (1-2秒)")
        print("  4. Agentic RAG: 智能处理复杂查询 (5-15秒)")
        print("  5. 混合优势: 平衡速度和准确性")

        print("\n推荐使用场景:")
        print("  • 简单事实查询 → 自动使用传统RAG")
        print("  • 复杂分析推理 → 自动使用Agentic RAG")
        print("  • 需要计算 → 自动使用Agentic RAG")

        print("\n下一步:")
        print("  1. 添加自己的文档到 data/documents/")
        print("  2. 运行 python hybrid_rag.py 体验完整系统")
        print("  3. 查看 README.md 了解更多用法")

        print("\n" + "="*70)

    except KeyboardInterrupt:
        print("\n\n⚠️ 演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
