"""
数据预处理和分块工具
Data Processing and Chunking Utilities
"""

import os
import re
from typing import List
from pathlib import Path
import config


class TextProcessor:
    """文本预处理器"""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        清洗文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 移除特殊字符（保留中文、英文、数字、标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、""''（）《》\n]', '', text)

        # 标准化换行
        text = re.sub(r'\n+', '\n', text)

        return text.strip()

    @staticmethod
    def split_by_sentence(text: str) -> List[str]:
        """
        按句子分割文本

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 中文句子分隔符
        sentences = re.split(r'[。！？；]', text)
        return [s.strip() for s in sentences if s.strip()]


class TextChunker:
    """文本分块器"""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP
    ):
        """
        初始化分块器

        Args:
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        分割文本为固定大小的块

        Args:
            text: 输入文本

        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # 计算结束位置
            end = min(start + self.chunk_size, text_length)

            # 提取文本块
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # 移动起始位置（考虑重叠）
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def chunk_by_semantic(self, text: str, max_chunk_size: int = None) -> List[str]:
        """
        按语义分割文本（基于句子边界）

        Args:
            text: 输入文本
            max_chunk_size: 最大块大小

        Returns:
            语义分块列表
        """
        if max_chunk_size is None:
            max_chunk_size = self.chunk_size

        # 分割成句子
        sentences = TextProcessor.split_by_sentence(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # 如果当前块加上新句子不超过最大大小
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + "。"
            else:
                # 保存当前块，开始新块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


class DocumentLoader:
    """文档加载器"""

    @staticmethod
    def load_text_file(file_path: str) -> str:
        """
        加载文本文件

        Args:
            file_path: 文件路径

        Returns:
            文件内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()

    @staticmethod
    def load_directory(directory: str, extensions: List[str] = None) -> List[dict]:
        """
        加载目录中的所有文档

        Args:
            directory: 目录路径
            extensions: 文件扩展名列表

        Returns:
            文档列表 [{"path": ..., "content": ...}, ...]
        """
        if extensions is None:
            extensions = ['.txt', '.md']

        documents = []
        path = Path(directory)

        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions:
                content = DocumentLoader.load_text_file(str(file_path))
                documents.append({
                    "path": str(file_path),
                    "filename": file_path.name,
                    "content": content
                })

        return documents


class DataPipeline:
    """完整的数据处理流水线"""

    def __init__(self):
        """初始化流水线"""
        self.processor = TextProcessor()
        self.chunker = TextChunker()

    def process_document(self, text: str, use_semantic: bool = True) -> List[str]:
        """
        处理单个文档

        Args:
            text: 原始文本
            use_semantic: 是否使用语义分块

        Returns:
            处理后的文本块列表
        """
        # 1. 清洗文本
        cleaned_text = self.processor.clean_text(text)

        # 2. 分块
        if use_semantic:
            chunks = self.chunker.chunk_by_semantic(cleaned_text)
        else:
            chunks = self.chunker.chunk_text(cleaned_text)

        return chunks

    def process_directory(
        self,
        input_dir: str,
        output_dir: str = None,
        use_semantic: bool = True
    ) -> List[str]:
        """
        批量处理目录中的文档

        Args:
            input_dir: 输入目录
            output_dir: 输出目录（可选）
            use_semantic: 是否使用语义分块

        Returns:
            所有文本块
        """
        # 加载文档
        documents = DocumentLoader.load_directory(input_dir)

        all_chunks = []

        for doc in documents:
            print(f"处理文档: {doc['filename']}")

            # 处理文档
            chunks = self.process_document(doc['content'], use_semantic)
            all_chunks.extend(chunks)

            # 可选：保存处理后的块
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(
                    output_dir,
                    f"{os.path.splitext(doc['filename'])[0]}_chunks.txt"
                )
                with open(output_file, 'w', encoding='utf-8') as f:
                    for i, chunk in enumerate(chunks, 1):
                        f.write(f"=== 块 {i} ===\n{chunk}\n\n")

                print(f"  → 生成 {len(chunks)} 个文本块")

        print(f"\n总共处理: {len(all_chunks)} 个文本块")
        return all_chunks


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    import sys

    print("="*60)
    print("数据处理工具演示")
    print("="*60)

    # 示例文本
    sample_text = """
    《西游记》是中国古代第一部浪漫主义章回体长篇神魔小说。现存明刊百回本《西游记》均无作者署名。
    清代学者吴玉搢等首先提出《西游记》作者是明代吴承恩。这部小说以"唐僧取经"这一历史事件为蓝本，
    通过作者的艺术加工，深刻地描绘了当时的社会现实。全书主要描写了孙悟空出世及大闹天宫后，遇见了
    唐僧、猪八戒和沙僧三人，西行取经，一路降妖伏魔，经历了九九八十一难，终于到达西天见到如来佛祖，
    最终五圣成真的故事。
    """

    # 1. 文本清洗
    print("\n1. 文本清洗")
    processor = TextProcessor()
    cleaned = processor.clean_text(sample_text)
    print(f"原文长度: {len(sample_text)}")
    print(f"清洗后长度: {len(cleaned)}")
    print(f"清洗后文本: {cleaned[:100]}...")

    # 2. 固定大小分块
    print("\n2. 固定大小分块")
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    fixed_chunks = chunker.chunk_text(cleaned)
    print(f"生成 {len(fixed_chunks)} 个固定大小块")
    for i, chunk in enumerate(fixed_chunks, 1):
        print(f"  块{i}: {chunk[:50]}...")

    # 3. 语义分块
    print("\n3. 语义分块")
    semantic_chunks = chunker.chunk_by_semantic(cleaned, max_chunk_size=150)
    print(f"生成 {len(semantic_chunks)} 个语义块")
    for i, chunk in enumerate(semantic_chunks, 1):
        print(f"  块{i}: {chunk}")

    # 4. 完整流水线
    if len(sys.argv) > 1:
        input_directory = sys.argv[1]
        print(f"\n4. 处理目录: {input_directory}")

        pipeline = DataPipeline()
        chunks = pipeline.process_directory(
            input_dir=input_directory,
            output_dir=config.OUTPUT_DIR,
            use_semantic=True
        )

        print(f"\n处理完成！共 {len(chunks)} 个文本块")
    else:
        print("\n提示: 运行 'python data_processor.py <目录路径>' 可以批量处理文档")
