"""
混合RAG系统配置文件
Hybrid RAG System Configuration
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ============================================
# 阿里云百炼平台配置
# ============================================

API_KEY = os.getenv("API_KEY", "sk-abe3417c96f6441b83efed38708bcfb6")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_ID = os.getenv("MODEL_ID", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")

# ============================================
# ChromaDB配置
# ============================================

CHROMA_DB_PATH = "./vector_db"
COLLECTION_NAME = "hybrid_rag_collection"

# ============================================
# RAG配置
# ============================================

# 检索参数
TOP_K = 3  # 检索文档数量
SIMILARITY_THRESHOLD = 0.7  # 相似度阈值

# 文本分块参数
CHUNK_SIZE = 500  # 分块大小（字符数）
CHUNK_OVERLAP = 50  # 分块重叠（字符数）

# ============================================
# LLM参数
# ============================================

# 温度参数
TEMPERATURE_REASONING = 0.3  # 用于推理和决策
TEMPERATURE_GENERATION = 0.7  # 用于答案生成

# Token限制
MAX_TOKENS_REASONING = 500
MAX_TOKENS_GENERATION = 1000
MAX_TOKENS_EVALUATION = 300

# ============================================
# Agentic RAG配置
# ============================================

# 最大迭代次数
MAX_AGENT_ITERATIONS = 5

# Web搜索配置
WEB_SEARCH_ENABLED = True
WEB_SEARCH_MAX_RESULTS = 3

# ============================================
# 路由器配置
# ============================================

# 简单查询关键词（触发传统RAG）
SIMPLE_QUERY_KEYWORDS = [
    "是什么", "什么是", "定义", "谁是", "在哪",
    "多少", "哪年", "哪里", "简介"
]

# 复杂查询关键词（触发Agentic RAG）
COMPLEX_QUERY_KEYWORDS = [
    "为什么", "如何", "分析", "比较", "评价",
    "原因", "影响", "关系", "区别", "优缺点"
]

# ============================================
# 日志配置
# ============================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================
# 数据路径
# ============================================

DATA_DIR = "./data/documents"  # 原始文档目录
OUTPUT_DIR = "./data/processed"  # 处理后的文档目录
