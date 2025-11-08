import os
from pathlib import Path
from typing import Dict, List, Any

class Settings:
    """系统配置类"""
    
    # 基础路径
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    VECTOR_STORE_DIR = BASE_DIR / "vector_store"
    CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
    LOG_DIR = BASE_DIR / "logs"
    
    # 文件路径
    VECTOR_STORE_PATH = VECTOR_STORE_DIR / "faiss_index"
    CHAT_HISTORY_PATH = CHAT_HISTORY_DIR / "chat_history.json"
    LOG_FILE = LOG_DIR / "app.log"
    
    # 模型配置
    AVAILABLE_MODELS = [
        "qwen:7b",
        "qwen:14b", 
        "qwen:32b",
        "llama2:7b",
        "llama2:13b",
        "llama2:70b",
        "mistral:7b",
        "codellama:7b",
        "vicuna:7b",
        "baichuan:7b"
    ]
    
    DEFAULT_MODEL = "qwen:7b"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TOP_K = 3
    DEFAULT_SEARCH_TYPE = "similarity"  # similarity 或 mmr
    
    # 向量存储配置
    VECTOR_DIMENSION = 768
    EMBEDDING_MODEL = "ollama"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # 天气API配置
    WEATHER_API_KEY = "你的高德地图API密钥"  # 请替换为实际的API密钥
    WEATHER_API_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
    WEATHER_CITY_URL = "https://restapi.amap.com/v3/config/district"
    
    # 文档处理配置
    SUPPORTED_FILE_TYPES = [
        ".pdf",
        ".txt", 
        ".md",
        ".docx"
    ]
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    CACHE_ENABLED = True
    CACHE_EXPIRE_TIME = 3600  # 1小时
    
    # 显示配置
    MAX_CHAT_HISTORY_DISPLAY = 100
    MESSAGE_TRUNCATE_LENGTH = 500
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # 系统配置
    ENABLE_FILE_CACHE = True
    ENABLE_CHAT_HISTORY = True
    ENABLE_VECTOR_STORE = True
    ENABLE_WEATHER_TOOL = True
    ENABLE_DOCUMENT_TOOL = True
    
    @classmethod
    def initialize_directories(cls):
        """初始化必要的目录"""
        directories = [
            cls.DATA_DIR,
            cls.VECTOR_STORE_DIR, 
            cls.CHAT_HISTORY_DIR,
            cls.LOG_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """获取模型配置"""
        model_configs = {
            "qwen:7b": {
                "temperature": cls.DEFAULT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "qwen:14b": {
                "temperature": cls.DEFAULT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS * 2,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "llama2:7b": {
                "temperature": 0.8,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "top_p": 0.95,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
        }
        
        return model_configs.get(model_name, {
            "temperature": cls.DEFAULT_TEMPERATURE,
            "max_tokens": cls.DEFAULT_MAX_TOKENS,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        })