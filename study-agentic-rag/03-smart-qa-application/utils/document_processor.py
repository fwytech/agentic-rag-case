import os
import hashlib
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from config.settings import Settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器类"""
    
    def __init__(self):
        self.settings = Settings()
        self.cache_dir = self.settings.DATA_DIR / "document_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_file_hash(self, file_content: bytes) -> str:
        """计算文件哈希值"""
        return hashlib.md5(file_content).hexdigest()
    
    def _get_cache_path(self, file_hash: str, file_name: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{file_hash}_{file_name}.json"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[List[Document]]:
        """从缓存加载文档"""
        try:
            if cache_path.exists() and self.settings.CACHE_ENABLED:
                import json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # 检查缓存是否过期
                import time
                current_time = time.time()
                cache_time = cache_data.get('timestamp', 0)
                
                if current_time - cache_time < self.settings.CACHE_EXPIRE_TIME:
                    # 重建Document对象
                    documents = []
                    for doc_data in cache_data.get('documents', []):
                        doc = Document(
                            page_content=doc_data['page_content'],
                            metadata=doc_data['metadata']
                        )
                        documents.append(doc)
                    
                    logger.info(f"从缓存加载文档成功: {len(documents)} 个文档")
                    return documents
                else:
                    logger.info("缓存已过期")
                    
        except Exception as e:
            logger.error(f"从缓存加载失败: {str(e)}")
        
        return None
    
    def _save_to_cache(self, cache_path: Path, documents: List[Document]):
        """保存文档到缓存"""
        try:
            if not self.settings.CACHE_ENABLED:
                return
                
            import json
            import time
            
            cache_data = {
                'timestamp': time.time(),
                'documents': [
                    {
                        'page_content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    for doc in documents
                ]
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存到缓存成功: {cache_path}")
            
        except Exception as e:
            logger.error(f"保存到缓存失败: {str(e)}")
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """处理上传的文件"""
        try:
            # 检查文件大小
            if uploaded_file.size > self.settings.MAX_FILE_SIZE:
                raise ValueError(f"文件大小超过限制: {uploaded_file.size} > {self.settings.MAX_FILE_SIZE}")
            
            # 读取文件内容
            file_content = uploaded_file.read()
            file_name = uploaded_file.name
            file_type = Path(file_name).suffix.lower()
            
            # 检查文件类型
            if file_type not in self.settings.SUPPORTED_FILE_TYPES:
                raise ValueError(f"不支持的文件类型: {file_type}")
            
            # 计算文件哈希
            file_hash = self._get_file_hash(file_content)
            cache_path = self._get_cache_path(file_hash, file_name)
            
            # 尝试从缓存加载
            cached_documents = self._load_from_cache(cache_path)
            if cached_documents is not None:
                return cached_documents
            
            # 处理文件
            documents = self._process_file_content(file_content, file_name, file_type)
            
            # 保存到缓存
            self._save_to_cache(cache_path, documents)
            
            logger.info(f"处理文件成功: {file_name}, 文档数量: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"处理上传文件失败: {str(e)}")
            raise
    
    def _process_file_content(self, file_content: bytes, file_name: str, file_type: str) -> List[Document]:
        """处理文件内容"""
        try:
            # 创建临时文件
            temp_dir = self.settings.DATA_DIR / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / file_name
            
            # 写入临时文件
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            try:
                # 根据文件类型选择加载器
                if file_type == '.pdf':
                    documents = self._load_pdf(temp_path)
                elif file_type == '.txt':
                    documents = self._load_text(temp_path)
                elif file_type == '.md':
                    documents = self._load_markdown(temp_path)
                elif file_type == '.docx':
                    documents = self._load_word(temp_path)
                else:
                    raise ValueError(f"不支持的文件类型: {file_type}")
                
                # 添加元数据
                for i, doc in enumerate(documents):
                    doc.metadata.update({
                        'source': file_name,
                        'file_type': file_type,
                        'chunk_index': i,
                        'total_chunks': len(documents),
                        'processing_timestamp': str(Path(temp_path).stat().st_mtime)
                    })
                
                return documents
                
            finally:
                # 清理临时文件
                if temp_path.exists():
                    temp_path.unlink()
                    
        except Exception as e:
            logger.error(f"处理文件内容失败: {str(e)}")
            raise
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """加载PDF文件"""
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # 添加页码信息
            for i, doc in enumerate(documents):
                if 'page' not in doc.metadata:
                    doc.metadata['page'] = i + 1
            
            logger.info(f"加载PDF成功: {file_path.name}, 页数: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"加载PDF失败: {str(e)}")
            raise
    
    def _load_text(self, file_path: Path) -> List[Document]:
        """加载文本文件"""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            
            logger.info(f"加载文本文件成功: {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"加载文本文件失败: {str(e)}")
            raise
    
    def _load_markdown(self, file_path: Path) -> List[Document]:
        """加载Markdown文件"""
        try:
            # Markdown文件也使用文本加载器
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            
            # 添加文件类型标识
            for doc in documents:
                doc.metadata['file_type'] = '.md'
            
            logger.info(f"加载Markdown文件成功: {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"加载Markdown文件失败: {str(e)}")
            raise
    
    def _load_word(self, file_path: Path) -> List[Document]:
        """加载Word文档"""
        try:
            loader = UnstructuredWordDocumentLoader(str(file_path))
            documents = loader.load()
            
            logger.info(f"加载Word文档成功: {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"加载Word文档失败: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document], chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
        """分割文档"""
        try:
            chunk_size = chunk_size or self.settings.CHUNK_SIZE
            chunk_overlap = chunk_overlap or self.settings.CHUNK_OVERLAP
            
            logger.info(f"分割文档，chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # 更新元数据
            for i, doc in enumerate(split_docs):
                doc.metadata['chunk_index'] = i
                doc.metadata['chunk_size'] = len(doc.page_content)
                doc.metadata['total_chunks'] = len(split_docs)
            
            logger.info(f"文档分割完成，片段数量: {len(split_docs)}")
            return split_docs
            
        except Exception as e:
            logger.error(f"文档分割失败: {str(e)}")
            return documents
    
    def process_documents_batch(self, uploaded_files: List) -> Dict[str, Any]:
        """批量处理文档"""
        try:
            results = {
                'total_files': len(uploaded_files),
                'processed_files': 0,
                'failed_files': 0,
                'total_documents': 0,
                'errors': []
            }
            
            all_documents = []
            
            for file in uploaded_files:
                try:
                    documents = self.process_uploaded_file(file)
                    all_documents.extend(documents)
                    results['processed_files'] += 1
                    results['total_documents'] += len(documents)
                    
                    logger.info(f"处理文件成功: {file.name}")
                    
                except Exception as e:
                    results['failed_files'] += 1
                    error_info = {
                        'file_name': file.name,
                        'error': str(e)
                    }
                    results['errors'].append(error_info)
                    logger.error(f"处理文件失败: {file.name}, 错误: {str(e)}")
            
            results['all_documents'] = all_documents
            return results
            
        except Exception as e:
            logger.error(f"批量处理文档失败: {str(e)}")
            raise
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            stats = {
                'cache_enabled': self.settings.CACHE_ENABLED,
                'cache_expire_time': self.settings.CACHE_EXPIRE_TIME,
                'cache_files_count': len(cache_files),
                'cache_total_size_bytes': total_size,
                'cache_total_size_mb': round(total_size / (1024 * 1024), 2)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取缓存统计信息失败: {str(e)}")
            return {'error': str(e)}
    
    def clear_cache(self) -> bool:
        """清空缓存"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            deleted_count = 0
            
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"删除缓存文件失败: {cache_file}, 错误: {str(e)}")
            
            logger.info(f"清空缓存成功，删除文件数: {deleted_count}")
            return True
            
        except Exception as e:
            logger.error(f"清空缓存失败: {str(e)}")
            return false