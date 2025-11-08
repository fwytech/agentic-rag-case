import streamlit as st
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from config.settings import Settings
from models.agent import AgenticRAGAgent
from services.vector_store import VectorStoreService
from services.weather_tools import WeatherTools
from utils.document_processor import DocumentProcessor
from utils.ui_components import UIComponents
from utils.chat_history import ChatHistoryManager
from utils.decorators import error_handler, log_execution

class AgenticRAGSystem:
    """主应用类 - Agentic RAG智能问答系统"""
    
    def __init__(self):
        self.settings = Settings()
        self.vector_store = VectorStoreService()
        self.weather_tools = WeatherTools()
        self.doc_processor = DocumentProcessor()
        self.ui_components = UIComponents()
        self.chat_history = ChatHistoryManager()
        self.agent = None
        self._initialize_system()
    
    @error_handler
    def _initialize_system(self):
        """初始化系统组件"""
        st.set_page_config(
            page_title="Agentic RAG智能问答系统",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 初始化会话状态
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.chat_history = []
            st.session_state.vector_store_ready = False
            st.session_state.current_model = self.settings.get_default_model()
            st.session_state.temperature = self.settings.DEFAULT_TEMPERATURE
            st.session_state.max_tokens = self.settings.DEFAULT_MAX_TOKENS
            st.session_state.top_k = self.settings.DEFAULT_TOP_K
            st.session_state.search_type = self.settings.DEFAULT_SEARCH_TYPE
            st.session_state.llm_provider = self.settings.LLM_PROVIDER
            
        # 创建必要的目录
        os.makedirs(self.settings.DATA_DIR, exist_ok=True)
        os.makedirs(self.settings.VECTOR_STORE_DIR, exist_ok=True)
        os.makedirs(self.settings.CHAT_HISTORY_DIR, exist_ok=True)
        
    @error_handler
    def _create_agent(self):
        """创建Agent实例"""
        tools = []
        
        # 如果向量存储已准备，添加文档搜索工具
        if st.session_state.vector_store_ready:
            tools.append(self._create_document_search_tool())
            
        # 添加天气查询工具
        tools.append(self._create_weather_tool())
        
        self.agent = AgenticRAGAgent(
            model_name=st.session_state.current_model,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            tools=tools if tools else None
        )
    
    def _create_document_search_tool(self):
        """创建文档搜索工具"""
        def document_search(query: str, top_k: Optional[int] = None) -> str:
            """搜索文档中的相关信息"""
            try:
                if not st.session_state.vector_store_ready:
                    return "向量存储未准备好，请先上传文档。"
                    
                top_k = top_k or st.session_state.top_k
                results = self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    search_type=st.session_state.search_type
                )
                
                if not results:
                    return "未找到相关文档信息。"
                    
                # 格式化搜索结果
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(
                        f"【文档{i}】\n内容: {result['content']}\n"
                        f"相似度: {result['score']:.3f}\n"
                    )
                
                return "\n".join(formatted_results)
                
            except Exception as e:
                return f"文档搜索出错: {str(e)}"
        
        return document_search
    
    def _create_weather_tool(self):
        """创建天气查询工具"""
        def weather_query(city: str, forecast_days: int = 1) -> str:
            """查询天气信息"""
            try:
                if forecast_days == 1:
                    return self.weather_tools.get_current_weather(city)
                else:
                    return self.weather_tools.get_weather_forecast(city, forecast_days)
            except Exception as e:
                return f"天气查询出错: {str(e)}"
        
        return weather_query
    
    @error_handler
    def process_uploaded_files(self, uploaded_files):
        """处理上传的文件"""
        if not uploaded_files:
            return 0
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            all_documents = []
            total_files = len(uploaded_files)
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"正在处理文件: {file.name} ({i+1}/{total_files})")
                
                # 处理文档
                documents = self.doc_processor.process_uploaded_file(file)
                if documents:
                    all_documents.extend(documents)
                    
                progress_bar.progress((i + 1) / total_files)
            
            if all_documents:
                status_text.text("正在构建向量存储...")
                
                # 添加到向量存储
                self.vector_store.add_documents(all_documents)
                
                # 保存向量存储
                self.vector_store.save_index(self.settings.VECTOR_STORE_PATH)
                
                st.session_state.vector_store_ready = True
                status_text.text(f"✅ 成功处理 {len(all_documents)} 个文档片段")
                
                return len(all_documents)
            else:
                status_text.text("⚠️ 没有有效的文档被处理")
                return 0
                
        except Exception as e:
            status_text.text(f"❌ 处理文件时出错: {str(e)}")
            return 0
        finally:
            progress_bar.empty()
            
    @error_handler
    def generate_response(self, query: str) -> str:
        """生成回答"""
        try:
            # 创建Agent（如果需要）
            if not self.agent:
                self._create_agent()
            
            # 生成回答
            response = self.agent.generate_response(query)
            
            return response
            
        except Exception as e:
            return f"生成回答时出错: {str(e)}"
    
    def run(self):
        """运行应用"""
        # 标题
        st.title("🤖 Agentic RAG智能问答系统")
        st.markdown("---")
        
        # 侧边栏
        with st.sidebar:
            st.header("⚙️ 系统配置")

            # LLM 提供商信息
            provider_info = self.settings.get_provider_info()
            st.info(f"🔧 **LLM 提供商**: {provider_info['provider']}\n\n"
                   f"📡 **服务地址**: {provider_info['base_url']}\n\n"
                   f"🎯 **嵌入模型**: {provider_info['embedding']}")

            st.markdown("---")

            # 模型设置
            st.subheader("模型设置")

            # 获取可用模型列表
            available_models = self.settings.get_available_models()

            # 确保当前模型在列表中
            if st.session_state.current_model not in available_models:
                st.session_state.current_model = self.settings.get_default_model()

            st.session_state.current_model = st.selectbox(
                "选择模型:",
                available_models,
                index=available_models.index(st.session_state.current_model) if st.session_state.current_model in available_models else 0
            )
            
            st.session_state.temperature = st.slider(
                "温度系数:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1
            )
            
            st.session_state.max_tokens = st.slider(
                "最大token数:",
                min_value=100,
                max_value=4000,
                value=st.session_state.max_tokens,
                step=100
            )
            
            # RAG设置
            st.subheader("RAG设置")
            st.session_state.top_k = st.slider(
                "检索数量:",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                step=1
            )
            
            st.session_state.search_type = st.selectbox(
                "搜索类型:",
                ["similarity", "mmr"],
                index=0 if st.session_state.search_type == "similarity" else 1
            )
            
            # 文档上传
            st.subheader("📄 文档上传")
            uploaded_files = st.file_uploader(
                "上传文档:",
                type=['pdf', 'txt', 'md', 'docx'],
                accept_multiple_files=True
            )
            
            if st.button("🔄 处理文档") and uploaded_files:
                with st.spinner("正在处理文档..."):
                    doc_count = self.process_uploaded_files(uploaded_files)
                    if doc_count > 0:
                        st.success(f"成功处理 {doc_count} 个文档片段")
                        st.rerun()
            
            # 向量存储状态
            st.subheader("📊 向量存储状态")
            if st.session_state.vector_store_ready:
                st.success("✅ 向量存储已准备")
                if st.button("🗑️ 清空向量存储"):
                    self.vector_store.clear()
                    st.session_state.vector_store_ready = False
                    st.rerun()
            else:
                st.warning("⚠️ 向量存储未准备")
                
                # 加载已有向量存储
                if os.path.exists(self.settings.VECTOR_STORE_PATH):
                    if st.button("📂 加载已有向量存储"):
                        try:
                            self.vector_store.load_index(self.settings.VECTOR_STORE_PATH)
                            st.session_state.vector_store_ready = True
                            st.success("✅ 向量存储加载成功")
                            st.rerun()
                        except Exception as e:
                            st.error(f"加载向量存储失败: {str(e)}")
            
            # 聊天记录管理
            st.subheader("💬 聊天记录")
            
            # 导出聊天记录
            if st.session_state.chat_history:
                if st.button("📥 导出聊天记录"):
                    csv_content = self.chat_history.export_to_csv()
                    st.download_button(
                        label="下载CSV文件",
                        data=csv_content,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            # 清空聊天记录
            if st.button("🗑️ 清空聊天记录"):
                st.session_state.chat_history = []
                self.chat_history.clear()
                st.rerun()
        
        # 主界面
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 聊天界面
            st.header("💬 智能问答")
            
            # 显示聊天记录
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # 用户输入
            if prompt := st.chat_input("请输入您的问题..."):
                # 添加用户消息
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # 显示用户消息
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # 生成回答
                with st.chat_message("assistant"):
                    with st.spinner("正在思考..."):
                        response = self.generate_response(prompt)
                        st.markdown(response)
                        
                        # 保存回答到聊天记录
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # 保存聊天记录到文件
                        self.chat_history.add_message("user", prompt)
                        self.chat_history.add_message("assistant", response)
        
        with col2:
            # 聊天统计
            st.header("📊 聊天统计")
            
            if st.session_state.chat_history:
                total_messages = len(st.session_state.chat_history)
                user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
                assistant_messages = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
                
                st.metric("总消息数", total_messages)
                st.metric("用户消息", user_messages)
                st.metric("助手消息", assistant_messages)
                
                # 显示最近的消息
                st.subheader("最近消息")
                recent_messages = st.session_state.chat_history[-5:]
                for msg in recent_messages:
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    st.text(f"{role_icon}: {content}")
            else:
                st.info("暂无聊天记录")


if __name__ == "__main__":
    app = AgenticRAGSystem()
    app.run()