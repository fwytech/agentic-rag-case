import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from config.settings import Settings

class UIComponents:
    """UI组件类 - 负责渲染各种Streamlit界面元素"""
    
    def __init__(self):
        self.settings = Settings()
    
    def render_model_selector(self, current_model: str, key_prefix: str = "") -> str:
        """渲染模型选择器"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_model = st.selectbox(
                "🤖 选择模型",
                options=self.settings.AVAILABLE_MODELS,
                index=self.settings.AVAILABLE_MODELS.index(current_model) 
                if current_model in self.settings.AVAILABLE_MODELS else 0,
                help="选择要使用的语言模型",
                key=f"{key_prefix}model_selector"
            )
        
        with col2:
            if st.button("🔄 刷新模型列表", key=f"{key_prefix}refresh_models"):
                st.rerun()
        
        return selected_model
    
    def render_temperature_slider(self, current_temp: float, key_prefix: str = "") -> float:
        """渲染温度系数滑块"""
        temperature = st.slider(
            "🌡️ 温度系数 (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=current_temp,
            step=0.1,
            help="控制回答的随机性。值越高，回答越随机；值越低，回答越确定。",
            key=f"{key_prefix}temperature_slider"
        )
        
        # 显示温度解释
        temp_explanation = self._get_temperature_explanation(temperature)
        st.caption(f"💡 {temp_explanation}")
        
        return temperature
    
    def render_max_tokens_slider(self, current_tokens: int, key_prefix: str = "") -> int:
        """渲染最大token数滑块"""
        max_tokens = st.slider(
            "📝 最大Token数",
            min_value=100,
            max_value=4000,
            value=current_tokens,
            step=100,
            help="控制生成回答的最大长度",
            key=f"{key_prefix}max_tokens_slider"
        )
        
        return max_tokens
    
    def render_rag_settings(self, current_top_k: int, current_search_type: str, key_prefix: str = "") -> tuple:
        """渲染RAG设置"""
        st.subheader("🔍 RAG设置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider(
                "检索数量 (Top-K)",
                min_value=1,
                max_value=10,
                value=current_top_k,
                step=1,
                help="从向量存储中检索的相关文档数量",
                key=f"{key_prefix}top_k_slider"
            )
        
        with col2:
            search_type = st.selectbox(
                "搜索类型",
                options=["similarity", "mmr"],
                index=0 if current_search_type == "similarity" else 1,
                help="similarity: 相似度搜索; mmr: 最大边际相关性搜索",
                key=f"{key_prefix}search_type_select"
            )
        
        # 显示搜索类型解释
        search_explanation = self._get_search_type_explanation(search_type)
        st.caption(f"💡 {search_explanation}")
        
        return top_k, search_type
    
    def render_file_uploader(self, key_prefix: str = "") -> Optional[List]:
        """渲染文件上传器"""
        st.subheader("📄 文档上传")
        
        uploaded_files = st.file_uploader(
            "选择要上传的文档",
            type=['pdf', 'txt', 'md', 'docx'],
            accept_multiple_files=True,
            help="支持 PDF、TXT、MD、DOCX 格式，可一次上传多个文件",
            key=f"{key_prefix}file_uploader"
        )
        
        if uploaded_files:
            st.success(f"✅ 已选择 {len(uploaded_files)} 个文件")
            
            # 显示文件列表
            with st.expander("📋 文件列表"):
                for i, file in enumerate(uploaded_files, 1):
                    file_size_mb = file.size / (1024 * 1024)
                    st.write(f"{i}. **{file.name}** ({file_size_mb:.2f} MB)")
        
        return uploaded_files
    
    def render_vector_store_status(self, is_ready: bool, stats: Optional[Dict] = None):
        """渲染向量存储状态"""
        if is_ready:
            st.success("✅ 向量存储已准备就绪")
            
            if stats:
                with st.expander("📊 向量存储统计"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("文档数量", stats.get('documents_count', 0))
                    
                    with col2:
                        st.metric("向量总数", stats.get('total_vectors', 0))
                    
                    with col3:
                        st.metric("向量维度", stats.get('dimension', 0))
                    
                    if stats.get('index_path'):
                        st.caption(f"📁 索引路径: {stats['index_path']}")
        
        else:
            st.warning("⚠️ 向量存储未准备")
    
    def render_chat_history(self, messages: List[Dict[str, str]], max_display: int = None):
        """渲染聊天记录"""
        max_display = max_display or self.settings.MAX_CHAT_HISTORY_DISPLAY
        
        if not messages:
            st.info("💬 暂无聊天记录")
            return
        
        # 显示最近的聊天记录
        recent_messages = messages[-max_display:]
        
        st.subheader(f"💬 聊天记录 (显示最近 {len(recent_messages)} 条)")
        
        # 创建可滚动的容器
        chat_container = st.container()
        
        with chat_container:
            for message in recent_messages:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(content)
                elif role == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(content)
        
        # 显示统计信息
        total_messages = len(messages)
        user_messages = len([m for m in messages if m.get("role") == "user"])
        assistant_messages = len([m for m in messages if m.get("role") == "assistant"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总消息数", total_messages)
        with col2:
            st.metric("用户消息", user_messages)
        with col3:
            st.metric("助手消息", assistant_messages)
    
    def render_chat_statistics(self, messages: List[Dict[str, str]]):
        """渲染聊天统计信息"""
        if not messages:
            st.info("📊 暂无统计数据")
            return
        
        st.subheader("📊 聊天统计")
        
        # 基本统计
        total_messages = len(messages)
        user_messages = len([m for m in messages if m.get("role") == "user"])
        assistant_messages = len([m for m in messages if m.get("role") == "assistant"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总消息数", total_messages)
        with col2:
            st.metric("用户消息", user_messages)
        with col3:
            st.metric("助手消息", assistant_messages)
        
        # 消息长度分析
        if user_messages > 0:
            user_lengths = [len(m.get("content", "")) for m in messages if m.get("role") == "user"]
            avg_user_length = sum(user_lengths) / len(user_lengths)
            max_user_length = max(user_lengths)
            
            st.write(f"💬 用户消息平均长度: {avg_user_length:.1f} 字符")
            st.write(f"🔤 用户消息最大长度: {max_user_length} 字符")
        
        if assistant_messages > 0:
            assistant_lengths = [len(m.get("content", "")) for m in messages if m.get("role") == "assistant"]
            avg_assistant_length = sum(assistant_lengths) / len(assistant_lengths)
            max_assistant_length = max(assistant_lengths)
            
            st.write(f"🤖 助手消息平均长度: {avg_assistant_length:.1f} 字符")
            st.write(f"📄 助手消息最大长度: {max_assistant_length} 字符")
        
        # 时间分布（如果有时间信息）
        if messages and any("timestamp" in m for m in messages):
            self._render_message_time_distribution(messages)
    
    def render_export_options(self, messages: List[Dict[str, str]]):
        """渲染导出选项"""
        st.subheader("📤 导出选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 导出为CSV"):
                csv_content = self._export_to_csv(messages)
                st.download_button(
                    label="下载CSV文件",
                    data=csv_content,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📄 导出为JSON"):
                json_content = self._export_to_json(messages)
                st.download_button(
                    label="下载JSON文件",
                    data=json_content,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    def render_clear_chat_button(self, key_prefix: str = "") -> bool:
        """渲染清空聊天按钮"""
        if st.button("🗑️ 清空聊天记录", key=f"{key_prefix}clear_chat", help="清空所有聊天记录"):
            return True
        return False
    
    def render_system_status(self, agent_info: Optional[Dict] = None, vector_store_info: Optional[Dict] = None):
        """渲染系统状态"""
        st.subheader("🔧 系统状态")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if agent_info:
                st.write("**🤖 Agent状态**")
                st.write(f"模型: {agent_info.get('model_name', '未知')}")
                st.write(f"工具数量: {agent_info.get('tools_count', 0)}")
                st.write(f"记忆启用: {agent_info.get('memory_enabled', False)}")
            else:
                st.write("**🤖 Agent状态**")
                st.write("未初始化")
        
        with col2:
            if vector_store_info:
                st.write("**📚 向量存储状态**")
                st.write(f"文档数量: {vector_store_info.get('documents_count', 0)}")
                st.write(f"向量总数: {vector_store_info.get('total_vectors', 0)}")
                st.write(f"状态: {'就绪' if vector_store_info.get('vector_store_initialized') else '未就绪'}")
            else:
                st.write("**📚 向量存储状态**")
                st.write("未初始化")
    
    def _get_temperature_explanation(self, temperature: float) -> str:
        """获取温度系数解释"""
        if temperature < 0.3:
            return "低温度：回答更确定、保守"
        elif temperature < 0.7:
            return "中等温度：平衡确定性和创造性"
        else:
            return "高温度：回答更随机、有创造性"
    
    def _get_search_type_explanation(self, search_type: str) -> str:
        """获取搜索类型解释"""
        if search_type == "similarity":
            return "相似度搜索：基于向量相似度检索最相关的文档"
        else:
            return "MMR搜索：在相关性和多样性之间取得平衡"
    
    def _export_to_csv(self, messages: List[Dict[str, str]]) -> str:
        """导出为CSV格式"""
        try:
            df = pd.DataFrame(messages)
            return df.to_csv(index=False)
        except Exception as e:
            st.error(f"CSV导出失败: {str(e)}")
            return ""
    
    def _export_to_json(self, messages: List[Dict[str, str]]) -> str:
        """导出为JSON格式"""
        try:
            import json
            return json.dumps(messages, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"JSON导出失败: {str(e)}")
            return ""
    
    def _render_message_time_distribution(self, messages: List[Dict[str, str]]):
        """渲染消息时间分布图"""
        try:
            # 提取时间戳
            timestamps = []
            for msg in messages:
                if "timestamp" in msg:
                    try:
                        ts = pd.to_datetime(msg["timestamp"])
                        timestamps.append(ts)
                    except:
                        continue
            
            if timestamps:
                st.subheader("⏰ 消息时间分布")
                
                # 创建时间分布图
                df = pd.DataFrame({'timestamp': timestamps})
                df['hour'] = df['timestamp'].dt.hour
                
                hourly_counts = df['hour'].value_counts().sort_index()
                
                fig = go.Figure(data=[
                    go.Bar(x=hourly_counts.index, y=hourly_counts.values)
                ])
                
                fig.update_layout(
                    title="每小时消息数量",
                    xaxis_title="小时",
                    yaxis_title="消息数量",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"时间分布图渲染失败: {str(e)}")
    
    def render_loading_spinner(self, text: str = "处理中..."):
        """渲染加载动画"""
        return st.spinner(text)
    
    def render_progress_bar(self, text: str = "进度"):
        """渲染进度条"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        return progress_bar, status_text
    
    def render_error_message(self, error: str, title: str = "错误"):
        """渲染错误消息"""
        st.error(f"❌ **{title}**: {error}")
    
    def render_success_message(self, message: str, title: str = "成功"):
        """渲染成功消息"""
        st.success(f"✅ **{title}**: {message}")
    
    def render_warning_message(self, message: str, title: str = "警告"):
        """渲染警告消息"""
        st.warning(f"⚠️ **{title}**: {message}")
    
    def render_info_message(self, message: str, title: str = "信息"):
        """渲染信息消息"""
        st.info(f"ℹ️ **{title}**: {message}")