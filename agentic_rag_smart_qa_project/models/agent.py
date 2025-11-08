from typing import List, Dict, Optional, Any, Callable
import logging
from langchain.llms import Ollama
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool, StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config.settings import Settings

logger = logging.getLogger(__name__)

class AgenticRAGAgent:
    """Agentic RAG智能问答代理"""
    
    def __init__(
        self,
        model_name: str = "qwen:7b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Callable]] = None,
        enable_memory: bool = True,
        system_prompt: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or []
        self.enable_memory = enable_memory
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # 初始化LLM
        self.llm = self._initialize_llm()
        
        # 初始化记忆
        if self.enable_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
        
        # 初始化代理
        self.agent = self._initialize_agent()
        
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return """你是一个智能问答助手，具备以下能力：

1. 文档问答：能够基于上传的文档回答相关问题
2. 天气查询：能够查询实时天气信息和天气预报
3. 多轮对话：能够理解上下文进行连续对话

行为准则：
- 回答要准确、简洁、有用
- 如果不确定答案，请说明
- 使用工具时要明确说明
- 保持友好和专业的态度

工具使用说明：
- 当需要搜索文档时，使用document_search工具
- 当需要查询天气时，使用weather_query工具
- 根据用户问题的具体需求选择合适的工具
"""

    def _initialize_llm(self) -> Ollama:
        """初始化语言模型"""
        try:
            llm = Ollama(
                model=self.model_name,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                callbacks=[StreamingStdOutCallbackHandler()] if Settings().LOG_LEVEL == "DEBUG" else None
            )
            return llm
        except Exception as e:
            logger.error(f"初始化LLM失败: {str(e)}")
            raise

    def _initialize_agent(self):
        """初始化代理"""
        try:
            # 创建工具
            langchain_tools = []
            for i, tool_func in enumerate(self.tools):
                tool = Tool(
                    name=f"tool_{i}",
                    func=tool_func,
                    description=f"工具{i+1}: {tool_func.__doc__ or '自定义工具'}"
                )
                langchain_tools.append(tool)
            
            # 创建提示词模板
            prompt_template = PromptTemplate(
                input_variables=["input", "chat_history", "agent_scratchpad"],
                template=self._create_agent_template()
            )
            
            # 创建代理
            if langchain_tools:
                agent = create_react_agent(
                    llm=self.llm,
                    tools=langchain_tools,
                    prompt=prompt_template
                )
                
                # 创建代理执行器
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=langchain_tools,
                    memory=self.memory if self.enable_memory else None,
                    verbose=True,
                    max_iterations=5,
                    handle_parsing_errors=True,
                    return_intermediate_steps=True
                )
                
                return agent_executor
            else:
                # 如果没有工具，直接返回LLM
                return self.llm
                
        except Exception as e:
            logger.error(f"初始化代理失败: {str(e)}")
            raise

    def _create_agent_template(self) -> str:
        """创建代理提示词模板"""
        return """{system_prompt}

当前对话历史:
{chat_history}

人类: {input}

助手: {agent_scratchpad}"""

    def generate_response(self, query: str) -> str:
        """生成回答"""
        try:
            logger.info(f"生成回答 - 查询: {query}")
            
            if isinstance(self.agent, AgentExecutor):
                # 使用代理执行器
                response = self.agent.invoke({
                    "input": query,
                    "system_prompt": self.system_prompt
                })
                
                output = response.get("output", "抱歉，我无法生成回答。")
                
                # 记录中间步骤
                if "intermediate_steps" in response:
                    for step in response["intermediate_steps"]:
                        logger.info(f"中间步骤: {step}")
                        
            else:
                # 直接使用LLM
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=query)
                ]
                
                output = self.agent.invoke(messages).content
            
            logger.info(f"生成回答成功: {output[:100]}...")
            return output
            
        except Exception as e:
            error_msg = f"生成回答时出错: {str(e)}"
            logger.error(error_msg)
            return f"抱歉，处理您的请求时出现了错误。{str(e)}"

    def add_tool(self, tool: Callable, name: str = None, description: str = None):
        """添加工具"""
        self.tools.append(tool)
        
        # 重新初始化代理
        self.agent = self._initialize_agent()
        
        logger.info(f"添加工具: {name or tool.__name__}")

    def clear_memory(self):
        """清空记忆"""
        if self.enable_memory and hasattr(self, 'memory'):
            self.memory.clear()
            logger.info("记忆已清空")

    def get_memory_history(self) -> List[Dict[str, str]]:
        """获取记忆历史"""
        if not self.enable_memory or not hasattr(self, 'memory'):
            return []
            
        try:
            messages = self.memory.chat_memory.messages
            history = []
            
            for message in messages:
                if isinstance(message, HumanMessage):
                    history.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    history.append({"role": "assistant", "content": message.content})
                    
            return history
        except Exception as e:
            logger.error(f"获取记忆历史失败: {str(e)}")
            return []

    def update_system_prompt(self, new_prompt: str):
        """更新系统提示词"""
        self.system_prompt = new_prompt
        
        # 重新初始化代理
        self.agent = self._initialize_agent()
        
        logger.info("系统提示词已更新")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools_count": len(self.tools),
            "memory_enabled": self.enable_memory,
            "system_prompt_length": len(self.system_prompt)
        }