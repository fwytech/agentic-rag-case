import functools
import logging
import time
import traceback
from typing import Callable, Any, Optional
import streamlit as st
from config.settings import Settings

logger = logging.getLogger(__name__)

def error_handler(
    func_name: str = None,
    show_in_ui: bool = True,
    log_level: str = "ERROR",
    return_on_error: Any = None,
    error_message: str = None
):
    """错误处理装饰器
    
    Args:
        func_name: 函数名称（用于日志记录）
        show_in_ui: 是否在Streamlit UI中显示错误信息
        log_level: 日志级别 (ERROR, WARNING, INFO, DEBUG)
        return_on_error: 发生错误时的返回值
        error_message: 自定义错误消息
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取函数名称
                actual_func_name = func_name or func.__name__
                
                # 构建错误信息
                error_msg = error_message or f"函数 '{actual_func_name}' 执行失败"
                full_error_msg = f"{error_msg}: {str(e)}"
                
                # 记录日志
                log_func = getattr(logger, log_level.lower(), logger.error)
                log_func(full_error_msg)
                
                # 记录详细错误信息
                logger.debug(f"错误详情:\n{traceback.format_exc()}")
                
                # 在UI中显示错误（如果使用Streamlit）
                if show_in_ui and hasattr(st, 'error'):
                    st.error(f"❌ {full_error_msg}")
                    
                    # 显示详细错误（在开发模式下）
                    settings = Settings()
                    if settings.LOG_LEVEL == "DEBUG":
                        with st.expander("🔍 查看详细错误信息"):
                            st.code(traceback.format_exc())
                
                # 返回错误时的默认值
                return return_on_error
                
        return wrapper
    return decorator

def log_execution(
    func_name: str = None,
    log_level: str = "INFO",
    log_args: bool = False,
    log_result: bool = False,
    log_time: bool = True
):
    """执行日志装饰器
    
    Args:
        func_name: 函数名称（用于日志记录）
        log_level: 日志级别 (INFO, DEBUG, WARNING, ERROR)
        log_args: 是否记录函数参数
        log_result: 是否记录函数返回值
        log_time: 是否记录执行时间
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 获取函数名称
            actual_func_name = func_name or func.__name__
            
            # 获取日志函数
            log_func = getattr(logger, log_level.lower(), logger.info)
            
            try:
                # 记录函数开始执行
                start_time = time.time()
                log_func(f"开始执行函数: {actual_func_name}")
                
                # 记录参数（如果启用）
                if log_args:
                    args_str = str(args) if args else ""
                    kwargs_str = str(kwargs) if kwargs else ""
                    log_func(f"函数参数 - args: {args_str}, kwargs: {kwargs_str}")
                
                # 执行函数
                result = func(*args, **kwargs)
                
                # 记录执行时间（如果启用）
                if log_time:
                    execution_time = time.time() - start_time
                    log_func(f"函数执行完成: {actual_func_name} (耗时: {execution_time:.3f}秒)")
                else:
                    log_func(f"函数执行完成: {actual_func_name}")
                
                # 记录返回值（如果启用）
                if log_result:
                    result_str = str(result) if result is not None else "None"
                    # 限制结果字符串长度
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "..."
                    log_func(f"函数返回值: {result_str}")
                
                return result
                
            except Exception as e:
                # 记录异常信息
                execution_time = time.time() - start_time if log_time else 0
                error_msg = f"函数执行异常: {actual_func_name}"
                if log_time:
                    error_msg += f" (耗时: {execution_time:.3f}秒)"
                error_msg += f" - {str(e)}"
                
                logger.error(error_msg)
                logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
                
                # 重新抛出异常，让上层处理
                raise
                
        return wrapper
    return decorator

def performance_monitor(
    func_name: str = None,
    warning_threshold: float = 1.0,
    error_threshold: float = 5.0,
    log_args: bool = False
):
    """性能监控装饰器
    
    Args:
        func_name: 函数名称（用于日志记录）
        warning_threshold: 警告时间阈值（秒）
        error_threshold: 错误时间阈值（秒）
        log_args: 是否记录函数参数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            actual_func_name = func_name or func.__name__
            start_time = time.time()
            
            try:
                # 记录参数（如果启用）
                if log_args:
                    args_str = str(args) if args else ""
                    kwargs_str = str(kwargs) if kwargs else ""
                    logger.debug(f"性能监控 - 函数: {actual_func_name}, 参数: args={args_str}, kwargs={kwargs_str}")
                
                # 执行函数
                result = func(*args, **kwargs)
                
                # 计算执行时间
                execution_time = time.time() - start_time
                
                # 根据执行时间记录不同级别的日志
                if execution_time >= error_threshold:
                    logger.error(f"性能告警 - 函数执行过慢: {actual_func_name} (耗时: {execution_time:.3f}秒)")
                elif execution_time >= warning_threshold:
                    logger.warning(f"性能警告 - 函数执行较慢: {actual_func_name} (耗时: {execution_time:.3f}秒)")
                else:
                    logger.info(f"性能正常 - 函数执行完成: {actual_func_name} (耗时: {execution_time:.3f}秒)")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"性能监控 - 函数执行异常: {actual_func_name} (耗时: {execution_time:.3f}秒) - {str(e)}")
                raise
                
        return wrapper
    return decorator

def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    func_name: str = None
):
    """失败重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟时间的增长因子
        exceptions: 需要重试的异常类型
        func_name: 函数名称（用于日志记录）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            actual_func_name = func_name or func.__name__
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    # 执行函数
                    result = func(*args, **kwargs)
                    
                    # 如果成功，记录日志并返回结果
                    if attempt > 0:
                        logger.info(f"重试成功 - 函数: {actual_func_name} (第{attempt + 1}次尝试)")
                    
                    return result
                    
                except exceptions as e:
                    # 如果还有重试机会
                    if attempt < max_retries:
                        logger.warning(f"函数执行失败，准备重试: {actual_func_name} (第{attempt + 1}次尝试，错误: {str(e)})")
                        
                        # 等待一段时间后重试
                        time.sleep(current_delay)
                        current_delay *= backoff
                        
                    else:
                        # 最后一次尝试仍然失败
                        logger.error(f"函数执行失败，已达到最大重试次数: {actual_func_name} (共{max_retries + 1}次尝试)")
                        raise
            
            return None
                
        return wrapper
    return decorator

def cache_result(
    cache_key: str = None,
    expire_time: int = 3600,
    func_name: str = None
):
    """结果缓存装饰器
    
    Args:
        cache_key: 缓存键名（如果不提供，则使用函数名和参数生成）
        expire_time: 缓存过期时间（秒）
        func_name: 函数名称（用于日志记录）
    """
    def decorator(func: Callable) -> Callable:
        # 简单的内存缓存（在实际应用中可以使用Redis等）
        cache = {}
        cache_timestamps = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            actual_func_name = func_name or func.__name__
            
            try:
                # 生成缓存键
                if cache_key:
                    key = cache_key
                else:
                    # 基于函数名和参数生成缓存键
                    key_parts = [actual_func_name] + [str(arg) for arg in args] + [f"{k}:{v}" for k, v in kwargs.items()]
                    key = "_".join(key_parts)
                
                current_time = time.time()
                
                # 检查缓存是否存在且未过期
                if key in cache:
                    if current_time - cache_timestamps[key] < expire_time:
                        logger.debug(f"使用缓存结果 - 函数: {actual_func_name}, 缓存键: {key}")
                        return cache[key]
                    else:
                        # 缓存过期，删除旧缓存
                        del cache[key]
                        del cache_timestamps[key]
                
                # 执行函数获取结果
                result = func(*args, **kwargs)
                
                # 缓存结果
                cache[key] = result
                cache_timestamps[key] = current_time
                
                logger.debug(f"缓存新结果 - 函数: {actual_func_name}, 缓存键: {key}")
                
                return result
                
            except Exception as e:
                logger.error(f"缓存装饰器出错 - 函数: {actual_func_name}, 错误: {str(e)}")
                # 如果缓存出错，直接返回函数结果
                return func(*args, **kwargs)
                
        return wrapper
    return decorator

# 组合装饰器 - 同时提供错误处理和执行日志
def safe_execute(
    func_name: str = None,
    show_in_ui: bool = True,
    log_level: str = "INFO",
    return_on_error: Any = None,
    error_message: str = None,
    log_args: bool = False,
    log_result: bool = False,
    log_time: bool = True
):
    """安全执行装饰器（组合错误处理和执行日志）"""
    def decorator(func: Callable) -> Callable:
        # 应用错误处理装饰器
        error_handler_decorator = error_handler(
            func_name=func_name,
            show_in_ui=show_in_ui,
            log_level=log_level,
            return_on_error=return_on_error,
            error_message=error_message
        )
        
        # 应用执行日志装饰器
        log_decorator = log_execution(
            func_name=func_name,
            log_level=log_level,
            log_args=log_args,
            log_result=log_result,
            log_time=log_time
        )
        
        # 组合装饰器
        return error_handler_decorator(log_decorator(func))
    
    return decorator