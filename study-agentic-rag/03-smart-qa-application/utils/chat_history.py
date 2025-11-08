import json
import os
import csv
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from config.settings import Settings

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """聊天记录管理器"""
    
    def __init__(self, history_file: str = None):
        self.settings = Settings()
        self.history_file = history_file or self.settings.CHAT_HISTORY_PATH
        self.history_dir = Path(self.history_file).parent
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.chat_history = []
        self.max_history_size = 10000  # 最大历史记录数
        
        # 加载现有历史记录
        self.load_history()
    
    def load_history(self) -> List[Dict[str, Any]]:
        """加载聊天记录"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                
                # 验证数据格式
                if not isinstance(self.chat_history, list):
                    logger.warning("聊天记录格式错误，重置为空列表")
                    self.chat_history = []
                
                # 清理过期记录（如果超过最大数量）
                if len(self.chat_history) > self.max_history_size:
                    logger.info(f"聊天记录超过最大数量限制 ({self.max_history_size})，清理旧记录")
                    self.chat_history = self.chat_history[-self.max_history_size:]
                    self.save_history()
                
                logger.info(f"加载聊天记录成功: {len(self.chat_history)} 条记录")
            else:
                logger.info("聊天记录文件不存在，创建新的历史记录")
                self.chat_history = []
                self.save_history()
            
            return self.chat_history
            
        except json.JSONDecodeError as e:
            logger.error(f"解析聊天记录JSON失败: {str(e)}")
            self.chat_history = []
            return self.chat_history
        except Exception as e:
            logger.error(f"加载聊天记录失败: {str(e)}")
            self.chat_history = []
            return self.chat_history
    
    def save_history(self) -> bool:
        """保存聊天记录"""
        try:
            # 确保目录存在
            self.history_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存历史记录
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存聊天记录成功: {len(self.chat_history)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"保存聊天记录失败: {str(e)}")
            return False
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """添加消息到历史记录"""
        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "id": self._generate_message_id()
            }
            
            if metadata:
                message["metadata"] = metadata
            
            self.chat_history.append(message)
            
            # 如果超过最大数量，移除最旧的记录
            if len(self.chat_history) > self.max_history_size:
                self.chat_history.pop(0)
            
            # 保存到文件
            self.save_history()
            
            logger.debug(f"添加消息成功: {role}")
            return True
            
        except Exception as e:
            logger.error(f"添加消息失败: {str(e)}")
            return False
    
    def get_history(self, limit: int = None, role_filter: str = None) -> List[Dict[str, Any]]:
        """获取历史记录"""
        try:
            history = self.chat_history.copy()
            
            # 角色过滤
            if role_filter:
                history = [msg for msg in history if msg.get("role") == role_filter]
            
            # 数量限制
            if limit and limit > 0:
                history = history[-limit:]
            
            return history
            
        except Exception as e:
            logger.error(f"获取历史记录失败: {str(e)}")
            return []
    
    def clear_history(self) -> bool:
        """清空历史记录"""
        try:
            self.chat_history = []
            self.save_history()
            
            logger.info("清空历史记录成功")
            return True
            
        except Exception as e:
            logger.error(f"清空历史记录失败: {str(e)}")
            return False
    
    def delete_message(self, message_id: str) -> bool:
        """删除指定消息"""
        try:
            original_length = len(self.chat_history)
            
            # 删除指定ID的消息
            self.chat_history = [
                msg for msg in self.chat_history 
                if msg.get("id") != message_id
            ]
            
            if len(self.chat_history) < original_length:
                self.save_history()
                logger.info(f"删除消息成功: {message_id}")
                return True
            else:
                logger.warning(f"未找到要删除的消息: {message_id}")
                return False
                
        except Exception as e:
            logger.error(f"删除消息失败: {str(e)}")
            return False
    
    def search_history(self, keyword: str, role_filter: str = None) -> List[Dict[str, Any]]:
        """搜索历史记录"""
        try:
            results = []
            keyword = keyword.lower()
            
            for message in self.chat_history:
                # 角色过滤
                if role_filter and message.get("role") != role_filter:
                    continue
                
                # 内容搜索
                content = message.get("content", "").lower()
                if keyword in content:
                    results.append(message)
            
            logger.info(f"搜索历史记录: '{keyword}' - 找到 {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索历史记录失败: {str(e)}")
            return []
    
    def export_to_csv(self, output_file: str = None) -> str:
        """导出为CSV格式"""
        try:
            if not output_file:
                # 返回CSV字符串
                import io
                output = io.StringIO()
                
                if self.chat_history:
                    writer = csv.DictWriter(output, fieldnames=self.chat_history[0].keys())
                    writer.writeheader()
                    writer.writerows(self.chat_history)
                
                csv_content = output.getvalue()
                output.close()
                
                logger.info("导出CSV成功")
                return csv_content
            else:
                # 保存到文件
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    if self.chat_history:
                        writer = csv.DictWriter(f, fieldnames=self.chat_history[0].keys())
                        writer.writeheader()
                        writer.writerows(self.chat_history)
                
                logger.info(f"导出CSV文件成功: {output_file}")
                return output_file
                
        except Exception as e:
            logger.error(f"导出CSV失败: {str(e)}")
            return ""
    
    def export_to_json(self, output_file: str = None, pretty_print: bool = True) -> str:
        """导出为JSON格式"""
        try:
            if not output_file:
                # 返回JSON字符串
                if pretty_print:
                    return json.dumps(self.chat_history, ensure_ascii=False, indent=2)
                else:
                    return json.dumps(self.chat_history, ensure_ascii=False)
            else:
                # 保存到文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    if pretty_print:
                        json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
                    else:
                        json.dump(self.chat_history, f, ensure_ascii=False)
                
                logger.info(f"导出JSON文件成功: {output_file}")
                return output_file
                
        except Exception as e:
            logger.error(f"导出JSON失败: {str(e)}")
            return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            total_messages = len(self.chat_history)
            
            if total_messages == 0:
                return {
                    "total_messages": 0,
                    "user_messages": 0,
                    "assistant_messages": 0,
                    "start_date": None,
                    "end_date": None,
                    "average_message_length": 0,
                    "longest_message": 0,
                    "shortest_message": 0
                }
            
            # 角色统计
            user_messages = len([msg for msg in self.chat_history if msg.get("role") == "user"])
            assistant_messages = len([msg for msg in self.chat_history if msg.get("role") == "assistant"])
            
            # 时间统计
            timestamps = []
            for msg in self.chat_history:
                if "timestamp" in msg:
                    try:
                        ts = datetime.fromisoformat(msg["timestamp"])
                        timestamps.append(ts)
                    except:
                        continue
            
            start_date = min(timestamps) if timestamps else None
            end_date = max(timestamps) if timestamps else None
            
            # 长度统计
            message_lengths = [len(msg.get("content", "")) for msg in self.chat_history]
            avg_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
            longest_message = max(message_lengths) if message_lengths else 0
            shortest_message = min(message_lengths) if message_lengths else 0
            
            stats = {
                "total_messages": total_messages,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "conversation_duration_days": (end_date - start_date).days if start_date and end_date else 0,
                "average_message_length": round(avg_length, 2),
                "longest_message": longest_message,
                "shortest_message": shortest_message,
                "history_file_size_bytes": os.path.getsize(self.history_file) if os.path.exists(self.history_file) else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {"error": str(e)}
    
    def backup_history(self, backup_dir: str = None) -> str:
        """备份历史记录"""
        try:
            if not backup_dir:
                backup_dir = self.history_dir / "backups"
            
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成备份文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"chat_history_backup_{timestamp}.json"
            
            # 复制当前历史记录到备份文件
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            
            logger.info(f"备份历史记录成功: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"备份历史记录失败: {str(e)}")
            return ""
    
    def restore_history(self, backup_file: str) -> bool:
        """从历史记录备份恢复"""
        try:
            backup_path = Path(backup_file)
            
            if not backup_path.exists():
                logger.error(f"备份文件不存在: {backup_file}")
                return False
            
            # 加载备份文件
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            if not isinstance(backup_data, list):
                logger.error("备份文件格式错误")
                return False
            
            # 先备份当前历史记录
            self.backup_history()
            
            # 恢复历史记录
            self.chat_history = backup_data
            self.save_history()
            
            logger.info(f"恢复历史记录成功: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"恢复历史记录失败: {str(e)}")
            return False
    
    def _generate_message_id(self) -> str:
        """生成消息ID"""
        import uuid
        return str(uuid.uuid4())
    
    def get_history_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """按日期范围获取历史记录"""
        try:
            filtered_history = []
            
            for message in self.chat_history:
                if "timestamp" in message:
                    try:
                        msg_date = datetime.fromisoformat(message["timestamp"])
                        if start_date <= msg_date <= end_date:
                            filtered_history.append(message)
                    except:
                        continue
            
            return filtered_history
            
        except Exception as e:
            logger.error(f"按日期范围获取历史记录失败: {str(e)}")
            return []