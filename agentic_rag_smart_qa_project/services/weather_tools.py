import requests
import json
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime
from config.settings import Settings

logger = logging.getLogger(__name__)

class WeatherService:
    """天气查询服务类"""
    
    def __init__(self):
        self.settings = Settings()
        self.api_key = self.settings.WEATHER_API_KEY
        self.weather_url = self.settings.WEATHER_API_URL
        self.city_url = self.settings.WEATHER_CITY_URL
        
        # 城市代码缓存
        self.city_cache = {}
        
    def get_city_code(self, city_name: str) -> Optional[str]:
        """获取城市代码"""
        try:
            # 检查缓存
            if city_name in self.city_cache:
                return self.city_cache[city_name]
            
            # 构建请求URL
            url = f"{self.city_url}"
            params = {
                "keywords": city_name,
                "subdistrict": 0,
                "key": self.api_key,
                "extensions": "base"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") == "1" and data.get("districts"):
                # 获取第一个匹配的城市
                districts = data["districts"]
                if districts and len(districts) > 0:
                    city_code = districts[0].get("adcode")
                    if city_code:
                        # 缓存结果
                        self.city_cache[city_name] = city_code
                        logger.info(f"获取城市代码成功: {city_name} -> {city_code}")
                        return city_code
            
            logger.warning(f"未找到城市: {city_name}")
            return None
            
        except requests.RequestException as e:
            logger.error(f"获取城市代码失败: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"获取城市代码出错: {str(e)}")
            return None
    
    def get_current_weather(self, city_name: str) -> str:
        """获取当前天气"""
        try:
            city_code = self.get_city_code(city_name)
            if not city_code:
                return f"抱歉，无法找到城市 '{city_name}' 的信息。请检查城市名称是否正确。"
            
            # 构建请求URL
            params = {
                "city": city_code,
                "key": self.api_key,
                "extensions": "base"
            }
            
            response = requests.get(self.weather_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") == "1" and data.get("lives"):
                weather_info = data["lives"][0]
                
                # 格式化天气信息
                result = self._format_current_weather(weather_info, city_name)
                logger.info(f"获取当前天气成功: {city_name}")
                return result
            else:
                error_msg = data.get("info", "未知错误")
                logger.warning(f"获取当前天气失败: {error_msg}")
                return f"获取天气信息失败: {error_msg}"
                
        except requests.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            logger.error(f"获取当前天气失败: {error_msg}")
            return f"获取天气信息失败，请稍后重试。"
        except Exception as e:
            error_msg = f"获取当前天气出错: {str(e)}"
            logger.error(error_msg)
            return f"获取天气信息时发生错误: {str(e)}"
    
    def get_weather_forecast(self, city_name: str, days: int = 3) -> str:
        """获取天气预报"""
        try:
            if days < 1 or days > 7:
                return "预报天数必须在1-7天之间。"
            
            city_code = self.get_city_code(city_name)
            if not city_code:
                return f"抱歉，无法找到城市 '{city_name}' 的信息。请检查城市名称是否正确。"
            
            # 构建请求URL
            params = {
                "city": city_code,
                "key": self.api_key,
                "extensions": "all"
            }
            
            response = requests.get(self.weather_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") == "1" and data.get("forecasts"):
                forecast_info = data["forecasts"][0]
                
                # 格式化预报信息
                result = self._format_weather_forecast(forecast_info, city_name, days)
                logger.info(f"获取天气预报成功: {city_name}, 天数: {days}")
                return result
            else:
                error_msg = data.get("info", "未知错误")
                logger.warning(f"获取天气预报失败: {error_msg}")
                return f"获取天气预报失败: {error_msg}"
                
        except requests.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            logger.error(f"获取天气预报失败: {error_msg}")
            return f"获取天气预报失败，请稍后重试。"
        except Exception as e:
            error_msg = f"获取天气预报出错: {str(e)}"
            logger.error(error_msg)
            return f"获取天气预报时发生错误: {str(e)}"
    
    def _format_current_weather(self, weather_data: Dict[str, Any], city_name: str) -> str:
        """格式化当前天气信息"""
        try:
            province = weather_data.get("province", "")
            city = weather_data.get("city", city_name)
            weather = weather_data.get("weather", "")
            temperature = weather_data.get("temperature", "")
            winddirection = weather_data.get("winddirection", "")
            windpower = weather_data.get("windpower", "")
            humidity = weather_data.get("humidity", "")
            reporttime = weather_data.get("reporttime", "")
            
            # 构建格式化输出
            result = f"🏙️ **{province} {city}** 当前天气\n\n"
            result += f"🌤️ **天气状况**: {weather}\n"
            result += f"🌡️ **气温**: {temperature}°C\n"
            result += f"💨 **风向风力**: {winddirection} {windpower}\n"
            result += f"💧 **湿度**: {humidity}%\n"
            result += f"📅 **发布时间**: {reporttime}\n"
            
            # 添加天气建议
            result += "\n💡 **温馨提示**:\n"
            
            if temperature and temperature.isdigit():
                temp = int(temperature)
                if temp < 10:
                    result += "• 天气较冷，请注意保暖。\n"
                elif temp > 30:
                    result += "• 天气较热，请注意防暑。\n"
                else:
                    result += "• 天气舒适，适合外出。\n"
            
            if humidity and humidity.isdigit():
                hum = int(humidity)
                if hum > 80:
                    result += "• 湿度较高，注意防潮。\n"
                elif hum < 30:
                    result += "• 湿度较低，注意补水。\n"
            
            return result
            
        except Exception as e:
            logger.error(f"格式化当前天气信息失败: {str(e)}")
            return f"天气数据格式化失败: {str(e)}"
    
    def _format_weather_forecast(self, forecast_data: Dict[str, Any], city_name: str, days: int) -> str:
        """格式化天气预报信息"""
        try:
            province = forecast_data.get("province", "")
            city = forecast_data.get("city", city_name)
            casts = forecast_data.get("casts", [])
            
            if not casts:
                return "未获取到预报数据。"
            
            # 限制预报天数
            casts = casts[:days]
            
            result = f"🏙️ **{province} {city}** {days}天天气预报\n\n"
            
            for i, cast in enumerate(casts, 1):
                date = cast.get("date", "")
                week = cast.get("week", "")
                dayweather = cast.get("dayweather", "")
                nightweather = cast.get("nightweather", "")
                daytemp = cast.get("daytemp", "")
                nighttemp = cast.get("nighttemp", "")
                daywind = cast.get("daywind", "")
                nightwind = cast.get("nightwind", "")
                daypower = cast.get("daypower", "")
                nightpower = cast.get("nightpower", "")
                
                # 转换星期
                week_map = {
                    "1": "周一", "2": "周二", "3": "周三", "4": "周四",
                    "5": "周五", "6": "周六", "7": "周日"
                }
                week_str = week_map.get(week, f"第{i}天")
                
                result += f"📅 **{date} {week_str}**\n"
                result += f"🌤️ **天气**: {dayweather} 转 {nightweather}\n"
                result += f"🌡️ **温度**: {nighttemp}°C ~ {daytemp}°C\n"
                result += f"💨 **风力**: {daywind}{daypower} 转 {nightwind}{nightpower}\n"
                
                if i < len(casts):
                    result += "\n"  # 添加分隔线
            
            # 添加总体建议
            result += "\n💡 **总体建议**:\n"
            
            # 分析温度趋势
            temps = []
            for cast in casts:
                try:
                    day_temp = int(cast.get("daytemp", "0"))
                    night_temp = int(cast.get("nighttemp", "0"))
                    temps.extend([day_temp, night_temp])
                except (ValueError, TypeError):
                    continue
            
            if temps:
                max_temp = max(temps)
                min_temp = min(temps)
                avg_temp = sum(temps) / len(temps)
                
                if avg_temp < 10:
                    result += "• 预报期间天气较冷，请注意保暖。\n"
                elif avg_temp > 30:
                    result += "• 预报期间天气较热，请注意防暑。\n"
                
                if max_temp - min_temp > 10:
                    result += "• 温差较大，请注意适时增减衣物。\n"
            
            return result
            
        except Exception as e:
            logger.error(f"格式化天气预报信息失败: {str(e)}")
            return f"预报数据格式化失败: {str(e)}"
    
    def get_supported_cities(self) -> List[str]:
        """获取支持的城市列表（示例）"""
        return [
            "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "成都",
            "重庆", "武汉", "西安", "天津", "青岛", "大连", "厦门", "宁波",
            "无锡", "福州", "济南", "长沙", "哈尔滨", "长春", "沈阳", "石家庄"
        ]


class WeatherTools:
    """天气工具包装类"""
    
    def __init__(self):
        self.weather_service = WeatherService()
        
    def get_current_weather(self, city: str) -> str:
        """获取当前天气 - 工具接口"""
        """
        查询指定城市的当前天气信息。
        
        Args:
            city: 城市名称，如"北京"、"上海"等
            
        Returns:
            格式化的天气信息字符串
        """
        return self.weather_service.get_current_weather(city)
    
    def get_weather_forecast(self, city: str, days: int = 3) -> str:
        """获取天气预报 - 工具接口"""
        """
        查询指定城市的天气预报信息。
        
        Args:
            city: 城市名称，如"北京"、"上海"等
            days: 预报天数，1-7天，默认为3天
            
        Returns:
            格式化的天气预报信息字符串
        """
        return self.weather_service.get_weather_forecast(city, days)
    
    def get_supported_cities(self) -> List[str]:
        """获取支持的城市列表"""
        return self.weather_service.get_supported_cities()