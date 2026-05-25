"""
虚拟 mindgo_api 模块 - 用于替换外部数据API，改为本地数据调用
当策略尝试调用mindgo_api函数时，会抛出明确的错误信息，提示使用本地数据。
"""

import sys
import warnings

class MindGoAPIError(ImportError):
    """MindGo API 错误"""
    pass

def __getattr__(name):
    """当访问未定义的属性时触发"""
    raise MindGoAPIError(
        f"mindgo_api.{name} 不可用。请改为使用本地数据调用。\n"
        "原策略依赖外部mindgo_api，现已改为本地数据源。\n"
        "请修改策略代码，使用本地CSV文件或数据库中的数据。\n"
        "项目数据目录: data/daily_data2/, data/week_data2/, data/5min/, data/30min/\n"
        "示例: 使用pandas读取CSV: pd.read_csv('data/daily_data2/000001.SZ.csv')"
    )

def __dir__():
    """返回虚拟的API函数列表"""
    return [
        'get_price', 'get_data', 'get_history', 'get_realtime',
        'get_ticks', 'get_bars', 'get_fundamental', 'get_industry',
        'get_index', 'get_futures', 'get_options', 'get_bonds'
    ]

# 发出警告
warnings.warn(
    "mindgo_api 已被虚拟模块替换。请将策略改为使用本地数据调用。",
    DeprecationWarning,
    stacklevel=2
)

# 提供一些占位函数，调用时抛出错误
def get_price(*args, **kwargs):
    raise MindGoAPIError("get_price 不可用。请使用本地数据调用。")

def get_data(*args, **kwargs):
    raise MindGoAPIError("get_data 不可用。请使用本地数据调用。")

def get_history(*args, **kwargs):
    raise MindGoAPIError("get_history 不可用。请使用本地数据调用。")

def get_realtime(*args, **kwargs):
    raise MindGoAPIError("get_realtime 不可用。请使用本地数据调用。")

# 导出这些函数
__all__ = ['get_price', 'get_data', 'get_history', 'get_realtime']