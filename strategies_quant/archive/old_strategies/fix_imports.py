#!/usr/bin/env python3
"""修复策略文件导入语句
"""

import os
import re
import sys

# 将项目根目录添加到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# BaseStrategy导入
try:
    from core.base_strategy import BaseStrategy
except ImportError:
    # 尝试相对导入
    from core.base_strategy import BaseStrategy

def fix_imports_in_file(file_path):
    """修复单个文件的导入语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否需要修复
    if 'from base_strategy import BaseStrategy' in content:
        # 修复导入
        new_content = content.replace(
            'from base_strategy import BaseStrategy',
            'try:\n    from core.base_strategy import BaseStrategy\nexcept ImportError:\n    from core.base_strategy import BaseStrategy'
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True, "导入已修复"
    elif 'import base_strategy' in content:
        # 另一种可能的导入方式
        new_content = content.replace(
            'import base_strategy',
            'try:\n    from core.base_strategy import BaseStrategy\nexcept ImportError:\n    from core.base_strategy import BaseStrategy'
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True, "导入已修复"
    else:
        return False, "无需修复"

def main():
    """主函数"""
    strategies_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🔧 开始修复策略文件导入语句")
    print("=" * 60)
    
    # 获取所有Python文件
    python_files = []
    for file_name in os.listdir(strategies_dir):
        if file_name.endswith('.py') and file_name not in ['__init__.py', 'base_strategy.py', 'fix_imports.py', 'simple_batch_test.sh', 'structure_validator.py', 'simple_test.py']:
            python_files.append(os.path.join(strategies_dir, file_name))
    
    print(f"发现 {len(python_files)} 个策略文件")
    print()
    
    fixed_count = 0
    for i, file_path in enumerate(sorted(python_files)):
        file_name = os.path.basename(file_path)
        print(f"[{i+1}/{len(python_files)}] 处理: {file_name[:30]:30s}", end='')
        
        try:
            fixed, message = fix_imports_in_file(file_path)
            if fixed:
                fixed_count += 1
                print(f" ✅ {message}")
            else:
                print(f" ⏭️ {message}")
        except Exception as e:
            print(f" ❌ 错误: {e}")
    
    print()
    print("=" * 60)
    print(f"修复完成: {fixed_count}/{len(python_files)} 个文件已修复")
    
    # 验证修复
    print()
    print("🔍 验证修复效果:")
    
    # 测试一个文件
    test_file = os.path.join(strategies_dir, 'ma_strategy.py')
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'from base_strategy import BaseStrategy' in content:
            print("  ma_strategy.py: ✅ 导入已修复")
        else:
            print("  ma_strategy.py: ❌ 导入未修复")

if __name__ == "__main__":
    main()


class FixImportsStrategy(BaseStrategy):
    """基于fix_imports的策略"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化代码
        self.name = "FixImportsStrategy"
        self.description = "基于fix_imports的策略"
        
    def calculate_signals(self, df):
        """计算交易信号"""
        # 策略逻辑
        return df
        
    def generate_signals(self, df):
        """生成交易信号"""
        # 信号生成逻辑
        return df