from core.base_strategy import BaseStrategy
#!/usr/bin/env python3
"""
策略文件结构验证器
只读取文件内容，不执行代码，避免安全限制
"""

import os
import re
import sys

def check_file_structure(file_path):
    """检查文件结构"""
    file_name = os.path.basename(file_path)
    result = {
        'file': file_name,
        'has_basestrategy_import': False,
        'has_basestrategy_class': False,
        'has_generate_signals': False,
        'strategy_classes': [],
        'line_count': 0,
        'file_size': 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result['file_size'] = len(content)
        result['line_count'] = content.count('\n') + 1
        
        # 检查BaseStrategy导入
        result['has_basestrategy_import'] = 'BaseStrategy' in content
        
        # 查找类定义
        class_pattern = r'class\s+(\w+)\s*\([^)]*BaseStrategy[^)]*\)'
        class_matches = re.findall(class_pattern, content)
        
        if class_matches:
            result['has_basestrategy_class'] = True
            result['strategy_classes'] = class_matches
        
        # 检查generate_signals方法
        if 'def generate_signals' in content:
            result['has_generate_signals'] = True
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result

def main():
    """主验证函数"""
    strategies_dir = "/Users/chengming/.openclaw/workspace/quant_trade-main/strategies"
    
    print("=" * 80)
    print("📋 策略文件结构验证报告")
    print("=" * 80)
    print(f"验证目录: {strategies_dir}")
    print()
    
    # 收集所有Python文件
    python_files = []
    for file_name in os.listdir(strategies_dir):
        if file_name.endswith('.py') and file_name not in ['__init__.py', 'base_strategy.py', 'structure_validator.py']:
            python_files.append(os.path.join(strategies_dir, file_name))
    
    print(f"发现 {len(python_files)} 个策略文件")
    print()
    
    # 验证每个文件
    results = []
    for i, file_path in enumerate(sorted(python_files)):
        file_name = os.path.basename(file_path)
        print(f"[{i+1:3d}/{len(python_files)}] 验证: {file_name[:40]:40s}", end='')
        
        result = check_file_structure(file_path)
        results.append(result)
        
        # 打印简单状态
        status_chars = []
        if result.get('has_basestrategy_import', False):
            status_chars.append('I')
        else:
            status_chars.append('-')
        
        if result.get('has_basestrategy_class', False):
            status_chars.append('C')
        else:
            status_chars.append('-')
        
        if result.get('has_generate_signals', False):
            status_chars.append('G')
        else:
            status_chars.append('-')
        
        print(f"  [{''.join(status_chars)}]")
        
        if 'error' in result:
            print(f"     错误: {result['error']}")
    
    print()
    print("=" * 80)
    print("📊 验证统计")
    print("=" * 80)
    
    # 统计
    total_files = len(results)
    has_import = sum(1 for r in results if r.get('has_basestrategy_import', False))
    has_class = sum(1 for r in results if r.get('has_basestrategy_class', False))
    has_method = sum(1 for r in results if r.get('has_generate_signals', False))
    
    print(f"总文件数: {total_files}")
    print(f"有BaseStrategy导入: {has_import} ({has_import/total_files*100:.1f}%)")
    print(f"有BaseStrategy类: {has_class} ({has_class/total_files*100:.1f}%)")
    print(f"有generate_signals方法: {has_method} ({has_method/total_files*100:.1f}%)")
    print()
    
    # 分类统计
    print("📁 文件分类:")
    
    complete_strategies = []
    partial_strategies = []
    incomplete_strategies = []
    
    for result in results:
        if result.get('has_basestrategy_class', False) and result.get('has_generate_signals', False):
            complete_strategies.append(result)
        elif result.get('has_basestrategy_class', False) or result.get('has_generate_signals', False):
            partial_strategies.append(result)
        else:
            incomplete_strategies.append(result)
    
    print(f"  完整策略 (有类+方法): {len(complete_strategies)}个")
    print(f"  部分策略 (有类或方法): {len(partial_strategies)}个")
    print(f"  不完整文件: {len(incomplete_strategies)}个")
    print()
    
    # 显示不完整文件
    if incomplete_strategies:
        print("⚠️ 不完整文件列表:")
        for result in incomplete_strategies[:10]:
            print(f"  - {result['file']} (I:{result.get('has_basestrategy_import', False)} C:{result.get('has_basestrategy_class', False)} G:{result.get('has_generate_signals', False)})")
        
        if len(incomplete_strategies) > 10:
            print(f"  ... 和其他 {len(incomplete_strategies) - 10} 个文件")
        print()
    
    # 显示完整策略示例
    if complete_strategies:
        print("✅ 完整策略示例 (前10个):")
        for result in complete_strategies[:10]:
            classes = result.get('strategy_classes', [])
            class_str = classes[0] if classes else '未知'
            print(f"  - {result['file']} ({class_str})")
        
        if len(complete_strategies) > 10:
            print(f"  ... 和其他 {len(complete_strategies) - 10} 个完整策略")
    
    print()
    print("=" * 80)
    print("🔑 状态说明:")
    print("  I = 有BaseStrategy导入")
    print("  C = 有BaseStrategy类")
    print("  G = 有generate_signals方法")
    print("  [ICG] = 完整策略文件")
    print("  [I--] = 只有导入，无类")
    print("  [-C-] = 有类，无方法")
    print("  [--G] = 有方法，无BaseStrategy类")
    print("=" * 80)

if __name__ == "__main__":
    main()


class StructureValidatorStrategy(BaseStrategy):
    """基于structure_validator的策略"""
    
    def __init__(self, data, params=None):
        super().__init__(data, params)
        # 初始化代码
        self.name = "StructureValidatorStrategy"
        self.description = "基于structure_validator的策略"
        
    def generate_signals(self):
        """Swing high/low breakout (BOS). Buy on break above swing high, sell on break below swing low."""
        df = self.data

        if len(df) < 10:
            return self.signals

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        n = len(df)
        lookback = 3

        # Detect swing highs and swing lows
        swing_highs = []
        swing_lows = []
        for i in range(lookback, n - lookback):
            is_swing_high = all(highs[i] >= highs[i + j] for j in range(-lookback, lookback + 1) if j != 0)
            is_swing_low = all(lows[i] <= lows[i + j] for j in range(-lookback, lookback + 1) if j != 0)
            if is_swing_high:
                swing_highs.append((i, highs[i]))
            if is_swing_low:
                swing_lows.append((i, lows[i]))

        # Track last swing high/low for BOS signals
        last_swing_high = None
        last_swing_low = None
        sh_idx = 0
        sl_idx = 0

        for i in range(2 * lookback, n):
            # Update last swing high
            while sh_idx < len(swing_highs) and swing_highs[sh_idx][0] < i:
                last_swing_high = swing_highs[sh_idx][1]
                sh_idx += 1
            # Update last swing low
            while sl_idx < len(swing_lows) and swing_lows[sl_idx][0] < i:
                last_swing_low = swing_lows[sl_idx][1]
                sl_idx += 1

            price = closes[i]
            # BOS bullish: close breaks above last swing high
            if last_swing_high is not None and price > last_swing_high and closes[i - 1] <= last_swing_high:
                self._record_signal(df.index[i], 'buy', price=float(price))
            # BOS bearish: close breaks below last swing low
            if last_swing_low is not None and price < last_swing_low and closes[i - 1] >= last_swing_low:
                self._record_signal(df.index[i], 'sell', price=float(price))

        return self.signals
