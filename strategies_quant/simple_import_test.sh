#!/bin/bash
# 简单导入测试脚本
# 测试策略能否导入和实例化

echo "🚀 策略导入测试"
echo "=========================="

STRATEGY_DIR="/Users/chengming/.openclaw/workspace/quant_trade-main/strategies"
RESULTS_FILE="import_test_results_$(date +%Y%m%d_%H%M%S).csv"

# 创建结果文件
echo "file_name,has_basestrategy,has_generate_signals,import_success,instance_success,strategy_class,error_message" > "$RESULTS_FILE"

# 测试文件列表（关键策略）
TEST_FILES="ma_strategy.py tradingview_strategy.py market_structure_identifier.py exit_strategy_optimizer.py advanced_entry_techniques.py advanced_risk_management_system.py case_study_analyzer.py common_errors_avoidance_system.py continuous_improvement_system.py csv_auto_select.py"

TOTAL_FILES=$(echo "$TEST_FILES" | wc -w)
COUNT=0

echo "测试文件数: $TOTAL_FILES"
echo ""

for FILE_NAME in $TEST_FILES; do
    COUNT=$((COUNT + 1))
    FILE_PATH="$STRATEGY_DIR/$FILE_NAME"
    
    echo "[$COUNT/$TOTAL_FILES] 测试: $FILE_NAME"
    
    if [ ! -f "$FILE_PATH" ]; then
        echo "  ❌ 文件不存在"
        echo "$FILE_NAME,0,0,0,0,,File not found" >> "$RESULTS_FILE"
        continue
    fi
    
    # 检查文件结构
    HAS_BASESTRATEGY=$(grep -c "BaseStrategy" "$FILE_PATH")
    HAS_GENERATE_SIGNALS=$(grep -c "def generate_signals" "$FILE_PATH")
    
    echo -n "  结构检查: "
    if [ "$HAS_BASESTRATEGY" -gt 0 ] && [ "$HAS_GENERATE_SIGNALS" -gt 0 ]; then
        echo "✅"
    else
        echo "❌"
        echo "$FILE_NAME,$HAS_BASESTRATEGY,$HAS_GENERATE_SIGNALS,0,0,,Structure incomplete" >> "$RESULTS_FILE"
        continue
    fi
    
    # 导入测试
    echo -n "  导入测试: "
    
    IMPORT_TEST=$(cd "$STRATEGY_DIR" && python3 -c "
import sys
sys.path.append('.')

try:
    module_name = '$FILE_NAME'.replace('.py', '')
    module = __import__(module_name)
    print('IMPORT_SUCCESS')
    
    # 查找策略类
    strategy_class = None
    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
            if hasattr(attr, '__mro__') and attr_name.endswith('Strategy') and attr_name != 'BaseStrategy':
                strategy_class = attr_name
                break
        except:
            continue
    
    if strategy_class:
        print('CLASS_FOUND:' + strategy_class)
        
        # 尝试实例化（不加载数据）
        try:
            # 创建简单数据
            import pandas as pd
            import numpy as np
            dates = pd.date_range('2021-01-01', periods=100, freq='D')
            data = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 101,
                'low': np.random.randn(100).cumsum() + 99,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
            
            StrategyClass = getattr(module, strategy_class)
            params = {'symbol': 'TEST', 'risk_per_trade': 0.02}
            instance = StrategyClass(data, params)
            print('INSTANCE_SUCCESS')
        except Exception as e:
            print('INSTANCE_FAILED:' + str(e))
    else:
        print('NO_CLASS_FOUND')
        
except ImportError as e:
    print('IMPORT_FAILED:' + str(e))
except Exception as e:
    print('GENERAL_ERROR:' + str(e))
" 2>&1)
    
    # 解析输出
    IMPORT_SUCCESS=0
    INSTANCE_SUCCESS=0
    STRATEGY_CLASS=""
    ERROR_MSG=""
    
    if echo "$IMPORT_TEST" | grep -q "IMPORT_SUCCESS"; then
        IMPORT_SUCCESS=1
        echo "✅"
        
        # 提取策略类
        if echo "$IMPORT_TEST" | grep -q "CLASS_FOUND:"; then
            STRATEGY_CLASS=$(echo "$IMPORT_TEST" | grep "CLASS_FOUND:" | cut -d':' -f2)
            echo "    策略类: $STRATEGY_CLASS"
            
            if echo "$IMPORT_TEST" | grep -q "INSTANCE_SUCCESS"; then
                INSTANCE_SUCCESS=1
                echo "    实例化: ✅"
            elif echo "$IMPORT_TEST" | grep -q "INSTANCE_FAILED:"; then
                ERROR_MSG=$(echo "$IMPORT_TEST" | grep "INSTANCE_FAILED:" | cut -d':' -f2-)
                echo "    实例化: ❌ ($ERROR_MSG)"
            fi
        else
            echo "    策略类: ❌ 未找到"
            ERROR_MSG="No strategy class found"
        fi
    elif echo "$IMPORT_TEST" | grep -q "IMPORT_FAILED:"; then
        ERROR_MSG=$(echo "$IMPORT_TEST" | grep "IMPORT_FAILED:" | cut -d':' -f2-)
        echo "❌ ($ERROR_MSG)"
    elif echo "$IMPORT_TEST" | grep -q "GENERAL_ERROR:"; then
        ERROR_MSG=$(echo "$IMPORT_TEST" | grep "GENERAL_ERROR:" | cut -d':' -f2-)
        echo "❌ ($ERROR_MSG)"
    else
        ERROR_MSG="Unknown error"
        echo "❌ (未知错误)"
    fi
    
    # 记录结果
    echo "$FILE_NAME,$HAS_BASESTRATEGY,$HAS_GENERATE_SIGNALS,$IMPORT_SUCCESS,$INSTANCE_SUCCESS,$STRATEGY_CLASS,$ERROR_MSG" >> "$RESULTS_FILE"
    
    echo ""
done

echo "=========================="
echo "🏁 导入测试完成"
echo "结果保存到: $RESULTS_FILE"

# 生成摘要
echo ""
echo "📊 测试摘要:"
TOTAL_TESTS=$TOTAL_FILES
STRUCTURE_OK=$(awk -F',' 'NR>1 && $2>0 && $3>0 {count++} END {print count+0}' "$RESULTS_FILE")
IMPORT_OK=$(awk -F',' 'NR>1 && $4==1 {count++} END {print count+0}' "$RESULTS_FILE")
INSTANCE_OK=$(awk -F',' 'NR>1 && $5==1 {count++} END {print count+0}' "$RESULTS_FILE")

echo "总测试策略: $TOTAL_TESTS"
echo "结构完整: $STRUCTURE_OK"
echo "导入成功: $IMPORT_OK"
echo "实例化成功: $INSTANCE_OK"

# 显示成功策略
echo ""
echo "✅ 成功策略:"
awk -F',' 'NR>1 && $5==1 {print "  - " $1 " (" $6 ")"}' "$RESULTS_FILE"

# 显示失败策略
echo ""
echo "❌ 失败策略:"
awk -F',' 'NR>1 && $5==0 {print "  - " $1 ": " $7}' "$RESULTS_FILE" | head -10