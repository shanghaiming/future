#!/bin/bash
# 简单批量策略测试脚本
# 逐个测试策略文件，统计信号数量
# 在后台运行，避免复杂Python脚本

echo "🚀 开始逐个测试策略文件"
echo "=========================="

STRATEGY_DIR="/Users/chengming/.openclaw/workspace/quant_trade-main/strategies"
DATA_FILE="/Users/chengming/.openclaw/workspace/quant_trade-main/data/daily_data2/000001.SZ.csv"
RESULTS_FILE="strategy_test_results_$(date +%Y%m%d_%H%M%S).csv"

# 创建结果文件
echo "file_name,has_basestrategy,has_generate_signals,signals_generated,import_success,test_result" > "$RESULTS_FILE"

# 获取所有Python文件
FILES=$(ls "$STRATEGY_DIR"/*.py | grep -v "__init__.py" | grep -v "base_strategy.py" | grep -v "simple_test.py" | grep -v "structure_validator.py")

TOTAL_FILES=$(echo "$FILES" | wc -l | tr -d ' ')
COUNT=0

echo "总文件数: $TOTAL_FILES"
echo ""

for FILE in $FILES; do
    COUNT=$((COUNT + 1))
    FILE_NAME=$(basename "$FILE")
    
    echo "[$COUNT/$TOTAL_FILES] 测试: $FILE_NAME"
    
    # 检查文件结构
    HAS_BASESTRATEGY=$(grep -c "BaseStrategy" "$FILE")
    HAS_GENERATE_SIGNALS=$(grep -c "def generate_signals" "$FILE")
    
    # 尝试导入测试
    IMPORT_SUCCESS=0
    SIGNALS_GENERATED="N/A"
    TEST_RESULT="SKIPPED"
    
    # 只测试有BaseStrategy和generate_signals的文件
    if [ "$HAS_BASESTRATEGY" -gt 0 ] && [ "$HAS_GENERATE_SIGNALS" -gt 0 ]; then
        echo "  结构检查: ✅ BaseStrategy + generate_signals"
        
        # 尝试简单导入测试
        IMPORT_TEST=$(cd "$STRATEGY_DIR" && python3 -c "
import sys
sys.path.append('.')
try:
    module_name = '$FILE_NAME'.replace('.py', '')
    spec = __import__('importlib.util').spec_from_file_location(module_name, '$FILE')
    if spec:
        module = __import__('importlib.util').module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        print('SUCCESS')
    else:
        print('FAILED: No spec')
except Exception as e:
    print(f'FAILED: {e}')
" 2>&1 | grep -E "SUCCESS|FAILED" | head -1)
        
        if echo "$IMPORT_TEST" | grep -q "SUCCESS"; then
            IMPORT_SUCCESS=1
            TEST_RESULT="IMPORT_OK"
            echo "  导入测试: ✅ 成功"
        else
            TEST_RESULT="IMPORT_FAILED"
            echo "  导入测试: ❌ 失败"
        fi
    else
        echo "  结构检查: ❌ 缺少BaseStrategy或generate_signals"
        TEST_RESULT="STRUCTURE_INCOMPLETE"
    fi
    
    # 记录结果
    echo "$FILE_NAME,$HAS_BASESTRATEGY,$HAS_GENERATE_SIGNALS,$SIGNALS_GENERATED,$IMPORT_SUCCESS,$TEST_RESULT" >> "$RESULTS_FILE"
    
    echo ""
done

echo "=========================="
echo "🏁 测试完成"
echo "结果保存到: $RESULTS_FILE"

# 生成摘要
echo ""
echo "📊 测试摘要:"
echo "总测试文件: $TOTAL_FILES"
COMPLETE_FILES=$(awk -F',' 'NR>1 && $2>0 && $3>0 {count++} END {print count+0}' "$RESULTS_FILE")
echo "完整结构文件: $COMPLETE_FILES"
IMPORT_OK_FILES=$(awk -F',' 'NR>1 && $5==1 {count++} END {print count+0}' "$RESULTS_FILE")
echo "导入成功文件: $IMPORT_OK_FILES"

# 显示不完整文件
echo ""
echo "⚠️ 不完整文件列表:"
awk -F',' 'NR>1 && ($2==0 || $3==0) {print "  - " $1}' "$RESULTS_FILE"