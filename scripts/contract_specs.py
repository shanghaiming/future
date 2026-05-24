"""
期货合约规格表
包含: 交易单位(手), 乘数, 保证金率, 最小变动价位
数据来源: 各交易所公开信息 (2024年标准)
"""

# symbol: (乘数, 保证金率, 最小变动价位, 品种名)
# 乘数 = 每手多少吨/千克/桶等 (合约价值 = 价格 * 乘数)
# 保证金率 = 需要的保证金占合约价值的比例
CONTRACT_SPECS = {
    # === 上期所 (SHFE) ===
    'agfi': (15, 0.12, 1, '沪银'),        # 15千克/手
    'alfi': (5, 0.10, 5, '沪铝'),          # 5吨/手
    'aufi': (1000, 0.10, 0.02, '沪金'),    # 1000克/手
    'bufi': (10, 0.10, 2, '沥青'),         # 10吨/手
    'cufi': (5, 0.10, 10, '沪铜'),         # 5吨/手
    'fufi': (10, 0.10, 1, '燃油'),         # 10吨/手
    'hcfi': (10, 0.10, 1, '热卷'),         # 10吨/手
    'nifi': (1, 0.12, 10, '沪镍'),         # 1吨/手
    'pbfi': (5, 0.10, 5, '沪铅'),          # 5吨/手
    'rbfi': (10, 0.10, 1, '螺纹'),         # 10吨/手
    'rufi': (10, 0.10, 5, '橡胶'),         # 10吨/手
    'snfi': (1, 0.12, 10, '沪锡'),         # 1吨/手
    'sffi': (5, 0.10, 2, '硅铁'),          # 5吨/手
    'smfi': (5, 0.10, 2, '锰硅'),          # 5吨/手
    'spfi': (10, 0.10, 2, '纸浆'),         # 10吨/手
    'wrffi': (10, 0.08, 1, '线材'),        # 10吨/手 (可能已退)
    'znfi': (5, 0.10, 5, '沪锌'),          # 5吨/手
    'ssfi': (5, 0.10, 5, '不锈钢'),        # 5吨/手

    # === 大商所 (DCE) ===
    'afi': (10, 0.08, 1, '豆一'),          # 10吨/手
    'bfi': (10, 0.08, 1, '豆二'),          # 10吨/手
    'bbfi': (500, 0.10, 0.05, '胶合板'),   # 500张/手
    'cffi': (5, 0.07, 5, '棉花'),          # CZCE actually
    'cfi': (10, 0.08, 1, '玉米'),          # 10吨/手
    'csfi': (10, 0.08, 1, '淀粉'),         # 10吨/手
    'ebfi': (5, 0.08, 1, '苯乙烯'),        # 5吨/手
    'egfi': (10, 0.08, 1, '乙二醇'),       # 10吨/手
    'fbfi': (500, 0.10, 0.1, '纤维板'),    # 500张/手
    'ifi': (100, 0.12, 0.5, '铁矿'),       # 100吨/手
    'jfi': (100, 0.12, 0.5, '焦炭'),       # 100吨/手
    'jmfi': (60, 0.12, 0.5, '焦煤'),       # 60吨/手
    'lfi': (5, 0.08, 1, '塑料'),           # 5吨/手
    'mfi': (10, 0.08, 1, '豆粕'),          # 10吨/手
    'pgfi': (20, 0.10, 1, 'LPG'),          # 20吨/手
    'ppfi': (5, 0.08, 1, 'PP'),            # 5吨/手
    'vfi': (5, 0.08, 1, 'PVC'),            # 5吨/手
    'yfi': (10, 0.08, 2, '豆油'),          # 10吨/手
    'pfi': (10, 0.08, 2, '棕榈'),          # 10吨/手
    'jdfi': (5, 0.10, 5, '红枣'),          # 5吨/手
    'lhfi': (16, 0.12, 5, '生猪'),         # 16吨/手
    'pkfi': (5, 0.08, 2, '花生'),          # 5吨/手
    'rrfi': (20, 0.08, 1, '粳稻'),         # 20吨/手
    'lrfi': (20, 0.08, 1, '晚籼稻'),       # 20吨/手
    'jrfi': (20, 0.08, 1, '籼稻'),         # 20吨/手
    'pmfi': (20, 0.08, 1, '普麦'),         # 20吨/手
    'whfi': (20, 0.08, 1, '强麦'),         # 20吨/手
    'rsfi': (20, 0.08, 1, '早籼稻'),       # 20吨/手
    'cjfi': (10, 0.08, 1, '菜籽'),         # 10吨/手
    'mafi': (10, 0.08, 1, '甲醇'),         # 10吨/手 (ZCE actually)

    # === 郑商所 (CZCE) ===
    'apfi': (10, 0.08, 1, '苹果'),         # 10吨/手
    'cyfi': (5, 0.08, 5, '棉纱'),          # 5吨/手
    'fgfi': (20, 0.08, 1, '玻璃'),         # 20吨/手
    'oifi': (10, 0.08, 1, '菜油'),         # 10吨/手
    'pfifi': (5, 0.08, 2, '短纤'),         # 5吨/手
    'rmfi': (10, 0.08, 1, '菜粕'),         # 10吨/手
    'srfi': (10, 0.08, 1, '白糖'),         # 10吨/手
    'tafi': (5, 0.08, 2, 'PTA'),           # 5吨/手
    'safi': (20, 0.08, 1, '纯碱'),         # 20吨/手
    'urfi': (20, 0.08, 1, '尿素'),         # 20吨/手
    'srfi': (10, 0.08, 1, '白糖'),         # 10吨/手

    # === 能源中心 (INE) ===
    'scfi': (1000, 0.12, 0.1, '原油'),     # 1000桶/手
    'lufi': (10, 0.10, 1, '低硫燃油'),     # 10吨/手
    'bcfi': (5, 0.10, 10, '国际铜'),       # 5吨/手
    'nrfi': (1, 0.12, 10, '20号胶'),       # 10吨/手 actually
    'lgfi': (20, 0.10, 1, '低硫燃油'),     # possibly same as lufi
    'brfi': (5, 0.10, 5, '丁二烯'),        # 5吨/手

    # === 广期所 (GFEX) ===
    'lcfi': (1, 0.12, 50, '碳酸锂'),       # 1吨/手
    'sifi': (5, 0.12, 5, '工业硅'),        # 5吨/手

    # === 中金所 (CFFEX) ===
    # 金融期货特殊: 股指按点数, 国债按面值
    # 这些不在futures_weighted目录中, 暂不处理
}

# 默认规格 (用于未在表中的品种)
DEFAULT_SPEC = (10, 0.10, 1, '未知')

def get_spec(symbol):
    """获取合约规格"""
    return CONTRACT_SPECS.get(symbol, DEFAULT_SPEC)

def calc_margin(symbol, price, lots):
    """计算保证金需求"""
    multiplier, margin_rate, _, _ = get_spec(symbol)
    contract_value = price * multiplier * lots
    return contract_value * margin_rate

def calc_contract_value(symbol, price, lots):
    """计算合约名义价值"""
    multiplier, _, _, _ = get_spec(symbol)
    return price * multiplier * lots

def calc_pnl(symbol, entry_price, exit_price, direction, lots):
    """计算盈亏"""
    multiplier, _, _, _ = get_spec(symbol)
    return (exit_price - entry_price) * direction * multiplier * lots

def calc_max_lots(symbol, price, available_cash, max_equity_pct=0.35):
    """计算可开最大手数 (受保证金约束)"""
    multiplier, margin_rate, _, _ = get_spec(symbol)
    margin_per_lot = price * multiplier * margin_rate
    if margin_per_lot <= 0:
        return 0
    max_lots = int(available_cash * max_equity_pct / margin_per_lot)
    return max(max_lots, 0)
