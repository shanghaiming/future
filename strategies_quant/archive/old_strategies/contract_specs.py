"""
中国期货合约规格表 (Contract Specifications)
=============================================
数据来源: 各交易所官网 + 7chabao.com 实时数据
更新日期: 2026-05-25

用途: paper_trading P&L 计算
  - margin = price * multiplier * margin_rate
  - tick_profit = tick_size * multiplier
  - pnl = (exit_price - entry_price) * multiplier

注意:
  - margin_rate 为交易所最低保证金率, 实际期货公司会加收2-5%
  - 交易所会根据市场波动/节假日/临近交割月临时调整保证金率
  - tick_size 和 multiplier 以交易所最新规则为准
"""

# ============================================================
# 合约规格主表
# 格式: (exchange, multiplier, tick_size, margin_rate_min, price_unit, name_cn)
#   exchange: SHFE / DCE / CZCE / CFFEX / GFEX / INE
#   multiplier: 合约乘数 (每手多少吨/克/桶/点)
#   tick_size: 最小变动价位 (元/吨, 元/克, 元/桶, 点)
#   margin_rate_min: 交易所最低保证金率 (小数)
#   price_unit: 报价单位说明
#   name_cn: 中文名称
# ============================================================

CONTRACT_SPECS = {
    # ==================== SHFE 上海期货交易所 ====================
    'rb':   ('SHFE', 10,  1,    0.07, '元/吨',   '螺纹钢'),
    'hc':   ('SHFE', 10,  1,    0.07, '元/吨',   '热轧卷板'),
    'cu':   ('SHFE', 5,   10,   0.05, '元/吨',   '阴极铜'),
    'al':   ('SHFE', 5,   5,    0.05, '元/吨',   '铝'),
    'zn':   ('SHFE', 5,   5,    0.05, '元/吨',   '锌'),
    'pb':   ('SHFE', 5,   5,    0.05, '元/吨',   '铅'),
    'ni':   ('SHFE', 1,   10,   0.05, '元/吨',   '镍'),
    'sn':   ('SHFE', 1,   10,   0.05, '元/吨',   '锡'),
    'au':   ('SHFE', 1000, 0.02, 0.05, '元/克',   '黄金'),    # 1000克/手
    'ag':   ('SHFE', 15,  1,    0.05, '元/千克', '白银'),    # 15千克/手
    'ru':   ('SHFE', 10,  5,    0.05, '元/吨',   '天然橡胶'),
    'fu':   ('SHFE', 10,  1,    0.05, '元/吨',   '燃料油'),
    'bu':   ('SHFE', 10,  2,    0.07, '元/吨',   '石油沥青'),
    'sp':   ('SHFE', 10,  2,    0.05, '元/吨',   '纸浆'),
    'ss':   ('SHFE', 5,   5,    0.05, '元/吨',   '不锈钢'),
    'wr':   ('SHFE', 10,  1,    0.07, '元/吨',   '线材'),
    'ao':   ('SHFE', 20,  1,    0.07, '元/吨',   '氧化铝'),
    'br':   ('SHFE', 5,   5,    0.05, '元/吨',   '丁二烯橡胶'),
    'op':   ('SHFE', 40,  1,    0.05, '元/吨',   '胶版印刷纸'),

    # ==================== DCE 大连商品交易所 ====================
    'a':    ('DCE', 10,  1,    0.05, '元/吨',   '豆一'),
    'b':    ('DCE', 10,  1,    0.05, '元/吨',   '豆二'),
    'c':    ('DCE', 10,  1,    0.05, '元/吨',   '玉米'),
    'cs':   ('DCE', 10,  1,    0.05, '元/吨',   '玉米淀粉'),
    'm':    ('DCE', 10,  1,    0.05, '元/吨',   '豆粕'),
    'y':    ('DCE', 10,  2,    0.05, '元/吨',   '豆油'),
    'p':    ('DCE', 10,  2,    0.05, '元/吨',   '棕榈油'),
    'i':    ('DCE', 100, 0.5,  0.05, '元/吨',   '铁矿石'),
    'j':    ('DCE', 100, 0.5,  0.05, '元/吨',   '焦炭'),
    'jm':   ('DCE', 60,  0.5,  0.05, '元/吨',   '焦煤'),
    'l':    ('DCE', 5,   1,    0.05, '元/吨',   '聚乙烯'),
    'v':    ('DCE', 5,   1,    0.05, '元/吨',   '聚氯乙烯'),
    'pp':   ('DCE', 5,   1,    0.05, '元/吨',   '聚丙烯'),
    'eg':   ('DCE', 10,  1,    0.05, '元/吨',   '乙二醇'),
    'eb':   ('DCE', 5,   1,    0.05, '元/吨',   '苯乙烯'),
    'pg':   ('DCE', 20,  1,    0.05, '元/吨',   '液化石油气'),
    'lh':   ('DCE', 16,  5,    0.05, '元/吨',   '生猪'),
    'jd':   ('DCE', 5,   1,    0.05, '元/500kg','鸡蛋'),    # 10吨/手 新规 vs 5吨旧规
    'fb':   ('DCE', 10,  0.5,  0.05, '元/立方米','纤维板'),  # 单位特殊
    'bb':   ('DCE', 500, 0.05, 0.05, '元/张',   '胶合板'),
    'lg':   ('DCE', 90,  0.5,  0.05, '元/立方米','原木'),

    # ==================== CZCE 郑州商品交易所 ====================
    'cf':   ('CZCE', 5,   5,    0.05, '元/吨',   '棉花'),
    'sr':   ('CZCE', 10,  1,    0.05, '元/吨',   '白糖'),
    'ta':   ('CZCE', 5,   2,    0.05, '元/吨',   'PTA'),
    'oi':   ('CZCE', 10,  1,    0.05, '元/吨',   '菜籽油'),
    'rm':   ('CZCE', 10,  1,    0.05, '元/吨',   '菜籽粕'),
    'fg':   ('CZCE', 20,  1,    0.05, '元/吨',   '玻璃'),
    'sa':   ('CZCE', 20,  1,    0.05, '元/吨',   '纯碱'),
    'sf':   ('CZCE', 5,   2,    0.05, '元/吨',   '硅铁'),
    'sm':   ('CZCE', 5,   2,    0.05, '元/吨',   '锰硅'),
    'ap':   ('CZCE', 10,  1,    0.05, '元/吨',   '苹果'),
    'cj':   ('CZCE', 5,   5,    0.05, '元/吨',   '红枣'),
    'ma':   ('CZCE', 10,  1,    0.05, '元/吨',   '甲醇'),
    'ur':   ('CZCE', 20,  1,    0.05, '元/吨',   '尿素'),
    'pf':   ('CZCE', 5,   2,    0.05, '元/吨',   '短纤(涤纶短纤)'),
    'sh':   ('CZCE', 30,  1,    0.05, '元/吨',   '烧碱'),
    'pk':   ('CZCE', 5,   2,    0.05, '元/吨',   '花生'),
    'cy':   ('CZCE', 5,   5,    0.05, '元/吨',   '棉纱'),
    'zc':   ('CZCE', 100, 0.2,  0.05, '元/吨',   '动力煤'),
    'pr':   ('CZCE', 15,  1,    0.05, '元/吨',   '瓶片PET'),    # 2025新上市
    'px':   ('CZCE', 5,   2,    0.05, '元/吨',   '对二甲苯'),

    # ==================== INE 上海国际能源交易中心 ====================
    'sc':   ('INE', 1000, 0.1,  0.05, '元/桶',   '原油'),      # 1000桶/手
    'lu':   ('INE', 10,  1,    0.05, '元/吨',   '低硫燃料油'),
    'nr':   ('INE', 10,  5,    0.07, '元/吨',   '20号胶'),
    'bc':   ('INE', 5,   10,   0.05, '元/吨',   '国际铜'),
    'ec':   ('INE', 50,  0.1,  0.12, '指数点',  '集运指数(欧线)'),  # 50元/点, 2026.5.11后tick调整为0.5点

    # ==================== GFEX 广州期货交易所 ====================
    'lc':   ('GFEX', 1,   20,   0.05, '元/吨',   '碳酸锂'),    # tick从10调整为20 (2024.12.17)
    'si':   ('GFEX', 5,   5,    0.05, '元/吨',   '工业硅'),
    'ps':   ('GFEX', 3,   5,    0.05, '元/吨',   '多晶硅'),    # 2025新上市

    # ==================== CFFEX 中国金融期货交易所 ====================
    'if':   ('CFFEX', 300,  0.2,  0.12, '指数点',  '沪深300股指'),
    'ih':   ('CFFEX', 300,  0.2,  0.12, '指数点',  '上证50股指'),
    'ic':   ('CFFEX', 200,  0.2,  0.12, '指数点',  '中证500股指'),
    'im':   ('CFFEX', 200,  0.2,  0.12, '指数点',  '中证1000股指'),
    # 国债期货: 报价=百元净价, 合约面值=100万(T/TF/TL) 或 200万(TS)
    # 乘数单位是"万元/点" (百元净价报价下, 1点=100元面值变化)
    # TS: 面值200万, 乘数20000元/点
    # TF: 面值100万, 乘数10000元/点
    # T:  面值100万, 乘数10000元/点
    # TL: 面值100万, 乘数10000元/点
    'ts':   ('CFFEX', 20000, 0.005, 0.005, '百元净价', '2年期国债'),
    'tf':   ('CFFEX', 10000, 0.005, 0.01,  '百元净价', '5年期国债'),
    't':    ('CFFEX', 10000, 0.005, 0.02,  '百元净价', '10年期国债'),
    'tl':   ('CFFEX', 10000, 0.01,  0.03,  '百元净价', '30年期国债'),
}


def get_multiplier(symbol):
    """获取合约乘数, 不存在则返回默认值10"""
    s = symbol.lower().replace('fi', '')
    if s in CONTRACT_SPECS:
        return CONTRACT_SPECS[s][1]
    return 10

def get_tick_size(symbol):
    """获取最小变动价位"""
    s = symbol.lower().replace('fi', '')
    if s in CONTRACT_SPECS:
        return CONTRACT_SPECS[s][2]
    return 1.0

def get_tick_profit(symbol):
    """计算一跳盈亏 = tick_size * multiplier"""
    return get_tick_size(symbol) * get_multiplier(symbol)

def get_margin_rate(symbol):
    """获取交易所最低保证金率"""
    s = symbol.lower().replace('fi', '')
    if s in CONTRACT_SPECS:
        return CONTRACT_SPECS[s][3]
    return 0.08

def get_exchange(symbol):
    """获取交易所"""
    s = symbol.lower().replace('fi', '')
    if s in CONTRACT_SPECS:
        return CONTRACT_SPECS[s][0]
    return 'UNKNOWN'

def calc_margin(price, symbol, lots=1, margin_rate=None):
    """计算保证金 = price * multiplier * margin_rate * lots"""
    m = get_multiplier(symbol)
    if margin_rate is None:
        margin_rate = get_margin_rate(symbol)
    return price * m * margin_rate * lots

def calc_pnl(entry_price, exit_price, symbol, lots=1):
    """计算盈亏 = (exit - entry) * multiplier * lots"""
    m = get_multiplier(symbol)
    return (exit_price - entry_price) * m * lots


# ============================================================
# 兼容 paper_trading.py 的 fi 后缀映射
# 注意: paper_trading.py 使用 fi 后缀, 有命名冲突需特殊处理:
#   cffi -> C (玉米), cfi -> CF (棉花)
#   jdfi -> JD (鸡蛋), 等等
# ============================================================
MULT_MAP_FI = {}
for _code, (_ex, _mult, _tick, _mr, _pu, _name) in CONTRACT_SPECS.items():
    MULT_MAP_FI[_code + 'fi'] = _mult

# 手动修正 paper_trading.py 中的 fi 命名冲突
# paper_trading.py 用 cffi 表示玉米(C), cfi 表示棉花(CF)
MULT_MAP_FI['cffi'] = CONTRACT_SPECS['c'][1]    # 玉米 10吨/手
MULT_MAP_FI['cfi']  = CONTRACT_SPECS['cf'][1]   # 棉花 5吨/手

# 补充 paper_trading.py 中有但 CONTRACT_SPECS 用不同key的品种
# 以下是 paper_trading.py 中有但使用旧名/小品种的映射
MULT_MAP_FI.update({
    'jrfi': 20,    # 粳稻 JR 20吨/手
    'lrfi': 20,    # 晚籼稻 LR 20吨/手
    'rrfi': 20,    # 早籼稻 RI 20吨/手
    'pmfi': 20,    # 普麦 PM 50吨/手 (paper_trading用20)
    'whfi': 20,    # 强麦 WH 20吨/手
    'rsfi': 20,    # 油菜籽 RS 10吨/手 (paper_trading用20)
    'lgfi': 90,    # 原木 LG 90立方米/手 (paper_trading用20, 错误)
    'wrffi': 10,   # 线材 WR 10吨/手 (paper_trading用wrffi而非wrfi)
})

# ============================================================
# paper_trading.py 中发现的乘数错误 (相比交易所官方数据)
# ============================================================
MULT_ERRORS_IN_PAPER_TRADING = {
    'cjfi':  (10, 5,   '红枣 CJ 应为5吨/手, paper_trading用10'),
    'fbfi':  (500, 10,  '纤维板 FB 已改为10立方米/手, paper_trading用旧值500张'),
    'nrfi':  (1, 10,   '20号胶 NR 应为10吨/手, paper_trading用1'),
    'lgfi':  (20, 90,  '原木 LG 应为90立方米/手, paper_trading用20'),
}


if __name__ == '__main__':
    print("=" * 90)
    print(f"{'代码':<6} {'交易所':<6} {'乘数':<8} {'Tick':<10} {'一跳盈亏':<10} {'保证金率':<8} {'报价单位':<10} {'名称'}")
    print("-" * 90)
    for code in sorted(CONTRACT_SPECS.keys()):
        ex, mult, tick, mr, pu, name = CONTRACT_SPECS[code]
        tp = tick * mult
        print(f"{code:<6} {ex:<6} {mult:<8} {tick:<10} {tp:<10.1f} {mr*100:<7.1f}% {pu:<10} {name}")
