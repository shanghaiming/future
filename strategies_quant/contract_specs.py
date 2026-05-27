"""
Contract specifications for Chinese commodity futures.
Source: Shanghai Futures Exchange (SHFE), Dalian Commodity Exchange (DCE),
Zhengzhou Commodity Exchange (CZCE), China Financial Futures Exchange (CFFEX),
Guangzhou Futures Exchange (GFEX).

Each contract: {symbol: (multiplier, tick_size, exchange, margin_rate)}
  multiplier: tons or units per contract
  tick_size: minimum price change per unit
  exchange: SHFE, DCE, CZCE, CFFEX, GFEX, INE
  margin_rate: typical exchange margin requirement (fraction)

All prices in CNY.
"""

# Standard contract multipliers for Chinese commodity futures
# Format: symbol -> (multiplier, tick_size, exchange, name, margin_rate)
CONTRACT_SPECS = {
    # Black metals (黑色) - SHFE/DCE
    'i':  (100,  0.5,  'DCE',  'iron ore 铁矿',       0.12),
    'j':  (100,  0.5,  'DCE',  'coke 焦炭',           0.12),
    'jm': (60,   0.5,  'DCE',  'coking coal 焦煤',    0.12),
    'hc': (10,   1.0,  'SHFE', 'hot coil 热卷',       0.10),
    'sf': (5,    2.0,  'DCE',  'silicon Fe 硅铁',     0.10),
    'sm': (5,    2.0,  'DCE',  'manganese Si 锰硅',   0.10),
    'wr': (10,   1.0,  'SHFE', 'wire rod 线材',       0.08),
    'rb': (10,   1.0,  'SHFE', 'rebar 螺纹钢',        0.10),

    # Non-ferrous metals (有色金属) - SHFE
    'cu': (5,    10.0, 'SHFE', 'copper 铜',           0.10),
    'al': (5,    5.0,  'SHFE', 'aluminum 铝',         0.10),
    'zn': (5,    5.0,  'SHFE', 'zinc 锌',             0.10),
    'pb': (5,    5.0,  'SHFE', 'lead 铅',             0.10),
    'ni': (1,    10.0, 'SHFE', 'nickel 镍',           0.12),
    'sn': (1,    10.0, 'SHFE', 'tin 锡',              0.10),
    'ss': (5,    5.0,  'SHFE', 'stainless SS 不锈钢', 0.10),
    'ao': (5,    5.0,  'SHFE', 'alumina 氧化铝',      0.12),

    # Energy (能源) - SHFE/INE
    'sc': (1000, 0.1,  'INE',  'crude oil 原油',      0.12),
    'fu': (10,   1.0,  'SHFE', 'fuel oil 燃油',       0.10),
    'bu': (10,   2.0,  'SHFE', 'bitumen 沥青',        0.10),
    'pg': (20,   1.0,  'DCE',  'LPG 液化气',          0.10),
    'lu': (10,   5.0,  'SHFE', 'rubber 橡胶',         0.10),

    # Chemicals (化工) - DCE/CZCE/SHFE
    'v':  (5,    5.0,  'DCE',  'PVC',                 0.10),
    'pp': (5,    1.0,  'DCE',  'polypropylene PP',    0.08),
    'l':  (5,    5.0,  'DCE',  'polyethylene L',      0.08),
    'eg': (10,   1.0,  'DCE',  'ethylene glycol EG',  0.10),
    'ma': (10,   1.0,  'CZCE', 'methanol 甲醇',       0.10),
    'ta': (5,    2.0,  'CZCE', 'PTA',                 0.08),
    'sa': (20,   1.0,  'CZCE', 'soda ash 纯碱',       0.10),
    'eb': (5,    1.0,  'DCE',  'benzene EB 苯乙烯',   0.10),
    'ur': (20,   1.0,  'CZCE', 'urea 尿素',           0.10),
    'pf': (5,    2.0,  'CZCE', 'short fiber PF 短纤', 0.08),
    'sh': (5,    1.0,  'DCE',  'styrene',             0.10),

    # Agriculture - Oils & Oilseeds (油脂油料) - DCE
    'm':  (10,   1.0,  'DCE',  'soybean meal 豆粕',   0.08),
    'y':  (10,   2.0,  'DCE',  'soybean oil 豆油',    0.08),
    'a':  (10,   1.0,  'DCE',  'soybean No.1 豆一',   0.08),
    'p':  (10,   2.0,  'DCE',  'palm oil 棕榈油',     0.10),

    # Agriculture - Grains (谷物) - DCE/CZCE
    'c':  (10,   1.0,  'DCE',  'corn 玉米',           0.08),
    'cs': (10,   1.0,  'DCE',  'corn starch 淀粉',    0.08),
    'rr': (20,   1.0,  'CZCE', 'early rice 早籼稻',   0.08),

    # Soft commodities (软商品) - CZCE
    'cf': (5,    5.0,  'CZCE', 'cotton 棉花',         0.08),
    'sr': (10,   1.0,  'CZCE', 'sugar 白糖',          0.08),
    'ap': (10,   1.0,  'CZCE', 'apple 苹果',          0.10),
    'cj': (5,    5.0,  'CZCE', 'jujube 红枣',         0.10),
    'pk': (5,    2.0,  'CZCE', 'peanut 花生',         0.08),

    # Livestock (畜牧) - DCE
    'lh': (16,   5.0,  'DCE',  'live hog 生猪',       0.12),
    'jd': (10,   1.0,  'DCE',  'egg 鸡蛋',           0.10),

    # New / special
    'lc': (1,    5.0,  'GFEX', 'lithium carbonate 碳酸锂', 0.15),
    'lrm': (20,  1.0,  'CZCE', 'late rice 晚籼稻',   0.08),
    'bc': (5,    10.0, 'INE',  'intl copper 国际铜',  0.10),
}


def get_contract_multiplier(symbol: str) -> int:
    """Return contract multiplier (tons/units per lot) for a symbol."""
    spec = CONTRACT_SPECS.get(symbol)
    if spec is None:
        # Default: assume 10 units per lot
        return 10
    return spec[0]


def get_tick_size(symbol: str) -> float:
    """Return minimum tick size for a symbol."""
    spec = CONTRACT_SPECS.get(symbol)
    if spec is None:
        return 1.0
    return spec[1]


def get_margin_rate(symbol: str) -> float:
    """Return typical margin rate for a symbol."""
    spec = CONTRACT_SPECS.get(symbol)
    if spec is None:
        return 0.10
    return spec[4]


def get_notional_value(symbol: str, price: float) -> float:
    """Calculate notional value of one contract.

    notional = price * multiplier
    Example: cu at 70000 → 70000 * 5 = 350,000 CNY per lot
    """
    return price * get_contract_multiplier(symbol)


def get_margin_required(symbol: str, price: float) -> float:
    """Calculate margin required for one contract.

    margin = notional_value * margin_rate
    Example: cu at 70000, margin=10% → 350,000 * 0.10 = 35,000 CNY per lot
    """
    return get_notional_value(symbol, price) * get_margin_rate(symbol)


def get_contracts_for_target(
    symbol: str, price: float, target_value: float,
) -> int:
    """Calculate number of contracts to reach target position value.

    Returns floor of target_value / notional_per_contract.
    Minimum 1 contract.
    """
    notional = get_notional_value(symbol, price)
    if notional <= 0:
        return 0
    return max(1, int(target_value / notional))


def get_all_multipliers(symbols: list) -> dict:
    """Return {symbol: multiplier} dict for a list of symbols."""
    return {s: get_contract_multiplier(s) for s in symbols}
