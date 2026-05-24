"""
Philosophy-based Trading Strategies
基于概率理论的高级交易策略
"""

from .martingale_stopping_strategy import MartingaleStoppingStrategy
from .random_matrix_filter_strategy import RandomMatrixFilterStrategy
from .wavelet_denoise_strategy import WaveletDenoiseStrategy
from .zscore_regime_strategy import ZScoreRegimeStrategy
from .adaptive_mean_reversion_strategy import AdaptiveMeanReversionStrategy
from .kalman_vdp_fusion_strategy import KalmanVDPFusionStrategy
from .dmd_regime_routing_strategy import DMDRegimeRoutingStrategy
from .wasserstein_regime_switch_strategy import WassersteinRegimeSwitchStrategy
from .copula_tail_risk_strategy import CopulaTailRiskStrategy
from .evt_extreme_risk_strategy import EVTExtremeRiskStrategy
from .fisher_information_strategy import FisherInformationStrategy

__all__ = [
    'MartingaleStoppingStrategy',
    'RandomMatrixFilterStrategy',
    'WaveletDenoiseStrategy',
    'ZScoreRegimeStrategy',
    'AdaptiveMeanReversionStrategy',
    'KalmanVDPFusionStrategy',
    'DMDRegimeRoutingStrategy',
    'WassersteinRegimeSwitchStrategy',
    'CopulaTailRiskStrategy',
    'EVTExtremeRiskStrategy',
    'FisherInformationStrategy',
]
