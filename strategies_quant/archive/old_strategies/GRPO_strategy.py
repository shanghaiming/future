try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import math
import warnings
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.distributions import Categorical
# 忽略警告
warnings.filterwarnings('ignore')

# 配置设置
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "grpo_model.pth"
    FEATURE_COLS = ['low', 'amount', 'high', 'close', 'volume', 'open', 'diff',
                    'dea', 'macd', 'ma5', 'ma8', 'ma13', 'ma21', 'ma34', 'ma55', 'vr',
                    'tsma5', 'tsma8', 'tsma13', 'er', 'var', 'sma', 'std', 'upper', 'lower',
                    'up_mark', 'down_mark', 'shadow_ratio', 'wr5', 'wr55',
                    'ma_trend_change_win1', 'ma_trend_change_win2', 'cross_events',
                    'divergence_status', 'divergence_status_macd', 'divergence_status_vr',
                    'divergence_status_wr5', 'trend_direction', 'is_range', 'start_type', 
                    'end_type', 'wave_number', 'move_type', 'subwaves', 'target', 
                    'returns', 'volatility']


# Transformer 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = self.PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            position = torch.arange(max_len).unsqueeze(1)
            pe = torch.zeros(max_len, 1, d_model)
            with torch.no_grad():
                pe[:, 0, 0::2] = torch.sin(position * div_term)
                pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)
        
    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        # 添加序列维度 (batch_size, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        last_output = x[-1]
        last_output = torch.nan_to_num(last_output, nan=0.0)
        normalized_output = self.layer_norm(last_output)
        return normalized_output

# GRPO 代理
class GRPOAgent:
    def __init__(self, feature_extractor, state_dim, action_dim):
        self.feature_extractor = feature_extractor
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = Config.DEVICE
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(self.device)
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        
        # 优化器
        self.policy_optim = optim.Adam(
            list(feature_extractor.parameters()) + list(self.policy_net.parameters()),
            lr=3e-4
        )
        
        self.value_optim = optim.Adam(
            self.value_net.parameters(),
            lr=1e-3
        )
        
        # 经验池
        self.memory = []
        self.epsilon = 0.0  # 测试时设为0，不使用探索

    def act(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            features = self.feature_extractor(state_tensor)
            logits = self.policy_net(features)
            dist = Categorical(logits=logits)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor).item()
            value = self.value_net(features).item()
        
        return action, log_prob, value

    def load_model(self, path):
        try:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=self.device)
                self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.value_net.load_state_dict(checkpoint['value_net'])
                return True
            else:
                raise FileNotFoundError(f"模型文件 {path} 不存在！请先训练模型。")
        except Exception as e:
            print(f"模型文件 {path} 不存在！请先训练模型。")
            raise
            return False



# GRPO策略实现
class GRPOStrategy(BaseStrategy):
    def __init__(self, data: pd.DataFrame, params: dict):
        super().__init__(data, params)
        self.scaler = params.get('scaler', None)
        self.model_path = params.get('model_path', Config.MODEL_PATH)
        self.feature_cols = params.get('feature_cols', Config.FEATURE_COLS)
        
        # 初始化特征提取器
        input_dim = len(self.feature_cols)
        self.feature_extractor = FeatureExtractor(
            input_dim=len(Config.FEATURE_COLS) ,
            d_model=64,
            nhead=4,
            num_layers=2
        ).to(Config.DEVICE)
        
        # 初始化代理
        self.agent = GRPOAgent(
            feature_extractor=self.feature_extractor,
            state_dim=64,
            action_dim=2  # 0=买入, 1=卖出
        )
        
        # 加载预训练模型
        self.agent.load_model(self.model_path)
        
    def generate_signals(self):
        """生成GRPO交易信号"""
        data = self.data.copy()
        
        # 确保数据按时间排序
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        # 标准化特征
        if self.scaler:
            data[self.feature_cols] = self.scaler.transform(data[self.feature_cols])
        
        # 初始化状态
        position = 0
        last_buy_date = None
        
        # 逐行生成信号
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # 获取当前状态
            state = row[self.feature_cols].values.astype(np.float32)
            
            # 获取代理动作
            action, _, _ = self.agent.act(state)
            
            # 根据动作和仓位生成信号
            close_price = row['close']
            
            # 买入信号 (action=0) 且当前无仓位
            if action == 0 and position == 0:
                self._record_signal(timestamp, 'buy', close_price)
                position += 1
                last_buy_date = timestamp
            
            # 卖出信号 (action=1) 且当前有仓位
            elif action == 1 and position == 1:
                # 确保不是同一K线内交易（至少间隔1分钟）
                if last_buy_date is None or (timestamp - last_buy_date) >= pd.Timedelta(minutes=1):
                    self._record_signal(timestamp, 'sell', close_price)
                    position -= 1
                    last_buy_date = None
        
        return self.signals