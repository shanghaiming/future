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
from sklearn.gaussian_process import GaussianProcessRegressor

# 忽略警告
warnings.filterwarnings('ignore')

# 配置设置
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "grpo_trading_model_20250811_sim.pth"
    FEATURE_COLS = [
        'low', 'amount', 'high', 'close', 'volume', 'open', 'diff',
       'dea', 'macd', 'ma5', 'ma8', 'ma13', 'ma21', 'rsi', 'returns', 'volatility'
    ]
    WINDOW_SIZE = 30



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
        
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)
        
    def forward(self, x):
        # 确保输入是3D张量 [batch, seq_len, features]        
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x[-1]  # 返回最后时间步的特征 [batch, d_model]

# GRPO 代理
class GRPOAgent:
    def __init__(self, feature_extractor, state_dim, action_dim, config):
        self.feature_extractor = feature_extractor
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )       
        
    
    def act(self, state):
        
        """根据状态选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, seq_len, features]
        
        
        with torch.no_grad():
            features = self.feature_extractor(state_tensor)  # [1, d_model]
            logits = self.policy_net(features)
            # 打印logits
            #rint("Logits:", logits)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            
        return action.item(), log_prob.item()
        
    
    def load_model(self, path):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            self.policy_net.load_state_dict(checkpoint['policy_net'])            
            print(f"已从 {path} 加载模型")
            return True
        else:
            raise FileNotFoundError
            return False



# GRPO策略实现
class GRPOStrategy_S(BaseStrategy):
    def __init__(self, data: pd.DataFrame, params: dict):
        super().__init__(data, params)
        # 获取关键参数
        self.feature_cols = params.get('feature_cols', Config.FEATURE_COLS)
        self.window_size = params.get('window_size', Config.WINDOW_SIZE)
        self.model_path = params.get('model_path', Config.MODEL_PATH)
        self.scaler = params.get('scaler', None)
        
        # 验证参数
        if not self.feature_cols:
            raise ValueError("必须提供feature_cols参数")
        if self.window_size <= 0:
            raise ValueError("window_size必须大于0")
        
        # 初始化模型
        self.feature_extractor = FeatureExtractor(
            input_dim=len(self.feature_cols),
            d_model=64,
            nhead=4,
            num_layers=2
        )
        self.agent = GRPOAgent(
            feature_extractor=self.feature_extractor,
            state_dim=64,
            action_dim=2,
            config=Config
        )
        
        # 加载预训练模型
        try:
            self.agent.load_model(self.model_path)
            print(f"策略初始化完成 - 特征数: {len(self.feature_cols)}, 窗口大小: {self.window_size}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            # 可以在这里添加回退逻辑
        
        
        
    def generate_signals(self):
        """生成GRPO交易信号"""
        data = self.data.copy()
       
        
        # 确保数据按时间排序
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        data = data.dropna(subset=self.feature_cols)
        print(len(data))
        print("检查数据中的NaN值...")
        nan_counts = data.isna().sum()
        print(nan_counts[nan_counts > 0])
        
        # 标准化特征
        if self.scaler:
            data[self.feature_cols] = self.scaler.transform(data[self.feature_cols])
        data['Date'] = data.index 
        # 检查数据长度
        if len(data) < self.window_size:
            print(f"警告: 数据长度({len(data)})小于窗口大小({self.window_size})")
            return []
        
        signals = []
        position = 0
        last_buy_date = None
        
        # 初始化状态缓冲区
        state_buffer = []
        for i in range(self.window_size):
            state_buffer.append(data.iloc[i][self.feature_cols].values.astype(np.float32))
        
        # 处理每个时间点
        for i in range(self.window_size, len(data)):
            row = data.iloc[i]
            timestamp = row.name
            
            # 更新状态窗口
            state_buffer.pop(0)
            state_buffer.append(row[self.feature_cols].values.astype(np.float32))
            
            # 构建正确的状态输入 [seq_len, features]
            state_input = np.array(state_buffer)  # 2D数组 [window_size, features]
            
            # 获取代理动作
            action, _= self.agent.act(state_input)
            #print (action)
            # 生成信号
            if action == 0 and position == 0:  # 买入信号且无仓位
                signals.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': row['close']
                })
                position = 1
                last_buy_date = timestamp
            
            elif action == 1 and position == 1:  # 卖出信号且有仓位
                # 确保不是同一K线内交易
                if last_buy_date is None or (timestamp - last_buy_date) >= pd.Timedelta(minutes=1):
                    signals.append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': row['close']
                    })
                    position = 0
                    last_buy_date = None
        
        return signals