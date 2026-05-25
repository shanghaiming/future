try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
from sklearn.preprocessing import StandardScaler
import joblib  # 用于保存和加载标准化器
import os

# Transformer模型定义（与训练时相同）
if _HAS_TORCH:
    class TimeSeriesTransformer(nn.Module):
        def __init__(self, num_features, num_classes, d_model=64, nhead=8, 
                     num_layers=3, dim_feedforward=256, dropout=0.1, 
                     use_cls_token=True, use_time_encoding=True, use_position_encoding=True):
            super(TimeSeriesTransformer, self).__init__()
            self.num_features = num_features
            self.d_model = d_model
            self.use_cls_token = use_cls_token
            self.use_time_encoding = use_time_encoding
            self.use_position_encoding = use_position_encoding

            # 特征嵌入层
            self.feature_embedding = nn.Linear(num_features, d_model)

            # 位置编码
            if use_position_encoding:
                self.position_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))

            # 时间位置编码
            if use_time_encoding:
                self.time_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))

            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # CLS token
            if use_cls_token:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

            # 分类器
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            batch_size, seq_len, num_features = x.shape

            # 嵌入每个时间步的特征
            x = self.feature_embedding(x)

            # 添加位置编码
            if self.use_position_encoding:
                x = x + self.position_encoding[:, :seq_len, :]

            # 添加时间编码
            if self.use_time_encoding:
                x = x + self.time_encoding[:, :seq_len, :]

            # 添加CLS token
            if self.use_cls_token:
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                seq_len = seq_len + 1

            # 通过Transformer编码器
            x = self.transformer_encoder(x)

            # 使用CLS token或最后一个时间步的输出进行分类
            if self.use_cls_token:
                output = x[:, 0, :]
            else:
                output = x[:, -1, :]

            # 分类
            return self.classifier(output)

    class TransformerStrategy(BaseStrategy):
        def __init__(self, data, params):
            super().__init__(data, params)

            # 模型参数
            self.model_path = params.get('model_path')
            self.scaler_path = params.get('scaler_path')
            self.window_size = params.get('window_size', 50)
            self.num_features = params.get('num_features', 20)
            self.num_classes = params.get('num_classes', 3)
            self.threshold = params.get('threshold', 0.5)  # 概率阈值
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 加载模型和标准化器
            self.model = self.load_model()
            self.scaler = self.load_scaler()

            # 特征列（根据你的数据调整）
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume', 
                'returns', 'volatility', 'rsi', 'macd', 'bollinger_upper',
                'bollinger_lower', 'atr', 'obv', 'vwap', 'momentum',
                'stochastic_k', 'stochastic_d', 'williams_r', 'adx', 'cci'
            ]

            # 确保数据包含所有需要的特征
            self.ensure_features()

            # 初始化窗口数据
            self.window_data = []

        def load_model(self):
            """加载训练好的模型"""
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            model = TimeSeriesTransformer(
                num_features=self.num_features,
                num_classes=self.num_classes,
                d_model=256,
                nhead=16,
                num_layers=5,
                dim_feedforward=1024,
                dropout=0.0326
            ).to(self.device)

            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()

            return model

        def load_scaler(self):
            """加载标准化器"""
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            return joblib.load(self.scaler_path)

        def ensure_features(self):
            """确保数据包含所有需要的特征"""
            missing_features = [col for col in self.feature_columns if col not in self.data.columns]
            if missing_features:
                raise ValueError(f"Data missing required features: {missing_features}")

        def prepare_features(self, data_point):
            """从数据点中提取特征"""
            features = []
            for col in self.feature_columns:
                features.append(data_point[col])
            return np.array(features).reshape(1, -1)

        def update_window(self, data_point):
            """更新窗口数据"""
            # 提取特征
            features = self.prepare_features(data_point)

            # 标准化特征
            scaled_features = self.scaler.transform(features)

            # 添加到窗口
            self.window_data.append(scaled_features[0])

            # 保持窗口大小
            if len(self.window_data) > self.window_size:
                self.window_data.pop(0)

        def predict(self):
            """使用模型进行预测"""
            if len(self.window_data) < self.window_size:
                return None, None

            # 准备输入数据
            window_array = np.array(self.window_data)
            input_tensor = torch.tensor(window_array, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 进行预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

            return predicted.cpu().numpy()[0], probabilities.cpu().numpy()[0]

        def generate_signals(self):
            """生成交易信号"""
            data = self.data.copy()
            self.signals = []

            # 初始化持仓状态
            position = 0
            last_buy_date = None

            for i in range(len(data)):
                current_date = data.index[i]
                current_data = data.iloc[i]

                # 更新窗口数据
                self.update_window(current_data)

                # 进行预测
                prediction, probabilities = self.predict()

                # 如果窗口数据不足，跳过
                if prediction is None:
                    continue

                # 获取类别1和2的概率
                prob_class_1 = probabilities[1]
                prob_class_2 = probabilities[2]

                # 生成信号
                signal = 'hold'

                # 类别1概率高且空仓时买入
                if prob_class_1 > self.threshold and position == 0:
                    signal = 'buy'
                    position += 1
                    last_buy_date = current_date

                # 类别2概率高且持仓时卖出
                elif prob_class_2 > self.threshold and position > 0:
                    # 确保不是同一K线内交易（至少间隔1分钟）
                    if last_buy_date is None or (current_date - last_buy_date) >= pd.Timedelta(minutes=5):
                        signal = 'sell'
                        position -= 1
                        last_buy_date = None

                # 记录信号
                if signal != 'hold':
                    self.signals.append({
                        'timestamp': current_date, 
                        'action': signal,
                        'prediction': prediction,
                        'prob_class_0': probabilities[0],
                        'prob_class_1': prob_class_1,
                        'prob_class_2': prob_class_2
                    })

            return self.signals