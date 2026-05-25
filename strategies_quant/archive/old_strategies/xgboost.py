#!/usr/bin/env python
# coding: utf-8

# In[12]:


from core.base_strategy import BaseStrategy
import numpy as np
import pandas as pd
import json
from datetime import datetime
try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import hashlib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# 配置常量
CONFIG = {
    "data_path": "/e/stock/csv_version/analysis_results/000001.SZ_analysis.csv",
    "annotation_path": "annotations.json",
    "model_path": "high_conviction_model.pkl",
    "feature_columns": [
        'macd'
    ],
    "label_definition": {
        "hold_period": 5,
        "profit_threshold": 0.08,
    },
    "model_params": {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "objective": "multi:softmax",
        "num_class": 3,
        "random_state": 42
    }
}

def load_data():
    """加载预处理好的股票数据"""
    ts_code = "000001.SZ" 
    file_path = fr"E:\stock\csv_version\analysis_results\{ts_code}_analysis.csv"
    df = pd.read_csv(file_path, index_col='trade_date', parse_dates=True)
    print(f"数据加载完成: {len(df)} 行, {len(df.columns)} 列")
    return df

def initialize_annotations():
    """初始化标注系统"""
    try:
        with open(CONFIG["annotation_path"], 'r') as f:
            annotations = json.load(f)
            print(f"标注记录加载: {len(annotations['annotations'])} 条历史标注")
    except FileNotFoundError:
        annotations = {
            "version": 1.0,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "annotations": {}
        }
        print("新建标注记录")
    return annotations

def save_annotations(annotations):
    """保存标注数据"""
    annotations["last_updated"] = datetime.now().isoformat()
    with open(CONFIG["annotation_path"], 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"标注已保存: {CONFIG['annotation_path']}")


def create_initial_labels(df):
    """创建初始标签（基于规则）"""
    # 计算未来收益
    future_returns = df['close'].pct_change(
        CONFIG["label_definition"]["hold_period"]
    ).shift(-CONFIG["label_definition"]["hold_period"])
    
    # 使用0-2标签体系
    labels = pd.Series(-1, index=df.index, name='label')  # 默认-1=未标注
    
    # 高胜率买入机会 -> 标记为2
    buy_mask = future_returns > CONFIG["label_definition"]["profit_threshold"]
    labels[buy_mask] = 2
    
    # 高胜率卖出机会 -> 标记为1
    sell_mask = future_returns < -CONFIG["label_definition"]["profit_threshold"]
    labels[sell_mask] = 1
    
    # 其余标记为持有(0)
    hold_mask = ~buy_mask & ~sell_mask
    labels[hold_mask] = 0
    
    print(f"初始标签分布: {labels.value_counts().to_dict()}")
    print("标签映射: -1=未标注, 0=持有, 1=卖出, 2=买入")
    return labels
def select_samples_for_annotation(df, labels, annotations, model=None, strategy="hybrid", n_samples=50):
    """智能选择标注样本"""
    # 已标注样本的哈希值
    annotated_hashes = {a['sample_hash'] for a in annotations['annotations'].values()}
    
    # 候选样本
    candidates = []
    
    # 策略1: 不确定样本（需要模型）
    if strategy in ("hybrid", "uncertainty") and model is not None and hasattr(model, 'predict_proba'):
        proba = model.predict_proba(df[CONFIG["feature_columns"]])
        uncertainty = 1 - np.max(proba, axis=1)
        uncertain_samples = df.index[uncertainty > 0.4]
        for idx in uncertain_samples:
            sample_data = df.loc[idx][CONFIG["feature_columns"]].to_dict()
            sample_hash = hashlib.md5(json.dumps(sample_data, sort_keys=True).encode()).hexdigest()
            if sample_hash not in annotated_hashes:
                candidates.append(("uncertain", idx))
    
    # 策略2: 关键市场事件
    if strategy in ("hybrid", "events"):
        key_events = [
            '2020-03-23', '2022-01-03', '2022-10-13',
            # 添加其他关键日期...
        ]
        for event in key_events:
            if event in df.index:
                idx = df.index.get_loc(event)
                sample_data = df.loc[event][CONFIG["feature_columns"]].to_dict()
                sample_hash = hashlib.md5(json.dumps(sample_data, sort_keys=True).encode()).hexdigest()
                if sample_hash not in annotated_hashes:
                    candidates.append(("event", event))
    
    # 策略3: 随机样本（确保多样性）
    if strategy in ("hybrid", "diversity"):
        random_samples = df.sample(min(n_samples//3, len(df))).index
        for idx in random_samples:
            sample_data = df.loc[idx][CONFIG["feature_columns"]].to_dict()
            sample_hash = hashlib.md5(json.dumps(sample_data, sort_keys=True).encode()).hexdigest()
            if sample_hash not in annotated_hashes:
                candidates.append(("random", idx))
    
    # 去重并选择top N
    unique_candidates = list({c[1]: c for c in candidates}.values())
    return unique_candidates[:n_samples]

def display_sample_info(df, idx):
    """显示样本详细信息"""
    sample = df.loc[idx]
    #print(f"\n日期: {idx.strftime('%Y-%m-%d')}")
    print(f"收盘价: {sample['close']:.2f}")
   

def manual_annotation_workflow(df, labels, annotations, model=None):
    """手动标注工作流"""
    # 选择标注样本（传递model参数）
    samples = select_samples_for_annotation(df, labels, annotations, model, n_samples=30)
    
    if not samples:
        print("没有需要标注的新样本")
        return labels, annotations
    
    print(f"\n=== 开始标注 ({len(samples)} 个样本) ===")
    
    for i, (sample_type, idx) in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] 样本类型: {sample_type.upper()}")
        display_sample_info(df, idx)
        
        # 显示初始标签
        current_label = labels.loc[idx]
        label_mapping = {0: "持有", 1: "卖出", 2: "买入"}
        print(f"\n当前标签: {current_label} ({label_mapping.get(current_label, '未知')})")
        
        # 获取用户标注
        while True:
            try:
                user_label = int(input("您的标注 (0=持有, 1=卖出, 2=买入): "))
                if user_label in (0, 1, 2):
                    break
                print("无效输入! 请输入0, 1或2")
            except ValueError:
                print("请输入数字!")
        
        # 更新标签
        labels.loc[idx] = user_label
        
        # 创建样本哈希
        sample_data = df.loc[idx][CONFIG["feature_columns"]].to_dict()
        sample_hash = hashlib.md5(json.dumps(sample_data, sort_keys=True).encode()).hexdigest()
        
        # 保存标注元数据
        annotations["annotations"][str(idx)] = {
            "date": str(idx),
            "label": user_label,
            "sample_type": sample_type,
            "sample_hash": sample_hash,
            "features": sample_data,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"已标注: {user_label}")
    
    save_annotations(annotations)
    return labels, annotations

def prepare_training_data(df, labels):
    """准备训练数据 - 包含所有样本"""
    # 包含所有已标注样本
    valid_doc_ids = labels[labels != -1].index
    
    X = df.loc[valid_doc_ids, CONFIG["feature_columns"]]
    y = labels.loc[valid_doc_ids]
    
    # 映射标签到0开始的连续整数
    unique_labels = sorted(y.unique())
    label_mapping = {orig: idx for idx, orig in enumerate(unique_labels)}
    y_mapped = y.map(label_mapping)
    
    # 划分训练集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
    )
    
    print(f"\n训练数据统计:")
    print(f" - 总样本: {len(X)} (持有: {sum(y == 0)}, 卖出: {sum(y == 1)}, 买入: {sum(y == 2)})")
    print(f" - 训练集: {len(X_train)}")
    print(f" - 验证集: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val, label_mapping

def train_model(X_train, y_train):
    """训练高胜率识别模型 - 自动识别类别数量"""
    print("\n=== 模型训练 ===")
    num_class = len(np.unique(y_train))
    
    # 动态设置模型参数
    model_params = CONFIG["model_params"].copy()
    model_params["num_class"] = num_class
    model_params["objective"] = "multi:softmax"
    
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """评估模型性能 - 支持动态标签"""
    y_pred = model.predict(X_val)
    
    print("\n=== 模型评估 ===")
    print(f"准确率: {accuracy_score(y_val, y_pred):.2%}")
    print("\n分类报告:")
    
    # 动态确定标签名称
    unique_labels = sorted(np.unique(np.concatenate([y_val, y_pred])))
    label_names = {0: "持有", 1: "卖出", 2: "买入"}
    
    print(classification_report(
        y_val, y_pred, 
        target_names=[label_names.get(i, f"Class{i}") for i in unique_labels]
    ))
    
    # 特征重要性
    feature_importance = pd.Series(
        model.feature_importances_,
        index=CONFIG["feature_columns"]
    ).sort_values(ascending=False)
    
    print("\n特征重要性:")
    print(feature_importance.head(10))
    
    return feature_importance

def active_learning_pipeline(df, labels, annotations, rounds=3):
    """主动学习流程"""
    model = None
    label_mapping = None
    
    for round in range(1, rounds+1):
        print(f"\n{'='*40}")
        print(f"主动学习轮次 {round}/{rounds}")
        print(f"{'='*40}")
        
        # 准备数据（获取标签映射）
        X_train, X_val, y_train, y_val, label_mapping = prepare_training_data(df, labels)
        
        # 训练模型
        model = train_model(X_train, y_train)
        
        # 评估模型
        evaluate_model(model, X_val, y_val)
        
        # 手动标注（传递当前模型）
        labels, annotations = manual_annotation_workflow(df, labels, annotations, model)
    
    # 最终训练（使用所有标注数据）
    print("\n=== 最终模型训练 ===")
    X_train, X_val, y_train, y_val, _ = prepare_training_data(df, labels)
    final_model = train_model(X_train, y_train)
    evaluate_model(final_model, X_val, y_val)
    
    # 保存标签映射（键转换为字符串）
    if label_mapping is not None:
        str_label_mapping = {str(k): v for k, v in label_mapping.items()}
        with open("label_mapping.json", "w") as f:
            json.dump(str_label_mapping, f)
    
    # 保存模型
    joblib.dump(final_model, CONFIG["model_path"])
    print(f"\n模型已保存至: {CONFIG['model_path']}")
    
    return final_model

def analyze_annotations(annotations):
    """分析标注模式"""
    if not annotations["annotations"]:
        print("没有标注记录可分析")
        return
    
    # 创建标注DataFrame
    ann_df = pd.DataFrame(annotations["annotations"]).T
    ann_df['label'] = ann_df['label'].astype(int)
    
    print("\n=== 标注分析 ===")
    print(f"总标注数: {len(ann_df)}")
    
    # 标注分布
    label_dist = ann_df['label'].value_counts().sort_index()
    print("\n标注分布:")
    print(label_dist)
    
    # 特征分析
    features_df = pd.json_normalize(ann_df['features'])
    features_df['label'] = ann_df['label']
    
    # 按标签分组分析
    print("\n特征均值 (按标签):")
    print(features_df.groupby('label').mean().T)
    
    # 标注一致性分析
    if 'initial_label' in features_df.columns:
        consistency = (features_df['label'] == features_df['initial_label']).mean()
        print(f"\n标注与初始标签一致率: {consistency:.2%}")
    
    # 标注模式可视化
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        sns.pairplot(
            features_df[['macd',  'label']], 
            hue='label',
            palette={-1: 'red', 0: 'gray', 1: 'green'},
            plot_kws={'alpha': 0.6}
        )
        plt.suptitle('标注样本特征分布', y=1.02)
        plt.savefig('annotation_analysis.png', dpi=300, bbox_inches='tight')
        print("标注分析图表已保存: annotation_analysis.png")
    except ImportError:
        print("未安装matplotlib/seaborn，跳过可视化")

# 主工作流
def main():
    # 加载数据
    df = load_data()
    
    # 初始化标注系统
    annotations = initialize_annotations()
    print("initial")
    
    # 创建初始标签
    labels = create_initial_labels(df)  
    print("label")
    
    # 初始标注（此时模型尚未存在，传递None）
    labels, annotations = manual_annotation_workflow(df, labels, annotations, model=None)
    print('annotations')

    # 主动学习流程
    final_model = active_learning_pipeline(df, labels, annotations, rounds=2)
    
    # 标注分析
    analyze_annotations(annotations)
    
    print("\n=== 系统执行完成 ===")
if __name__ == "__main__":    
    main()

# In[ ]:






class XgboostStrategy(BaseStrategy):
    """基于xgboost的策略"""

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)

    def generate_signals(self):
        """生成交易信号"""
        df = self.data
        if len(df) < 2:
            return self.signals

        close = df['close']
        ma_short = close.rolling(5).mean()
        ma_long = close.rolling(20).mean()

        for i in range(1, len(df)):
            if pd.isna(ma_long.iloc[i]):
                continue
            if ma_short.iloc[i] > ma_long.iloc[i] and ma_short.iloc[i-1] <= ma_long.iloc[i-1]:
                self._record_signal(timestamp=df.index[i], action='buy', price=float(close.iloc[i]))
            elif ma_short.iloc[i] < ma_long.iloc[i] and ma_short.iloc[i-1] >= ma_long.iloc[i-1]:
                self._record_signal(timestamp=df.index[i], action='sell', price=float(close.iloc[i]))

        return self.signals
