# Isolation Forest 模型改進計劃

> **文檔目的**：追蹤 Isolation Forest 模型的改進策略，按照優先級排序，方便持續檢查和溝通。

**最後更新**：2024-12-XX  
**狀態**：🟡 進行中（部分已完成）

---

## 📊 執行摘要

### 當前問題診斷

1. **XGBoost 準確率 99.99%**：可能是 Overfitting 或 Label Leakage
2. **Isolation Forest 表現改善**：Accuracy 已達 80%+，但仍有改進空間
3. **False Positive 過多**：108,520 個 FP，正常流量被誤判為異常
4. **False Negative 過多**：178,694 個 FN，異常流量被漏掉
5. **特徵選擇問題**：`hour` 特徵重要性過高，可能造成時間偏差
6. **Contamination 參數**：固定 5% 可能太低，導致 False Negative 過多
7. **協議混合訓練**：UDP 和 TCP 混在一起訓練，行為模式不同
8. **單一模型限制**：僅使用 Isolation Forest，可能無法捕捉所有異常模式

---

## 🔴 高優先級（立即實施）

### 1. 調整 Contamination 參數策略

**問題描述**：
- 當前固定使用 5% 的 contamination，但 CTU-13 Botnet 流量佔比可能更高（10-20%）
- 固定 5% 可能導致 False Negative 過多，無法捕捉足夠的異常流量

**當前狀況**：
- 📍 `scripts/train_unsupervised.py:337`：固定使用 `contamination = 0.05`
- 📍 `scripts/visualize_feature_distributions.py:114`：使用 `min(actual_contamination * 1.2, 0.2)`

**建議方案**：
```python
# 改進的 contamination 策略
if 'Label' in features_df.columns:
    actual_contamination = y_true.sum() / len(y_true)
    # 使用實際比例的 1.3 倍，但不超過 20%
    contamination = min(actual_contamination * 1.3, 0.2)
    # 如果實際比例很高（>15%），直接使用實際比例
    if actual_contamination > 0.15:
        contamination = actual_contamination
    print(f"   實際異常比例：{actual_contamination:.4f} ({actual_contamination*100:.2f}%)")
    print(f"   設定 contamination：{contamination:.4f} ({contamination*100:.2f}%)")
else:
    # 無標籤時，使用更積極的策略
    contamination = 0.1  # 從 0.05 提高到 0.1
    print(f"   使用預設 contamination：{contamination:.4f} (無標籤環境)")
```

**相關文件**：
- `scripts/train_unsupervised.py:333-342`
- `scripts/visualize_feature_distributions.py:109-119`

**實施狀態**：✅ 已實施（2024-12-19）

**實施細節**：
- 在 `scripts/train_unsupervised.py:388-396` 實施動態 contamination 策略
- 有標籤時：使用實際比例的 1.3 倍（上限 20%），如果實際比例 >15% 則直接使用實際比例
- 無標籤時：從 0.05 提高到 0.1

---

### 2. 改用異常分數排序（Scoring 模式）而非硬閾值

**問題描述**：
- 當前使用硬閾值（基於 contamination）進行二分類
- 無法靈活調整，且可能錯過邊緣案例
- 建議改用異常分數排序，取 Top N% 最異常的流量

**當前狀況**：
- 📍 `scripts/train_unsupervised.py:428-438`：使用 `predict_labels()` 返回硬分類結果
- 📍 `scripts/train_unsupervised.py:432`：已有異常分數計算，但未充分利用

**建議方案**：
```python
# 獲取異常分數
anomaly_scores = model.predict(X)
anomaly_scores_normalized = -anomaly_scores  # 轉為正數（越高越異常）

# 策略1：基於分數的 Top N%（推薦用於無標籤環境）
top_n_percent = contamination  # 使用 contamination 作為 Top N%
threshold_score = np.percentile(anomaly_scores_normalized, 100 * (1 - top_n_percent))
y_pred_score = (anomaly_scores_normalized >= threshold_score).astype(int)

# 策略2：動態調整（如果有標籤，最大化 F1）
if y_true is not None:
    from sklearn.metrics import f1_score
    thresholds = np.percentile(anomaly_scores_normalized, np.arange(90, 100, 0.5))
    best_threshold = 0
    best_f1 = 0
    for thresh in thresholds:
        y_pred_temp = (anomaly_scores_normalized >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred_temp)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    y_pred_score = (anomaly_scores_normalized >= best_threshold).astype(int)
    print(f"   最佳閾值（基於 F1）：{best_threshold:.4f}，F1={best_f1:.4f}")
```

**相關文件**：
- `scripts/train_unsupervised.py:426-450`
- `src/models.py:IsolationForestModel.predict_labels()`

**實施狀態**：⬜ 待實施

---

### 3. 檢查並移除可能洩漏標籤的特徵（IP/Port）

**問題描述**：
- XGBoost 準確率 99.99% 可能是因為模型學到了特定 IP 或 Port，而不是攻擊行為
- 需要檢查是否有 IP/Port 特徵直接洩漏標籤信息

**當前狀況**：
- 📍 需要檢查特徵工程階段是否包含 `SrcAddr`, `DstAddr`, `Sport`, `Dport` 等原始欄位
- 📍 `src/feature_engineer.py`：特徵工程模組，需要確認是否包含這些欄位

**建議方案**：
```python
# 在特徵選擇階段添加檢查
def check_label_leakage(X: pd.DataFrame, features_df: pd.DataFrame) -> List[str]:
    """檢查可能洩漏標籤的特徵"""
    leakage_features = []
    
    # 檢查 IP 地址特徵
    ip_features = ['SrcAddr', 'DstAddr', 'SrcIP', 'DstIP']
    for feat in ip_features:
        if feat in X.columns:
            leakage_features.append(feat)
            print(f"   ⚠️  警告：發現 IP 地址特徵 '{feat}'，可能洩漏標籤信息")
    
    # 檢查 Port 特徵（原始 Port 可能洩漏，但聚合特徵可以使用）
    port_features = ['Sport', 'Dport', 'SrcPort', 'DstPort']
    for feat in port_features:
        if feat in X.columns:
            # 檢查是否為原始 Port（而非聚合特徵）
            if feat not in ['unique_dport_per_minute_by_src']:  # 聚合特徵可以使用
                leakage_features.append(feat)
                print(f"   ⚠️  警告：發現原始 Port 特徵 '{feat}'，可能洩漏標籤信息")
    
    return leakage_features

# 在特徵選擇後移除洩漏特徵
leakage_features = check_label_leakage(X, features_df)
if leakage_features:
    X = X[[col for col in X.columns if col not in leakage_features]]
    print(f"   ✅ 已移除 {len(leakage_features)} 個可能洩漏標籤的特徵：{leakage_features}")
```

**相關文件**：
- `scripts/train_unsupervised.py:92-107`（特徵選擇階段）
- `scripts/train_supervised.py:70-90`（特徵選擇階段）
- `src/feature_engineer.py`（特徵工程模組）

**實施狀態**：⬜ 待實施

---

## 🟡 中優先級（短期實施）

### 4. 按協議分組訓練（TCP/UDP 分開）

**問題描述**：
- UDP 和 TCP 的行為模式完全不同（DNS vs HTTP）
- 混在一起訓練會影響模型效果
- 建議分別訓練 TCP 和 UDP 模型

**當前狀況**：
- 📍 代碼中未見協議分組實現
- 📍 需要確認數據中是否有 `Proto` 欄位

**建議方案**：
```python
# 在 train_unsupervised.py 中添加協議分組
if 'Proto' in features_df.columns:
    print("\n[步驟 4.6] 按協議分組訓練...")
    
    protocol_models = {}
    
    # TCP 流量
    tcp_mask = features_df['Proto'] == 'tcp'
    X_tcp = X[tcp_mask]
    features_df_tcp = features_df[tcp_mask]
    
    if len(X_tcp) > 1000:  # 確保有足夠樣本
        print(f"   TCP 流量：{len(X_tcp):,} 筆 ({len(X_tcp)/len(X)*100:.2f}%)")
        
        # 計算 TCP 的 contamination
        if 'Label' in features_df_tcp.columns:
            y_true_tcp = (features_df_tcp['Label'].str.contains('Botnet', case=False, na=False)).astype(int)
            contamination_tcp = min(y_true_tcp.sum() / len(y_true_tcp) * 1.3, 0.2)
        else:
            contamination_tcp = 0.1
        
        # 訓練 TCP 專用模型
        model_tcp = ModelFactory.create(ModelType.ISOLATION_FOREST)
        model_tcp.train(X_tcp, contamination=contamination_tcp, random_state=42, ...)
        protocol_models['tcp'] = model_tcp
    
    # UDP 流量
    udp_mask = features_df['Proto'] == 'udp'
    X_udp = X[udp_mask]
    features_df_udp = features_df[udp_mask]
    
    if len(X_udp) > 1000:
        print(f"   UDP 流量：{len(X_udp):,} 筆 ({len(X_udp)/len(X)*100:.2f}%)")
        
        # 計算 UDP 的 contamination
        if 'Label' in features_df_udp.columns:
            y_true_udp = (features_df_udp['Label'].str.contains('Botnet', case=False, na=False)).astype(int)
            contamination_udp = min(y_true_udp.sum() / len(y_true_udp) * 1.3, 0.2)
        else:
            contamination_udp = 0.1
        
        # 訓練 UDP 專用模型
        model_udp = ModelFactory.create(ModelType.ISOLATION_FOREST)
        model_udp.train(X_udp, contamination=contamination_udp, random_state=42, ...)
        protocol_models['udp'] = model_udp
    
    # 預測時根據協議選擇對應模型
    # ...
else:
    print("   ⚠️  未找到 'Proto' 欄位，跳過協議分組")
```

**相關文件**：
- `scripts/train_unsupervised.py:344-401`（模型訓練階段）
- 需要確認數據載入階段是否包含 `Proto` 欄位

**實施狀態**：✅ 已實施（2024-12-19）

**實施細節**：
- 在 `scripts/train_unsupervised.py:403-520` 實施協議分組訓練
- 自動檢測 `Proto` 欄位，按協議（TCP/UDP）分組訓練
- 每個協議分別計算 contamination
- 預測時根據協議選擇對應模型
- 樣本數不足 1000 的協議會自動跳過

**注意事項**：
- 已實現自動檢測 `Proto` 欄位
- 如果樣本數不足，會自動跳過該協議並回退到單一模型

---

### 5. 條件移除 `hour` 特徵（如果重要性過高）

**問題描述**：
- `hour` 特徵重要性過高，可能導致模型只學會「特定時間就是異常」
- 攻擊可能發生在任何時間，不應該依賴時間特徵

**當前狀況**：
- 📍 `src/feature_engineer.py:42`：包含 `hour` 特徵
- 📍 `src/feature_engineer.py:324-336`：時間特徵列表包含 `hour`, `sin_hour`, `cos_hour`

**建議方案**：
```python
# 在特徵選擇階段添加時間特徵過濾
def filter_time_features(X: pd.DataFrame, features_df: pd.DataFrame, 
                        importance_threshold: float = 0.1) -> pd.DataFrame:
    """條件移除可能造成偏差的時間特徵"""
    
    time_features_to_remove = []
    
    # 如果有標籤，檢查 hour 的重要性
    if 'Label' in features_df.columns and 'hour' in X.columns:
        try:
            from src.models import ModelFactory, ModelType
            y_temp = (features_df['Label'].str.contains('Botnet', case=False, na=False)).astype(int)
            
            # 快速訓練一個簡單模型檢查重要性
            xgb_model = ModelFactory.create(ModelType.XGBOOST)
            sample_size = min(10000, len(X))
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y_temp.iloc[sample_idx]
            
            xgb_model.train(X_sample, y_sample, test_size=0.2, random_state=42,
                          n_estimators=50, max_depth=4, learning_rate=0.1)
            
            feature_importance = xgb_model.get_feature_importance()
            hour_importance = feature_importance.get('hour', 0)
            
            if hour_importance > importance_threshold:
                time_features_to_remove.append('hour')
                print(f"   ⚠️  'hour' 特徵重要性過高 ({hour_importance:.4f})，將移除以避免時間偏差")
        except Exception as e:
            print(f"   ⚠️  無法檢查時間特徵重要性：{e}")
    
    # 移除可能造成偏差的時間特徵
    # 保留週期性編碼（sin_hour, cos_hour），因為它們更通用
    time_features_to_remove.extend(['day_of_month'])  # day_of_month 也可能造成偏差
    
    if time_features_to_remove:
        X = X[[col for col in X.columns if col not in time_features_to_remove]]
        print(f"   ✅ 已移除 {len(time_features_to_remove)} 個可能造成偏差的時間特徵：{time_features_to_remove}")
    
    return X
```

**相關文件**：
- `src/feature_engineer.py:18-66`（時間特徵提取）
- `scripts/train_unsupervised.py:92-107`（特徵選擇階段）

**實施狀態**：✅ 已實施（2024-12-19）

**實施細節**：
- 在 `scripts/train_unsupervised.py:325-371` 實施條件移除 hour 特徵
- 使用 XGBoost 快速評估 hour 特徵重要性（閾值 0.1）
- 如果重要性過高，自動移除 `hour` 和 `day_of_month` 特徵
- 保留週期性編碼（`sin_hour`, `cos_hour`）

**注意事項**：
- 已實現自動檢查和條件移除
- 如果沒有標籤，會提示但保留所有時間特徵

---

### 6. 新增 Local Outlier Factor (LOF) 模型

**問題描述**：
- 當前僅使用 Isolation Forest，可能無法捕捉所有異常模式
- LOF 基於局部密度，對局部異常更敏感，可與 Isolation Forest 互補
- 適合網路流量中的局部異常檢測（如特定 IP 的異常行為）

**當前狀況**：
- 📍 `src/models.py`：目前僅支援 Isolation Forest 和 XGBoost
- 📍 需要新增 LOF 模型類別

**建議方案**：
```python
# 在 src/models.py 中新增 LOFModel 類別
from sklearn.neighbors import LocalOutlierFactor

class LOFModel(BaseModel):
    """Local Outlier Factor 模型包裝器（無監督學習）"""
    
    def __init__(self):
        self.model: Optional[LocalOutlierFactor] = None
        self.scaler: Optional[RobustScaler] = None
    
    def train(self, X: pd.DataFrame, contamination: float = 0.1, 
              n_neighbors: int = 20, **kwargs):
        """訓練 LOF 模型"""
        # LOF 對尺度敏感，需要標準化
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False,  # 訓練時使用
            **kwargs
        )
        self.model.fit(X_scaled)
        return self.model, self.scaler
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """獲取異常分數（負分數表示異常，轉為正數）"""
        X_scaled = self.scaler.transform(X)
        scores = -self.model.score_samples(X_scaled)  # 轉為正數，越高越異常
        return scores
```

**相關文件**：
- `src/models.py`（需要新增 LOFModel 類別）
- `scripts/train_unsupervised.py`（需要整合 LOF 模型訓練）

**實施狀態**：✅ 已實施

**實施細節**：
- 在 `src/models.py:585-702` 實作 `LOFModel` 類別
- 繼承 `BaseModel`，實作 `train()` 和 `predict()` 方法
- 已註冊到 `ModelFactory`（`ModelType.LOCAL_OUTLIER_FACTOR`）
- 支援 `contamination` 和 `n_neighbors` 參數
- 使用 `RobustScaler` 進行特徵標準化

**預期效果**：
- 可降低 10-20% 的 False Negative（捕捉局部異常）
- 與 Isolation Forest 集成後，整體性能提升 5-10%

---

### 7. 新增 One-Class SVM 模型

**問題描述**：
- One-Class SVM 在高維特徵空間中表現良好
- 對非線性邊界有較好的處理能力
- 可作為 Isolation Forest 和 LOF 的補充

**當前狀況**：
- 📍 `src/models.py`：已支援 One-Class SVM
- 📍 已實作 `OneClassSVMModel` 類別

**建議方案**：
```python
# 在 src/models.py 中新增 OCSVMModel 類別
from sklearn.svm import OneClassSVM

class OCSVMModel(BaseModel):
    """One-Class SVM 模型包裝器（無監督學習）"""
    
    def __init__(self):
        self.model: Optional[OneClassSVM] = None
        self.scaler: Optional[RobustScaler] = None
    
    def train(self, X: pd.DataFrame, nu: float = 0.1, 
              kernel: str = 'rbf', gamma: str = 'scale', **kwargs):
        """訓練 One-Class SVM 模型"""
        # One-Class SVM 對尺度敏感，需要標準化
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = OneClassSVM(
            nu=nu,  # 異常比例的上界
            kernel=kernel,
            gamma=gamma,
            **kwargs
        )
        self.model.fit(X_scaled)
        return self.model, self.scaler
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """獲取異常分數（決策函數值，負值表示異常）"""
        X_scaled = self.scaler.transform(X)
        scores = -self.model.decision_function(X_scaled)  # 轉為正數，越高越異常
        return scores
```

**相關文件**：
- `src/models.py:817-938`（已實作 `OneClassSVMModel` 類別）
- `scripts/train_unsupervised.py`（可整合 One-Class SVM 模型訓練）

**實施狀態**：✅ 已實施

**實施細節**：
- 在 `src/models.py:817-938` 實作 `OneClassSVMModel` 類別
- 繼承 `BaseModel`，實作 `train()` 和 `predict()` 方法
- 已註冊到 `ModelFactory`（`ModelType.ONE_CLASS_SVM`）
- 支援 `nu`、`kernel`、`gamma` 等參數
- 使用 `RobustScaler` 進行特徵標準化

**注意事項**：
- One-Class SVM 計算成本較高，適合小到中等規模數據
- 對於大數據集，可能需要取樣訓練

---

## 🟢 低優先級（長期優化）

### 8. 特徵重要性分析與自動過濾

**問題描述**：
- 當前特徵選擇主要基於統計方法（常數、低變異數、高度相關）
- 缺少基於模型重要性的特徵選擇
- 建議添加自動化的特徵重要性分析

**當前狀況**：
- 📍 `scripts/train_unsupervised.py:92-200`：已有特徵選擇邏輯，但主要基於統計方法
- 📍 `scripts/visualize_feature_distributions.py:1149-1187`：有基於 XGBoost 的特徵重要性分析，但僅用於可視化

**建議方案**：
```python
# 在特徵選擇階段添加基於重要性的過濾
def feature_selection_by_importance(X: pd.DataFrame, features_df: pd.DataFrame,
                                   min_features: int = 15, max_features: int = 20,
                                   importance_threshold: float = 0.98) -> pd.DataFrame:
    """基於 XGBoost 特徵重要性進行特徵選擇"""
    
    if 'Label' not in features_df.columns:
        print("   ⚠️  無標籤，跳過基於重要性的特徵選擇")
        return X
    
    try:
        from src.models import ModelFactory, ModelType
        y_temp = (features_df['Label'].str.contains('Botnet', case=False, na=False)).astype(int)
        
        # 取樣以加快計算
        sample_size = min(100000, len(X))
        if sample_size < len(X):
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y_temp.iloc[sample_idx]
        else:
            X_sample = X
            y_sample = y_temp
        
        # 訓練模型獲取重要性
        xgb_model = ModelFactory.create(ModelType.XGBOOST)
        xgb_model.train(X_sample, y_sample, test_size=0.2, random_state=42,
                      n_estimators=50, max_depth=4, learning_rate=0.1)
        
        feature_importance = xgb_model.get_feature_importance()
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 選擇累積重要性達到閾值的特徵
        total_importance = sum(imp for _, imp in sorted_importance)
        cumulative_importance = 0
        important_features = []
        
        for feature, importance in sorted_importance:
            cumulative_importance += importance
            important_features.append(feature)
            if (cumulative_importance / total_importance >= importance_threshold and 
                len(important_features) >= min_features) or len(important_features) >= max_features:
                break
        
        if len(important_features) < min_features:
            important_features = [f[0] for f in sorted_importance[:min_features]]
        
        X_selected = X[[col for col in X.columns if col in important_features]]
        print(f"   ✅ 基於重要性選擇了 {len(important_features)} 個特徵（累積重要性：{cumulative_importance/total_importance*100:.2f}%）")
        
        return X_selected
    except Exception as e:
        print(f"   ⚠️  特徵重要性分析失敗：{e}")
        return X
```

**相關文件**：
- `scripts/train_unsupervised.py:92-200`（特徵選擇階段）
- `scripts/visualize_feature_distributions.py:1149-1187`（已有類似實現）

**實施狀態**：⬜ 待實施

---

### 9. 多模型集成策略（協議分組 + 多算法集成）

**問題描述**：
- 單一模型（Isolation Forest）可能無法捕捉所有異常模式
- 不同模型對不同類型的異常敏感度不同
- 通過模型集成可以提高整體性能和穩定性

**當前狀況**：
- 📍 需要先實施協議分組（項目 4）
- 📍 需要先實施 LOF 和 One-Class SVM 模型（項目 6、7）

**建議方案**：
```python
# 多模型集成框架
def ensemble_anomaly_detection(
    X: pd.DataFrame, 
    features_df: pd.DataFrame,
    contamination: float = 0.1
) -> tuple:
    """
    整合多個協議和多個模型的預測結果
    
    策略：
    1. 按協議分組（TCP/UDP）
    2. 每個協議訓練多個模型（Isolation Forest, LOF, One-Class SVM）
    3. 使用加權平均整合異常分數
    4. 使用統一的閾值進行最終預測
    """
    from src.models import ModelFactory, ModelType
    
    protocol_models = {}
    all_scores = {}
    
    # 按協議分組
    if 'Proto' in features_df.columns:
        protocols = features_df['Proto'].unique()
    else:
        protocols = ['all']  # 如果沒有協議欄位，使用全部數據
    
    for protocol in protocols:
        if protocol == 'all':
            mask = pd.Series([True] * len(features_df), index=features_df.index)
        else:
            mask = features_df['Proto'] == protocol
        
        X_proto = X[mask]
        features_df_proto = features_df[mask]
        
        if len(X_proto) < 1000:  # 樣本數不足，跳過
            continue
        
        print(f"\n   訓練 {protocol.upper()} 協議模型...")
        
        # 計算該協議的 contamination
        if 'Label' in features_df_proto.columns:
            y_true_proto = (features_df_proto['Label'].str.contains('Botnet', case=False, na=False)).astype(int)
            contamination_proto = min(y_true_proto.sum() / len(y_true_proto) * 1.3, 0.2)
        else:
            contamination_proto = contamination
        
        # 訓練多個模型
        models = {}
        scores = {}
        
        # 1. Isolation Forest（權重 0.5）
        print(f"      訓練 Isolation Forest...")
        if_model = ModelFactory.create(ModelType.ISOLATION_FOREST)
        if_model.train(X_proto, contamination=contamination_proto, random_state=42)
        scores['isolation_forest'] = -if_model.predict(X_proto)  # 轉為正數
        models['isolation_forest'] = if_model
        
        # 2. LOF（權重 0.3）
        try:
            print(f"      訓練 LOF...")
            lof_model = ModelFactory.create(ModelType.LOF)
            lof_model.train(X_proto, contamination=contamination_proto, n_neighbors=20)
            scores['lof'] = lof_model.predict_scores(X_proto)
            models['lof'] = lof_model
        except Exception as e:
            print(f"      ⚠️  LOF 訓練失敗：{e}")
            scores['lof'] = None
        
        # 3. One-Class SVM（權重 0.2）
        try:
            print(f"      訓練 One-Class SVM...")
            ocsvm_model = ModelFactory.create(ModelType.ONE_CLASS_SVM)
            ocsvm_model.train(X_proto, nu=contamination_proto, kernel='rbf')
            scores['ocsvm'] = ocsvm_model.predict_scores(X_proto)
            models['ocsvm'] = ocsvm_model
        except Exception as e:
            print(f"      ⚠️  One-Class SVM 訓練失敗：{e}")
            scores['ocsvm'] = None
        
        # 加權平均異常分數
        weights = {'isolation_forest': 0.5, 'lof': 0.3, 'ocsvm': 0.2}
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        valid_weights = {k: weights[k] for k in valid_scores.keys()}
        
        # 正規化權重
        total_weight = sum(valid_weights.values())
        normalized_weights = {k: v / total_weight for k, v in valid_weights.items()}
        
        # 標準化每個模型的分數到 [0, 1] 區間
        normalized_scores = {}
        for name, score in valid_scores.items():
            score_min, score_max = score.min(), score.max()
            if score_max > score_min:
                normalized_scores[name] = (score - score_min) / (score_max - score_min)
            else:
                normalized_scores[name] = score
        
        # 加權平均
        ensemble_score = np.zeros(len(X_proto))
        for name, score in normalized_scores.items():
            ensemble_score += normalized_weights[name] * score
        
        # 存儲結果
        protocol_models[protocol] = models
        all_scores[protocol] = {
            'scores': ensemble_score,
            'indices': X_proto.index
        }
    
    # 整合所有協議的分數
    final_scores = np.zeros(len(X))
    for protocol, score_data in all_scores.items():
        final_scores[score_data['indices']] = score_data['scores']
    
    # 使用統一的閾值（基於 contamination）
    threshold = np.percentile(final_scores, 100 * (1 - contamination))
    y_pred = (final_scores >= threshold).astype(int)
    
    return y_pred, final_scores, protocol_models
```

**相關文件**：
- 需要先實施項目 4（協議分組）
- 需要先實施項目 6（LOF 模型）
- 需要先實施項目 7（One-Class SVM 模型）
- `src/models.py`（需要新增 ModelType.LOF 和 ModelType.ONE_CLASS_SVM）

**實施狀態**：⬜ 待實施（依賴項目 4、6、7）

**預期效果**：
- 通過多模型集成，可降低 15-25% 的 False Positive
- 可降低 20-30% 的 False Negative
- 整體 Accuracy 提升 5-10%

---

## 📋 實施檢查清單

### 高優先級
- [x] 1. 調整 Contamination 參數策略 ✅ 已實施（2024-12-19）
- [ ] 2. 改用異常分數排序（Scoring 模式）
- [ ] 3. 檢查並移除可能洩漏標籤的特徵（IP/Port）

### 中優先級
- [x] 4. 按協議分組訓練（TCP/UDP 分開） ✅ 已實施（2024-12-19）
- [x] 5. 條件移除 `hour` 特徵（如果重要性過高） ✅ 已實施（2024-12-19）
- [x] 6. 新增 Local Outlier Factor (LOF) 模型 ✅ 已實施
- [x] 7. 新增 One-Class SVM 模型 ✅ 已實施

### 低優先級
- [ ] 8. 特徵重要性分析與自動過濾
- [ ] 9. 多模型集成策略（協議分組 + 多算法集成）

---

## 📝 實施記錄

### 2024-12-XX
- ✅ 更新實施狀態：項目 6（LOF 模型）和項目 7（One-Class SVM 模型）已完成
- ✅ 確認協議分組訓練功能正常運作
- ✅ 更新實施檢查清單和優先順序建議

### 2024-12-19
- ✅ 創建改進計劃文檔
- ✅ 更新執行摘要：加入當前狀況（Accuracy 80%+，但 FP/FN 仍高）
- ✅ 新增項目 6：Local Outlier Factor (LOF) 模型
- ✅ 新增項目 7：One-Class SVM 模型
- ✅ 擴展項目 9：多模型集成策略（協議分組 + 多算法集成）
- ✅ **實施項目 1**：調整 Contamination 參數策略（動態調整，根據實際異常比例）
- ✅ **實施項目 4**：按協議分組訓練（TCP/UDP 分開，自動檢測和分組）
- ✅ **實施項目 5**：條件移除 hour 特徵（基於重要性檢查，自動移除）

### 當前狀況（2024-12-XX）
- **Accuracy**: 80%+ ✅
- **False Positive**: 108,520 ⚠️（需要降低）
- **False Negative**: 178,694 ⚠️（需要降低）
- **已完成項目**：
  - ✅ 動態 Contamination 參數策略（項目 1）
  - ✅ 協議分組訓練（項目 4）
  - ✅ 條件移除 hour 特徵（項目 5）
  - ✅ LOF 模型（項目 6）
  - ✅ One-Class SVM 模型（項目 7）
- **下一步重點**：實施異常分數排序（項目 2）和模型集成策略（項目 9）

---

## 🔗 相關文檔

- [XGBoost 過擬合檢查](./XGBOOST_OVERFITTING_CHECK.md)
- [特徵工程文檔](../src/feature_engineer.py)
- [模型訓練腳本](../scripts/train_unsupervised.py)

---

## 💡 備註

1. **優先級說明**：
   - 🔴 高優先級：影響模型核心性能，建議立即實施
   - 🟡 中優先級：能顯著改善模型效果，建議短期內實施
   - 🟢 低優先級：優化項目，可以長期規劃

2. **實施建議**：
   - 按照優先級順序逐步實施
   - 每個項目實施後進行測試和驗證
   - 記錄實施效果，必要時調整策略

3. **測試驗證**：
   - 每個改進項目實施後，需要重新評估模型性能
   - 比較改進前後的指標（Accuracy, Precision, Recall, F1）
   - 特別關注 False Positive 和 False Negative 的變化

4. **實施優先順序建議**：
   - **第一階段**（已完成）：項目 1、4、5、6、7 ✅
   - **第二階段**（進行中）：項目 2（異常分數排序）- 提升預測靈活性
   - **第三階段**（待實施）：項目 3（標籤洩漏檢查）- 確保模型泛化能力
   - **第四階段**（未來）：項目 9（模型集成）- 結合 Isolation Forest + LOF + One-Class SVM

