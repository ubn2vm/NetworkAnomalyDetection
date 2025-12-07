# XGBoost 過擬合驗證報告

> **文檔狀態**：✅ 功能已實作完成，文檔保持更新

**最後更新**：2024-12-XX

---

## 概述

已為 XGBoost 模型加入過擬合檢測功能，包括：
1. Early Stopping 機制
2. 訓練集與驗證集性能對比
3. 過擬合風險評估
4. HTML 報告中的過擬合診斷部分

## 已完成的改進

### 1. 模型訓練改進 (`src/models.py`)

- ✅ 加入 `early_stopping_rounds` 參數（預設 10 輪）
- ✅ 使用 `eval_set` 監控訓練集和驗證集性能
- ✅ 計算訓練集和測試集的準確率差異
- ✅ 自動評估過擬合風險（low/medium/high）

**過擬合風險評估標準：**
- **Low**: 準確率差異 < 1%
- **Medium**: 準確率差異 1% - 5%
- **High**: 準確率差異 > 5%

### 2. 訓練腳本改進

#### `scripts/train_supervised.py`
- ✅ 加入 `early_stopping_rounds=10` 參數
- ✅ 輸出過擬合診斷信息（訓練集 vs 驗證集準確率）

#### `scripts/visualize_feature_distributions.py`
- ✅ 加入 `early_stopping_rounds=10` 參數
- ✅ 在控制台輸出過擬合診斷信息
- ✅ 在 HTML 報告中顯示過擬合診斷部分

### 3. HTML 報告改進

HTML 報告現在包含：
- 訓練集準確率
- 驗證集準確率
- 準確率差異
- 過擬合風險等級（帶顏色標示）
- 最佳迭代次數
- 改進建議（如果存在過擬合風險）

## 使用方法

### 方法 1：運行完整的特徵分析腳本（推薦）

```bash
python scripts/visualize_feature_distributions.py
```

這會：
1. 載入數據並進行特徵工程
2. 訓練 XGBoost 模型（帶過擬合檢測）
3. 生成包含過擬合診斷的 HTML 報告

### 方法 2：運行監督學習訓練腳本

```bash
python scripts/train_supervised.py
```

這會在控制台輸出過擬合診斷信息。

### 方法 3：運行快速過擬合檢查腳本

```bash
python scripts/evaluation/check_supervised_overfitting.py
```

## 過擬合診斷結果解讀

### 低風險 (Low Risk) ✅
- **準確率差異 < 1%**
- 模型泛化能力良好
- 不需要調整參數

### 中等風險 (Medium Risk) ⚠️
- **準確率差異 1% - 5%**
- 建議：
  - 降低模型複雜度（`max_depth`）
  - 增加正則化參數

### 高風險 (High Risk) ❌
- **準確率差異 > 5%**
- 建議：
  1. 降低 `max_depth`（例如從 6 降到 4）
  2. 增加 `subsample` 和 `colsample_bytree` 的隨機性（例如從 0.8 降到 0.7）
  3. 降低 `learning_rate` 並增加 `n_estimators`（例如 `learning_rate=0.05, n_estimators=300`）
  4. 增加 `early_stopping_rounds`（例如從 10 增加到 20）

## 當前模型參數

```python
{
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'early_stopping_rounds': 10,
    'scale_pos_weight': <自動計算>
}
```

## 預期結果

如果模型沒有過擬合，您應該看到：
- 訓練集準確率 ≈ 驗證集準確率（差異 < 1%）
- 過擬合風險：LOW
- 最佳迭代次數 < `n_estimators`（表示 early stopping 生效）

如果模型存在過擬合，您應該看到：
- 訓練集準確率 > 驗證集準確率（差異 > 1%）
- 過擬合風險：MEDIUM 或 HIGH
- HTML 報告中會顯示改進建議

## 注意事項

1. **數據分割**：模型內部會再次分割數據（80% 訓練，20% 驗證），這是在外部 train_test_split 之後的額外分割
2. **Early Stopping**：如果驗證集性能在 10 輪內沒有改善，訓練會提前停止
3. **最佳迭代次數**：`best_iteration` 顯示模型實際使用的迭代次數，可能小於 `n_estimators`

## 下一步

運行 `scripts/visualize_feature_distributions.py` 來生成包含過擬合診斷的完整報告。

報告位置：`output/visualizations/feature_analysis_report.html`

