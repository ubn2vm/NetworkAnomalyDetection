# 報告生成器使用說明

## 概述

`scripts/generate_report.py` 是一個統一的 HTML 報告生成器，整合了以下功能：

1. **EDA 和特徵轉換分析**（來自 `visualize_feature_distributions.py`）
2. **特徵處理流程可視化**（來自 `visualize_feature_pipeline.py`）
3. **模型選擇理由說明**（Isolation Forest vs 其他模型）
4. **XGBoost 和 Isolation Forest 特徵對比**
5. **白名單機制說明和效果展示**
6. **視覺化圖表**：
   - 漏斗圖（原始異常 → 白名單過濾 → 最終異常）
   - 特徵重要性圖（Top 15 特徵）

## 使用方法

### 基本使用

```bash
# 生成完整的報告（包含所有內容）
python scripts/generate_report.py

# 指定輸出路徑
python scripts/generate_report.py --output output/my_report.html

# 不包含白名單資訊
python scripts/generate_report.py --include-whitelist=False

# 不包含 XGBoost 特徵重要性
python scripts/generate_report.py --include-xgb=False
```

### 完整流程（建議執行順序）

```bash
# 1. 先運行模型評估（生成 JSON 結果）
python scripts/unsupervised_model_selection/quick_model_benchmark.py

# 2. 運行監督學習訓練（生成特徵重要性）
python scripts/train_supervised.py

# 3. 運行無監督學習訓練（生成模型結果）
python scripts/train_unsupervised.py

# 4. 運行白名單後處理（生成白名單資訊）
python scripts/postprocess_with_whitelist.py

# 5. 生成統一的報告（讀取上述步驟的結果）
python scripts/generate_report.py
```

## 報告內容

生成的 HTML 報告包含以下章節：

1. **專案概述與方法論**
   - 為什麼選擇 Isolation Forest
   - 方法論流程

2. **模型選擇：小樣本快速評估**
   - Isolation Forest vs LOF vs One-Class SVM 對比

3. **特徵重要性分析（監督學習輔助）**
   - 為什麼使用監督學習分析特徵重要性
   - Top 15 特徵重要性圖
   - 關鍵發現

4. **特徵工程與轉換**
   - 特徵處理流程
   - Log-Transformation + RobustScaler 說明
   - 設計模式應用

5. **白名單機制：False Positive 優化**
   - 為什麼需要白名單
   - 漏斗圖（視覺化過濾效果）
   - 白名單效果統計

6. **架構設計與設計模式**
   - Factory Pattern
   - Strategy Pattern
   - Abstract Base Class

7. **分散式處理與性能考量**
   - 已實作的 PySpark 支援
   - 為什麼沒有使用 PySpark（誠實說明）
   - 性能優化措施

8. **最終成果與總結**
   - 關鍵指標
   - 改進歷程
   - 架構優勢

## 視覺化圖表

### 1. 漏斗圖（Funnel Chart）

展示白名單過濾效果：
- 原始預測異常數量
- 白名單過濾掉的數量
- 最終異常數量

**生成條件**：需要白名單資訊（從 `postprocess_with_whitelist.py` 獲取）

### 2. 特徵重要性圖（Feature Importance Chart）

展示 Top 15 特徵的重要性：
- 使用 XGBoost 監督學習模型分析
- 證明選取的特徵（如 `unique_dst_per_minute_by_src`）是有意義的

**生成條件**：需要先運行 `train_supervised.py` 生成特徵重要性文件（`output/evaluations/xgb_feature_importance.json`）

## 輸出文件

- **HTML 報告**：`output/report/report.html`（預設）
- **漏斗圖**：`output/report/visualizations/whitelist_funnel_chart.png`
- **特徵重要性圖**：`output/report/visualizations/feature_importance_chart.png`

所有報告相關文件都會保存在 `output/report/` 目錄下，方便管理和查看。

## 注意事項

1. **白名單資訊**：如果沒有運行 `postprocess_with_whitelist.py`，腳本會嘗試從模型結果推斷（僅供演示）

2. **特徵重要性**：需要先運行 `train_supervised.py` 生成特徵重要性文件。報告生成器**不會自動訓練**，只讀取已保存的結果

3. **模型結果**：需要先運行 `quick_model_benchmark.py` 生成模型評估結果

4. **特徵資料**：需要先運行特徵處理流程生成特徵數據

## 技術特點

- **誠實謙虛**：誠實說明 PySpark 的狀況（已實作但未使用）
- **展現思考**：每個章節都說明設計決策和理由
- **展現架構**：詳細說明設計模式的應用
- **視覺化豐富**：包含漏斗圖和特徵重要性圖

## 故障排除

### 問題：找不到模型結果文件

**解決方法**：
```bash
python scripts/unsupervised_model_selection/quick_model_benchmark.py
```

### 問題：無法生成漏斗圖

**解決方法**：
```bash
python scripts/postprocess_with_whitelist.py
```

### 問題：無法獲取特徵重要性

**原因**：需要標籤資料
**解決方法**：確保資料包含 `Label` 欄位

