## Network Anomaly Detection - 專案執行計劃與進度追蹤

### 專案目標
建立端到端的網路異常流量偵測系統，展示：
- Big Data 處理能力（PySpark）
- 無監督學習應用（Isolation Forest）
- 特徵工程與模型比較（XGBoost vs Isolation Forest）
- 生產級系統考量（誤報降低、協議特定特徵）

---

## ✅ 已完成階段

### 階段一：資料攝取與預處理 ✅
**狀態：** 已完成  
**檔案：** `notebooks/01_EDA_and_Cleaning.ipynb`, `scripts/load_raw_data.py`

**實作內容：**
- [x] 讀取 CTU-13 NetFlow 資料（`.labeled` 格式）
- [x] 資料型別轉換與清洗
- [x] 標籤二元化（Background/LEGIT → 0, Botnet → 1）
- [x] 資料品質檢查（缺失值、異常值、極端值）
- [x] 儲存為 Parquet 格式（`data/processed/capture20110817_cleaned.parquet`）

**技術細節：**
- 使用 Pandas 進行資料處理
- 處理 8,087,512 筆 NetFlow 記錄
- 標籤分布：Background 90.48%, LEGITIMATE 4.78%, Botnet 4.74%

---

### 階段二：特徵工程（時間窗口聚合）✅
**狀態：** 已完成（基礎功能）  
**檔案：** `notebooks/02_Feature_Engineering_Spark.ipynb`

**實作內容：**
- [x] PySpark Session 初始化與環境設定
- [x] IP:Port 欄位拆分（`src_ip`, `src_port`, `dst_ip`, `dst_port`）
- [x] 時間窗口聚合（Source IP × 1 分鐘窗口）

**已實作特徵類別：**

1. **基礎統計特徵** (`features_base`)
   - `flow_count`: 流量數量
   - `total_bytes`: 總位元組數
   - `total_packets`: 總封包數
   - `avg_duration`, `min_duration`, `max_duration`: 持續時間統計
   - `label`: 窗口標籤（窗口內有任一 Botnet 流量則標記為 1）

2. **Diversity 特徵** (`features_diversity`)
   - `dst_ip_diversity`: `countDistinct(dst_ip)` - unique destination IP count
   - `dst_port_diversity`: `countDistinct(dst_port)` - unique destination port count
   - `protocol_diversity`: `countDistinct(Prot)` - unique protocol count
   - **時間窗口：** 1 分鐘滑動窗口
   - **聚合維度：** `(src_ip, time_window)`

3. **Ratio 特徵** (`features_ratio`)
   - `bytes_per_flow`: 平均每個流量的位元組數
   - `packets_per_flow`: 平均每個流量的封包數

4. **Entropy 特徵** (`features_entropy`)
   - `port_entropy`: Port 分布的熵值（衡量隨機性）
   - `protocol_entropy`: Protocol 分布的熵值

**技術細節：**
- 使用 PySpark Window Functions (`window(timestamp, "1 minute")`)
- **混合式存儲策略：** Spark 運算 → Pandas/Arrow 序列化
  - ⚠️ **技術債務：** 因 Windows 環境 Hadoop winutils.exe 權限問題，採用此短期解法
  - 長期改善：見「技術債務與改進項目」區塊
- 輸出：`data/processed/features.parquet`

---

### 階段三：模型建立與比較 ✅
**狀態：** 已完成  
**檔案：** `notebooks/03_Model_Training.ipynb`

**實作內容：**
- [x] Isolation Forest（無監督學習）
  - 不使用標籤進行訓練
  - 異常分數分布視覺化（KDE + 獨立歸一化）
  - 結果：Botnet 平均分數 0.52 vs 正常流量 0.35（差異 46%）

- [x] XGBoost（監督學習）
  - 使用標籤進行訓練
  - Feature Importance 分析
  - 關鍵特徵：`total_bytes` (0.6681), `max_duration` (0.0840), `avg_duration` (0.0776)

**技術細節：**
- 資料清理：處理無限值 (inf) 與缺失值
- 特徵標準化：StandardScaler
- 視覺化：Density Estimation (KDE) 解決極度不平衡資料的視覺化問題

---

### 階段四：驗證與指標 ✅
**狀態：** 部分完成（在 `03_Model_Training.ipynb` 中）

**實作內容：**
- [x] 模型訓練與評估
- [x] 視覺化結果（Anomaly Score Distribution, Feature Importance）

---

### 測試框架 ✅
**狀態：** 基礎框架已建立  
**檔案：** `tests/`

**實作內容：**
- [x] `conftest.py`: pytest 設定
- [x] `test_data_quality.py`: 資料品質測試
- [x] `test_label_processor.py`: 標籤處理測試

---

## 🚧 各階段待完成項目

### 階段二：特徵工程 - 待完成項目

#### 2.1 SMB 專用特徵
**優先級：** 高  

**執行步驟：**

1. **在 `02_Feature_Engineering_Spark.ipynb` 中新增 SMB 專用特徵計算**
   ```python
   # 過濾 SMB 流量（port 445 或 139）
   smb_flows = df.filter(
       (col("dst_port").isin([445, 139])) | 
       (col("Prot") == "SMB")
   )
   
   # 計算 SMB 專用的目標 IP 多樣性
   features_smb = smb_flows.groupBy(
       "src_ip",
       window(col("timestamp"), "1 minute").alias("time_window")
   ).agg(
       countDistinct("dst_ip").alias("smb_dst_ip_diversity"),
       count("dst_ip").alias("smb_connection_count"),
       countDistinct("dst_port").alias("smb_port_diversity")
   )
   ```

2. **合併 SMB 特徵到最終特徵表**
   - 修改特徵合併邏輯，加入 `features_smb`
   - 確保與其他特徵的 join key 一致（`src_ip`, `time_window`）

3. **驗證與測試**
   - 檢查 SMB 特徵的統計分布
   - 確認特徵表完整性

**技術考量：**
- 正常情況下，員工電腦很少同時連接大量其他員工電腦的 SMB
- 結合 `protocol` 和 `dst_port` 特徵，提升 SMB 橫向移動偵測準確度

---

#### 2.2 特徵工程文件補充
**優先級：** 中  

**執行步驟：**

1. **更新 `02_Feature_Engineering_Spark.ipynb` 的 Markdown Cell**
   - 在 "### 4.2 Diversity 特徵" 區塊中補充：
     - 明確說明 `dst_ip_diversity` 計算方式：`countDistinct(dst_ip)` = unique count（非 entropy）
     - 時間窗口：1 分鐘
     - 聚合維度：`(src_ip, time_window)`
     - Diversity vs Entropy 的區別

2. **更新程式碼註解**
   - 在計算 Diversity 的程式碼區塊中加入詳細註解

**文件內容範例：**
```markdown
**dst_ip_diversity（目標 IP 多樣性）**
- **計算方式：** `countDistinct(dst_ip)` - 即 **unique destination IP count**（非 entropy）
- **時間窗口：** 每個 `src_ip` 在每個 1 分鐘窗口內連接的不同 `dst_ip` 數量
- **數值意義：**
  - 高值（例如 > 50）：可能表示掃描行為、DDoS 攻擊、橫向移動
  - 低值（例如 = 1）：正常的一對一連接行為
```

---

### 階段四：驗證與指標 - 待完成項目

#### 4.1 Time Series Cross-Validation
**優先級：** 中  

**執行步驟：**

1. **實作 Rolling Window Cross-Validation**
   - 在 `03_Model_Training.ipynb` 中新增時間序列切分邏輯
   - 避免 Look-ahead bias（使用未來資料預測過去）

2. **驗證邏輯**
   - 確保訓練集時間早於測試集時間
   - 實作滑動窗口機制

---

#### 4.2 詳細評估指標計算
**優先級：** 中  

**執行步驟：**

1. **計算分類指標**
   - Precision, Recall, F1-Score
   - 針對不平衡資料調整閾值

2. **視覺化評估結果**
   - PR Curve (Precision-Recall Curve)
   - ROC Curve (Receiver Operating Characteristic Curve)
   - Confusion Matrix

3. **分析與報告**
   - 比較 Isolation Forest 與 XGBoost 的表現
   - 說明為何 Accuracy 不適用於高度不平衡資料

---

#### 4.3 模型性能優化
**優先級：** 高  
**狀態：** 待實作  
**檔案：** `notebooks/03_Model_Training.ipynb`

**背景分析：**
根據預測概率分佈分析，XGBoost 模型表現如下：
- ✅ 正常樣本：99.85% 有高信心度（概率 < 0.3），誤判率極低
- ⚠️ 異常樣本：僅 39.68% 有高信心度（概率 > 0.7），約 30% 被漏判
- 📊 不確定預測：僅 0.12%，模型很少猶豫

**執行步驟：**

1. **分類閾值優化**
   - 實作閾值掃描（Threshold Sweeping），測試 0.1 到 0.9 的不同閾值
   - 計算每個閾值下的 Precision、Recall、F1-Score
   - 繪製 Precision-Recall 曲線，找出最佳平衡點
   - 目標：從預設 0.5 調整至 0.3-0.4，提升召回率

2. **異常樣本增強**
   - 分析被漏判的異常樣本特徵（預測概率 < 0.1 的異常樣本）
   - 實作 SMOTE (Synthetic Minority Oversampling Technique) 或 ADASYN
   - 增加訓練集中異常樣本的數量與多樣性
   - 目標：減少 30% 的漏判率

3. **針對性特徵工程**
   - 分析難以識別的異常樣本（預測概率在 0.1-0.7 區間）
   - 設計專用特徵捕捉這些邊界情況的模式
   - 例如：時間序列特徵、頻率域特徵、交互特徵
   - 目標：提升模型對邊界情況的識別能力

4. **集成方法實作**
   - 結合 Isolation Forest 和 XGBoost 的預測結果
   - 實作投票機制（Voting Ensemble）：
     - 如果任一模型預測為異常，則標記為異常（提高召回率）
     - 或使用加權投票（根據模型信心度）
   - 比較集成方法與單一模型的性能
   - 目標：結合無監督與監督學習的優勢

**預期成果：**
- 異常樣本召回率從目前水平提升至 70% 以上
- 維持正常樣本的高精確率（> 99%）
- 建立可調整的閾值機制，適應不同業務場景需求

**技術細節：**
- 使用 `sklearn.metrics.precision_recall_curve` 進行閾值優化
- 使用 `imbalanced-learn` 套件進行樣本增強
- 使用 `sklearn.ensemble.VotingClassifier` 實作集成方法

---

### 工程化項目

#### 模組化與封裝
**優先級：** 中  

**執行步驟：**

1. **建立 `src/features.py`**
   - 將 `02_Feature_Engineering_Spark.ipynb` 中的特徵工程邏輯封裝為函數
   - 函數應可重複使用，支援批次處理
   - 包含完整的 docstring 和類型提示

2. **建立 `src/pipeline.py`**
   - 模擬 Airflow 呼叫的 Script
   - 端到端流程：資料攝取 → 特徵工程 → 模型訓練 → 預測
   - 支援命令列參數與配置檔案

3. **測試與文件**
   - 為封裝的函數撰寫單元測試
   - 更新 README.md 說明使用方式

---

### 階段五：降低誤報（False Positive Reduction）
**狀態：** 待規劃

**說明：** 此階段目前尚未有具體實作項目，未來可考慮加入：
- 白名單機制（IT 管理員、檔案伺服器、備份系統）
- 後處理規則（Post-processing Rules）
- Ensemble Voting
- Human-in-the-loop 模擬

---

## 📋 技術債務與改進項目

### 文件改進
- [ ] 補充 README.md 中的技術細節說明
- [ ] 建立 API 文件（如果實作 Model Serving）

### 測試覆蓋率
- [ ] 增加特徵工程的單元測試
- [ ] 增加模型訓練的整合測試

### 效能優化
- [ ] Spark 運算效能調優（partition 策略）
- [ ] 特徵表儲存格式優化（壓縮率、讀取速度）

### 技術債務
- [ ] **混合式存儲策略優化（Windows 環境限制）**
  - **現況：** 使用 Spark 進行記憶體內運算，最終階段轉由 Pandas/Arrow 引擎進行序列化存儲
  - **背景：**
    - Windows 開發環境下，Spark 寫入 Parquet 時遭遇 Hadoop winutils.exe 權限衝突 (WinError 5)
    - 採用短期解法：利用 Spark 進行記憶體內運算，最終階段轉由 Pandas/Arrow 引擎進行序列化存儲
  - **影響：**
    - ✅ **優點：** 成功解決 Windows 環境問題，維持 Spark 分散式運算優勢
    - ⚠️ **限制：**
      - 無法充分利用 Spark 的分散式寫入優勢
      - 需要將整個 DataFrame 轉換為 Pandas（記憶體瓶頸）
      - 不適合處理超大資料集（超過單機記憶體容量）
  - **長期改善方向：**
    1. **生產環境部署（推薦）：** 在 Linux 環境部署，避免 Windows 限制
    2. **開發環境配置：** 配置 Hadoop winutils.exe，解決權限問題
    3. **替代方案：** 使用 Delta Lake 或 Iceberg，提供更好的寫入策略
    4. **Spark 寫入優化：** 使用 `coalesce(1)` + 直接寫入 Parquet（需解決權限問題）
  - **優先級：** 低（目前方案可滿足開發需求，生產環境應使用 Linux）

---

## 🔗 相關文件

- `README.md`: 專案概述與對外展示
- `notebooks/01_EDA_and_Cleaning.ipynb`: 資料探索與清理
- `notebooks/02_Feature_Engineering_Spark.ipynb`: 特徵工程（核心邏輯）
- `notebooks/03_Model_Training.ipynb`: 模型訓練與視覺化
