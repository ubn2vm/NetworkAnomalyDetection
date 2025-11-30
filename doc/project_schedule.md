## Network Anomaly Detection Side Project Plan

### 專案架構

network-anomaly-detection/
├── data/
│   └── capture20110817.pcap.netflow.labeled
|   └── capture20110817_cleaned.parquet
├── notebooks/
│   ├── 01_EDA_and_Cleaning.ipynb  (展示您對資料的理解)
│   ├── 02_Feature_Engineering_Spark.ipynb (展示 Spark 能力)
│   └── 03_Model_Training_IsoForest_vs_XGB.ipynb (展示模型比較)
├── src/
│   ├── features.py (封裝特徵工程邏輯，證明您會寫模組化程式碼)
│   └── pipeline.py (模擬由 Airflow 呼叫的 Script)
├── README.md (最重要的部分！寫下您的思考流程、遇到的困難、如何解決 FP)
└── requirements.txt

### 落地建議
先新增 notebooks/01_EDA_and_Cleaning.ipynb，使用 Parquet 做欄位/資料量分析（完成階段一剩餘任務）。
建立 src/features.py，把 Notebook 中的特徵工程轉成函數。
更新 README，描述新的目錄與 demo 流程。
後續 Notebook 逐步完成 Feature Engineering、模型比較，並在 README/doc/ 中記錄觀察與決策。


### Guiding Priorities
- 建立具 repeatable 的端到端流程，並以「階段一到階段五」逐步滿足 JD 所需之資料、特徵、模型、驗證與誤報抑制能力。
- 以 pytest 撰寫單元/整合測試。
- 每一階段完成後即更新 README 與 `/doc`，留下決策紀錄與下一步規畫。

### Phase Overview
1. **階段一：資料攝取與預處理**  
   - 模擬 Data Lake → 讀取 `.labeled`、清洗欄位型別。  
   - 標籤二元化（Background/LEGIT → 0，Botnet → 1），並視需要排除不明背景流量。  
   - 工具：Pandas 為主。
2. **階段二：特徵工程（時間窗口聚合）**  
   - 以 Source IP + 1 分鐘窗口計算 Count、Diversity、Bytes Ratio、Port Entropy 等統計。  
   - 建立可重用的特徵表輸出與測試，證明 Raw NetFlow 不可直接進模型。
3. **階段三：模型建立與比較**  
   - Model A（Supervised XGBoost）：作為上限，提供 Feature Importance。  
   - Model B（Unsupervised Isolation Forest / Autoencoder）：設定 contamination=1% 等參數，評估無標籤實力。
4. **階段四：驗證與指標**  
   - 以時間序列切分 Train/Test；指標聚焦 Precision、Recall、PR/ROC、Confusion Matrix。  
   - 清楚說明為何 Accuracy 不適用於高度不平衡資料。
5. **階段五：降低誤報**  
   - Post-processing Rules（白名單/黑名單）、Ensemble Voting、Human-in-the-loop 模擬。  
   - 每項策略皆需程式化，並以 pytest 撰寫單元測試驗證。
