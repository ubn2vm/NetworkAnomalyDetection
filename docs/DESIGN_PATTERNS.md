# 設計模式總覽

本文件記錄 Network Anomaly Detection 專案中使用的所有設計模式，包括實作細節、設計決策與使用指南。

**目標**：統一管理設計模式文檔，提升程式碼的可維護性與擴展性。

---

## 📋 目錄

1. [Factory Pattern](#factory-pattern)
2. [Strategy Pattern](#strategy-pattern)
3. [Abstract Base Class Pattern](#abstract-base-class-pattern)
4. [設計原則](#設計原則)

---

## 🏭 Factory Pattern

### 概述

使用 Factory Pattern 簡化並統一資料載入、模型創建的流程，實現解耦與擴展性。

### 1. 模型工廠 (Model Factory)

#### 設計目標
- 統一不同模型的訓練與預測介面
- 支援無監督學習（Isolation Forest、LOF、One-Class SVM）和監督學習（XGBoost）
- 易於擴展新模型類型

#### 類別結構

```
BaseModel (抽象基類)
├── train(X, y=None, **kwargs) -> Any
├── predict(X) -> np.ndarray
└── (其他共用方法)

IsolationForestModel (實作)
├── train(X, y=None, contamination=0.1, ...) -> (model, scaler)
└── predict(X) -> anomaly_scores

XGBoostModel (實作)
├── train(X, y, test_size=0.2, ...) -> (model, metrics)
└── predict(X) -> predictions

LOFModel (實作)
├── train(X, y=None, n_neighbors=20, ...) -> (model, scaler)
└── predict(X) -> anomaly_scores

OneClassSVMModel (實作)
├── train(X, y=None, nu=0.1, ...) -> (model, scaler)
└── predict(X) -> anomaly_scores

ModelFactory (工廠類別)
└── create(model_type: ModelType) -> BaseModel
```

#### 檔案位置
- `src/models.py` - 包含所有模型類別和工廠

#### 使用範例

```python
from src.models import ModelFactory, ModelType

# 創建模型
model = ModelFactory.create(ModelType.ISOLATION_FOREST)

# 訓練模型
trained_model, scaler = model.train(X_train, contamination=0.1)

# 預測
anomaly_scores = model.predict(X_test)
```

#### 實作狀態
- [x] 定義 `ModelType` 枚舉（ISOLATION_FOREST, XGBOOST, LOCAL_OUTLIER_FACTOR, ONE_CLASS_SVM）
- [x] 建立 `BaseModel` 抽象基類
- [x] 實作 `IsolationForestModel` 類別
- [x] 實作 `XGBoostModel` 類別
- [x] 實作 `LOFModel` 類別
- [x] 實作 `OneClassSVMModel` 類別
- [x] 建立 `ModelFactory` 工廠類別
- [x] 添加 doctest 範例
- [ ] 撰寫單元測試 (`tests/test_models.py`)

---

### 2. 資料載入器工廠 (Data Loader Factory)

#### 設計目標
- 統一不同資料來源的載入介面
- 支援 binetflow 格式（雙向流）和 API 來源
- 支援 Spark 分散式載入
- 簡化資料清洗流程

#### 類別結構

```
BaseDataLoader (抽象基類)
├── load(file_path=None) -> pd.DataFrame
├── clean(df) -> pd.DataFrame
└── save_cleaned_data(df, output_path=None) -> Path

BidirectionalBinetflowLoader (實作)
├── load() - 讀取 .binetflow 格式（CSV）
└── clean() - 轉換 StartTime，處理數值欄位

BidirectionalBinetflowLoaderSpark (實作)
├── load() - 使用 Spark 讀取 .binetflow 格式
└── clean() - Spark 分散式資料清洗

APIDataLoader (實作)
├── load() - 從 API 載入資料（框架實作，待完善）
└── clean() - API 資料清洗（待實作）

DataLoaderFactory (工廠類別)
└── create(source_type: DataSourceType) -> BaseDataLoader
```

#### 檔案位置
- `src/data_loader.py` - 包含所有資料載入器類別和工廠

#### 使用範例

```python
from src.data_loader import DataLoaderFactory, DataSourceType

# 創建載入器
loader = DataLoaderFactory.create(DataSourceType.BIDIRECTIONAL_BINETFLOW)

# 載入資料
df = loader.load(file_path="data/raw/capture20110817.binetflow")

# 清洗資料
cleaned_df = loader.clean(df)
```

#### 實作狀態
- [x] 定義 `DataSourceType` 枚舉（BIDIRECTIONAL_BINETFLOW, BIDIRECTIONAL_BINETFLOW_SPARK, API）
- [x] 建立 `BaseDataLoader` 抽象基類
- [x] 實作 `BidirectionalBinetflowLoader` 類別
- [x] 實作 `BidirectionalBinetflowLoaderSpark` 類別
- [x] 實作 `APIDataLoader` 類別（框架實作，待完善）
- [x] 建立 `DataLoaderFactory` 工廠類別
- [x] 添加 doctest 範例
- [ ] 撰寫單元測試 (`tests/test_data_loader.py`)

---

### 3. 特徵處理器工廠函數

#### 設計決策
**使用工廠函數而非完整 Factory Pattern**，原因：
- 目前僅有一種實作（StandardFeatureProcessor）
- 未來如有需要可擴展為完整 Factory Pattern
- 避免過度設計

#### 檔案位置
- `src/feature_processor.py` - 包含特徵處理器類別和工廠函數

#### 使用範例

```python
from src.feature_processor import create_feature_processor

# 創建處理器
processor = create_feature_processor("standard")

# 提取特徵
features_df = processor.extract(cleaned_df)

# 轉換特徵
transformed_df, scaler, transformed_columns = processor.transform(features_df)
```

#### 實作狀態
- [x] 建立 `BaseFeatureProcessor` 抽象基類
- [x] 實作 `StandardFeatureProcessor` 類別
- [x] 建立 `create_feature_processor()` 工廠函數
- [x] 添加 doctest 範例

---

## 🎯 Strategy Pattern

### 概述

使用 Strategy Pattern 支援多種特徵選擇策略，允許在執行時動態選擇不同的演算法。

### 特徵選擇器 (Feature Selector)

#### 設計目標
- 支援多種特徵選擇策略（品質檢查、相關性分析、重要性選擇）
- 遵循單一職責原則，專門負責特徵選擇邏輯
- 可組合使用多種策略

#### 類別結構

```
FeatureSelectionStrategy (枚舉)
├── QUALITY_CHECK - 品質檢查
├── CORRELATION - 相關性分析
├── IMPORTANCE - 基於重要性
└── ALL - 全部策略

FeatureSelector (策略上下文)
├── select_features(X, strategies=None) -> (X_selected, removed_features)
├── _quality_check() - 品質檢查策略
├── _correlation_analysis() - 相關性分析策略
└── _importance_selection() - 重要性選擇策略
```

#### 檔案位置
- `src/feature_selector.py` - 包含特徵選擇器類別

#### 使用範例

```python
from src.feature_selector import FeatureSelector, FeatureSelectionStrategy

# 創建選擇器
selector = FeatureSelector(
    remove_constant=True,
    remove_low_variance=True,
    remove_high_correlation=True
)

# 使用特定策略
X_selected, removed = selector.select_features(
    X,
    strategies=[FeatureSelectionStrategy.QUALITY_CHECK, 
                FeatureSelectionStrategy.CORRELATION]
)

# 或使用全部策略
X_selected, removed = selector.select_features(
    X,
    strategies=[FeatureSelectionStrategy.ALL]
)
```

#### 支援的策略

1. **品質檢查 (QUALITY_CHECK)**
   - 移除常數特徵
   - 移除低變異數特徵
   - 移除無限值比例過高的特徵
   - 移除高缺失值特徵

2. **相關性分析 (CORRELATION)**
   - 移除高度相關的特徵（預設閾值：0.98）

3. **重要性選擇 (IMPORTANCE)**
   - 基於 XGBoost 特徵重要性
   - 移除重要性過低的特徵
   - 需要提供標籤資料

#### 實作狀態
- [x] 定義 `FeatureSelectionStrategy` 枚舉
- [x] 實作 `FeatureSelector` 類別
- [x] 實作品質檢查策略
- [x] 實作相關性分析策略
- [x] 實作重要性選擇策略
- [x] 添加 doctest 範例

---

## 🔷 Abstract Base Class Pattern

### 概述

使用 Abstract Base Class (ABC) 定義統一介面，確保所有實作類別遵循相同的契約。

### 1. BaseModel

#### 設計目標
- 定義所有模型必須實作的統一介面
- 確保模型訓練與預測的一致性
- 支援無監督和監督學習模型

#### 抽象方法

```python
@abstractmethod
def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> Any:
    """訓練模型"""
    pass

@abstractmethod
def predict(self, X: pd.DataFrame) -> np.ndarray:
    """預測"""
    pass
```

#### 實作類別
- `IsolationForestModel`
- `XGBoostModel`
- `LOFModel`
- `OneClassSVMModel`

#### 檔案位置
- `src/models.py`

---

### 2. BaseDataLoader

#### 設計目標
- 定義所有資料載入器必須實作的統一介面
- 確保資料載入與清洗的一致性

#### 抽象方法

```python
@abstractmethod
def load(self, file_path: Optional[Path] = None) -> pd.DataFrame:
    """載入資料"""
    pass

@abstractmethod
def clean(self, df: pd.DataFrame) -> pd.DataFrame:
    """清洗資料"""
    pass
```

#### 實作類別
- `BidirectionalBinetflowLoader`
- `BidirectionalBinetflowLoaderSpark`
- `APIDataLoader`

#### 檔案位置
- `src/data_loader.py`

---

### 3. BaseFeatureProcessor

#### 設計目標
- 定義所有特徵處理器必須實作的統一介面
- 確保特徵提取與轉換的一致性

#### 抽象方法

```python
@abstractmethod
def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """提取特徵"""
    pass

@abstractmethod
def transform(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Any, List[str]]:
    """轉換特徵"""
    pass
```

#### 實作類別
- `StandardFeatureProcessor`

#### 檔案位置
- `src/feature_processor.py`

---

## 📚 設計原則

### Factory Pattern 的優勢
1. **解耦**：使用者不需要知道具體實作類別
2. **擴展性**：新增模型或資料來源只需新增類別並註冊
3. **統一介面**：所有模型/載入器使用相同的介面
4. **易於測試**：可以輕鬆 mock 和替換實作

### Strategy Pattern 的優勢
1. **靈活性**：可以在執行時選擇不同的策略
2. **可擴展性**：新增策略只需實作新的方法
3. **單一職責**：每個策略專注於特定的選擇邏輯
4. **可組合性**：可以組合使用多種策略

### Abstract Base Class 的優勢
1. **契約保證**：確保所有實作類別遵循相同介面
2. **類型安全**：編譯時檢查介面實作
3. **文檔化**：明確定義類別必須實作的方法
4. **多型支援**：支援多型操作

### 簡化原則
1. **避免過度設計**：只在必要的地方使用設計模式
2. **保持簡單**：特徵工程使用統一函數而非 Factory Pattern
3. **向後相容**：盡量保持與現有程式碼的相容性
4. **漸進式重構**：逐步引入設計模式，不一次性重構

---

## 🔍 擴展指南

### 新增模型類型

1. 在 `ModelType` 枚舉中新增類型
2. 建立新的模型類別，繼承 `BaseModel`
3. 實作 `train()` 和 `predict()` 方法
4. 在 `ModelFactory._models` 中註冊新模型

```python
# 1. 新增枚舉
class ModelType(Enum):
    NEW_MODEL = "new_model"

# 2. 實作模型類別
class NewModel(BaseModel):
    def train(self, X, y=None, **kwargs):
        # 實作訓練邏輯
        pass
    
    def predict(self, X):
        # 實作預測邏輯
        pass

# 3. 註冊到工廠
ModelFactory._models[ModelType.NEW_MODEL] = NewModel
```

### 新增資料載入器

1. 在 `DataSourceType` 枚舉中新增類型
2. 建立新的載入器類別，繼承 `BaseDataLoader`
3. 實作 `load()` 和 `clean()` 方法
4. 在 `DataLoaderFactory._loaders` 中註冊新載入器

### 新增特徵選擇策略

1. 在 `FeatureSelectionStrategy` 枚舉中新增策略
2. 在 `FeatureSelector` 中實作對應的私有方法（如 `_new_strategy()`）
3. 在 `select_features()` 方法中加入策略判斷邏輯

---

## 📖 參考資料

- [Factory Pattern - Python Design Patterns](https://refactoring.guru/design-patterns/factory-method/python/example)
- [Strategy Pattern - Python Design Patterns](https://refactoring.guru/design-patterns/strategy/python/example)
- [Abstract Base Classes in Python](https://docs.python.org/3/library/abc.html)

---

## 📅 更新記錄

| 日期 | 更新內容 | 更新人 |
|------|---------|--------|
| 2024-XX-XX | 建立整合設計模式文檔，整合 Factory Pattern、Strategy Pattern 和 Abstract Base Class | - |

