"""
Network Anomaly Detection - 核心模組

本套件提供網路異常流量偵測的核心功能，使用 Factory Pattern 設計：
- 模型工廠：統一 Isolation Forest 和 XGBoost 的介面
- 資料載入器工廠：統一不同資料來源的載入邏輯
"""

__version__ = "0.1.0"

# 匯出模型相關類別
from src.models import (
    ModelType,
    BaseModel,
    IsolationForestModel,
    XGBoostModel,
    LOFModel,
    OneClassSVMModel,
    ModelFactory,
)

# 匯出資料載入器相關類別
from src.data_loader import (
    DataSourceType,
    BaseDataLoader,
    BidirectionalBinetflowLoader,
    BidirectionalBinetflowLoaderSpark,
    APIDataLoader,
    DataLoaderFactory,
)

# 匯出特徵工程相關函數
from src.feature_engineer import (
    extract_features,
    get_feature_columns,
)

# 匯出特徵轉換相關函數
from src.feature_transformer import (
    DEFAULT_SKEWED_FEATURES,
    apply_log_transformation,
    apply_robust_scaling,
    transform_features_for_unsupervised,
    get_transformed_feature_names,
    apply_sqrt_transformation,
    apply_boxcox_transformation,
    calculate_transformation_metrics,
)

# 匯出標籤處理相關函數
from src.label_processor import (
    convert_label_to_binary,
    get_label_statistics,
    validate_labels,
)

# 匯出特徵選擇相關類別和函數
from src.feature_selector import (
    BASE_STATISTICAL_FEATURES,
    FeatureSelectionStrategy,
    FeatureSelector,
    prepare_feature_set,
    get_base_statistical_features,
)

# 匯出特徵處理器相關類別
from src.feature_processor import (
    BaseFeatureProcessor,
    StandardFeatureProcessor,
    create_feature_processor,
)

# 匯出訓練工具相關函數
from src.training_utils import (
    calculate_contamination,
    train_single_model,
    train_protocol_grouped_models,
)

# 匯出白名單相關類別
from src.whitelist import (
    WhitelistRuleType,
    WhitelistAnalyzer,
    WhitelistApplier,
)

# 匯出評估工具相關類別和函數
from src.evaluator import (
    EvaluationMetrics,
    calculate_metrics,
    print_confusion_matrix,
    print_metrics_summary,
    print_metrics_detailed,
    evaluate_and_print,
    compare_metrics,
    compare_train_test_metrics,
)

__all__ = [
    # 模型相關
    'ModelType',
    'BaseModel',
    'IsolationForestModel',
    'XGBoostModel',
    'LOFModel',
    'OneClassSVMModel',
    'ModelFactory',
    # 資料載入器相關
    'DataSourceType',
    'BaseDataLoader',
    'BidirectionalBinetflowLoader',
    'BidirectionalBinetflowLoaderSpark',
    'APIDataLoader',
    'DataLoaderFactory',
    # 特徵工程相關
    'extract_features',
    'get_feature_columns',
    # 特徵轉換相關
    'DEFAULT_SKEWED_FEATURES',
    'apply_log_transformation',
    'apply_robust_scaling',
    'transform_features_for_unsupervised',
    'get_transformed_feature_names',
    'apply_sqrt_transformation',
    'apply_boxcox_transformation',
    'calculate_transformation_metrics',
    # 標籤處理相關
    'convert_label_to_binary',
    'get_label_statistics',
    'validate_labels',
    # 特徵選擇相關
    'BASE_STATISTICAL_FEATURES',
    'FeatureSelectionStrategy',
    'FeatureSelector',
    'prepare_feature_set',
    'get_base_statistical_features',
    # 特徵處理器相關
    'BaseFeatureProcessor',
    'StandardFeatureProcessor',
    'create_feature_processor',
    # 訓練工具相關
    'calculate_contamination',
    'train_single_model',
    'train_protocol_grouped_models',
    # 白名單相關
    'WhitelistRuleType',
    'WhitelistAnalyzer',
    'WhitelistApplier',
    # 評估工具相關
    'EvaluationMetrics',
    'calculate_metrics',
    'print_confusion_matrix',
    'print_metrics_summary',
    'print_metrics_detailed',
    'evaluate_and_print',
    'compare_metrics',
    'compare_train_test_metrics',
]

