"""
監督學習模型過擬合檢查工具（XGBoost）

檢查 XGBoost 監督學習模型是否存在過擬合問題。
支援兩種模式：
1. 真實資料模式（預設）：使用專案資料進行完整過擬合檢查
2. 快速測試模式：使用模擬資料快速驗證過擬合檢測功能

對應訓練腳本：scripts/training/train_supervised.py

使用方法：
    # 使用真實資料（預設）
    python scripts/evaluation/check_supervised_overfitting.py
    
    # 快速測試模式（使用模擬資料）
    python scripts/evaluation/check_supervised_overfitting.py --quick-test
    python scripts/evaluation/check_supervised_overfitting.py -q
    
    # 指定特徵階段（僅真實資料模式）
    python scripts/evaluation/check_supervised_overfitting.py --feature-stage 4
"""
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# 將專案根目錄加入 Python 路徑
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    DataLoaderFactory,
    DataSourceType,
    ModelFactory,
    ModelType,
    extract_features,
    prepare_feature_set
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np


def create_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    創建模擬資料（用於快速測試）
    
    Args:
        n_samples: 樣本數量
        n_features: 特徵數量
        random_state: 隨機種子
    
    Returns:
        (X, y): 特徵資料和標籤
    
    >>> X, y = create_synthetic_data(n_samples=100, n_features=5)
    >>> len(X), len(y)
    (100, 100)
    >>> X.shape[1]
    5
    """
    np.random.seed(random_state)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    # 創建一個簡單的分類任務
    y = pd.Series((X.iloc[:, 0] > 0).astype(int))
    return X, y


def load_real_data(feature_stage: int = 3) -> Tuple[pd.DataFrame, pd.Series]:
    """
    載入真實資料（完整流程）
    
    Args:
        feature_stage: 特徵階段（1-4）
    
    Returns:
        (X, y): 特徵資料和標籤
    """
    # 1. 載入資料
    print("\n[步驟 1] 載入資料...")
    loader = DataLoaderFactory.create(DataSourceType.BIDIRECTIONAL_BINETFLOW)
    
    parquet_path = Path("data/processed/capture20110817_cleaned_spark.parquet")
    if parquet_path.exists():
        print(f"   使用 Parquet: {parquet_path}")
        raw_df = pd.read_parquet(parquet_path, engine='pyarrow')
    else:
        print("   從 CSV 載入...")
        raw_df = loader.load()
    
    print(f"✅ 載入完成：{len(raw_df):,} 筆資料")
    
    # 2. 清洗資料
    print("\n[步驟 2] 清洗資料...")
    cleaned_df = loader.clean(raw_df)
    print(f"✅ 清洗完成：{len(cleaned_df):,} 筆資料")
    
    # 3. 特徵工程
    print("\n[步驟 3] 特徵工程...")
    features_df = extract_features(
        cleaned_df,
        include_time_features=True,
        time_feature_stage=feature_stage
    )
    print(f"✅ 特徵工程完成：{features_df.shape[1]} 個特徵")
    
    # 4. 準備特徵
    print("\n[步驟 4] 準備特徵...")
    X = prepare_feature_set(
        features_df,
        include_base_features=True,
        include_time_features=True,
        time_feature_stage=feature_stage
    )
    
    # 簡單的特徵選擇
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        X = X[[col for col in X.columns if col not in constant_features]]
    
    print(f"✅ 特徵準備完成：{len(X.columns)} 個特徵")
    
    # 5. 準備標籤
    print("\n[步驟 5] 準備標籤...")
    if 'Label' not in features_df.columns:
        raise ValueError("❌ 錯誤：缺少 'Label' 欄位")
    
    y = (features_df['Label'].str.contains('Botnet', case=False, na=False)).astype(int)
    print(f"   正常 (0): {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.2f}%)")
    print(f"   異常 (1): {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.2f}%)")
    
    return X, y


def diagnose_overfitting(
    train_metrics: Dict[str, Any],
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    model = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    過擬合診斷（共用函數）
    
    Args:
        train_metrics: 訓練指標字典
        X_test: 測試集特徵（可選，用於最終性能評估）
        y_test: 測試集標籤（可選）
        model: 訓練好的模型（可選，用於預測）
        verbose: 是否輸出詳細資訊
    
    Returns:
        dict: 診斷結果
    
    >>> metrics = {'train_accuracy': 0.95, 'test_accuracy': 0.85, 'accuracy_gap': 0.10, 'overfitting_risk': 'high'}
    >>> result = diagnose_overfitting(metrics, verbose=False)
    >>> result['risk']
    'high'
    """
    if verbose:
        print("\n" + "=" * 60)
        print("📊 過擬合診斷結果")
        print("=" * 60)
    
    if 'train_accuracy' not in train_metrics or 'test_accuracy' not in train_metrics:
        if verbose:
            print("❌ 無法獲取過擬合診斷信息")
            print(f"   可用的指標：{list(train_metrics.keys())}")
        return {'error': 'Missing required metrics'}
    
    train_acc = train_metrics['train_accuracy']
    test_acc = train_metrics['test_accuracy']
    gap = train_metrics.get('accuracy_gap', train_acc - test_acc)
    risk = train_metrics.get('overfitting_risk', 'unknown')
    best_iter = train_metrics.get('best_iteration', 'N/A')
    
    result = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'accuracy_gap': gap,
        'risk': risk,
        'best_iteration': best_iter
    }
    
    if verbose:
        print(f"\n訓練集準確率：{train_acc:.6f} ({train_acc*100:.4f}%)")
        print(f"驗證集準確率：{test_acc:.6f} ({test_acc*100:.4f}%)")
        print(f"準確率差異：{gap:.6f} ({gap*100:.4f}%)")
        print(f"過擬合風險：{risk.upper()}")
        print(f"最佳迭代次數：{best_iter}")
        
        print("\n" + "-" * 60)
        if risk == 'high':
            print("⚠️  警告：存在高過擬合風險！")
            print("   建議：")
            print("   1. 降低 max_depth（例如從 6 降到 4）")
            print("   2. 增加 subsample 和 colsample_bytree 的隨機性")
            print("   3. 降低 learning_rate 並增加 n_estimators")
            print("   4. 增加 early_stopping_rounds")
        elif risk == 'medium':
            print("⚠️  注意：存在中等過擬合風險")
            print("   建議：")
            print("   1. 考慮降低模型複雜度")
            print("   2. 增加正則化參數")
        else:
            print("✅ 過擬合風險較低，模型泛化能力良好")
        
        # 計算測試集最終性能（如果有提供）
        if X_test is not None and y_test is not None and model is not None:
            print("\n" + "-" * 60)
            print("最終測試集性能：")
            y_pred_final = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_final)
            final_accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
            print(f"   準確率：{final_accuracy:.6f} ({final_accuracy*100:.4f}%)")
            print(f"   TN: {cm[0,0]:,}, FP: {cm[0,1]:,}")
            print(f"   FN: {cm[1,0]:,}, TP: {cm[1,1]:,}")
            result['final_test_accuracy'] = final_accuracy
            result['confusion_matrix'] = cm.tolist()
    
    return result


def run_quick_test_mode() -> int:
    """
    快速測試模式（使用模擬資料）
    
    Returns:
        exit_code: 0 表示成功，非 0 表示失敗
    """
    print("=" * 60)
    print("XGBoost 過擬合檢查（快速測試模式）")
    print("=" * 60)
    
    # 創建模擬資料
    print("\n[步驟 1] 創建模擬資料...")
    X, y = create_synthetic_data()
    print(f"✅ 資料創建完成：{len(X)} 筆，{len(X.columns)} 個特徵")
    print(f"   標籤分布：正常 {(y == 0).sum()}, 異常 {(y == 1).sum()}")
    
    # 分割資料
    print("\n[步驟 2] 分割資料...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ 訓練集：{len(X_train)} 筆，測試集：{len(X_test)} 筆")
    
    # 訓練模型
    print("\n[步驟 3] 訓練 XGBoost 模型...")
    model = ModelFactory.create(ModelType.XGBOOST)
    
    trained_model, metrics = model.train(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        early_stopping_rounds=10
    )
    
    # 診斷
    diagnose_overfitting(metrics, X_test, y_test, model, verbose=True)
    print("\n✅ 過擬合檢測功能正常運作！")
    
    return 0


def run_real_data_mode(feature_stage: int = 3) -> int:
    """
    真實資料模式（使用專案資料）
    
    Args:
        feature_stage: 特徵階段（1-4）
    
    Returns:
        exit_code: 0 表示成功，非 0 表示失敗
    """
    print("=" * 60)
    print("XGBoost 過擬合檢查（真實資料模式）")
    print("=" * 60)
    
    try:
        # 載入真實資料
        X, y = load_real_data(feature_stage=feature_stage)
        
        # 6. 分割資料集
        print("\n[步驟 6] 分割資料集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        print(f"✅ 訓練集：{len(X_train):,} 筆")
        print(f"✅ 測試集：{len(X_test):,} 筆")
        
        # 7. 訓練模型（帶過擬合診斷）
        print("\n[步驟 7] 訓練 XGBoost 模型（帶過擬合診斷）...")
        model = ModelFactory.create(ModelType.XGBOOST)
        
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        trained_model, train_metrics = model.train(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=10
        )
        
        # 8. 過擬合診斷
        diagnose_overfitting(train_metrics, X_test, y_test, model, verbose=True)
        
        print("\n" + "=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='監督學習模型過擬合檢查工具（XGBoost）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 使用真實資料（預設）
  python scripts/evaluation/check_supervised_overfitting.py
  
  # 快速測試模式（使用模擬資料）
  python scripts/evaluation/check_supervised_overfitting.py --quick-test
  python scripts/evaluation/check_supervised_overfitting.py -q
  
  # 指定特徵階段（僅真實資料模式）
  python scripts/evaluation/check_supervised_overfitting.py --feature-stage 4
        """
    )
    parser.add_argument(
        '--quick-test',
        '-q',
        action='store_true',
        help='快速測試模式（使用模擬資料）'
    )
    parser.add_argument(
        '--feature-stage',
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help='特徵階段（僅真實資料模式，預設：3）'
    )
    
    args = parser.parse_args()
    
    if args.quick_test:
        return run_quick_test_mode()
    else:
        return run_real_data_mode(feature_stage=args.feature_stage)


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷執行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未預期的錯誤：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

