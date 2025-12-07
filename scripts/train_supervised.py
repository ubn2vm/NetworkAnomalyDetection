"""
監督學習模型訓練：XGBoost

使用 Pandas 載入器載入資料，並使用 XGBoost 進行監督學習異常檢測。
僅支援從 Parquet 檔案載入。
使用標籤進行訓練，通常比無監督學習表現更好。
"""
import sys
import time
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

# 將專案根目錄加入 Python 路徑（必須在匯入 src 模組之前）
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    DataLoaderFactory,
    DataSourceType,
    ModelFactory,
    ModelType,
    extract_features,
    prepare_feature_set,
    FeatureSelector,
    FeatureSelectionStrategy,
    convert_label_to_binary,
    StandardFeatureProcessor,
    evaluate_and_print
)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def load_and_prepare_features(
    parquet_path: Path,
    processor: StandardFeatureProcessor
) -> pd.DataFrame:
    """
    載入和準備特徵數據（支援快取）
    
    Args:
        parquet_path: Parquet 文件路徑
        processor: 特徵處理器
    
    Returns:
        包含所有特徵的 DataFrame
    """
    features_stage4_path = Path("data/processed/features_stage4.parquet")
    
    if features_stage4_path.exists():
        print(f"\n   💾 發現已處理的特徵快取，直接載入...")
        cache_start_time = time.time()
        
        # 載入原始特徵
        features_df = processor.load_features()
        
        # 檢查並添加 bidirectional_window_flow_ratio（如果不存在但應該存在）
        if (processor.time_feature_stage == 4 and 
            'bidirectional_window_flow_ratio' not in features_df.columns and
            'bidirectional_total_src_bytes' in features_df.columns and
            'bidirectional_total_dst_bytes' in features_df.columns):
            print("   🔧 檢測到缺少 bidirectional_window_flow_ratio，正在計算並添加...")
            features_df['bidirectional_window_flow_ratio'] = (
                features_df['bidirectional_total_src_bytes'].astype(float) / 
                (features_df['bidirectional_total_dst_bytes'].astype(float) + 1)
            ).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            print("   ✅ bidirectional_window_flow_ratio 已添加")
        
        cache_load_time = time.time() - cache_start_time
        print(f"   ✅ 快取載入完成（耗時 {cache_load_time:.2f} 秒）")
        print(f"   📊 原始特徵數：{features_df.shape[1]} 個")
    else:
        print("   ⚠️  未發現特徵快取，執行完整特徵工程流程...")
        print("   ⚠️  階段4需要 PySpark，計算成本較高，請耐心等待...")
        features_start_time = time.time()
        
        # 讀取 Parquet 文件
        raw_df = pd.read_parquet(parquet_path, engine='pyarrow')
        
        # 創建載入器用於清洗資料
        loader = DataLoaderFactory.create(DataSourceType.BIDIRECTIONAL_BINETFLOW)
        cleaned_df = loader.clean(raw_df)
        
        # 執行完整特徵工程
        features_df = extract_features(
            cleaned_df,
            include_time_features=True,
            time_feature_stage=4  # 階段4：包含所有階段特徵（最完整）
        )
        
        features_time = time.time() - features_start_time
        print(f"   ✅ 特徵工程完成（耗時 {features_time:.2f} 秒）")
        print(f"   📊 原始特徵數：{features_df.shape[1]} 個")
    
    return features_df


def perform_feature_selection(
    X: pd.DataFrame,
    features_df: pd.DataFrame,
    initial_feature_count: int
) -> pd.DataFrame:
    """
    執行完整的特徵選擇流程
    
    Args:
        X: 特徵 DataFrame
        features_df: 包含標籤的完整 DataFrame
        initial_feature_count: 初始特徵數量
    
    Returns:
        選擇後的特徵 DataFrame
    """
    print("\n[步驟 4.5] 特徵選擇（品質檢查和相關性分析）...")
    
    # 使用 FeatureSelector 進行品質檢查和相關性分析
    selector = FeatureSelector(
        remove_constant=True,
        remove_low_variance=True,
        variance_threshold=1e-6,
        remove_inf=True,
        inf_ratio_threshold=0.1,
        remove_high_missing=True,
        missing_ratio_threshold=0.5,
        remove_high_correlation=True,
        correlation_threshold=0.98
    )
    
    X_selected, removed_features = selector.select_features(
        X,
        features_df=None,  # 品質檢查和相關性分析不需要標籤
        strategies=[FeatureSelectionStrategy.QUALITY_CHECK, FeatureSelectionStrategy.CORRELATION],
        verbose=True
    )
    
    print(f"\n✅ 基本特徵選擇完成：從 {initial_feature_count} 個特徵減少到 {len(X_selected.columns)} 個特徵")
    
    # 基於重要性的特徵選擇（需要標籤）
    print("\n[步驟 4.6] 基於 XGBoost 特徵重要性的特徵選擇...")
    if 'Label' in features_df.columns:
        try:
            # 準備標籤
            if 'label_binary' not in features_df.columns:
                features_df_temp = convert_label_to_binary(features_df, verbose=False)
            else:
                features_df_temp = features_df.copy()
            
            # 使用 FeatureSelector 的內部方法進行重要性選擇（需要自定義參數）
            # 注意：由於 FeatureSelector._importance_selection 的參數與我們的需求略有不同，
            # 我們直接調用它並傳入自定義參數。這是合理的，因為我們需要特定的參數值。
            # 未來可以考慮在 FeatureSelector 中添加支持自定義參數的公共方法。
            try:
                X_selected, removed_importance = selector._importance_selection(
                    X_selected,
                    features_df_temp,
                    verbose=True,
                    min_features=15,
                    max_features=30,  # train_supervised.py 使用 30，而不是 FeatureSelector 預設的 25
                    importance_threshold=0.95  # train_supervised.py 使用 0.95，而不是 FeatureSelector 預設的 0.98
                )
            except TypeError:
                # 如果 _importance_selection 不支持這些參數，回退到使用 select_features
                # 但這會使用預設參數（min_features=15, max_features=25, importance_threshold=0.98）
                print("   ⚠️  使用 FeatureSelector 預設參數進行重要性選擇...")
                X_selected, removed_features_dict = selector.select_features(
                    X_selected,
                    features_df=features_df_temp,
                    strategies=[FeatureSelectionStrategy.IMPORTANCE],
                    verbose=True
                )
            
            print(f"\n   ✅ 基於重要性選擇完成：保留 {len(X_selected.columns)} 個最重要特徵")
            print(f"   最終特徵列表：{list(X_selected.columns)}")
            
        except Exception as e:
            print(f"   ⚠️  基於重要性的特徵選擇失敗：{e}")
            print(f"   將使用基本特徵選擇的結果繼續執行")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠️  缺少標籤，跳過基於重要性的特徵選擇")
    
    return X_selected




def train_model_with_overfitting_check(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float
) -> Tuple[any, Dict]:
    """
    訓練模型並進行過擬合檢查和動態參數調整
    
    Args:
        X_train: 訓練特徵
        y_train: 訓練標籤
        scale_pos_weight: 不平衡權重
    
    Returns:
        (訓練好的模型, 訓練指標)
    """
    print("\n[步驟 7] 訓練 XGBoost 模型（初始參數）...")
    model = ModelFactory.create(ModelType.XGBOOST)
    
    # XGBoost 初始參數（保守設置，防止過擬合）
    initial_params = {
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 10
    }
    
    print(f"   初始參數：max_depth={initial_params['max_depth']}, learning_rate={initial_params['learning_rate']}, "
          f"n_estimators={initial_params['n_estimators']}, subsample={initial_params['subsample']}, "
          f"colsample_bytree={initial_params['colsample_bytree']}")
    
    trained_model, train_metrics = model.train(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        **initial_params
    )
    
    print("✅ 初始模型訓練完成")
    
    # 過擬合診斷
    print(f"\n   📊 過擬合診斷：")
    if 'train_accuracy' in train_metrics and 'test_accuracy' in train_metrics:
        train_acc = train_metrics['train_accuracy']
        test_acc = train_metrics['test_accuracy']
        gap = train_metrics.get('accuracy_gap', train_acc - test_acc)
        risk = train_metrics.get('overfitting_risk', 'unknown')
        best_iter = train_metrics.get('best_iteration', 'N/A')
        
        print(f"     訓練集準確率：{train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"     驗證集準確率：{test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"     準確率差異：{gap:.4f} ({gap*100:.2f}%)")
        print(f"     過擬合風險：{risk.upper()}")
        print(f"     最佳迭代次數：{best_iter}")
        
        # 根據過擬合風險動態調整參數
        if risk == 'high':
            print(f"\n     ⚠️  警告：存在高過擬合風險！將調整參數降低模型複雜度...")
            adjusted_params = {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'early_stopping_rounds': 20
            }
            print(f"     調整後參數：max_depth={adjusted_params['max_depth']}, learning_rate={adjusted_params['learning_rate']}, "
                  f"n_estimators={adjusted_params['n_estimators']}, subsample={adjusted_params['subsample']}, "
                  f"colsample_bytree={adjusted_params['colsample_bytree']}")
            
            # 使用調整後的參數重新訓練
            print(f"\n   🔄 使用調整後的參數重新訓練模型...")
            model_adjusted = ModelFactory.create(ModelType.XGBOOST)
            trained_model, train_metrics = model_adjusted.train(
                X_train,
                y_train,
                test_size=0.2,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                **adjusted_params
            )
            
            # 重新診斷
            train_acc = train_metrics['train_accuracy']
            test_acc = train_metrics['test_accuracy']
            gap = train_metrics.get('accuracy_gap', train_acc - test_acc)
            risk = train_metrics.get('overfitting_risk', 'unknown')
            best_iter = train_metrics.get('best_iteration', 'N/A')
            
            print(f"   ✅ 調整後模型訓練完成")
            print(f"     訓練集準確率：{train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"     驗證集準確率：{test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"     準確率差異：{gap:.4f} ({gap*100:.2f}%)")
            print(f"     過擬合風險：{risk.upper()}")
            print(f"     最佳迭代次數：{best_iter}")
            
            if risk == 'high':
                print(f"     ⚠️  警告：調整後仍存在高過擬合風險！")
            elif risk == 'medium':
                print(f"     ⚠️  注意：調整後仍存在中等過擬合風險")
            else:
                print(f"     ✅ 調整後過擬合風險降低，模型泛化能力改善")
            
            # 更新 model 對象
            model = model_adjusted
            
        elif risk == 'medium':
            print(f"\n     ⚠️  注意：存在中等過擬合風險，建議調整模型參數")
            print(f"     可以考慮：降低 max_depth 或增加 subsample/colsample_bytree 的隨機性")
        else:
            print(f"     ✅ 過擬合風險較低，模型泛化能力良好")
    else:
        print(f"     準確率：{train_metrics.get('accuracy', 'N/A'):.4f}")
    
    print(f"\n   內部驗證集性能：")
    if 'classification_report' in train_metrics:
        report = train_metrics['classification_report']
        if isinstance(report, dict):
            print(f"     正常類別 - 精確率：{report.get('0', {}).get('precision', 'N/A'):.4f}, 召回率：{report.get('0', {}).get('recall', 'N/A'):.4f}, F1：{report.get('0', {}).get('f1-score', 'N/A'):.4f}")
            print(f"     異常類別 - 精確率：{report.get('1', {}).get('precision', 'N/A'):.4f}, 召回率：{report.get('1', {}).get('recall', 'N/A'):.4f}, F1：{report.get('1', {}).get('f1-score', 'N/A'):.4f}")
        else:
            print(f"     分類報告：\n{report}")
    
    return model, train_metrics


def main():
    print("=" * 60)
    print("監督學習模型訓練：XGBoost")
    print("=" * 60)
    start_time = time.time()
    
    # 1. 載入資料（僅支援 Parquet 檔案）
    print("\n[步驟 1] 載入資料...")
    parquet_path = Path("data/processed/capture20110817_cleaned_spark.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"找不到 Parquet 檔案: {parquet_path}\n"
            f"請先執行資料處理腳本生成 Parquet 檔案。"
        )
    
    print(f"   使用 Pandas 讀取 Parquet: {parquet_path}")
    load_time = time.time() - start_time
    
    # 2-3. 特徵處理（使用 FeatureProcessor，支援快取）
    print("\n[步驟 2-3] 特徵處理...")
    print("   使用階段4時間特徵（最完整：包含所有階段特徵）")
    print("   - 階段1：基本時間特徵")
    print("   - 階段2：時間間隔特徵")
    print("   - 階段3：時間窗口聚合特徵（按 SrcAddr）")
    print("   - 階段4：雙向流 Pair 聚合特徵（按 IP Pair，需要 PySpark）")
    
    processor = StandardFeatureProcessor(time_feature_stage=4)
    features_df = load_and_prepare_features(parquet_path, processor)
    
    # 4. 準備訓練資料
    print("\n[步驟 4] 準備訓練資料...")
    X = prepare_feature_set(
        features_df,
        include_base_features=True,
        include_time_features=True,
        time_feature_stage=4
    )
    initial_feature_count = len(X.columns)
    print(f"✅ 初始特徵欄位（共 {initial_feature_count} 個）")
    
    # 4.5-4.6 特徵選擇
    X = perform_feature_selection(X, features_df, initial_feature_count)
    
    # 4.7 時間特徵檢查（使用 FeatureSelector 統一方法）
    print("\n[步驟 4.7] 檢查時間特徵重要性（避免時間偏差）...")
    selector = FeatureSelector()
    X, time_importance_dict = selector.check_time_feature_bias(
        X,
        features_df,
        time_features=['hour', 'cos_hour', 'sin_hour'],
        importance_threshold=0.05,  # 時間特徵總重要性閾值 5%
        sample_size=10000,
        verbose=True
    )
    
    print(f"\n✅ 特徵選擇完成：從 {initial_feature_count} 個特徵減少到 {len(X.columns)} 個特徵")
    
    # 5. 準備標籤
    print("\n[步驟 5] 準備標籤...")
    if 'Label' not in features_df.columns:
        print("❌ 錯誤：缺少 'Label' 欄位，無法進行監督學習")
        print("   請使用包含標籤的資料集")
        return 1
    
    features_df = convert_label_to_binary(features_df, verbose=True)
    y = features_df['label_binary']
    
    # 顯示標籤分布
    print(f"\n   標籤分布統計：")
    print(f"     正常 (0): {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.2f}%)")
    print(f"     異常 (1): {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.2f}%)")
    
    # 計算不平衡比例
    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
    print(f"   不平衡比例：{scale_pos_weight:.2f}:1（正常:異常）")
    print(f"   將使用 scale_pos_weight={scale_pos_weight:.2f} 來處理不平衡資料")
    
    # 6. 分割資料集
    print("\n[步驟 6] 分割資料集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"✅ 訓練集：{len(X_train):,} 筆（{len(X_train)/len(X)*100:.1f}%）")
    print(f"✅ 測試集：{len(X_test):,} 筆（{len(X_test)/len(X)*100:.1f}%）")
    print(f"   訓練集標籤分布：正常 {(y_train == 0).sum():,}，異常 {(y_train == 1).sum():,}")
    print(f"   測試集標籤分布：正常 {(y_test == 0).sum():,}，異常 {(y_test == 1).sum():,}")
    
    # 7. 訓練模型（包含過擬合檢查和動態參數調整）
    model, train_metrics = train_model_with_overfitting_check(
        X_train, y_train, scale_pos_weight
    )
    
    # 8. 特徵重要性分析
    print("\n[步驟 8] 特徵重要性分析...")
    feature_importance = model.get_feature_importance()
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n   Top 10 最重要特徵：")
    for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
        print(f"     {i:2d}. {feature:30s}: {importance:.4f}")
    
    # 保存特徵重要性到文件（供報告生成器使用）
    output_dir = Path("output/evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)
    xgb_feature_importance_path = output_dir / "xgb_feature_importance.json"
    with open(xgb_feature_importance_path, 'w', encoding='utf-8') as f:
        json.dump(feature_importance, f, indent=2, ensure_ascii=False)
    print(f"\n   ✅ 特徵重要性已保存至: {xgb_feature_importance_path}")
    
    # 9. 預測
    print("\n[步驟 9] 進行預測...")
    y_pred_labels = model.predict(X_test)
    print(f"   預測異常數量：{y_pred_labels.sum():,} ({y_pred_labels.sum()/len(y_pred_labels)*100:.2f}%)")
    
    # 10. 評估
    print("\n[步驟 10] 模型評估（測試集）...")
    evaluate_and_print(
        y_test,
        y_pred_labels,
        show_confusion_matrix=True,
        show_detailed=True,
        show_classification_report=True,
        indent="  "
    )
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ 執行完成（總耗時：{total_time:.2f} 秒）")
    print(f"   資料載入：{load_time:.2f} 秒")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

