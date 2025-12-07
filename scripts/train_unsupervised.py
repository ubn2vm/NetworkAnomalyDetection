"""
無監督學習模型訓練：Isolation Forest

使用 Pandas 載入器載入資料，並使用 sklearn 的 Isolation Forest 進行異常檢測。
僅支援從 Parquet 檔案載入。

此腳本專注於模型訓練，遵循單一職責原則。
白名單後處理請使用 postprocess_with_whitelist.py。
"""
import sys
import time
import pickle
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# 將專案根目錄加入 Python 路徑（必須在匯入 src 模組之前）
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    DataLoaderFactory,
    DataSourceType,
    extract_features,
    transform_features_for_unsupervised,
    DEFAULT_SKEWED_FEATURES,
    prepare_feature_set,
    FeatureSelector,
    FeatureSelectionStrategy,
    StandardFeatureProcessor,
    calculate_contamination,
    train_single_model,
    train_protocol_grouped_models,
    evaluate_and_print
)
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def load_and_prepare_features(
    processor: StandardFeatureProcessor,
    cleaned_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    載入和準備特徵數據（支援快取，優先使用階段4）
    
    Args:
        processor: 特徵處理器
        cleaned_df: 清洗後的 DataFrame（如果快取不存在，則需要此參數）
    
    Returns:
        包含所有特徵的 DataFrame
    """
    features_stage3_path = Path("data/processed/features_stage3.parquet")
    features_stage4_path = Path("data/processed/features_stage4.parquet")
    
    # 優先檢查階段4快取
    if features_stage4_path.exists():
        print(f"\n   💾 發現階段4特徵快取，直接載入...")
        cache_start_time = time.time()
        
        # 載入原始特徵（不傳 stage 參數，會自動優先載入階段4）
        features_df = processor.load_features()
        
        cache_load_time = time.time() - cache_start_time
        print(f"   ✅ 階段4快取載入完成（耗時 {cache_load_time:.2f} 秒）")
        print(f"   📊 原始特徵數：{features_df.shape[1]} 個")
    elif features_stage3_path.exists():
        # 有階段3但沒有階段4，可以增量執行階段4
        print(f"\n   📂 發現階段3快取，但未發現階段4快取...")
        print(f"   💡 提示：階段4包含雙向流 Pair 聚合特徵，能提升模型效果")
        print(f"   ⏱️  階段4需要 PySpark，預計需要 30-60 分鐘")
        
        if cleaned_df is None:
            print(f"   ⚠️  未提供 cleaned_df，無法執行階段4，將使用階段3特徵")
            features_df = processor.load_features(stage=3)
        else:
            # 自動執行階段4（可以改為互動式）
            execute_stage4 = True  # 預設執行階段4
            
            if execute_stage4:
                print(f"\n   🔄 開始執行階段4特徵工程（PySpark）...")
                features_start_time = time.time()
                
                # 增量執行階段4（從階段3到階段4）
                features_df, _, _, _ = processor.process(
                    cleaned_df,
                    save_features=True,
                    save_transformed=False,  # 不保存轉換後的特徵（因為後續會重新轉換）
                    incremental=True  # 增量模式：從階段3到階段4
                )
                
                features_time = time.time() - features_start_time
                print(f"   ✅ 階段4特徵處理完成（耗時 {features_time:.2f} 秒）")
                print(f"   💾 階段4特徵已儲存，下次執行將直接載入")
                print(f"   📊 原始特徵數：{features_df.shape[1]} 個")
            else:
                print(f"   ⏸️  已跳過階段4，使用階段3特徵")
                features_df = processor.load_features(stage=3)
    else:
        # 都沒有快取，執行完整特徵工程
        if cleaned_df is None:
            raise ValueError(
                "快取不存在且未提供 cleaned_df，無法執行特徵工程。"
                "請先執行資料清洗步驟。"
            )
        
        print("   ⚠️  未發現特徵快取，執行完整特徵工程流程...")
        print("   ⚠️  階段4需要 PySpark，計算成本較高，請耐心等待...")
        features_start_time = time.time()
        
        # 執行完整特徵工程（階段4）
        features_df = extract_features(
            cleaned_df,
            include_time_features=True,
            time_feature_stage=4  # 階段4：包含所有階段特徵（最完整）
        )
        
        # 保存快取供後續使用
        print("   💾 保存階段4特徵快取...")
        processor.save_features(features_df, stage=4)
        print("   ✅ 階段4特徵快取已保存")
        
        features_time = time.time() - features_start_time
        print(f"   ✅ 特徵工程完成（耗時 {features_time:.2f} 秒）")
        print(f"   📊 原始特徵數：{features_df.shape[1]} 個")
    
    return features_df


def save_training_results(
    output_dir: Path,
    model: Any,
    scaler: Any,
    protocol_models: Optional[Dict[str, Any]] = None,
    protocol_scalers: Optional[Dict[str, Any]] = None,
    X_train: Optional[pd.DataFrame] = None,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    features_df_train: Optional[pd.DataFrame] = None,
    features_df_val: Optional[pd.DataFrame] = None,
    features_df_test: Optional[pd.DataFrame] = None,
    y_train: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    train_anomaly_scores: Optional[np.ndarray] = None,
    val_anomaly_scores: Optional[np.ndarray] = None,
    test_anomaly_scores: Optional[np.ndarray] = None,
    best_threshold: Optional[float] = None,
    contamination: Optional[float] = None,
    use_protocol_grouping: bool = False,
    feature_robust_scaler: Any = None,
    transformed_feature_cols: Optional[list] = None,
    final_feature_cols: Optional[list] = None
):
    """
    保存訓練結果供後續白名單後處理使用
    
    Args:
        output_dir: 輸出目錄
        model: 訓練好的模型（單一模型）
        scaler: 特徵標準化器（單一模型）
        protocol_models: 協議分組模型字典（可選）
        protocol_scalers: 協議分組標準化器字典（可選）
        X_train, X_val, X_test: 訓練/驗證/測試集特徵
        features_df_train, features_df_val, features_df_test: 訓練/驗證/測試集完整特徵
        y_train, y_val, y_test: 訓練/驗證/測試集標籤
        train_anomaly_scores, val_anomaly_scores, test_anomaly_scores: 異常分數
        best_threshold: 最佳閾值
        contamination: contamination 參數
        use_protocol_grouping: 是否使用協議分組
        feature_robust_scaler: 特徵 RobustScaler
        transformed_feature_cols: 轉換後的特徵欄位列表（28個：用於重要性選擇）
        final_feature_cols: 最終用於模型訓練的特徵欄位列表（15個：經過重要性選擇後）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    if use_protocol_grouping and protocol_models:
        with open(output_dir / "protocol_models.pkl", "wb") as f:
            pickle.dump(protocol_models, f)
        if protocol_scalers:
            with open(output_dir / "protocol_scalers.pkl", "wb") as f:
                pickle.dump(protocol_scalers, f)
    else:
        with open(output_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        if scaler:
            with open(output_dir / "scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
    
    # 保存特徵資料
    if X_train is not None:
        X_train.to_parquet(output_dir / "X_train.parquet")
    if X_val is not None:
        X_val.to_parquet(output_dir / "X_val.parquet")
    if X_test is not None:
        X_test.to_parquet(output_dir / "X_test.parquet")
    
    if features_df_train is not None:
        features_df_train.to_parquet(output_dir / "features_df_train.parquet")
    if features_df_val is not None:
        features_df_val.to_parquet(output_dir / "features_df_val.parquet")
    if features_df_test is not None:
        features_df_test.to_parquet(output_dir / "features_df_test.parquet")
    
    # 保存標籤和異常分數
    if y_train is not None:
        np.save(output_dir / "y_train.npy", y_train)
    if y_val is not None:
        np.save(output_dir / "y_val.npy", y_val)
    if y_test is not None:
        np.save(output_dir / "y_test.npy", y_test)
    
    if train_anomaly_scores is not None:
        np.save(output_dir / "train_anomaly_scores.npy", train_anomaly_scores)
    if val_anomaly_scores is not None:
        np.save(output_dir / "val_anomaly_scores.npy", val_anomaly_scores)
    if test_anomaly_scores is not None:
        np.save(output_dir / "test_anomaly_scores.npy", test_anomaly_scores)
    
    # 保存配置
    config = {
        "best_threshold": best_threshold,
        "contamination": contamination,
        "use_protocol_grouping": use_protocol_grouping,
        "transformed_feature_cols": transformed_feature_cols,  # 28個：轉換後的特徵
        "final_feature_cols": final_feature_cols  # 15個：最終用於模型訓練的特徵
    }
    
    if feature_robust_scaler is not None:
        with open(output_dir / "feature_robust_scaler.pkl", "wb") as f:
            pickle.dump(feature_robust_scaler, f)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"\n✅ 訓練結果已保存至：{output_dir}")


def main():
    print("=" * 60)
    print("無監督學習模型訓練：Isolation Forest")
    print("=" * 60)
    
    # 輸出目錄
    output_dir = Path("data/models/unsupervised_training")
    
    # 1. 載入資料（僅支援 Parquet 檔案）
    print("\n[步驟 1] 載入資料...")
    start_time = time.time()
    
    # 直接讀取 Parquet 檔案
    parquet_path = Path("data/processed/capture20110817_cleaned_spark.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"找不到 Parquet 檔案: {parquet_path}\n"
            f"請先執行資料處理腳本生成 Parquet 檔案。"
        )
    
    print(f"   使用 Pandas 讀取 Parquet: {parquet_path}")
    raw_df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    # 創建載入器用於清洗資料
    loader = DataLoaderFactory.create(DataSourceType.BIDIRECTIONAL_BINETFLOW)
    
    load_time = time.time() - start_time
    print(f"✅ 載入完成：{len(raw_df):,} 筆資料（耗時 {load_time:.2f} 秒）")
    
    # 2. 清洗資料
    print("\n[步驟 2] 清洗資料...")
    cleaned_df = loader.clean(raw_df)
    print(f"✅ 清洗完成：{len(cleaned_df):,} 筆資料")
    
    # 3. 特徵工程（使用快取機制）
    print("\n[步驟 3] 特徵工程...")
    # 使用時間特徵（階段4：最完整，包含所有階段特徵）
    # 階段4包含雙向流 Pair 聚合特徵，能識別更複雜的異常模式
    # - 階段1：基本時間特徵
    # - 階段2：時間間隔特徵
    # - 階段3：時間窗口聚合特徵（按 SrcAddr）
    # - 階段4：雙向流 Pair 聚合特徵（按 IP Pair，需要 PySpark）
    print("   ⚠️  使用階段4時間特徵（最完整，包含所有階段特徵）...")
    
    # 創建特徵處理器（使用階段4）
    processor = StandardFeatureProcessor(time_feature_stage=4)
    
    # 使用快取機制載入或生成特徵
    features_df = load_and_prepare_features(processor, cleaned_df)
    
    print(f"✅ 特徵工程完成：{features_df.shape[1]} 個特徵")
    
    # 4. 準備訓練資料
    print("\n[步驟 4] 準備訓練資料...")
    # 使用統一的特徵準備接口
    X = prepare_feature_set(
        features_df,
        include_base_features=True,
        include_time_features=True,
        time_feature_stage=4  # 與 extract_features 的階段保持一致（階段4）
    )
    print(f"✅ 初始特徵欄位（共 {len(X.columns)} 個）：{list(X.columns)}")
    
    # 4.5 無監督特徵選擇
    print("\n[步驟 4.5] 無監督特徵選擇...")
    initial_feature_count = len(X.columns)
    
    # 使用統一的特徵選擇器
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
    
    # 執行品質檢查和相關性分析
    X, removed_features = selector.select_features(
        X,
        strategies=[
            FeatureSelectionStrategy.QUALITY_CHECK,
            FeatureSelectionStrategy.CORRELATION
        ],
        verbose=True
    )
    
    print(f"\n✅ 特徵選擇完成：從 {initial_feature_count} 個特徵減少到 {len(X.columns)} 個特徵")
    print(f"   最終特徵列表：{list(X.columns)}")
    
    # 4.5.3 特徵轉換：Log-Transformation + RobustScaler（優化 Unsupervised Model）
    print("\n  [4.5.3] 特徵轉換（Log-Transformation + RobustScaler）...")
    print("   使用新的特徵轉換模組優化長尾分佈特徵...")
    
    # 使用新的特徵轉換模組
    # 這會自動：
    # 1. 對長尾分佈特徵進行 log1p 轉換
    # 2. 使用 RobustScaler 進行標準化（對極端值更穩健）
    skewed_features = [col for col in DEFAULT_SKEWED_FEATURES if col in X.columns]
    
    if skewed_features:
        print(f"   對 {len(skewed_features)} 個長尾分佈特徵進行轉換：{skewed_features[:5]}...")
        
        # 應用 Log-Transformation + RobustScaler
        # 注意：這裡我們需要先將 X 轉換回完整的 DataFrame（包含所有欄位）
        # 然後再提取特徵欄位
        features_df_temp = features_df.copy()
        features_df_temp[X.columns] = X
        
        transformed_df, robust_scaler, transformed_feature_cols = transform_features_for_unsupervised(
            features_df_temp,
            skewed_features=skewed_features,
            feature_columns=list(X.columns),  # 使用當前選擇的特徵
            replace_original=False  # 創建 log_ 前綴的新欄位
        )
        
        # 更新 X 為轉換後的特徵（只包含被標準化的欄位）
        X = transformed_df[transformed_feature_cols].copy()
        
        print(f"   ✅ 特徵轉換完成")
        print(f"      - 對數轉換：{len(skewed_features)} 個特徵")
        print(f"      - RobustScaler 標準化：{len(transformed_feature_cols)} 個特徵")
        print(f"      - 使用 RobustScaler（中位數 + IQR）而非 StandardScaler，對極端值更穩健")
        # 保存 robust_scaler 供後續使用
        feature_robust_scaler = robust_scaler
    else:
        print("   ⚠️  未找到需要轉換的長尾分佈特徵，跳過轉換")
        feature_robust_scaler = None
        transformed_feature_cols = list(X.columns)
    
    # 4.6 基於 XGBoost 特徵重要性進一步優化（可選）
    # 如果資料有標籤，可以使用 XGBoost 來識別最重要的特徵
    if 'Label' in features_df.columns:
        print("\n[步驟 4.6] 基於 XGBoost 特徵重要性優化特徵選擇...")
        try:
            # 使用統一的特徵選擇器進行重要性選擇
            X, removed = selector.select_features(
                X,
                features_df=features_df,
                strategies=[FeatureSelectionStrategy.IMPORTANCE],
                verbose=True
            )
            print(f"   優化後特徵列表：{list(X.columns)}")
        except Exception as e:
            print(f"   ⚠️  特徵重要性分析失敗：{e}")
            print("   繼續使用所有特徵")
    
    # 4.7 條件移除時間特徵（項目 5：條件移除 hour 特徵）
    print("\n[步驟 4.7] 檢查時間特徵重要性（避免時間偏差）...")
    X, time_importance_dict = selector.check_time_feature_bias(
        X,
        features_df,
        time_features=['hour', 'cos_hour', 'sin_hour'],
        importance_threshold=0.05,  # 時間特徵總重要性閾值 5%
        sample_size=10000,
        verbose=True
    )
    
    # 保存最終用於模型訓練的特徵列表（在特徵選擇完成後）
    final_feature_cols = list(X.columns)  # 最終用於模型訓練的特徵（15個）
    print(f"\n✅ 最終特徵選擇完成：使用 {len(final_feature_cols)} 個特徵進行模型訓練")
    print(f"   轉換後特徵數：{len(transformed_feature_cols)} 個")
    print(f"   最終訓練特徵數：{len(final_feature_cols)} 個")
    print(f"   移除特徵數：{len(transformed_feature_cols) - len(final_feature_cols)} 個")
    
    # 計算 contamination 參數（使用統一函數）
    contamination, y_true = calculate_contamination(
        features_df,
        multiplier=1.3,
        max_contamination=0.2,
        high_threshold=0.15,
        min_contamination=0.01,
        default=0.1,
        verbose=True
    )
    
    # 4.8 資料分割（避免 data leakage）
    print("\n[步驟 4.8] 資料分割（避免偷看答案）...")
    use_data_split = y_true is not None
    
    if use_data_split:
        # 先分割 train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_true, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_true if y_true is not None else None
        )
        # 再從訓練集中分割出驗證集 (80/20 of train = 64/16/20)
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_train
        )
        
        # 同時需要分割 features_df 以保持索引對齊
        train_idx = X_train_final.index
        val_idx = X_val.index
        test_idx = X_test.index
        
        features_df_train = features_df.loc[train_idx]
        features_df_val = features_df.loc[val_idx]
        features_df_test = features_df.loc[test_idx]
        
        print(f"   ✅ 資料分割完成：")
        print(f"      訓練集：{len(X_train_final):,} 筆 ({len(X_train_final)/len(X)*100:.1f}%)")
        print(f"      驗證集：{len(X_val):,} 筆 ({len(X_val)/len(X)*100:.1f}%)")
        print(f"      測試集：{len(X_test):,} 筆 ({len(X_test)/len(X)*100:.1f}%)")
        print(f"      訓練集異常比例：{(y_train_final == 1).sum()/len(y_train_final)*100:.2f}%")
        print(f"      驗證集異常比例：{(y_val == 1).sum()/len(y_val)*100:.2f}%")
        print(f"      測試集異常比例：{(y_test == 1).sum()/len(y_test)*100:.2f}%")
        print(f"   💡 將在訓練集上訓練模型，在驗證集上優化閾值，在測試集上評估")
    else:
        # 無標籤時，使用全部資料
        print(f"   ⚠️  無標籤資料，使用全部資料進行訓練和預測")
        X_train_final = X
        X_val = None
        X_test = X
        y_train_final = None
        y_val = None
        y_test = None
        features_df_train = features_df
        features_df_val = None
        features_df_test = features_df
        train_idx = X.index
        val_idx = None
        test_idx = X.index
    
    # 5. 訓練模型（項目 4：按協議分組訓練）
    print("\n[步驟 5] 訓練 Isolation Forest 模型...")
    
    # 檢查是否有協議欄位
    use_protocol_grouping = 'Proto' in features_df_train.columns
    
    if use_protocol_grouping:
        print("\n[步驟 5.1] 按協議分組訓練（僅主要協議：TCP/UDP）...")
        print("   💡 使用訓練集進行模型訓練...")
        
        # 使用統一的協議分組訓練函數（在訓練集上）
        protocol_models, protocol_scalers, protocol_contaminations, success = train_protocol_grouped_models(
            X_train_final,
            features_df_train,
            contamination=contamination,
            feature_robust_scaler=feature_robust_scaler,
            main_protocols=['tcp', 'udp'],
            min_samples=1000,
            random_state=42,
            n_estimators=300,
            max_samples='auto',
            bootstrap=True,
            verbose=True
        )
        
        if not success:
            use_protocol_grouping = False
            protocol_models = None
            protocol_scalers = None
            model = None
            scaler = None
        else:
            model = None
            scaler = None
    else:
        use_protocol_grouping = False
        protocol_models = None
        protocol_scalers = None
    
    # 如果沒有協議分組或協議分組失敗，使用單一模型
    if not use_protocol_grouping:
        print("\n[步驟 5.2] 單一模型訓練（未使用協議分組）...")
        print("   💡 使用訓練集進行模型訓練...")
        
        # 使用統一的單一模型訓練函數（在訓練集上）
        model, scaler = train_single_model(
            X_train_final,
            contamination=contamination,
            feature_robust_scaler=feature_robust_scaler,
            random_state=42,
            n_estimators=300,
            max_samples='auto',
            bootstrap=True,
            verbose=True
        )
    
    # 6. 在訓練集上預測（用於後續白名單分析）
    print("\n[步驟 6] 在訓練集上預測...")
    
    if use_protocol_grouping:
        train_anomaly_scores = np.zeros(len(X_train_final))
        
        for protocol, model_proto in protocol_models.items():
            if protocol == 'other':
                proto_mask = ~features_df_train['Proto'].str.lower().isin(['tcp', 'udp'])
            else:
                proto_mask = features_df_train['Proto'].str.lower() == protocol.lower()
            
            X_train_proto = X_train_final[proto_mask]
            if len(X_train_proto) == 0:
                continue
            
            anomaly_scores_proto = model_proto.predict(X_train_proto)
            anomaly_scores_proto_normalized = -anomaly_scores_proto
            
            if len(anomaly_scores_proto_normalized) > 0:
                score_mean = anomaly_scores_proto_normalized.mean()
                score_std = anomaly_scores_proto_normalized.std()
                if score_std > 1e-6:
                    scores_normalized = (anomaly_scores_proto_normalized - score_mean) / score_std
                    scores_normalized = 1 / (1 + np.exp(-scores_normalized))
                else:
                    scores_normalized = np.zeros_like(anomaly_scores_proto_normalized)
                train_anomaly_scores[proto_mask] = scores_normalized
        
        train_anomaly_scores_normalized = -train_anomaly_scores
    else:
        train_anomaly_scores = model.predict(X_train_final)
        train_anomaly_scores_normalized = -train_anomaly_scores
    
    print(f"   訓練集異常分數範圍：[{train_anomaly_scores_normalized.min():.4f}, {train_anomaly_scores_normalized.max():.4f}]")
    
    # 7. 在驗證集上優化閾值（如果有標籤）
    best_threshold = None
    val_anomaly_scores_normalized = None
    
    if use_data_split and y_val is not None:
        print("\n[步驟 7] 在驗證集上優化閾值...")
        print("   💡 使用驗證集優化閾值（避免在測試集上偷看答案）...")
        
        # 在驗證集上預測
        if use_protocol_grouping:
            # 協議分組預測（驗證集）
            val_anomaly_scores = np.zeros(len(X_val))
            
            for protocol, model_proto in protocol_models.items():
                if protocol == 'other':
                    proto_mask = ~features_df_val['Proto'].str.lower().isin(['tcp', 'udp'])
                else:
                    proto_mask = features_df_val['Proto'].str.lower() == protocol.lower()
                
                X_val_proto = X_val[proto_mask]
                if len(X_val_proto) == 0:
                    continue
                
                anomaly_scores_proto = model_proto.predict(X_val_proto)
                anomaly_scores_proto_normalized = -anomaly_scores_proto
                
                # 標準化（使用訓練集的統計量）
                if len(anomaly_scores_proto_normalized) > 0:
                    # 獲取訓練集上該協議的統計量（需要從訓練時保存）
                    # 簡化處理：使用驗證集自己的統計量
                    score_mean = anomaly_scores_proto_normalized.mean()
                    score_std = anomaly_scores_proto_normalized.std()
                    if score_std > 1e-6:
                        scores_normalized = (anomaly_scores_proto_normalized - score_mean) / score_std
                        scores_normalized = 1 / (1 + np.exp(-scores_normalized))
                    else:
                        scores_normalized = np.zeros_like(anomaly_scores_proto_normalized)
                    val_anomaly_scores[proto_mask] = scores_normalized
            
            val_anomaly_scores_normalized = -val_anomaly_scores
        else:
            # 單一模型預測（驗證集）
            val_anomaly_scores = model.predict(X_val)
            val_anomaly_scores_normalized = -val_anomaly_scores
        
        # 使用 precision_recall_curve 在驗證集上找最佳閾值
        precision, recall, thresholds = precision_recall_curve(y_val, val_anomaly_scores_normalized)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        
        valid_f1_scores = f1_scores[:len(thresholds)]
        best_idx = np.nanargmax(valid_f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else None
        
        if best_threshold is not None:
            # 檢查最佳閾值對應的異常比例
            y_pred_val = (val_anomaly_scores_normalized >= best_threshold).astype(int)
            anomaly_ratio_val = y_pred_val.sum() / len(y_pred_val)
            
            # 計算驗證集上的指標
            tp_val = ((y_pred_val == 1) & (y_val == 1)).sum()
            fp_val = ((y_pred_val == 1) & (y_val == 0)).sum()
            fn_val = ((y_pred_val == 0) & (y_val == 1)).sum()
            precision_val = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
            recall_val = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
            f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-10)
            
            print(f"   ✅ 在驗證集上找到最佳閾值：{best_threshold:.4f}")
            print(f"      驗證集指標：Precision={precision_val:.4f}, Recall={recall_val:.4f}, F1={f1_val:.4f}")
            print(f"      驗證集異常比例：{anomaly_ratio_val*100:.2f}%")
        else:
            # 回退到使用 contamination 的百分位數
            best_threshold = np.percentile(val_anomaly_scores_normalized, 100 * (1 - contamination))
            print(f"   ⚠️  無法從 PR 曲線找到最佳閾值，使用 contamination 百分位數：{best_threshold:.4f}")
    
    # 8. 在測試集上預測
    print("\n[步驟 8] 在測試集上預測...")
    print("   💡 使用驗證集找到的最佳閾值進行預測（避免在測試集上偷看答案）...")
    
    # 在測試集上預測
    if use_protocol_grouping:
        # 協議分組預測（測試集）
        test_anomaly_scores = np.zeros(len(X_test))
        
        for protocol, model_proto in protocol_models.items():
            if protocol == 'other':
                proto_mask = ~features_df_test['Proto'].str.lower().isin(['tcp', 'udp'])
            else:
                proto_mask = features_df_test['Proto'].str.lower() == protocol.lower()
            
            X_test_proto = X_test[proto_mask]
            if len(X_test_proto) == 0:
                continue
            
            anomaly_scores_proto = model_proto.predict(X_test_proto)
            anomaly_scores_proto_normalized = -anomaly_scores_proto
            
            # 標準化（使用訓練集的統計量，但這裡簡化使用測試集自己的統計量）
            # 注意：理想情況下應該使用訓練集的統計量，但為了簡化，這裡使用測試集自己的
            if len(anomaly_scores_proto_normalized) > 0:
                score_mean = anomaly_scores_proto_normalized.mean()
                score_std = anomaly_scores_proto_normalized.std()
                if score_std > 1e-6:
                    scores_normalized = (anomaly_scores_proto_normalized - score_mean) / score_std
                    scores_normalized = 1 / (1 + np.exp(-scores_normalized))
                else:
                    scores_normalized = np.zeros_like(anomaly_scores_proto_normalized)
                test_anomaly_scores[proto_mask] = scores_normalized
        
        test_anomaly_scores_normalized = -test_anomaly_scores
    else:
        # 單一模型預測（測試集）
        test_anomaly_scores = model.predict(X_test)
        test_anomaly_scores_normalized = -test_anomaly_scores
    
    print(f"   測試集異常分數範圍：[{test_anomaly_scores_normalized.min():.4f}, {test_anomaly_scores_normalized.max():.4f}]")
    print(f"   測試集平均異常分數：{test_anomaly_scores_normalized.mean():.4f}")
    
    # 使用驗證集找到的最佳閾值（如果有），否則使用 contamination 百分位數
    if best_threshold is not None:
        print(f"   ✅ 使用驗證集找到的最佳閾值：{best_threshold:.4f}")
        y_pred_test = (test_anomaly_scores_normalized >= best_threshold).astype(int)
    else:
        # 無標籤或無法優化閾值時，使用 contamination 百分位數
        threshold_test = np.percentile(test_anomaly_scores_normalized, 100 * (1 - contamination))
        print(f"   ⚠️  使用 contamination 百分位數閾值：{threshold_test:.4f}")
        y_pred_test = (test_anomaly_scores_normalized >= threshold_test).astype(int)
    
    print(f"   測試集預測異常數量：{y_pred_test.sum():,} ({y_pred_test.sum()/len(y_pred_test)*100:.2f}%)")
    
    # 9. 保存訓練結果
    print("\n[步驟 9] 保存訓練結果...")
    save_training_results(
        output_dir=output_dir,
        model=model,
        scaler=scaler,
        protocol_models=protocol_models,
        protocol_scalers=protocol_scalers,
        X_train=X_train_final,
        X_val=X_val,
        X_test=X_test,
        features_df_train=features_df_train,
        features_df_val=features_df_val,
        features_df_test=features_df_test,
        y_train=y_train_final,
        y_val=y_val,
        y_test=y_test,
        train_anomaly_scores=train_anomaly_scores_normalized,
        val_anomaly_scores=val_anomaly_scores_normalized,
        test_anomaly_scores=test_anomaly_scores_normalized,
        best_threshold=best_threshold,
        contamination=contamination,
        use_protocol_grouping=use_protocol_grouping,
        feature_robust_scaler=feature_robust_scaler,
        transformed_feature_cols=transformed_feature_cols,  # 28個：轉換後的特徵
        final_feature_cols=final_feature_cols  # 15個：最終用於模型訓練的特徵
    )
    
    # 10. 基本評估（如果有標籤）
    if use_data_split and y_test is not None:
        print("\n[步驟 10] 測試集基本評估...")
        print("   💡 在測試集上評估基本性能（未應用白名單）...")
        print("   💡 完整評估（含白名單）請使用 postprocess_with_whitelist.py")
        
        if best_threshold is not None:
            y_pred_test = (test_anomaly_scores_normalized >= best_threshold).astype(int)
        else:
            threshold_test = np.percentile(test_anomaly_scores_normalized, 100 * (1 - contamination))
            y_pred_test = (test_anomaly_scores_normalized >= threshold_test).astype(int)
        
        evaluate_and_print(
            y_test, y_pred_test,
            show_confusion_matrix=True,
            show_summary=True
        )
    elif y_true is not None:
        # 無資料分割但有標籤（舊的評估方式，僅用於向後兼容）
        print("\n[步驟 10] 模型評估（警告：未進行資料分割）...")
        print("   ⚠️  警告：未進行 train/test 分割，評估結果可能過於樂觀")
        
        if best_threshold is not None:
            y_pred_test = (test_anomaly_scores_normalized >= best_threshold).astype(int)
        else:
            threshold_test = np.percentile(test_anomaly_scores_normalized, 100 * (1 - contamination))
            y_pred_test = (test_anomaly_scores_normalized >= threshold_test).astype(int)
        
        evaluate_and_print(
            y_true, y_pred_test,
            show_confusion_matrix=True,
            show_detailed=True,
            indent="  "
        )
    
    # 舊的評估代碼（保留用於向後兼容，但已簡化）
    # 8. 在測試集上評估（如果有真實標籤）
    if False and use_data_split and y_test is not None:
        print("\n[步驟 8] 測試集模型評估...")
        print("   💡 在測試集上評估最終性能（未使用測試集標籤優化閾值）...")
        pass
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ 模型訓練完成（總耗時：{total_time:.2f} 秒）")
    print(f"   資料載入：{load_time:.2f} 秒")
    print(f"   訓練結果已保存至：{output_dir}")
    print(f"   💡 下一步：執行 postprocess_with_whitelist.py 進行白名單後處理")
    print("=" * 60)

if __name__ == "__main__":
    main()

