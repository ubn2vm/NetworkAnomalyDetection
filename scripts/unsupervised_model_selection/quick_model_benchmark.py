"""
快速模型基準測試工具（使用抽樣資料）

使用抽樣資料快速評估和比較多個無監督異常檢測模型，
協助選擇最佳模型進行完整訓練。

設計模式：
- Factory Pattern: 使用 ModelFactory 創建不同模型
- Benchmark Pattern: 統一資料和評估標準進行公平比較

使用方法：
    # 執行所有模型（預設）
    python scripts/unsupervised_model_selection/quick_model_benchmark.py
    
    # 執行單一模型
    python scripts/unsupervised_model_selection/quick_model_benchmark.py --model isolation_forest
    python scripts/unsupervised_model_selection/quick_model_benchmark.py --model lof
    python scripts/unsupervised_model_selection/quick_model_benchmark.py --model one_class_svm
    
    # 使用別名
    python scripts/unsupervised_model_selection/quick_model_benchmark.py --model if
    python scripts/unsupervised_model_selection/quick_model_benchmark.py --model ocsvm
    
    # 比較結果
    python scripts/unsupervised_model_selection/compare_model_results.py
"""
import sys
import time
import json
from pathlib import Path
import argparse

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
    transform_features_for_unsupervised,
    DEFAULT_SKEWED_FEATURES,
    convert_label_to_binary,
    prepare_feature_set,
    FeatureSelector,
    FeatureSelectionStrategy,
    StandardFeatureProcessor,
    calculate_metrics
)
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
import pandas as pd

# 模型名稱映射
MODEL_MAP = {
    'isolation_forest': ModelType.ISOLATION_FOREST,
    'if': ModelType.ISOLATION_FOREST,
    'lof': ModelType.LOCAL_OUTLIER_FACTOR,
    'local_outlier_factor': ModelType.LOCAL_OUTLIER_FACTOR,
    'one_class_svm': ModelType.ONE_CLASS_SVM,
    'ocsvm': ModelType.ONE_CLASS_SVM,
    'svm': ModelType.ONE_CLASS_SVM,
}

# 所有可用的模型列表（用於預設執行所有模型）
ALL_MODELS = ['isolation_forest', 'lof', 'one_class_svm']


def print_feature_breakdown(features_df, transformed_feature_cols, actual_stage):
    """
    拆分並列出每個階段的特徵
    
    Args:
        features_df: 原始特徵 DataFrame
        transformed_feature_cols: 轉換後的特徵欄位列表
        actual_stage: 實際使用的特徵階段
    """
    print("\n" + "=" * 60)
    print("📊 特徵階段拆分與清單")
    print("=" * 60)
    
    # 定義各階段特徵
    base_features = [
        'DstBytes', 'flow_ratio', 'bytes_symmetry', 'is_scanning',
        'src_ratio', 'dst_ratio', 'packet_size', 'bytes_per_second', 'packets_per_second'
    ]
    
    time_features_stage1 = [
        'hour', 'day_of_week', 'day_of_month', 'is_weekend', 'is_work_hour', 'is_night',
        'sin_hour', 'cos_hour', 'sin_day_of_week', 'cos_day_of_week', 
        'sin_day_of_month', 'cos_day_of_month'
    ]
    
    time_features_stage2 = [
        'time_since_last_flow', 'time_to_next_flow'
    ]
    
    time_features_stage3 = [
        'flows_per_minute_by_src', 'unique_dst_per_minute_by_src',
        'unique_dport_per_minute_by_src', 'total_bytes_per_minute_by_src'
    ]
    
    time_features_stage4 = [
        'bidirectional_flow_count', 'bidirectional_total_bytes', 'bidirectional_total_packets',
        'bidirectional_total_src_bytes', 'bidirectional_total_dst_bytes', 'bidirectional_symmetry',
        'bidirectional_avg_bytes_per_flow', 'bidirectional_avg_packets_per_flow', 'bidirectional_avg_duration',
        'bidirectional_window_flow_ratio'  # 時間窗口內聚合後的上下行流量比
    ]
    
    # 從實際 DataFrame 中找出存在的特徵
    all_stage_features = {
        '基礎特徵': base_features,
        '階段1（基本時間特徵）': time_features_stage1,
        '階段2（時間間隔特徵）': time_features_stage2,
        '階段3（時間窗口聚合特徵）': time_features_stage3,
        '階段4（雙向流 Pair 聚合特徵）': time_features_stage4
    }
    
    # 原始特徵拆分
    print("\n[原始特徵拆分]")
    original_by_stage = {}
    for stage_name, stage_features in all_stage_features.items():
        if '階段4' in stage_name and actual_stage < 4:
            continue
        if '階段3' in stage_name and actual_stage < 3:
            continue
        if '階段2' in stage_name and actual_stage < 2:
            continue
        
        available = [f for f in stage_features if f in features_df.columns]
        if available:
            original_by_stage[stage_name] = available
            print(f"\n  {stage_name} ({len(available)} 個):")
            for feat in available:
                print(f"    - {feat}")
    
    # 找出其他原始特徵（不在定義列表中的）
    defined_features = set()
    for features in all_stage_features.values():
        defined_features.update(features)
    
    other_original = [col for col in features_df.columns 
                     if col not in defined_features 
                     and col not in ['Label', 'StartTime', 'SrcAddr', 'DstAddr', 'Sport', 'Dport', 'State', 'Proto']
                     and pd.api.types.is_numeric_dtype(features_df[col])]
    
    if other_original:
        print(f"\n  其他原始特徵 ({len(other_original)} 個):")
        for feat in sorted(other_original)[:20]:  # 只顯示前20個
            print(f"    - {feat}")
        if len(other_original) > 20:
            print(f"    ... 還有 {len(other_original) - 20} 個特徵")
    
    # 轉換後特徵拆分
    print("\n[轉換後特徵拆分]")
    if transformed_feature_cols:
        # 分類轉換後的特徵
        log_features = [f for f in transformed_feature_cols if f.startswith('log_')]
        stage4_transformed = [f for f in transformed_feature_cols if 'bidirectional' in f]
        stage3_transformed = [f for f in transformed_feature_cols if any(s in f for s in ['per_minute_by_src'])]
        stage2_transformed = [f for f in transformed_feature_cols if any(s in f for s in ['time_since', 'time_to'])]
        stage1_transformed = [f for f in transformed_feature_cols if f in time_features_stage1]
        base_transformed = [f for f in transformed_feature_cols if f in base_features]
        other_transformed = [f for f in transformed_feature_cols 
                           if f not in log_features 
                           and f not in stage4_transformed 
                           and f not in stage3_transformed 
                           and f not in stage2_transformed 
                           and f not in stage1_transformed 
                           and f not in base_transformed]
        
        print(f"\n  Log 轉換特徵 ({len(log_features)} 個):")
        for feat in sorted(log_features):
            print(f"    - {feat}")
        
        if base_transformed:
            print(f"\n  基礎特徵 ({len(base_transformed)} 個):")
            for feat in sorted(base_transformed):
                print(f"    - {feat}")
        
        if stage1_transformed:
            print(f"\n  階段1特徵 ({len(stage1_transformed)} 個):")
            for feat in sorted(stage1_transformed):
                print(f"    - {feat}")
        
        if stage2_transformed:
            print(f"\n  階段2特徵 ({len(stage2_transformed)} 個):")
            for feat in sorted(stage2_transformed):
                print(f"    - {feat}")
        
        if stage3_transformed:
            print(f"\n  階段3特徵 ({len(stage3_transformed)} 個):")
            for feat in sorted(stage3_transformed):
                print(f"    - {feat}")
        
        if stage4_transformed:
            print(f"\n  階段4特徵 ({len(stage4_transformed)} 個):")
            for feat in sorted(stage4_transformed):
                print(f"    - {feat}")
        
        if other_transformed:
            print(f"\n  其他轉換特徵 ({len(other_transformed)} 個):")
            for feat in sorted(other_transformed):
                print(f"    - {feat}")
    
    # 統計摘要
    print("\n[統計摘要]")
    print(f"  原始特徵總數: {features_df.shape[1]} 個")
    print(f"  轉換後特徵總數: {len(transformed_feature_cols) if transformed_feature_cols else 0} 個")
    print(f"  實際使用階段: {actual_stage}")
    
    # 完整特徵清單
    print("\n" + "=" * 60)
    print("📋 完整轉換後特徵清單")
    print("=" * 60)
    if transformed_feature_cols:
        for i, feat in enumerate(sorted(transformed_feature_cols), 1):
            print(f"  {i:2d}. {feat}")
    else:
        print("  （無轉換後特徵）")
    
    print("=" * 60)


def prepare_data():
    """準備訓練和測試資料"""
    print("=" * 60)
    print("準備資料")
    print("=" * 60)
    
    # 1. 載入資料
    print("\n[步驟 1] 載入資料...")
    start_time = time.time()
    
    parquet_path = Path("data/processed/capture20110817_cleaned_spark.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"找不到 Parquet 檔案: {parquet_path}\n"
            f"請先執行資料處理腳本生成 Parquet 檔案。"
        )
    
    print(f"   使用 Pandas 讀取 Parquet: {parquet_path}")
    raw_df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    load_time = time.time() - start_time
    print(f"✅ 載入完成：{len(raw_df):,} 筆資料（耗時 {load_time:.2f} 秒）")
    
    # 2. 清洗資料
    print("\n[步驟 2] 清洗資料...")
    loader = DataLoaderFactory.create(DataSourceType.BIDIRECTIONAL_BINETFLOW)
    cleaned_df = loader.clean(raw_df)
    print(f"✅ 清洗完成：{len(cleaned_df):,} 筆資料")
    
    # 3. 特徵處理（使用 FeatureProcessor）
    print("\n[步驟 3] 特徵處理...")
    print("   使用階段4時間特徵（最完整：包含所有階段特徵）")
    print("   - 階段1：基本時間特徵")
    print("   - 階段2：時間間隔特徵")
    print("   - 階段3：時間窗口聚合特徵（按 SrcAddr）")
    print("   - 階段4：雙向流 Pair 聚合特徵（按 IP Pair，需要 PySpark）")
    
    processor = StandardFeatureProcessor(time_feature_stage=4)
    
    # 檢查是否已有處理好的特徵（分階段檢查）
    features_stage3_path = Path("data/processed/features_stage3.parquet")
    features_stage4_path = Path("data/processed/features_stage4.parquet")
    transformed_cache_path = Path("data/processed/features_transformed.parquet")
    
    if features_stage4_path.exists() and transformed_cache_path.exists():
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
        
        # 載入轉換後的特徵和 scaler
        X_transformed, robust_scaler, transformed_feature_cols = processor.load_transformed_features()

        # 確保只保留轉換後的特徵欄位（重要：只保留 scaler 訓練時使用的特徵）
        if transformed_feature_cols:
            # 只保留 scaler 訓練時使用的特徵
            available_cols = [col for col in transformed_feature_cols if col in X_transformed.columns]
            if len(available_cols) != len(transformed_feature_cols):
                print(f"   ⚠️  警告：部分轉換特徵不存在於 DataFrame 中")
                print(f"      預期：{len(transformed_feature_cols)} 個，實際可用：{len(available_cols)} 個")
            X_transformed = X_transformed[available_cols].copy()
        else:
            # 如果沒有 transformed_feature_cols，至少確保只保留數值欄位
            X_transformed = X_transformed.select_dtypes(include=[np.number])
        
        # 確保索引對齊（使用 features_df 的索引）
        if len(X_transformed) == len(features_df):
            # 如果長度相同，使用 features_df 的索引重新索引 X_transformed
            X_transformed = X_transformed.reindex(features_df.index)
            # 填充可能出現的 NaN（理論上不應該有，但為了安全）
            X_transformed = X_transformed.fillna(0)
        
        cache_load_time = time.time() - cache_start_time
        print(f"   ✅ 快取載入完成（耗時 {cache_load_time:.2f} 秒）")
        print(f"   📊 原始特徵數：{features_df.shape[1]} 個")
        print(f"   📊 轉換特徵數：{len(transformed_feature_cols)} 個")
        
        # 輸出特徵拆分與清單
        print_feature_breakdown(features_df, transformed_feature_cols, 4)
    elif features_stage3_path.exists() and not features_stage4_path.exists():
        # 有階段3但沒有階段4，增量執行階段4
        print(f"\n   📂 發現階段3快取，將在此基礎上執行階段4（PySpark）...")
        print(f"   ⏱️  階段4預計需要 30-60 分鐘，請耐心等待...")
        
        # 刪除可能存在的舊 scaler（避免特徵數量不匹配）
        old_scaler_path = Path("data/processed/features_transformed.scaler.pkl")
        old_transformed_path = Path("data/processed/features_transformed.parquet")
        old_info_path = Path("data/processed/features_transformed.info.json")
        
        if old_scaler_path.exists():
            print("   🗑️  刪除舊的 scaler 檔案（避免特徵數量不匹配）...")
            try:
                old_scaler_path.unlink()
                if old_transformed_path.exists():
                    old_transformed_path.unlink()
                if old_info_path.exists():
                    old_info_path.unlink()
                print("   ✅ 舊的 scaler 檔案已刪除")
            except Exception as e:
                print(f"   ⚠️  刪除舊 scaler 失敗: {e}（將繼續執行）")
        
        features_start_time = time.time()
        
        # 增量執行階段4
        features_df, X_transformed, robust_scaler, transformed_feature_cols = processor.process(
            cleaned_df,
            save_features=True,
            save_transformed=True,
            incremental=True  # 增量模式：從階段3到階段4
        )
        
        # 檢查是否成功產生階段4特徵
        stage4_features = [
            'bidirectional_flow_count',
            'bidirectional_total_bytes',
            'bidirectional_symmetry'
        ]
        has_stage4 = any(col in features_df.columns for col in stage4_features)
        
        # 確保只保留數值欄位（移除可能的 Timestamp 欄位）
        X_transformed = X_transformed.select_dtypes(include=[np.number])
        
        features_time = time.time() - features_start_time
        
        if has_stage4:
            print(f"   ✅ 階段4特徵處理完成（耗時 {features_time:.2f} 秒）")
            print(f"   💾 特徵已自動儲存，下次執行將直接載入")
        else:
            print(f"   ⚠️  階段4特徵處理失敗，使用階段3特徵（耗時 {features_time:.2f} 秒）")
            print(f"   💡 提示：可能是 PySpark 在 Windows 上不穩定，已自動回退到階段3")
        
        print(f"   📊 原始特徵數：{features_df.shape[1]} 個")
        print(f"   📊 轉換特徵數：{len(transformed_feature_cols)} 個")
        
        # 輸出特徵拆分與清單
        actual_stage = 4 if has_stage4 else 3
        print_feature_breakdown(features_df, transformed_feature_cols, actual_stage)
    elif not features_stage3_path.exists() and not features_stage4_path.exists():
        # 都沒有，先執行階段3（快速）
        print(f"\n   🔄 未發現快取檔案，先執行階段3特徵工程（約 10-15 分鐘）...")
        print(f"   💡 階段3完成後，可以選擇執行階段4（約 30-60 分鐘）")
        
        # 先執行階段3
        processor_stage3 = StandardFeatureProcessor(time_feature_stage=3)
        features_start_time = time.time()
        
        features_df, X_transformed, robust_scaler, transformed_feature_cols = processor_stage3.process(
            cleaned_df,
            save_features=True,
            save_transformed=False  # 階段3先不儲存轉換後的特徵
        )
        
        # 確保只保留數值欄位
        X_transformed = X_transformed.select_dtypes(include=[np.number])
        
        features_time = time.time() - features_start_time
        print(f"   ✅ 階段3特徵處理完成（耗時 {features_time:.2f} 秒）")
        print(f"   💾 階段3特徵已儲存")
        print(f"   📊 原始特徵數：{features_df.shape[1]} 個")
        print(f"   📊 轉換特徵數：{len(transformed_feature_cols)} 個")
        
        # 輸出特徵拆分與清單
        print_feature_breakdown(features_df, transformed_feature_cols, 3)
        
        # 詢問是否執行階段4
        print(f"\n   ❓ 是否要繼續執行階段4（PySpark，約 30-60 分鐘）？")
        print(f"   💡 提示：階段4會增加雙向流 Pair 聚合特徵，提升模型效果")
        print(f"   ⏸️  如果選擇跳過，可以稍後重新執行此腳本，會自動從階段3繼續")
        
        # 自動執行階段4（可以改為互動式）
        execute_stage4 = True  # 預設執行階段4
        
        if execute_stage4:
            print(f"\n   🔄 開始執行階段4特徵工程（PySpark）...")
            stage4_start_time = time.time()
            
            # 增量執行階段4
            features_df, X_transformed, robust_scaler, transformed_feature_cols = processor.process(
                cleaned_df,
                save_features=True,
                save_transformed=True,
                incremental=True  # 增量模式：從階段3到階段4
            )
            
            # 確保只保留數值欄位
            X_transformed = X_transformed.select_dtypes(include=[np.number])
            
            stage4_time = time.time() - stage4_start_time
            total_time = time.time() - features_start_time
            print(f"   ✅ 階段4特徵處理完成（耗時 {stage4_time:.2f} 秒）")
            print(f"   ✅ 總計耗時：{total_time:.2f} 秒")
            print(f"   💾 階段4特徵已儲存，下次執行將直接載入")
            print(f"   📊 最終原始特徵數：{features_df.shape[1]} 個")
            print(f"   📊 最終轉換特徵數：{len(transformed_feature_cols)} 個")
            
            # 輸出特徵拆分與清單
            print_feature_breakdown(features_df, transformed_feature_cols, 4)
        else:
            print(f"   ⏸️  已跳過階段4，使用階段3特徵繼續執行")
            # 重新載入階段3特徵並轉換
            features_df = processor_stage3.load_features(stage=3)
            # 準備特徵集並轉換
            from src import prepare_feature_set, FeatureSelector, FeatureSelectionStrategy
            X = prepare_feature_set(
                features_df,
                include_base_features=True,
                include_time_features=True,
                time_feature_stage=3
            )
            selector = FeatureSelector(
                remove_constant=True,
                remove_low_variance=True,
                remove_high_correlation=True,
                remove_inf=True,
                remove_high_missing=True,
                correlation_threshold=0.98
            )
            X, _ = selector.select_features(
                X,
                features_df=features_df,
                strategies=[FeatureSelectionStrategy.ALL],
                verbose=False
            )
            X_transformed, robust_scaler, transformed_feature_cols = processor_stage3.transform(
                features_df,
                feature_columns=list(X.columns)
            )
            X_transformed = X_transformed.select_dtypes(include=[np.number])
            
            # 輸出特徵拆分與清單
            print_feature_breakdown(features_df, transformed_feature_cols, 3)
    else:
        # 其他情況：執行完整流程
        print(f"\n   🔄 執行完整特徵處理流程（階段1-4）...")
        features_start_time = time.time()
        
        features_df, X_transformed, robust_scaler, transformed_feature_cols = processor.process(
            cleaned_df,
            save_features=True,
            save_transformed=True
        )
        
        # 確保只保留數值欄位
        X_transformed = X_transformed.select_dtypes(include=[np.number])
        
        features_time = time.time() - features_start_time
        print(f"   ✅ 特徵處理完成（耗時 {features_time:.2f} 秒）")
        print(f"   💾 特徵已自動儲存，下次執行將直接載入")
        print(f"   📊 原始特徵數：{features_df.shape[1]} 個")
        print(f"   📊 轉換特徵數：{len(transformed_feature_cols)} 個")
        
        # 輸出特徵拆分與清單
        print_feature_breakdown(features_df, transformed_feature_cols, processor.time_feature_stage)
    
    print(f"✅ 特徵處理完成")
    
    # 5.5. 統一抽樣 50,000 筆（確保三個模型使用相同資料）
    print("\n[步驟 5.5] 統一抽樣資料（確保模型可比較性）...")
    SAMPLE_SIZE = 50000
    if len(X_transformed) > SAMPLE_SIZE:
        print(f"   原始資料量: {len(X_transformed):,} 筆")
        print(f"   目標抽樣量: {SAMPLE_SIZE:,} 筆")
        
        if 'Label' in features_df.columns:
            # 使用統一的標籤轉換函數
            features_df = convert_label_to_binary(features_df, verbose=True)
            y_binary = features_df['label_binary']
            
            if (y_binary == 1).sum() == 0:
                print(f"   ⚠️  警告：轉換後沒有異常樣本！請檢查標籤轉換邏輯")
                # 如果沒有異常樣本，無法使用 stratify，改用隨機抽樣
                print(f"   ⚠️  改用隨機抽樣（無法使用分層抽樣）")
                X_sampled = X_transformed.sample(n=SAMPLE_SIZE, random_state=42)
                y_sampled = y_binary.loc[X_sampled.index].copy()
                features_df_sampled = features_df.loc[X_sampled.index].copy()
            else:
                print(f"   ✅ 標籤轉換正常，異常樣本: {(y_binary == 1).sum():,} 筆")
                
                # 使用 stratify 抽樣，確保異常比例保留
                # 使用 StratifiedShuffleSplit 進行分層抽樣以確保精確數量
                sss = StratifiedShuffleSplit(n_splits=1, train_size=SAMPLE_SIZE, random_state=42)
                train_idx, _ = next(sss.split(X_transformed, y_binary))
                # 使用 iloc 獲取位置索引對應的資料，保持原始索引
                X_sampled = X_transformed.iloc[train_idx].copy()
                y_sampled = y_binary.iloc[train_idx].copy()
                print(f"   ✅ 分層抽樣完成：{len(X_sampled):,} 筆")
                print(f"   抽樣後異常比例: {y_sampled.sum()/len(y_sampled)*100:.2f}%")
                print(f"   抽樣後異常樣本數: {y_sampled.sum():,} 筆")
                
                # 更新 features_df 索引以匹配抽樣後的資料（使用相同的索引）
                features_df_sampled = features_df.loc[X_sampled.index].copy()
        else:
            # 無標籤時使用簡單隨機抽樣
            X_sampled = X_transformed.sample(n=SAMPLE_SIZE, random_state=42)
            y_sampled = None
            features_df_sampled = features_df.loc[X_sampled.index].copy()
            print(f"   ✅ 隨機抽樣完成：{len(X_sampled):,} 筆（無標籤）")
        
        X_transformed = X_sampled
        features_df = features_df_sampled
        # 更新 y_binary 為抽樣後的標籤
        if 'Label' in features_df.columns:
            y_binary = y_sampled
    else:
        print(f"   資料量 ({len(X_transformed):,} 筆) 小於目標抽樣量 ({SAMPLE_SIZE:,} 筆)，跳過抽樣")
        if 'Label' in features_df.columns:
            # 使用統一的標籤轉換函數
            if 'label_binary' not in features_df.columns:
                features_df = convert_label_to_binary(features_df, verbose=False)
            y_binary = features_df['label_binary']
        else:
            y_binary = None
    
    # 6. 分割資料
    print("\n[步驟 6] 分割資料...")
    if 'Label' in features_df.columns:
        # 確保標籤已轉換
        if 'label_binary' not in features_df.columns:
            features_df = convert_label_to_binary(features_df, verbose=False)
        y_binary = features_df['label_binary']
        
        # 確保 X_transformed 和 y_binary 的長度一致
        if len(X_transformed) != len(y_binary):
            print(f"   ⚠️  警告：X_transformed ({len(X_transformed):,} 筆) 和 y_binary ({len(y_binary):,} 筆) 長度不一致")
            print(f"   使用 X_transformed 的索引來對齊 y_binary")
            y_binary = y_binary.loc[X_transformed.index]
        
        # 檢查是否有異常樣本，如果沒有則不使用 stratify
        if (y_binary == 1).sum() == 0:
            print(f"   ⚠️  警告：沒有異常樣本，無法使用 stratify，改用隨機分割")
            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, y_binary, test_size=0.3, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, y_binary, test_size=0.3, random_state=42, stratify=y_binary
            )
        print(f"✅ 資料分割完成：訓練集 {len(X_train):,} 筆，測試集 {len(X_test):,} 筆")
        print(f"   訓練集異常比例: {y_train.sum()/len(y_train)*100:.2f}% ({y_train.sum():,} 筆異常)")
        print(f"   測試集異常比例: {y_test.sum()/len(y_test)*100:.2f}% ({y_test.sum():,} 筆異常)")
        
        # 診斷：檢查測試集是否有異常樣本
        if y_test.sum() == 0:
            print(f"   ⚠️  警告：測試集中沒有異常樣本！")
            print(f"      這會導致 TP=0, FN=0，精確率和召回率都為 0")
        else:
            print(f"   ✅ 測試集包含 {y_test.sum():,} 筆異常樣本，可用於評估")
    else:
        X_train, X_test = train_test_split(
            X_transformed, test_size=0.3, random_state=42
        )
        y_test = None
        print(f"✅ 資料分割完成：訓練集 {len(X_train):,} 筆，測試集 {len(X_test):,} 筆（無標籤）")
    
    # 收集特徵統計信息
    # 過濾掉標籤欄位和非特徵欄位
    non_feature_columns = ['Label', 'label_binary', 'StartTime', 'SrcAddr', 'DstAddr', 
                          'Sport', 'Dport', 'State', 'Proto']
    feature_columns = [col for col in features_df.columns 
                      if col not in non_feature_columns]
    
    feature_info = {
        'original_feature_count': len(feature_columns),
        'original_feature_names': feature_columns,  # 只包含真正的特徵
        'transformed_feature_count': len(transformed_feature_cols) if transformed_feature_cols else 0,
        'transformed_feature_names': transformed_feature_cols if transformed_feature_cols else []
    }
    
    return X_train, X_test, y_test, robust_scaler, transformed_feature_cols, feature_info


def evaluate_and_save_model(model_type_name, X_train, X_test, y_test, robust_scaler):
    """評估單個模型並保存結果"""
    model_type = MODEL_MAP.get(model_type_name.lower())
    if model_type is None:
        raise ValueError(f"未知的模型類型: {model_type_name}。可用類型: {list(MODEL_MAP.keys())}")
    
    model_name_map = {
        ModelType.ISOLATION_FOREST: "Isolation Forest",
        ModelType.LOCAL_OUTLIER_FACTOR: "Local Outlier Factor (LOF)",
        ModelType.ONE_CLASS_SVM: "One-Class SVM",
    }
    model_name = model_name_map[model_type]
    
    print("\n" + "=" * 60)
    print(f"評估模型: {model_name}")
    print("=" * 60)
    
    # 創建模型
    model = ModelFactory.create(model_type)
    
    # 所有模型現在都使用相同的訓練資料（已在 prepare_data() 中統一抽樣）
    # 不再需要個別抽樣，確保三個模型使用完全相同的資料
    X_train_actual = X_train
    print(f"  📌 使用統一抽樣的訓練資料：{len(X_train_actual):,} 筆")
    
    # 模型特定參數設定
    train_kwargs = {}
    if model_type == ModelType.LOCAL_OUTLIER_FACTOR:
        # LOF 需要設定 n_neighbors
        n_neighbors = min(20, max(5, len(X_train_actual) // 100))
        train_kwargs['n_neighbors'] = n_neighbors
        print(f"  📌 使用 n_neighbors={n_neighbors}")
    elif model_type == ModelType.ONE_CLASS_SVM:
        print(f"  💡 提示：One-Class SVM 計算複雜度較高，已使用統一抽樣加速訓練")
    
    # 訓練模型
    print(f"\n[訓練階段]")
    start_time = time.time()
    
    if robust_scaler is not None:
        X_train_scaled = robust_scaler.transform(X_train_actual.values)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_actual.columns, index=X_train_actual.index)
        train_params = {
            'contamination': 0.1,
            'use_external_scaler': True,
            'external_scaler': robust_scaler,
            **train_kwargs
        }
        trained_model, model_scaler = model.train(X_train_scaled_df, **train_params)
    else:
        train_params = {
            'contamination': 0.1,
            **train_kwargs
        }
        trained_model, model_scaler = model.train(X_train_actual, **train_params)
    
    train_time = time.time() - start_time
    print(f"✅ 訓練完成（耗時 {train_time:.2f} 秒）")
    
    # 預測
    print(f"\n[預測階段]")
    start_time = time.time()
    
    if robust_scaler is not None:
        X_test_scaled = robust_scaler.transform(X_test.values)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        scores = model.predict(X_test_scaled_df)
    else:
        scores = model.predict(X_test)
    
    predict_time = time.time() - start_time
    print(f"✅ 預測完成（耗時 {predict_time:.2f} 秒）")
    
    # 計算異常標籤
    contamination = 0.1
    threshold = np.percentile(scores, contamination * 100)
    predictions = (scores <= threshold).astype(int)
    
    y_test_binary = (y_test == 1).astype(int) if y_test is not None else None
    
    # 獲取最終用於模型的特徵欄位
    feature_columns = list(X_train_actual.columns)
    
    # 計算指標
    result = {
        'model_name': model_name,
        'model_type': model_type_name.lower(),
        'train_time': train_time,
        'predict_time': predict_time,
        'scores': scores.tolist(),  # 轉換為列表以便 JSON 序列化
        'predictions': predictions.tolist(),
        'threshold': float(threshold),
        'contamination': contamination,
        'feature_columns': feature_columns,  # 最終用於模型的特徵欄位列表
        'feature_count': len(feature_columns),  # 特徵數量
        'feature_info': {
            'model_feature_count': len(feature_columns),
            'model_feature_names': feature_columns
        }
    }
    
    if y_test_binary is not None:
        print(f"\n[效能指標]")
        print(f"  異常分數範圍: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  異常分數平均值: {scores.mean():.4f}")
        print(f"  異常分數標準差: {scores.std():.4f}")
        print(f"  預測異常數量: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")
        print(f"  實際異常數量: {y_test_binary.sum()} ({y_test_binary.sum()/len(y_test_binary)*100:.2f}%)")
        
        # 使用評估模組計算指標
        metrics = calculate_metrics(y_test_binary, predictions)
        
        # 輸出混淆矩陣（簡化格式，符合原有風格）
        print(f"\n  混淆矩陣:")
        print(f"    TN={metrics.tn}, FP={metrics.fp}")
        print(f"    FN={metrics.fn}, TP={metrics.tp}")
        
        # 輸出基本指標
        print(f"\n  準確率 (Accuracy): {metrics.accuracy:.4f}")
        print(f"  精確率 (Precision): {metrics.precision:.4f}")
        print(f"  召回率 (Recall): {metrics.recall:.4f}")
        print(f"  F1 分數: {metrics.f1:.4f}")
        
        # 輸出詳細指標（包含公式說明）
        print(f"\n  異常類別（正類）指標:")
        print(f"    精確率 (Precision): {metrics.precision:.4f}  [TP/(TP+FP) = {metrics.tp}/({metrics.tp}+{metrics.fp})]")
        print(f"    召回率 (Recall): {metrics.recall:.4f}  [TP/(TP+FN) = {metrics.tp}/({metrics.tp}+{metrics.fn})]")
        print(f"    F1 分數: {metrics.f1:.4f}")
        
        print(f"\n  正常類別（負類）指標:")
        print(f"    精確率 (Precision): {metrics.precision_normal:.4f}  [TN/(TN+FN) = {metrics.tn}/({metrics.tn}+{metrics.fn})]")
        print(f"    召回率 (Recall): {metrics.recall_normal:.4f}  [TN/(TN+FP) = {metrics.tn}/({metrics.tn}+{metrics.fp})]")
        
        # 計算 ROC AUC
        try:
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            scores_prob = 1 - scores_normalized
            roc_auc = roc_auc_score(y_test_binary, scores_prob)
            print(f"  ROC AUC: {roc_auc:.4f}")
        except Exception as e:
            print(f"  ROC AUC: 無法計算 ({str(e)})")
            roc_auc = None
        
        # 更新 result 字典
        result.update({
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,  # 異常類別的精確率（保持向後兼容）
            'recall': metrics.recall,  # 異常類別的召回率（保持向後兼容）
            'f1': metrics.f1,
            'roc_auc': roc_auc,
            'precision_anomaly': metrics.precision_anomaly,
            'recall_anomaly': metrics.recall_anomaly,
            'precision_normal': metrics.precision_normal,
            'recall_normal': metrics.recall_normal,
            'tn': metrics.tn,
            'fp': metrics.fp,
            'fn': metrics.fn,
            'tp': metrics.tp,
            'has_labels': True,
        })
        
        # 保存詳細分類報告到結果中（不打印）
        report = classification_report(y_test_binary, predictions, target_names=['正常', '異常'], zero_division=0, output_dict=True)
        result['classification_report'] = report
    else:
        result['has_labels'] = False
    
    # 保存結果
    output_dir = Path("output/unsupervised_model_selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{model_type_name.lower()}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 結果已保存至: {output_file}")
    print(f"📊 模型使用的特徵數量: {len(feature_columns)} 個")
    print(f"📋 特徵欄位列表已保存至結果檔案")
    
    return result


def normalize_model_name(model_name: str) -> str:
    """
    將模型名稱標準化為主要名稱
    
    Args:
        model_name: 模型名稱（可能是別名）
    
    Returns:
        標準化的模型名稱（主要名稱）
    
    >>> normalize_model_name('if')
    'isolation_forest'
    >>> normalize_model_name('isolation_forest')
    'isolation_forest'
    >>> normalize_model_name('ocsvm')
    'one_class_svm'
    """
    model_name_lower = model_name.lower()
    
    # 如果已經是主要名稱，直接返回
    if model_name_lower in ALL_MODELS:
        return model_name_lower
    
    # 將別名映射到主要名稱
    alias_map = {
        'if': 'isolation_forest',
        'local_outlier_factor': 'lof',
        'ocsvm': 'one_class_svm',
        'svm': 'one_class_svm',
    }
    
    # 先檢查別名映射
    if model_name_lower in alias_map:
        return alias_map[model_name_lower]
    
    # 如果 MODEL_MAP 中有，嘗試找到對應的主要名稱
    if model_name_lower in MODEL_MAP:
        model_type = MODEL_MAP[model_name_lower]
        # 反向查找主要名稱
        for main_name in ALL_MODELS:
            if MODEL_MAP[main_name] == model_type:
                return main_name
    
    # 如果都找不到，返回原始名稱（會在下層檢查時報錯）
    return model_name_lower


def main():
    parser = argparse.ArgumentParser(
        description='快速模型基準測試工具（使用抽樣資料）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 執行所有模型（預設）
  python scripts/unsupervised_model_selection/quick_model_benchmark.py
  
  # 執行單一模型
  python scripts/unsupervised_model_selection/quick_model_benchmark.py --model isolation_forest
  python scripts/unsupervised_model_selection/quick_model_benchmark.py --model lof
  python scripts/unsupervised_model_selection/quick_model_benchmark.py --model one_class_svm
  
  # 使用別名
  python scripts/unsupervised_model_selection/quick_model_benchmark.py --model if
  python scripts/unsupervised_model_selection/quick_model_benchmark.py --model ocsvm
  
  # 比較結果
  python scripts/unsupervised_model_selection/compare_model_results.py
        """
    )
    parser.add_argument(
        '--model',
        '-m',
        choices=['isolation_forest', 'if', 'lof', 'local_outlier_factor', 'one_class_svm', 'ocsvm', 'svm', 'all'],
        default='all',
        help='要運行的模型類型（預設：all，執行所有模型）'
    )
    
    args = parser.parse_args()
    
    # 確定要執行的模型列表
    if args.model.lower() == 'all':
        models_to_run = ALL_MODELS
        print("=" * 60)
        print("執行所有模型")
        print("=" * 60)
        print(f"將執行以下模型：{', '.join(models_to_run)}")
    else:
        normalized_name = normalize_model_name(args.model)
        if normalized_name not in ALL_MODELS:
            raise ValueError(
                f"未知的模型類型: {args.model}。\n"
                f"可用類型: {', '.join(ALL_MODELS)}\n"
                f"可用別名: if, local_outlier_factor, ocsvm, svm"
            )
        
        models_to_run = [normalized_name]
        print("=" * 60)
        print(f"執行單一模型: {normalized_name}")
        print("=" * 60)
    
    # 準備資料（所有模型共用相同資料，只需準備一次）
    print("\n準備資料（所有模型共用）...")
    X_train, X_test, y_test, robust_scaler, feature_cols, feature_info = prepare_data()
    
    # 執行每個模型
    results = {}
    total_start_time = time.time()
    
    for i, model_name in enumerate(models_to_run, 1):
        print("\n" + "=" * 60)
        print(f"[{i}/{len(models_to_run)}] 執行模型: {model_name}")
        print("=" * 60)
        
        try:
            # 評估並保存模型
            result = evaluate_and_save_model(model_name, X_train, X_test, y_test, robust_scaler)
            results[model_name] = result
            
            # 更新結果文件，添加完整的特徵信息
            output_dir = Path("output/unsupervised_model_selection")
            output_file = output_dir / f"{model_name}_results.json"
            
            # 讀取現有結果並更新
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
            else:
                result_data = result.copy()
            
            # 確保 feature_info 鍵存在
            if 'feature_info' not in result_data:
                result_data['feature_info'] = {}
            
            # 添加完整的特徵信息
            result_data['feature_info'].update({
                'original_feature_count': feature_info['original_feature_count'],
                'original_feature_names': feature_info['original_feature_names'],
                'transformed_feature_count': feature_info['transformed_feature_count'],
                'transformed_feature_names': feature_info['transformed_feature_names']
            })
            
            # 保存更新後的結果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ 模型 {model_name} 執行完成")
            print(f"📊 特徵統計信息已更新到結果文件")
            print(f"   原始特徵數: {feature_info['original_feature_count']} 個")
            print(f"   轉換後特徵數: {feature_info['transformed_feature_count']} 個")
            print(f"   模型使用特徵數: {result_data['feature_info']['model_feature_count']} 個")
            print(f"   結果已保存至: {output_file}")
            
        except Exception as e:
            print(f"\n❌ 模型 {model_name} 執行失敗: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
            continue
    
    total_time = time.time() - total_start_time
    
    # 總結
    print("\n" + "=" * 60)
    print("執行總結")
    print("=" * 60)
    print(f"總耗時: {total_time:.2f} 秒")
    print(f"成功執行: {len([r for r in results.values() if 'error' not in r])}/{len(models_to_run)} 個模型")
    
    if len(models_to_run) > 1:
        print(f"\n所有結果已保存至: output/unsupervised_model_selection/")
        print(f"使用以下命令比較結果:")
        print(f"  python scripts/unsupervised_model_selection/compare_model_results.py")
    else:
        model_name = models_to_run[0]
        print(f"\n結果已保存至: output/unsupervised_model_selection/{model_name}_results.json")
        print(f"使用以下命令比較結果:")
        print(f"  python scripts/unsupervised_model_selection/compare_model_results.py")


if __name__ == "__main__":
    main()

