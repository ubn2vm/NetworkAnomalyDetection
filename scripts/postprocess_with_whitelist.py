"""
白名單後處理：分析 False Positives 並應用白名單規則

此腳本專注於模型預測的後處理，遵循單一職責原則。
需要先執行 train_unsupervised.py 生成訓練結果。

此腳本支援三種白名單規則生成方法：
1. 固定閾值方法（原有方法）
2. 評分方法 + Top-N
3. 評分方法 + 評分閾值
"""
import sys
import time
import pickle
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# 將專案根目錄加入 Python 路徑（必須在匯入 src 模組之前）
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    DataLoaderFactory,
    DataSourceType,
    WhitelistAnalyzer,
    WhitelistApplier,
    evaluate_and_print,
    compare_metrics
)
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

# ============================================================================
# 🔧 測試配置：選擇要測試的方法（修改此變量來切換方法）
# ============================================================================
# 選項：
#   "threshold"     - 方法 1：固定閾值方法（原有方法）
#   "scoring_topn"  - 方法 2：評分方法 + Top-N
#   "scoring_threshold" - 方法 3：評分方法 + 評分閾值
WHITELIST_METHOD = "scoring_threshold"  # 預設使用評分閾值方法

# 方法 2 的參數（當 WHITELIST_METHOD == "scoring_topn" 時使用）
SCORING_TOP_N = 20  # Top-N 個組合
SCORING_MIN_SAMPLES = 50  # 最小樣本量要求

# 方法 3 的參數（當 WHITELIST_METHOD == "scoring_threshold" 時使用）
SCORING_THRESHOLD = 0.3  # 評分閾值
SCORING_MIN_SAMPLES_THRESHOLD = 50  # 最小樣本量要求

# ============================================================================


def load_training_results(input_dir: Path) -> Dict[str, Any]:
    """
    載入訓練結果
    
    Args:
        input_dir: 訓練結果目錄
    
    Returns:
        包含所有訓練結果的字典
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"找不到訓練結果目錄: {input_dir}\n"
            f"請先執行 train_unsupervised.py 生成訓練結果。"
        )
    
    results = {}
    
    # 載入配置
    with open(input_dir / "config.json", "r") as f:
        results["config"] = json.load(f)
    
    use_protocol_grouping = results["config"]["use_protocol_grouping"]
    
    # 載入模型
    if use_protocol_grouping:
        with open(input_dir / "protocol_models.pkl", "rb") as f:
            results["protocol_models"] = pickle.load(f)
        if (input_dir / "protocol_scalers.pkl").exists():
            with open(input_dir / "protocol_scalers.pkl", "rb") as f:
                results["protocol_scalers"] = pickle.load(f)
    else:
        with open(input_dir / "model.pkl", "rb") as f:
            results["model"] = pickle.load(f)
        if (input_dir / "scaler.pkl").exists():
            with open(input_dir / "scaler.pkl", "rb") as f:
                results["scaler"] = pickle.load(f)
    
    # 載入特徵資料
    if (input_dir / "X_train.parquet").exists():
        results["X_train"] = pd.read_parquet(input_dir / "X_train.parquet")
    if (input_dir / "X_val.parquet").exists():
        results["X_val"] = pd.read_parquet(input_dir / "X_val.parquet")
    if (input_dir / "X_test.parquet").exists():
        results["X_test"] = pd.read_parquet(input_dir / "X_test.parquet")
    
    if (input_dir / "features_df_train.parquet").exists():
        results["features_df_train"] = pd.read_parquet(input_dir / "features_df_train.parquet")
    if (input_dir / "features_df_val.parquet").exists():
        results["features_df_val"] = pd.read_parquet(input_dir / "features_df_val.parquet")
    if (input_dir / "features_df_test.parquet").exists():
        results["features_df_test"] = pd.read_parquet(input_dir / "features_df_test.parquet")
    
    # 載入標籤和異常分數
    if (input_dir / "y_train.npy").exists():
        results["y_train"] = np.load(input_dir / "y_train.npy")
    if (input_dir / "y_val.npy").exists():
        results["y_val"] = np.load(input_dir / "y_val.npy")
    if (input_dir / "y_test.npy").exists():
        results["y_test"] = np.load(input_dir / "y_test.npy")
    
    if (input_dir / "train_anomaly_scores.npy").exists():
        results["train_anomaly_scores"] = np.load(input_dir / "train_anomaly_scores.npy")
    if (input_dir / "val_anomaly_scores.npy").exists():
        results["val_anomaly_scores"] = np.load(input_dir / "val_anomaly_scores.npy")
    if (input_dir / "test_anomaly_scores.npy").exists():
        results["test_anomaly_scores"] = np.load(input_dir / "test_anomaly_scores.npy")
    
    # 載入 feature_robust_scaler
    if (input_dir / "feature_robust_scaler.pkl").exists():
        with open(input_dir / "feature_robust_scaler.pkl", "rb") as f:
            results["feature_robust_scaler"] = pickle.load(f)
    
    return results


def main():
    print("=" * 60)
    print("白名單後處理：分析 False Positives 並應用白名單規則")
    print("=" * 60)
    
    # 顯示當前使用的白名單方法
    print(f"\n🔧 當前白名單方法：{WHITELIST_METHOD}")
    if WHITELIST_METHOD == "threshold":
        print("   方法 1：固定閾值方法（原有方法）")
    elif WHITELIST_METHOD == "scoring_topn":
        print(f"   方法 2：評分方法 + Top-N (top_n={SCORING_TOP_N}, min_samples={SCORING_MIN_SAMPLES})")
    elif WHITELIST_METHOD == "scoring_threshold":
        print(f"   方法 3：評分方法 + 評分閾值 (threshold={SCORING_THRESHOLD}, min_samples={SCORING_MIN_SAMPLES_THRESHOLD})")
    print("=" * 60)
    
    # 1. 載入訓練結果
    print("\n[步驟 1] 載入訓練結果...")
    start_time = time.time()
    
    input_dir = Path("data/models/unsupervised_training")
    results = load_training_results(input_dir)
    
    config = results["config"]
    use_protocol_grouping = config["use_protocol_grouping"]
    best_threshold = config.get("best_threshold")
    contamination = config.get("contamination", 0.1)
    
    load_time = time.time() - start_time
    print(f"✅ 載入完成（耗時 {load_time:.2f} 秒）")
    print(f"   使用協議分組：{use_protocol_grouping}")
    print(f"   最佳閾值：{best_threshold}")
    print(f"   Contamination：{contamination}")
    
    # 2. 載入原始資料（用於白名單分析）
    print("\n[步驟 2] 載入原始資料（用於白名單分析）...")
    
    parquet_path = Path("data/processed/capture20110817_cleaned_spark.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"找不到 Parquet 檔案: {parquet_path}\n"
            f"請先執行資料處理腳本生成 Parquet 檔案。"
        )
    
    raw_df = pd.read_parquet(parquet_path, engine='pyarrow')
    loader = DataLoaderFactory.create(DataSourceType.BIDIRECTIONAL_BINETFLOW)
    cleaned_df = loader.clean(raw_df)
    
    print(f"✅ 原始資料載入完成：{len(cleaned_df):,} 筆資料")
    
    # 3. 準備資料
    print("\n[步驟 3] 準備資料...")
    
    X_train = results.get("X_train")
    X_test = results.get("X_test")
    features_df_train = results.get("features_df_train")
    features_df_test = results.get("features_df_test")
    y_train = results.get("y_train")
    y_test = results.get("y_test")
    train_anomaly_scores = results.get("train_anomaly_scores")
    test_anomaly_scores = results.get("test_anomaly_scores")
    
    if X_train is None or features_df_train is None:
        raise ValueError("缺少訓練集資料，無法進行白名單分析")
    if X_test is None or features_df_test is None:
        raise ValueError("缺少測試集資料，無法進行評估")
    
    # 獲取索引（用於從 cleaned_df 獲取資料）
    train_idx = features_df_train.index if hasattr(features_df_train.index, 'tolist') else None
    test_idx = features_df_test.index if hasattr(features_df_test.index, 'tolist') else None
    
    print(f"✅ 資料準備完成")
    print(f"   訓練集：{len(X_train):,} 筆")
    print(f"   測試集：{len(X_test):,} 筆")
    
    # 4. 在訓練集上生成預測（用於 FP 分析）
    print("\n[步驟 4] 在訓練集上生成預測（用於 FP 分析）...")
    
    if train_anomaly_scores is None:
        raise ValueError("缺少訓練集異常分數，無法進行白名單分析")
    
    # 使用 contamination 百分位數作為臨時閾值（用於分析）
    temp_threshold = np.percentile(train_anomaly_scores, 100 * (1 - contamination))
    y_pred_train = (train_anomaly_scores >= temp_threshold).astype(int)
    
    print(f"   訓練集預測異常數量：{y_pred_train.sum():,} ({y_pred_train.sum()/len(y_pred_train)*100:.2f}%)")
    
    # 5. 分析訓練集上的 False Positives（歸納白名單規則）
    print("\n[步驟 5] 分析訓練集上的 False Positives 模式...")
    
    whitelist_rules = []
    if y_train is not None:
        # 計算異常分數閾值（用於更精確的白名單應用）
        anomaly_score_threshold = np.percentile(train_anomaly_scores, 25)
        
        # 使用 WhitelistAnalyzer 分析 FP 模式
        print(f"\n   🔧 使用白名單方法：{WHITELIST_METHOD}")
        
        if WHITELIST_METHOD == "threshold":
            analyzer = WhitelistAnalyzer(
                fp_ratio_threshold=0.01,
                normal_ratio_threshold=0.01,
                attack_ratio_threshold=0.03,
                anomaly_score_threshold=anomaly_score_threshold,
                use_scoring_method=False,
                verbose=True
            )
        elif WHITELIST_METHOD == "scoring_topn":
            analyzer = WhitelistAnalyzer(
                normal_ratio_threshold=0.01,
                attack_ratio_threshold=0.03,
                anomaly_score_threshold=anomaly_score_threshold,
                use_scoring_method=True,
                top_n_combos=SCORING_TOP_N,
                min_combo_samples=SCORING_MIN_SAMPLES,
                verbose=True
            )
        elif WHITELIST_METHOD == "scoring_threshold":
            analyzer = WhitelistAnalyzer(
                normal_ratio_threshold=0.01,
                attack_ratio_threshold=0.03,
                anomaly_score_threshold=anomaly_score_threshold,
                use_scoring_method=True,
                score_threshold=SCORING_THRESHOLD,
                min_combo_samples=SCORING_MIN_SAMPLES_THRESHOLD,
                verbose=True
            )
        else:
            raise ValueError(f"未知的白名單方法：{WHITELIST_METHOD}")
        
        whitelist_rules = analyzer.analyze_fp_patterns(
            features_df_train,
            y_pred_train,
            y_train,
            anomaly_scores=train_anomaly_scores,
            cleaned_df=cleaned_df,
            train_idx=train_idx
        )
        
        print(f"\n   ✅ 白名單規則生成完成：共 {len(whitelist_rules)} 條規則")
    else:
        print("\n[步驟 5] 跳過 FP 分析（無標籤資料）...")
    
    # 6. 在測試集上生成預測
    print("\n[步驟 6] 在測試集上生成預測...")
    
    if test_anomaly_scores is None:
        raise ValueError("缺少測試集異常分數，無法進行評估")
    
    # 使用最佳閾值（如果有），否則使用 contamination 百分位數
    if best_threshold is not None:
        print(f"   ✅ 使用訓練時找到的最佳閾值：{best_threshold:.4f}")
        y_pred_test = (test_anomaly_scores >= best_threshold).astype(int)
    else:
        threshold_test = np.percentile(test_anomaly_scores, 100 * (1 - contamination))
        print(f"   ⚠️  使用 contamination 百分位數閾值：{threshold_test:.4f}")
        y_pred_test = (test_anomaly_scores >= threshold_test).astype(int)
    
    print(f"   測試集預測異常數量：{y_pred_test.sum():,} ({y_pred_test.sum()/len(y_pred_test)*100:.2f}%)")
    
    # 7. 應用白名單規則
    print("\n[步驟 7] 應用白名單規則...")
    
    # 確保 features_df_test 按照 X_test 的順序排列
    features_df_test_aligned = features_df_test.loc[X_test.index].reset_index(drop=True) if hasattr(X_test, 'index') else features_df_test.reset_index(drop=True)
    
    applier = WhitelistApplier(
        verbose=True,
        use_anomaly_score_filter=False,
        anomaly_score_percentile=90.0
    )
    
    y_pred_test_filtered, whitelist_stats = applier.apply_rules(
        y_pred_test,
        features_df_test_aligned,
        whitelist_rules,
        anomaly_scores=test_anomaly_scores,
        cleaned_df=cleaned_df,
        test_idx=test_idx,
        y_true=y_test
    )
    
    # 驗證順序
    if y_test is not None:
        assert len(y_pred_test_filtered) == len(y_test), \
            f"長度不匹配：y_pred_test_filtered ({len(y_pred_test_filtered)}) vs y_test ({len(y_test)})"
    
    # 8. 在測試集上評估
    if y_test is not None:
        print("\n[步驟 8] 測試集模型評估...")
        print("   💡 在測試集上評估最終性能（未使用測試集標籤優化閾值）...")
        
        # 評估原始預測
        print("\n   📊 原始預測結果（未應用白名單）：")
        metrics_original = evaluate_and_print(
            y_test, y_pred_test,
            show_confusion_matrix=True,
            show_summary=True
        )
        
        # 評估應用白名單後的預測
        print("\n   📊 應用白名單後的預測結果：")
        metrics_filtered = evaluate_and_print(
            y_test, y_pred_test_filtered,
            show_confusion_matrix=True,
            show_summary=True
        )
        
        # 比較改進
        print("\n   📈 白名單效果比較：")
        compare_metrics(metrics_original, metrics_filtered)
        
        print("\n分類報告（應用白名單後）：")
        print(classification_report(y_test, y_pred_test_filtered, target_names=['正常', '異常'], zero_division=0))
    else:
        print("\n[步驟 8] 無真實標籤，跳過評估")
    
    # 9. 保存白名單後處理結果（供報告生成器使用）
    print("\n[步驟 9] 保存白名單後處理結果...")
    
    output_dir = Path("data/models/whitelist_rules")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 準備白名單後處理結果
    whitelist_postprocess_results = {
        'original_anomalies': int(whitelist_stats.get('original_anomalies', 0)),
        'final_anomalies': int(whitelist_stats.get('filtered_anomalies', 0)),
        'filtered_count': int(whitelist_stats.get('reduced_anomalies', 0)),
        'total_samples': len(y_pred_test),
        'rule_count': len(whitelist_rules),
        'whitelist_method': WHITELIST_METHOD,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 如果有測試集評估結果，也保存指標
    if y_test is not None:
        whitelist_postprocess_results['test_metrics'] = {
            'original': {
                'tn': int(metrics_original.tn),
                'fp': int(metrics_original.fp),
                'fn': int(metrics_original.fn),
                'tp': int(metrics_original.tp),
                'accuracy': float(metrics_original.accuracy),
                'precision': float(metrics_original.precision),
                'recall': float(metrics_original.recall),
                'f1': float(metrics_original.f1)
            },
            'filtered': {
                'tn': int(metrics_filtered.tn),
                'fp': int(metrics_filtered.fp),
                'fn': int(metrics_filtered.fn),
                'tp': int(metrics_filtered.tp),
                'accuracy': float(metrics_filtered.accuracy),
                'precision': float(metrics_filtered.precision),
                'recall': float(metrics_filtered.recall),
                'f1': float(metrics_filtered.f1)
            }
        }
        print(f"   ✅ 測試集評估指標已包含在結果中")
    
    # 保存到 JSON 文件
    results_path = output_dir / "whitelist_postprocess_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(whitelist_postprocess_results, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ 白名單後處理結果已保存至: {results_path}")
    print(f"      原始異常數量: {whitelist_postprocess_results['original_anomalies']:,}")
    print(f"      過濾後異常數量: {whitelist_postprocess_results['final_anomalies']:,}")
    print(f"      過濾掉的數量: {whitelist_postprocess_results['filtered_count']:,}")
    print(f"      白名單規則數: {whitelist_postprocess_results['rule_count']}")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ 白名單後處理完成（總耗時：{total_time:.2f} 秒）")
    print(f"   載入訓練結果：{load_time:.2f} 秒")
    print(f"   使用的白名單方法：{WHITELIST_METHOD}")
    print(f"   生成的白名單規則數：{len(whitelist_rules)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

