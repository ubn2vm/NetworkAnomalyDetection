"""
比較多個無監督異常檢測模型的結果

讀取 output/unsupervised_model_selection/ 目錄下的結果檔案並進行比較。
"""
import sys
import json
from pathlib import Path
import pandas as pd

# 將專案根目錄加入 Python 路徑
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_results():
    """載入所有模型的評估結果"""
    output_dir = Path("output/unsupervised_model_selection")
    
    if not output_dir.exists():
        print(f"❌ 結果目錄不存在: {output_dir}")
        print("   請先運行快速模型基準測試腳本:")
        print("     python scripts/unsupervised_model_selection/quick_model_benchmark.py")
        print("     # 或執行單一模型:")
        print("     python scripts/unsupervised_model_selection/quick_model_benchmark.py --model isolation_forest")
        print("     python scripts/unsupervised_model_selection/quick_model_benchmark.py --model lof")
        print("     python scripts/unsupervised_model_selection/quick_model_benchmark.py --model one_class_svm")
        return None
    
    results = {}
    model_files = {
        'isolation_forest': ['isolation_forest_results.json', 'if_results.json'],
        'lof': ['lof_results.json', 'local_outlier_factor_results.json'],
        'one_class_svm': ['one_class_svm_results.json', 'ocsvm_results.json', 'svm_results.json'],
    }
    
    for model_type, filenames in model_files.items():
        for filename in filenames:
            filepath = output_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    results[model_type] = json.load(f)
                print(f"✅ 載入 {model_type}: {filename}")
                break
    
    return results


def compare_results(results):
    """比較模型結果"""
    if not results:
        print("❌ 沒有可用的結果進行比較")
        return
    
    print("\n" + "=" * 60)
    print("模型比較總結")
    print("=" * 60)
    
    # 檢查是否有標籤
    has_labels = any(r.get('has_labels', False) for r in results.values())
    
    if has_labels:
        # 有標籤的比較
        comparison_data = []
        for model_type, result in results.items():
            if result.get('has_labels', False):
                comparison_data.append({
                    '模型': result['model_name'],
                    '訓練時間(秒)': result['train_time'],
                    '預測時間(秒)': result['predict_time'],
                    '準確率': result.get('accuracy', None),
                    '精確率': result.get('precision', None),
                    '召回率': result.get('recall', None),
                    'F1分數': result.get('f1', None),
                    'ROC AUC': result.get('roc_auc', None),
                    'False Negative': result.get('fn', None),
                    'False Positive': result.get('fp', None),
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n效能比較表:")
        print(comparison_df.to_string(index=False))
        
        # 找出最佳模型
        print("\n最佳模型:")
        if '準確率' in comparison_df.columns and comparison_df['準確率'].notna().any():
            best_acc = comparison_df.loc[comparison_df['準確率'].idxmax()]
            print(f"  最高準確率: {best_acc['模型']} ({best_acc['準確率']:.4f})")
        
        if '精確率' in comparison_df.columns and comparison_df['精確率'].notna().any():
            best_precision = comparison_df.loc[comparison_df['精確率'].idxmax()]
            print(f"  最高精確率: {best_precision['模型']} ({best_precision['精確率']:.4f})")
        
        if '召回率' in comparison_df.columns and comparison_df['召回率'].notna().any():
            best_recall = comparison_df.loc[comparison_df['召回率'].idxmax()]
            print(f"  最高召回率: {best_recall['模型']} ({best_recall['召回率']:.4f})")
        
        if 'F1分數' in comparison_df.columns and comparison_df['F1分數'].notna().any():
            best_f1 = comparison_df.loc[comparison_df['F1分數'].idxmax()]
            print(f"  最高F1分數: {best_f1['模型']} ({best_f1['F1分數']:.4f})")
        
        if 'ROC AUC' in comparison_df.columns and comparison_df['ROC AUC'].notna().any():
            best_auc = comparison_df.loc[comparison_df['ROC AUC'].idxmax()]
            print(f"  最高ROC AUC: {best_auc['模型']} ({best_auc['ROC AUC']:.4f})")
        
        if '訓練時間(秒)' in comparison_df.columns:
            fastest_train = comparison_df.loc[comparison_df['訓練時間(秒)'].idxmin()]
            print(f"  最快訓練: {fastest_train['模型']} ({fastest_train['訓練時間(秒)']:.2f}秒)")
        
        if '預測時間(秒)' in comparison_df.columns:
            fastest_predict = comparison_df.loc[comparison_df['預測時間(秒)'].idxmin()]
            print(f"  最快預測: {fastest_predict['模型']} ({fastest_predict['預測時間(秒)']:.2f}秒)")
        
        if 'False Negative' in comparison_df.columns and comparison_df['False Negative'].notna().any():
            lowest_fn = comparison_df.loc[comparison_df['False Negative'].idxmin()]
            print(f"  最低False Negative: {lowest_fn['模型']} ({lowest_fn['False Negative']})")
    else:
        # 無標籤的比較
        comparison_data = []
        for model_type, result in results.items():
            pred_ratio = result['predictions'].count(1) / len(result['predictions']) * 100 if result.get('predictions') else 0
            comparison_data.append({
                '模型': result['model_name'],
                '訓練時間(秒)': result['train_time'],
                '預測時間(秒)': result['predict_time'],
                '預測異常比例': pred_ratio,
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n效能比較表:")
        print(comparison_df.to_string(index=False))
    
    # 保存比較結果
    output_file = Path("output/unsupervised_model_selection/comparison_summary.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型比較總結\n")
        f.write("=" * 60 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        if has_labels:
            f.write("最佳模型:\n")
            if '準確率' in comparison_df.columns and comparison_df['準確率'].notna().any():
                best_acc = comparison_df.loc[comparison_df['準確率'].idxmax()]
                f.write(f"  最高準確率: {best_acc['模型']} ({best_acc['準確率']:.4f})\n")
            if '精確率' in comparison_df.columns and comparison_df['精確率'].notna().any():
                best_precision = comparison_df.loc[comparison_df['精確率'].idxmax()]
                f.write(f"  最高精確率: {best_precision['模型']} ({best_precision['精確率']:.4f})\n")
            if '召回率' in comparison_df.columns and comparison_df['召回率'].notna().any():
                best_recall = comparison_df.loc[comparison_df['召回率'].idxmax()]
                f.write(f"  最高召回率: {best_recall['模型']} ({best_recall['召回率']:.4f})\n")
            if 'F1分數' in comparison_df.columns and comparison_df['F1分數'].notna().any():
                best_f1 = comparison_df.loc[comparison_df['F1分數'].idxmax()]
                f.write(f"  最高F1分數: {best_f1['模型']} ({best_f1['F1分數']:.4f})\n")
    
    print(f"\n✅ 比較結果已保存至: {output_file}")


def main():
    print("=" * 60)
    print("比較無監督異常檢測模型結果")
    print("=" * 60)
    
    # 載入結果
    results = load_results()
    
    if results:
        # 比較結果
        compare_results(results)
    else:
        print("\n請先運行快速模型基準測試腳本:")
        print("  python scripts/unsupervised_model_selection/quick_model_benchmark.py")
        print("  # 或執行單一模型:")
        print("  python scripts/unsupervised_model_selection/quick_model_benchmark.py --model isolation_forest")
        print("  python scripts/unsupervised_model_selection/quick_model_benchmark.py --model lof")
        print("  python scripts/unsupervised_model_selection/quick_model_benchmark.py --model one_class_svm")


if __name__ == "__main__":
    main()

