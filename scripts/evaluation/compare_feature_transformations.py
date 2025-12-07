"""
特徵轉換方式比較腳本

比較多種特徵轉換方式（原始、log1p、sqrt、boxcox、yeo-johnson），
自動選擇最佳轉換方式，然後應用 RobustScaler 標準化。

評估指標：
1. 分離度（Cohen's d）：正常與異常樣本的分離程度
2. 分佈形狀（偏度、峰度）：是否接近正態分佈
3. 綜合評分：自動選擇最佳轉換方式

使用方法：
# 比較所有長尾分佈特徵（預設，自動使用多進程並行處理）
python scripts/compare_feature_transformations.py

# 比較所有特徵並應用最佳轉換
python scripts/compare_feature_transformations.py --apply-best

# 比較單個特徵
python scripts/compare_feature_transformations.py --feature DstBytes

# 比較單個特徵並應用最佳轉換
python scripts/compare_feature_transformations.py --feature DstBytes --apply-best

# 跳過生成比較圖（加速處理）
python scripts/compare_feature_transformations.py --no-plot

# 指定並行處理的進程數
python scripts/compare_feature_transformations.py --n-jobs 4
"""
import sys
import time
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import logging
import base64
import platform
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy import stats
from scipy.stats import gaussian_kde
from multiprocessing import Pool, cpu_count
from functools import partial

# 設置中文字體
try:
    if platform.system() == 'Windows':
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
        plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
except Exception:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, message=r'.*Glyph.*missing from font.*')
warnings.filterwarnings('ignore', category=UserWarning, message=r'.*glyph.*U\+.*')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    StandardFeatureProcessor,
    convert_label_to_binary,
    DEFAULT_SKEWED_FEATURES,
    apply_log_transformation,
    apply_sqrt_transformation,
    apply_boxcox_transformation,
    calculate_transformation_metrics,
    apply_robust_scaling
)


def calculate_cohens_d(normal_values: pd.Series, anomaly_values: pd.Series) -> float:
    """計算 Cohen's d（分離度指標）"""
    normal_clean = normal_values.replace([np.inf, -np.inf], np.nan).dropna()
    anomaly_clean = anomaly_values.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(normal_clean) == 0 or len(anomaly_clean) == 0:
        return 0.0
    
    normal_mean = normal_clean.mean()
    anomaly_mean = anomaly_clean.mean()
    normal_std = normal_clean.std()
    anomaly_std = anomaly_clean.std()
    
    if pd.isna(normal_mean) or pd.isna(anomaly_mean):
        return 0.0
    
    pooled_std = np.sqrt((normal_std**2 + anomaly_std**2) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = abs(normal_mean - anomaly_mean) / pooled_std
    return cohens_d


def sample_data_for_plotting(
    values: pd.Series,
    y_labels: pd.Series,
    max_samples: int = 50000,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series]:
    """
    對資料進行分層採樣，用於繪圖和計算優化
    
    Args:
        values: 特徵值
        y_labels: 標籤（0=正常, 1=異常）
        max_samples: 最大採樣數量
        random_state: 隨機種子
    
    Returns:
        (採樣後的值, 採樣後的標籤)
    """
    if len(values) <= max_samples:
        return values, y_labels
    
    # 分層採樣：保持正常和異常樣本的比例
    normal_mask = y_labels == 0
    anomaly_mask = y_labels == 1
    
    normal_values = values[normal_mask]
    anomaly_values = values[anomaly_mask]
    
    normal_count = len(normal_values)
    anomaly_count = len(anomaly_values)
    total_count = normal_count + anomaly_count
    
    # 計算採樣比例
    normal_ratio = normal_count / total_count
    anomaly_ratio = anomaly_count / total_count
    
    normal_samples = min(int(max_samples * normal_ratio), normal_count)
    anomaly_samples = min(int(max_samples * anomaly_ratio), anomaly_count)
    
    # 如果採樣後總數不足，調整
    if normal_samples + anomaly_samples < max_samples:
        remaining = max_samples - normal_samples - anomaly_samples
        if normal_count > normal_samples:
            normal_samples += min(remaining, normal_count - normal_samples)
        remaining = max_samples - normal_samples - anomaly_samples
        if anomaly_count > anomaly_samples:
            anomaly_samples += min(remaining, anomaly_count - anomaly_samples)
    
    # 隨機採樣
    if normal_samples < normal_count:
        normal_indices = normal_values.sample(n=normal_samples, random_state=random_state).index
    else:
        normal_indices = normal_values.index
    
    if anomaly_samples < anomaly_count:
        anomaly_indices = anomaly_values.sample(n=anomaly_samples, random_state=random_state).index
    else:
        anomaly_indices = anomaly_values.index
    
    # 合併採樣結果
    sampled_indices = normal_indices.union(anomaly_indices)
    return values.loc[sampled_indices], y_labels.loc[sampled_indices]


def calculate_density_overlap(normal_values: pd.Series, anomaly_values: pd.Series) -> float:
    """計算密度圖重疊度"""
    normal_clean = normal_values.replace([np.inf, -np.inf], np.nan).dropna()
    anomaly_clean = anomaly_values.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(normal_clean) == 0 or len(anomaly_clean) == 0:
        return 1.0
    
    try:
        normal_kde = gaussian_kde(normal_clean)
        anomaly_kde = gaussian_kde(anomaly_clean)
        
        min_val = min(normal_clean.min(), anomaly_clean.min())
        max_val = max(normal_clean.max(), anomaly_clean.max())
        x_range = np.linspace(min_val, max_val, 1000)
        
        normal_density = normal_kde(x_range)
        anomaly_density = anomaly_kde(x_range)
        
        overlap = np.minimum(normal_density, anomaly_density)
        total_area = np.trapezoid(normal_density, x_range) + np.trapezoid(anomaly_density, x_range)
        overlap_area = np.trapezoid(overlap, x_range)
        
        overlap_ratio = overlap_area / total_area if total_area > 0 else 1.0
        return overlap_ratio
    except:
        return 1.0


def evaluate_transformation(
    feature_name: str,
    values: pd.Series,
    y_labels: pd.Series,
    transformation_name: str
) -> Dict[str, float]:
    """評估單個轉換方式"""
    normal_values = values[y_labels == 0]
    anomaly_values = values[y_labels == 1]
    
    if len(normal_values) == 0 or len(anomaly_values) == 0:
        return {
            'transformation': transformation_name,
            'cohens_d': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'is_normal_like': False,
            'density_overlap': 1.0,
            'score': 0.0
        }
    
    # 計算指標
    cohens_d = calculate_cohens_d(normal_values, anomaly_values)
    metrics = calculate_transformation_metrics(values, y_labels)
    density_overlap = calculate_density_overlap(normal_values, anomaly_values)
    
    # 綜合評分（0-100）
    # 分離度（50分）+ 分佈形狀（30分）+ 重疊度（20分）
    separation_score = min(cohens_d / 2.0 * 50, 50)  # Cohen's d > 2.0 得滿分
    distribution_score = (metrics['is_normal_like'] * 30) + (abs(metrics['skewness']) < 1.0) * 10
    overlap_score = (1.0 - density_overlap) * 20
    
    total_score = separation_score + distribution_score + overlap_score
    
    return {
        'transformation': transformation_name,
        'cohens_d': cohens_d,
        'skewness': metrics['skewness'],
        'kurtosis': metrics['kurtosis'],
        'is_normal_like': metrics['is_normal_like'],
        'density_overlap': density_overlap,
        'separation_score': separation_score,
        'distribution_score': distribution_score,
        'overlap_score': overlap_score,
        'total_score': total_score
    }


def compare_transformations_for_feature(
    feature_name: str,
    original_values: pd.Series,
    y_labels: pd.Series,
    output_dir: Path,
    verbose: bool = True,
    max_samples: int = 50000
) -> Tuple[Dict[str, Dict], Optional[Path], str]:
    """
    比較單個特徵的多種轉換方式
    
    Returns:
        (評估結果字典, 比較圖路徑, 特徵名稱)
    """
    if verbose:
        print(f"      比較特徵：{feature_name}")
    start_time = time.time()
    
    # 準備數據
    df_temp = pd.DataFrame({feature_name: original_values})
    
    # 1. 原始值
    t0 = time.time()
    original_eval = evaluate_transformation(
        feature_name, original_values, y_labels, 'original'
    )
    if verbose:
        print(f"         原始值評估：{time.time() - t0:.2f} 秒")
    
    # 2. Log1p 轉換
    t0 = time.time()
    df_log = apply_log_transformation(df_temp, [feature_name], prefix='log_')
    log_values = df_log[f'log_{feature_name}']
    log_eval = evaluate_transformation(
        feature_name, log_values, y_labels, 'log1p'
    )
    if verbose:
        print(f"         Log1p 轉換評估：{time.time() - t0:.2f} 秒")
    
    # 3. 平方根轉換
    t0 = time.time()
    df_sqrt = apply_sqrt_transformation(df_temp, [feature_name], prefix='sqrt_')
    sqrt_values = df_sqrt[f'sqrt_{feature_name}']
    sqrt_eval = evaluate_transformation(
        feature_name, sqrt_values, y_labels, 'sqrt'
    )
    if verbose:
        print(f"         平方根轉換評估：{time.time() - t0:.2f} 秒")
    
    # 4. Box-Cox 轉換（優化：使用採樣來找 lambda）
    boxcox_eval = None
    boxcox_values = None
    t0 = time.time()
    try:
        # 對大量資料進行採樣以加速 lambda 尋找
        if len(df_temp) > max_samples:
            sampled_df, sampled_labels = sample_data_for_plotting(
                df_temp[feature_name], y_labels, max_samples=max_samples
            )
            df_for_lambda = pd.DataFrame({feature_name: sampled_df})
        else:
            df_for_lambda = df_temp
        
        # 只在採樣資料上找 lambda
        _, lambdas = apply_boxcox_transformation(
            df_for_lambda, [feature_name], prefix='boxcox_', method='box-cox'
        )
        
        # 用找到的 lambda 對全部資料進行轉換
        if feature_name in lambdas:
            lambda_param = lambdas[feature_name]
            col_data = df_temp[feature_name]
            
            # Box-Cox 需要正數
            if col_data.min() <= 0:
                shift = -col_data.min() + 1
                full_shifted = col_data + shift
                full_shifted = full_shifted.clip(lower=1e-10)
            else:
                full_shifted = col_data
                shift = 0
            
            # 使用找到的 lambda 對全部資料進行轉換
            valid_mask = ~col_data.isna()
            boxcox_values = col_data.copy()
            if valid_mask.sum() > 0:
                # 使用 scipy.stats.boxcox 的內部邏輯手動計算（因為我們已經知道 lambda）
                shifted_valid = full_shifted[valid_mask]
                if abs(lambda_param) < 1e-10:
                    # lambda = 0 時使用 log
                    boxcox_values.loc[valid_mask] = np.log(shifted_valid)
                else:
                    # 標準 Box-Cox 轉換公式
                    boxcox_values.loc[valid_mask] = (shifted_valid ** lambda_param - 1) / lambda_param
            
            boxcox_eval = evaluate_transformation(
                feature_name, boxcox_values, y_labels, f'boxcox(lambda={lambda_param:.3f})'
            )
        
        if verbose:
            print(f"         Box-Cox 轉換評估：{time.time() - t0:.2f} 秒")
    except Exception as e:
        if verbose:
            print(f"         ⚠️  Box-Cox 轉換失敗：{e} ({time.time() - t0:.2f} 秒)")
    
    # 5. Yeo-Johnson 轉換（優化：使用採樣來找 lambda）
    yeojohnson_eval = None
    yeojohnson_values = None
    t0 = time.time()
    try:
        # 對大量資料進行採樣以加速 lambda 尋找
        if len(df_temp) > max_samples:
            sampled_df, sampled_labels = sample_data_for_plotting(
                df_temp[feature_name], y_labels, max_samples=max_samples
            )
            df_for_lambda = pd.DataFrame({feature_name: sampled_df})
        else:
            df_for_lambda = df_temp
        
        # 只在採樣資料上找 lambda
        _, lambdas_yj = apply_boxcox_transformation(
            df_for_lambda, [feature_name], prefix='yeoj_', method='yeo-johnson'
        )
        
        # 用找到的 lambda 對全部資料進行轉換
        if feature_name in lambdas_yj:
            from sklearn.preprocessing import PowerTransformer
            lambda_param = lambdas_yj[feature_name]
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            # 在採樣資料上 fit 以獲取 transformer
            pt.fit(df_for_lambda[[feature_name]].values)
            # 對全部資料進行轉換
            full_col_2d = df_temp[[feature_name]].values
            yeojohnson_values = pd.Series(
                pt.transform(full_col_2d).flatten(),
                index=df_temp.index
            )
            
            yeojohnson_eval = evaluate_transformation(
                feature_name, yeojohnson_values, y_labels, f'yeo-johnson(lambda={lambda_param:.3f})'
            )
        
        if verbose:
            print(f"         Yeo-Johnson 轉換評估：{time.time() - t0:.2f} 秒")
    except Exception as e:
        if verbose:
            print(f"         ⚠️  Yeo-Johnson 轉換失敗：{e} ({time.time() - t0:.2f} 秒)")
    
    # 收集所有評估結果
    all_evaluations = {
        'original': original_eval,
        'log1p': log_eval,
        'sqrt': sqrt_eval
    }
    
    if boxcox_eval:
        all_evaluations['boxcox'] = boxcox_eval
    if yeojohnson_eval:
        all_evaluations['yeo-johnson'] = yeojohnson_eval
    
    # 選擇最佳轉換方式
    best_transformation = max(all_evaluations.items(), key=lambda x: x[1]['total_score'])
    if verbose:
        print(f"         ✅ 最佳轉換：{best_transformation[0]} (分數：{best_transformation[1]['total_score']:.1f}/100)")
    
    # 生成比較圖（如果沒有禁用）
    image_path = None
    no_plot = getattr(compare_transformations_for_feature, '_no_plot', False)
    if not no_plot:
        t0 = time.time()
        image_path = plot_transformation_comparison(
            feature_name,
            original_values,
            log_values,
            sqrt_values,
            boxcox_values,
            yeojohnson_values,
            y_labels,
            all_evaluations,
            output_dir,
            max_plot_samples=max_samples
        )
        if verbose:
            print(f"         生成比較圖：{time.time() - t0:.2f} 秒")
    
    if verbose:
        print(f"       ✅ 完成：{feature_name} (總計：{time.time() - start_time:.2f} 秒)")
    
    return all_evaluations, image_path, feature_name


def plot_transformation_comparison(
    feature_name: str,
    original_values: pd.Series,
    log_values: pd.Series,
    sqrt_values: pd.Series,
    boxcox_values: Optional[pd.Series],
    yeojohnson_values: Optional[pd.Series],
    y_labels: pd.Series,
    evaluations: Dict[str, Dict],
    output_dir: Path,
    max_plot_samples: int = 50000
) -> Optional[Path]:
    """
    生成轉換方式比較圖
    
    Args:
        max_plot_samples: 繪圖時的最大採樣數量（預設 50000）
    """
    try:
        # 對資料進行採樣以加速繪圖
        sampled_original, sampled_labels = sample_data_for_plotting(
            original_values, y_labels, max_samples=max_plot_samples
        )
        
        # 對其他轉換值也進行對應的採樣
        sampled_indices = sampled_original.index
        sampled_log = log_values.loc[sampled_indices] if log_values is not None else None
        sampled_sqrt = sqrt_values.loc[sampled_indices] if sqrt_values is not None else None
        sampled_boxcox = boxcox_values.loc[sampled_indices] if boxcox_values is not None else None
        sampled_yeoj = yeojohnson_values.loc[sampled_indices] if yeojohnson_values is not None else None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        transformations = [
            ('original', sampled_original, '原始特徵'),
            ('log1p', sampled_log, 'Log1p 轉換'),
            ('sqrt', sampled_sqrt, '平方根轉換'),
            ('boxcox', sampled_boxcox, 'Box-Cox 轉換'),
            ('yeo-johnson', sampled_yeoj, 'Yeo-Johnson 轉換'),
            (None, None, '統計摘要')
        ]
        
        for idx, (trans_name, values, title) in enumerate(transformations):
            ax = axes[idx]
            
            if trans_name is None:
                # 統計摘要
                ax.axis('off')
                summary_text = "評估結果摘要\n\n"
                for name, eval_result in sorted(evaluations.items(), key=lambda x: x[1]['total_score'], reverse=True):
                    summary_text += f"{name}:\n"
                    summary_text += f"  Cohen's d: {eval_result['cohens_d']:.3f}\n"
                    summary_text += f"  偏度: {eval_result['skewness']:.3f}\n"
                    summary_text += f"  峰度: {eval_result['kurtosis']:.3f}\n"
                    summary_text += f"  總分: {eval_result['total_score']:.1f}/100\n\n"
                
                best = max(evaluations.items(), key=lambda x: x[1]['total_score'])
                summary_text += f"推薦：{best[0]}"
                ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='center', family='monospace')
                continue
            
            if values is None:
                ax.text(0.5, 0.5, '轉換失敗', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=10)
                continue
            
            # 繪製密度圖（使用採樣後的資料）
            normal_values = values[sampled_labels == 0].dropna()
            anomaly_values = values[sampled_labels == 1].dropna()
            
            if len(normal_values) > 0 and len(anomaly_values) > 0:
                try:
                    normal_kde = gaussian_kde(normal_values)
                    anomaly_kde = gaussian_kde(anomaly_values)
                    
                    x_range = np.linspace(
                        min(normal_values.min(), anomaly_values.min()),
                        max(normal_values.max(), anomaly_values.max()),
                        1000
                    )
                    
                    ax.plot(x_range, normal_kde(x_range), label='正常', color='blue', linewidth=2)
                    ax.plot(x_range, anomaly_kde(x_range), label='異常', color='red', linewidth=2)
                    ax.fill_between(x_range, normal_kde(x_range), alpha=0.3, color='blue')
                    ax.fill_between(x_range, anomaly_kde(x_range), alpha=0.3, color='red')
                except:
                    # 如果 KDE 失敗，使用直方圖
                    ax.hist(normal_values, bins=50, alpha=0.6, label='正常', color='blue', density=True)
                    ax.hist(anomaly_values, bins=50, alpha=0.6, label='異常', color='red', density=True)
            
            # 添加評估指標到標題
            if trans_name in evaluations:
                eval_result = evaluations[trans_name]
                title += f"\nCohen's d={eval_result['cohens_d']:.3f}, 分數={eval_result['total_score']:.1f}"
            
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('值', fontsize=9)
            ax.set_ylabel('密度', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'特徵轉換方式比較：{feature_name} (採樣：{len(sampled_original):,}/{len(original_values):,})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        safe_feature_name = feature_name.replace('/', '_').replace('\\', '_')
        image_path = output_dir / f"comparison_{safe_feature_name}.png"
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return image_path
    except Exception as e:
        print(f"         ⚠️  生成比較圖失敗：{e}")
        return None


def apply_best_transformation_and_scale(
    features_df: pd.DataFrame,
    best_transformations: Dict[str, str],
    y_labels: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, any], List[str]]:
    """
    應用最佳轉換方式並進行 RobustScaler 標準化
    
    Returns:
        (轉換並標準化後的 DataFrame, scaler 字典, 轉換後的特徵列名)
    """
    result_df = features_df.copy()
    scalers = {}
    transformed_columns = []
    
    for feature_name, best_trans in best_transformations.items():
        if feature_name not in result_df.columns:
            continue
        
        original_values = result_df[feature_name]
        
        # 應用最佳轉換
        if best_trans == 'original':
            transformed_values = original_values
            new_col_name = feature_name
        elif best_trans == 'log1p':
            df_temp = pd.DataFrame({feature_name: original_values})
            df_log = apply_log_transformation(df_temp, [feature_name], prefix='')
            transformed_values = df_log[feature_name]
            new_col_name = feature_name  # 替換原欄位
        elif best_trans == 'sqrt':
            df_temp = pd.DataFrame({feature_name: original_values})
            df_sqrt = apply_sqrt_transformation(df_temp, [feature_name], prefix='')
            transformed_values = df_sqrt[feature_name]
            new_col_name = feature_name
        elif best_trans.startswith('boxcox'):
            df_temp = pd.DataFrame({feature_name: original_values})
            df_boxcox, _ = apply_boxcox_transformation(df_temp, [feature_name], prefix='', method='box-cox')
            if feature_name in df_boxcox.columns:
                transformed_values = df_boxcox[feature_name]
            else:
                transformed_values = original_values
            new_col_name = feature_name
        elif best_trans.startswith('yeo-johnson'):
            df_temp = pd.DataFrame({feature_name: original_values})
            df_yeoj, _ = apply_boxcox_transformation(df_temp, [feature_name], prefix='', method='yeo-johnson')
            if feature_name in df_yeoj.columns:
                transformed_values = df_yeoj[feature_name]
            else:
                transformed_values = original_values
            new_col_name = feature_name
        else:
            transformed_values = original_values
            new_col_name = feature_name
        
        # 更新 DataFrame
        result_df[new_col_name] = transformed_values
        transformed_columns.append(new_col_name)
    
    # 應用 RobustScaler 標準化
    if transformed_columns:
        result_df, robust_scaler = apply_robust_scaling(result_df, transformed_columns)
        scalers['robust'] = robust_scaler
    
    return result_df, scalers, transformed_columns


def main():
    parser = argparse.ArgumentParser(
        description='比較多種特徵轉換方式並選擇最佳方式',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--feature',
        type=str,
        help='要比較的單個特徵名稱（例如：DstBytes）。如果不指定，則比較所有長尾分佈特徵'
    )
    parser.add_argument(
        '--time-feature-stage',
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help='使用的時間特徵階段（預設：4）'
    )
    parser.add_argument(
        '--apply-best',
        action='store_true',
        help='應用最佳轉換方式並保存結果'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,
        help=f'並行處理的進程數（預設：CPU 核心數，當前：{cpu_count()}）'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='跳過生成比較圖（加速處理）'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50000,
        help='繪圖和 Box-Cox lambda 尋找時的最大採樣數量（預設：50000）'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("特徵轉換方式比較")
    print("=" * 60)
    
    # 載入數據
    print("\n[步驟 1] 載入特徵數據...")
    processor = StandardFeatureProcessor(time_feature_stage=args.time_feature_stage)
    
    try:
        X_original = processor.load_features()
        print(f"   ✅ 原始特徵：{len(X_original):,} 筆，{len(X_original.columns)} 個特徵")
        
        # 檢查標籤
        if 'Label' not in X_original.columns:
            print("   ❌ 錯誤：沒有標籤，無法計算分離度指標")
            return 1
        
        X_original = convert_label_to_binary(X_original, verbose=False)
        y_labels = X_original['label_binary']
        print(f"   ✅ 標籤：正常 {len(y_labels[y_labels==0]):,} 筆，異常 {len(y_labels[y_labels==1]):,} 筆")
    except Exception as e:
        print(f"   ❌ 載入原始特徵失敗：{e}")
        return 1
    
    # 決定要比較的特徵（預設比較所有長尾分佈特徵）
    if args.feature:
        features_to_compare = [args.feature]
        print(f"\n   將比較單個特徵：{args.feature}")
    else:
        features_to_compare = [f for f in DEFAULT_SKEWED_FEATURES if f in X_original.columns]
        print(f"\n   將比較所有 {len(features_to_compare)} 個長尾分佈特徵")
    
    # 輸出目錄
    output_dir = Path("output/visualizations/transformation_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 過濾存在的特徵
    valid_features = [f for f in features_to_compare if f in X_original.columns]
    if len(valid_features) == 0:
        print("   ❌ 沒有可用的特徵進行比較")
        return 1
    
    # 決定是否使用多進程
    n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
    use_parallel = len(valid_features) > 1 and n_jobs > 1
    
    # 如果禁用繪圖，設置標記
    if args.no_plot:
        compare_transformations_for_feature._no_plot = True
    
    # 比較結果
    all_results = []
    best_transformations = {}
    
    print("\n[步驟 2] 比較特徵轉換方式...")
    if use_parallel:
        print(f"   使用 {n_jobs} 個進程並行處理 {len(valid_features)} 個特徵...")
        start_time = time.time()
        
        # 準備數據（只傳遞需要的部分，避免傳遞整個 DataFrame）
        feature_data = []
        for feature_name in valid_features:
            feature_data.append({
                'feature_name': feature_name,
                'values': X_original[feature_name].values,
                'y_labels': y_labels.values,
                'output_dir': str(output_dir),
                'verbose': False  # 多進程時不輸出詳細信息
            })
        
        # 並行處理
        def process_feature_wrapper(data):
            """包裝函數用於多進程"""
            feature_name = data['feature_name']
            original_values = pd.Series(data['values'])
            y_labels = pd.Series(data['y_labels'])
            output_dir = Path(data['output_dir'])
            verbose = data['verbose']
            no_plot = data.get('no_plot', False)
            
            try:
                # 臨時設置全局標記（用於跳過繪圖）
                if no_plot:
                    compare_transformations_for_feature._no_plot = True
                
                max_samples = data.get('max_samples', 50000)
                evaluations, image_path, _ = compare_transformations_for_feature(
                    feature_name, original_values, y_labels, output_dir, verbose=verbose, max_samples=max_samples
                )
                
                if no_plot:
                    compare_transformations_for_feature._no_plot = False
                
                return (feature_name, evaluations, image_path, None)
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                return (feature_name, None, None, error_msg)
        
        # 添加 no_plot 和 max_samples 標記到數據中
        for data in feature_data:
            data['no_plot'] = args.no_plot
            data['max_samples'] = args.max_samples
        
        # 並行處理（Windows 和 Linux 都使用相同的方式）
        with Pool(processes=n_jobs) as pool:
            # 使用 imap 以便顯示進度
            results = []
            for idx, result in enumerate(pool.imap(process_feature_wrapper, feature_data), 1):
                results.append(result)
                feature_name = result[0]
                if result[3] is None:  # 沒有錯誤
                    print(f"   [{idx}/{len(valid_features)}] ✅ {feature_name} 完成")
                else:
                    print(f"   [{idx}/{len(valid_features)}] ⚠️  {feature_name} 失敗")
        
        # 收集結果
        for feature_name, evaluations, image_path, error in results:
            if error:
                print(f"   ⚠️  {feature_name} 處理失敗：{error}")
                continue
            
            if evaluations is None:
                continue
            
            # 選擇最佳轉換方式
            best_trans = max(evaluations.items(), key=lambda x: x[1]['total_score'])
            best_transformations[feature_name] = best_trans[0]
            
            # 保存結果
            for trans_name, eval_result in evaluations.items():
                # 創建結果字典，確保使用簡短名稱作為 transformation
                # 先複製 eval_result，移除其中的 transformation 欄位（因為它可能是完整字串）
                eval_result_clean = {k: v for k, v in eval_result.items() if k != 'transformation'}
                result_dict = {
                    'feature': feature_name,
                    'transformation': trans_name,  # 使用簡短名稱（字典的 key）
                    **eval_result_clean
                }
                all_results.append(result_dict)
        
        elapsed_time = time.time() - start_time
        print(f"   ✅ 並行處理完成（耗時：{elapsed_time:.2f} 秒，平均每個特徵：{elapsed_time/len(valid_features):.2f} 秒）")
    else:
        # 單進程處理（用於單個特徵或禁用並行時）
        for idx, feature_name in enumerate(valid_features, 1):
            print(f"\n   [{idx}/{len(valid_features)}] 處理特徵：{feature_name}")
            original_values = X_original[feature_name]
            
            # 比較轉換方式
            evaluations, image_path, _ = compare_transformations_for_feature(
                feature_name, original_values, y_labels, output_dir, verbose=True, max_samples=args.max_samples
            )
            
            # 選擇最佳轉換方式
            best_trans = max(evaluations.items(), key=lambda x: x[1]['total_score'])
            best_transformations[feature_name] = best_trans[0]
            
            # 保存結果
            for trans_name, eval_result in evaluations.items():
                # 創建結果字典，確保使用簡短名稱作為 transformation
                # 先複製 eval_result，移除其中的 transformation 欄位（因為它可能是完整字串）
                eval_result_clean = {k: v for k, v in eval_result.items() if k != 'transformation'}
                result_dict = {
                    'feature': feature_name,
                    'transformation': trans_name,  # 使用簡短名稱（字典的 key）
                    **eval_result_clean
                }
                all_results.append(result_dict)
    
    # 生成摘要報告
    print("\n[步驟 3] 生成摘要報告...")
    results_df = pd.DataFrame(all_results)
    
    csv_path = output_dir / "comparison_results.csv"
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 結果已保存：{csv_path}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("比較結果摘要")
    print("=" * 60)
    print("\n推薦轉換方式：")
    for feature_name, best_trans in best_transformations.items():
        feature_results = results_df[results_df['feature'] == feature_name]
        matching_results = feature_results[feature_results['transformation'] == best_trans]
        
        if len(matching_results) == 0:
            # 如果找不到，嘗試使用字串匹配（處理可能的格式不一致）
            matching_results = feature_results[
                feature_results['transformation'].str.startswith(best_trans, na=False)
            ]
        
        if len(matching_results) > 0:
            best_result = matching_results.iloc[0]
            print(f"  {feature_name:30s} -> {best_trans:15s} "
                  f"(Cohen's d={best_result['cohens_d']:.3f}, "
                  f"偏度={best_result['skewness']:.3f}, "
                  f"總分={best_result['total_score']:.1f}/100)")
        else:
            print(f"  {feature_name:30s} -> {best_trans:15s} (⚠️ 無法找到對應結果)")
    
    # 應用最佳轉換方式
    if args.apply_best:
        print("\n[步驟 4] 應用最佳轉換方式並標準化...")
        
        # 過濾非特徵欄位
        non_feature_columns = ['Label', 'label_binary', 'StartTime', 'SrcAddr', 'DstAddr', 
                              'Sport', 'Dport', 'State', 'Proto']
        feature_columns = [col for col in X_original.columns 
                          if col not in non_feature_columns 
                          and pd.api.types.is_numeric_dtype(X_original[col])]
        
        # 只對要比較的特徵應用最佳轉換，其他特徵保持原樣
        features_to_transform = {k: v for k, v in best_transformations.items() 
                                if k in feature_columns}
        
        # 創建轉換後的 DataFrame
        X_transformed = X_original[feature_columns].copy()
        
        for feature_name, best_trans in features_to_transform.items():
            original_values = X_transformed[feature_name]
            
            if best_trans == 'original':
                continue  # 保持原樣
            elif best_trans == 'log1p':
                df_temp = pd.DataFrame({feature_name: original_values})
                df_log = apply_log_transformation(df_temp, [feature_name], prefix='')
                X_transformed[feature_name] = df_log[feature_name]
            elif best_trans == 'sqrt':
                df_temp = pd.DataFrame({feature_name: original_values})
                df_sqrt = apply_sqrt_transformation(df_temp, [feature_name], prefix='')
                X_transformed[feature_name] = df_sqrt[feature_name]
            elif best_trans.startswith('boxcox'):
                df_temp = pd.DataFrame({feature_name: original_values})
                df_boxcox, _ = apply_boxcox_transformation(df_temp, [feature_name], prefix='', method='box-cox')
                X_transformed[feature_name] = df_boxcox[feature_name]
            elif best_trans.startswith('yeo-johnson'):
                df_temp = pd.DataFrame({feature_name: original_values})
                df_yeoj, _ = apply_boxcox_transformation(df_temp, [feature_name], prefix='', method='yeo-johnson')
                X_transformed[feature_name] = df_yeoj[feature_name]
        
        # 應用 RobustScaler 標準化
        X_scaled, robust_scaler = apply_robust_scaling(X_transformed, list(X_transformed.columns))
        
        # 保存結果
        scaled_path = output_dir / "best_transformed_features.parquet"
        X_scaled.to_parquet(scaled_path, engine='pyarrow')
        print(f"   ✅ 轉換並標準化後的特徵已保存：{scaled_path}")
        
        # 保存 scaler
        import pickle
        scaler_path = output_dir / "best_transformation_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(robust_scaler, f)
        print(f"   ✅ Scaler 已保存：{scaler_path}")
        
        # 保存最佳轉換方式配置
        import json
        config_path = output_dir / "best_transformations.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(best_transformations, f, indent=2, ensure_ascii=False)
        print(f"   ✅ 最佳轉換配置已保存：{config_path}")
    
    print("\n✅ 完成！")
    print(f"   結果文件：{output_dir}")
    if not args.apply_best:
        print(f"\n💡 提示：使用 --apply-best 參數應用最佳轉換方式並保存結果")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

