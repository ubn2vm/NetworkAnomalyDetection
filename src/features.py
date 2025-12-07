"""
特徵提取與分析模組

提供長尾分佈分析和 UDP/TCP 切分分析功能。
用於 EDA 和特徵工程。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import contextlib
import sys
import io

from src.feature_engineer import extract_features as _extract_features_original
from src.feature_transformer import (
    DEFAULT_SKEWED_FEATURES,
    apply_log_transformation,
    calculate_transformation_metrics
)

# ===== 字體設定 =====
# 設定 matplotlib 以正確顯示中文
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 設定中文字型（Windows）
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
except:
    # 如果找不到中文字體，使用 DejaVu Sans（至少可以顯示英文和數字）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# ===== 字體設定結束 =====


class FilteredOutput:
    """
    過濾 stdout 輸出，隱藏 PySpark 相關的冗長訊息。
    
    用於在 notebook 中隱藏系統記憶體、Spark 配置和詳細錯誤訊息。
    """
    def __init__(self, original_stream, filter_keywords):
        self.original_stream = original_stream
        self.filter_keywords = filter_keywords
        self.buffer = io.StringIO()
    
    def write(self, text):
        # 檢查是否包含要過濾的關鍵字
        if not any(keyword in text for keyword in self.filter_keywords):
            self.original_stream.write(text)
        # 所有輸出都寫入 buffer（用於調試）
        self.buffer.write(text)
    
    def flush(self):
        self.original_stream.flush()
        self.buffer.flush()
    
    def getvalue(self):
        return self.buffer.getvalue()


def _extract_features_quietly(
    df: pd.DataFrame,
    flow_type: str = 'auto',
    include_time_features: bool = True,
    time_feature_stage: int = 1,
    include_bidirectional_features: bool = False
) -> pd.DataFrame:
    """
    安靜地提取特徵，過濾 PySpark 相關的冗長輸出。
    
    Args:
        df: 輸入的 NetFlow DataFrame
        flow_type: 流類型 ('auto', 'bidirectional')
        include_time_features: 是否包含時間特徵
        time_feature_stage: 時間特徵階段（1-4）
        include_bidirectional_features: 是否包含雙向流特徵
    
    Returns:
        包含特徵的 DataFrame
    """
    # 定義要過濾的關鍵字
    filter_keywords = [
        '💾 系統記憶體',
        '⚙️  Spark 記憶體配置',
        '⚠️  PySpark 執行失敗',
        '錯誤訊息：',
        '💡 提示：Pandas 版本較慢',
        '⚠️  PySpark 未安裝',
        'collectToPython',
        'SparkException',
        'Job aborted'
    ]
    
    # 創建過濾器
    filtered_stdout = FilteredOutput(sys.stdout, filter_keywords)
    
    # 使用過濾器執行特徵提取
    with contextlib.redirect_stdout(filtered_stdout):
        features_df = _extract_features_original(
            df,
            flow_type=flow_type,
            include_time_features=include_time_features,
            time_feature_stage=time_feature_stage,
            include_bidirectional_features=include_bidirectional_features
        )
    
    return features_df


def analyze_long_tail_distribution(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    show_plots: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    分析特徵的長尾分佈特性。
    
    計算偏度（skewness）、峰度（kurtosis）等統計指標，
    並可視化分佈情況。
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'TotBytes': [100, 200, 300, 1000, 10000],
    ...     'Dur': [1.0, 2.0, 3.0, 10.0, 100.0]
    ... })
    >>> result = analyze_long_tail_distribution(df, show_plots=False)
    >>> 'TotBytes' in result
    True
    >>> result['TotBytes']['skewness'] > 0  # 正偏（長尾在右側）
    True
    
    Args:
        df: 輸入的 DataFrame
        features: 要分析的特徵列表，如果為 None 則使用 DEFAULT_SKEWED_FEATURES
        show_plots: 是否顯示分佈圖
    
    Returns:
        包含每個特徵統計指標的字典
    """
    if features is None:
        features = [f for f in DEFAULT_SKEWED_FEATURES if f in df.columns]
    
    results = {}
    
    for feature in features:
        if feature not in df.columns:
            continue
        
        values = df[feature].dropna()
        if len(values) == 0:
            continue
        
        # 計算統計指標
        skewness = stats.skew(values)
        kurtosis = stats.kurtosis(values, fisher=False)  # fisher=False: 正態分佈的峰度為3
        
        # 計算分位數
        q25 = values.quantile(0.25)
        q50 = values.quantile(0.50)
        q75 = values.quantile(0.75)
        q95 = values.quantile(0.95)
        q99 = values.quantile(0.99)
        
        # 判斷是否為長尾分佈
        # 長尾分佈的特徵：偏度 > 1 或峰度 > 5
        is_long_tail = (abs(skewness) > 1) or (kurtosis > 5)
        
        results[feature] = {
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'mean': float(values.mean()),
            'median': float(q50),
            'std': float(values.std()),
            'q25': float(q25),
            'q75': float(q75),
            'q95': float(q95),
            'q99': float(q99),
            'is_long_tail': is_long_tail
        }
        
        if show_plots:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # 左圖：原始分佈（對數尺度）
            ax1 = axes[0]
            ax1.hist(values, bins=50, edgecolor='black', alpha=0.7)
            ax1.set_xlabel(f'{feature} (Original Scale)', fontsize=10)
            ax1.set_ylabel('Frequency', fontsize=10)
            ax1.set_title(f'{feature} Original Distribution\nSkewness={skewness:.2f}, Kurtosis={kurtosis:.2f}', 
                         fontsize=11, fontweight='bold')
            ax1.set_yscale('log')
            if values.max() / (values.min() + 1) > 100:
                ax1.set_xscale('log')
            ax1.grid(True, alpha=0.3)
            
            # 右圖：Log 轉換後的分佈
            ax2 = axes[1]
            log_values = np.log1p(values)
            ax2.hist(log_values, bins=50, edgecolor='black', alpha=0.7, color='green')
            ax2.set_xlabel(f'log({feature} + 1)', fontsize=10)
            ax2.set_ylabel('Frequency', fontsize=10)
            log_skewness = stats.skew(log_values)
            log_kurtosis = stats.kurtosis(log_values, fisher=False)
            ax2.set_title(f'{feature} After Log Transformation\nSkewness={log_skewness:.2f}, Kurtosis={log_kurtosis:.2f}', 
                         fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    return results


def analyze_protocol_split(
    df: pd.DataFrame,
    protocol_col: str = 'Proto',
    show_plots: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    分析 UDP/TCP 切分的統計資訊。
    
    分別統計 UDP 和 TCP 流量的特徵分佈，並比較差異。
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Proto': ['udp', 'tcp', 'udp', 'tcp'],
    ...     'TotBytes': [100, 200, 150, 300],
    ...     'TotPkts': [10, 20, 15, 30]
    ... })
    >>> result = analyze_protocol_split(df, show_plots=False)
    >>> 'udp' in result
    True
    >>> 'tcp' in result
    True
    
    Args:
        df: 輸入的 DataFrame
        protocol_col: 協議欄位名稱，預設為 'Proto'
        show_plots: 是否顯示比較圖
    
    Returns:
        包含 UDP 和 TCP 統計資訊的字典
    """
    if protocol_col not in df.columns:
        raise ValueError(f"找不到協議欄位: {protocol_col}")
    
    results = {}
    
    # 分離 UDP 和 TCP
    udp_df = df[df[protocol_col].str.lower() == 'udp'].copy()
    tcp_df = df[df[protocol_col].str.lower() == 'tcp'].copy()
    
    print("=" * 60)
    print("【UDP/TCP 切分分析】")
    print("=" * 60)
    print(f"\nUDP 流量: {len(udp_df):,} 筆 ({len(udp_df)/len(df)*100:.2f}%)")
    print(f"TCP 流量: {len(tcp_df):,} 筆 ({len(tcp_df)/len(df)*100:.2f}%)")
    
    # 計算基本統計
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(udp_df) > 0:
        udp_stats = udp_df[numeric_cols].describe()
        results['udp'] = udp_stats
        print("\n【UDP 流量統計】")
        print(udp_stats)
    
    if len(tcp_df) > 0:
        tcp_stats = tcp_df[numeric_cols].describe()
        results['tcp'] = tcp_stats
        print("\n【TCP 流量統計】")
        print(tcp_stats)
    
    # 比較關鍵特徵
    key_features = ['TotBytes', 'TotPkts', 'Dur', 'SrcBytes', 'DstBytes']
    available_features = [f for f in key_features if f in numeric_cols]
    
    if show_plots and len(available_features) > 0:
        n_features = len(available_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, feature in enumerate(available_features):
            ax = axes[idx]
            
            # 繪製分佈比較
            if len(udp_df) > 0:
                udp_values = udp_df[feature].dropna()
                if len(udp_values) > 0:
                    ax.hist(udp_values, bins=50, alpha=0.5, label='UDP', 
                           color='blue', edgecolor='black')
            
            if len(tcp_df) > 0:
                tcp_values = tcp_df[feature].dropna()
                if len(tcp_values) > 0:
                    ax.hist(tcp_values, bins=50, alpha=0.5, label='TCP', 
                           color='red', edgecolor='black')
            
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{feature} Distribution Comparison (UDP vs TCP)', fontsize=11, fontweight='bold')
            ax.set_yscale('log')
            if feature in ['TotBytes', 'TotPkts', 'SrcBytes', 'DstBytes']:
                ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隱藏多餘的子圖
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return results


def extract_features_with_analysis(
    df: pd.DataFrame,
    analyze_long_tail: bool = True,
    analyze_protocol: bool = True,
    show_plots: bool = True,
    time_feature_stage: int = 1,
    quiet: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    提取特徵並進行分析。
    
    這是一個便利函數，結合了特徵提取、長尾分佈分析和協議切分分析。
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'TotBytes': [1000, 2000, 3000],
    ...     'SrcBytes': [600, 1200, 1800],
    ...     'TotPkts': [10, 20, 30],
    ...     'Dur': [1.0, 2.0, 3.0],
    ...     'Proto': ['tcp', 'udp', 'tcp']
    ... })
    >>> features_df, analysis = extract_features_with_analysis(df, show_plots=False)
    >>> 'flow_ratio' in features_df.columns
    True
    
    Args:
        df: 輸入的 NetFlow DataFrame
        analyze_long_tail: 是否進行長尾分佈分析
        analyze_protocol: 是否進行協議切分分析
        show_plots: 是否顯示圖表
        time_feature_stage: 時間特徵階段（1-4），預設為 1
        quiet: 是否隱藏 PySpark 相關的冗長輸出（預設 True）
    
    Returns:
        (特徵 DataFrame, 分析結果字典)
    """
    # 提取特徵（使用過濾器隱藏 PySpark 冗長訊息）
    if quiet:
        features_df = _extract_features_quietly(df, time_feature_stage=time_feature_stage)
    else:
        features_df = _extract_features_original(df, time_feature_stage=time_feature_stage)
    
    analysis_results = {}
    
    # 長尾分佈分析
    if analyze_long_tail:
        print("\n" + "=" * 60)
        print("【長尾分佈分析】")
        print("=" * 60)
        long_tail_results = analyze_long_tail_distribution(
            features_df, 
            show_plots=show_plots
        )
        analysis_results['long_tail'] = long_tail_results
        
        # 顯示摘要
        print("\n【長尾分佈摘要】")
        for feature, stats_dict in long_tail_results.items():
            if stats_dict['is_long_tail']:
                print(f"  {feature}: 偏度={stats_dict['skewness']:.2f}, "
                      f"峰度={stats_dict['kurtosis']:.2f} (長尾分佈)")
    
    # 協議切分分析
    if analyze_protocol and 'Proto' in features_df.columns:
        protocol_results = analyze_protocol_split(
            features_df,
            show_plots=show_plots
        )
        analysis_results['protocol'] = protocol_results
    
    return features_df, analysis_results


def extract_features(
    df: pd.DataFrame,
    flow_type: str = 'auto',
    include_time_features: bool = True,
    time_feature_stage: int = 1,
    include_bidirectional_features: bool = False,
    quiet: bool = True
) -> pd.DataFrame:
    """
    提取特徵（包裝版本，自動過濾 PySpark 冗長輸出）。
    
    這是 `src.feature_engineer.extract_features()` 的包裝函數，
    用於在 notebook 中自動過濾 PySpark 相關的冗長訊息。
    
    Args:
        df: 輸入的 NetFlow DataFrame
        flow_type: 流類型 ('auto', 'bidirectional')
        include_time_features: 是否包含時間特徵
        time_feature_stage: 時間特徵階段（1-4）
        include_bidirectional_features: 是否包含雙向流特徵
        quiet: 是否隱藏 PySpark 相關的冗長輸出（預設 True）
    
    Returns:
        包含特徵的 DataFrame
    """
    if quiet:
        # 使用安靜模式提取特徵
        return _extract_features_quietly(
            df,
            flow_type=flow_type,
            include_time_features=include_time_features,
            time_feature_stage=time_feature_stage,
            include_bidirectional_features=include_bidirectional_features
        )
    else:
        # 直接調用原始函數（不過濾輸出）
        return _extract_features_original(
            df,
            flow_type=flow_type,
            include_time_features=include_time_features,
            time_feature_stage=time_feature_stage,
            include_bidirectional_features=include_bidirectional_features
        )


def visualize_time_window_features(
    df: pd.DataFrame,
    feature_name: str = 'flows_per_minute_by_src',
    figsize: Tuple[int, int] = (12, 4),
    use_log_scale: bool = True
) -> None:
    """
    視覺化時間窗口聚合特徵的分佈。
    
    用於階段3特徵（按源IP的每分鐘聚合特徵）的視覺化。
    顯示直方圖和箱線圖，使用英文標題。
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'flows_per_minute_by_src': [1, 2, 3, 10, 100, 1000]
    ... })
    >>> visualize_time_window_features(df, 'flows_per_minute_by_src')
    
    Args:
        df: 包含特徵的 DataFrame
        feature_name: 要視覺化的特徵名稱
        figsize: 圖表大小
        use_log_scale: 是否使用對數尺度（預設 True，適合長尾分佈）
    """
    if feature_name not in df.columns:
        print(f"⚠️  Feature {feature_name} not found in DataFrame")
        return
    
    values = df[feature_name].dropna()
    if len(values) == 0:
        print(f"⚠️  Feature {feature_name} has no valid values")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Histogram
    axes[0].hist(values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel(feature_name, fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].set_title(f'Distribution of {feature_name}', fontsize=11, fontweight='bold')
    if use_log_scale:
        axes[0].set_yscale('log')
        axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Box plot
    bp = axes[1].boxplot([values], tick_labels=[feature_name], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[1].set_ylabel(feature_name, fontsize=10)
    axes[1].set_title(f'Box Plot of {feature_name}', fontsize=11, fontweight='bold')
    if use_log_scale:
        axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def visualize_bidirectional_symmetry(
    df: pd.DataFrame,
    feature_name: str = 'bidirectional_symmetry',
    figsize: Tuple[int, int] = (12, 4),
    symmetric_threshold: float = 0.9,
    asymmetric_threshold: float = 0.1
) -> None:
    """
    視覺化雙向流對稱性特徵的分佈。
    
    用於階段4特徵（雙向流 Pair 聚合特徵）的視覺化。
    顯示直方圖和箱線圖，並標記對稱性閾值，使用英文標題。
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'bidirectional_symmetry': [0.1, 0.5, 0.9, 0.95, 1.0]
    ... })
    >>> visualize_bidirectional_symmetry(df, 'bidirectional_symmetry')
    
    Args:
        df: 包含特徵的 DataFrame
        feature_name: 要視覺化的特徵名稱（預設為 'bidirectional_symmetry'）
        figsize: 圖表大小
        symmetric_threshold: 完全對稱的閾值（預設 0.9）
        asymmetric_threshold: 不對稱的閾值（預設 0.1）
    """
    if feature_name not in df.columns:
        print(f"⚠️  Feature {feature_name} not found in DataFrame")
        return
    
    values = df[feature_name].dropna()
    if len(values) == 0:
        print(f"⚠️  Feature {feature_name} has no valid values")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Histogram with symmetry thresholds
    axes[0].hist(values, bins=50, alpha=0.7, color='coral', edgecolor='black')
    axes[0].set_xlabel(feature_name, fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].set_title(f'Distribution of {feature_name}', fontsize=11, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=symmetric_threshold, color='green', linestyle='--', 
                    label=f'Fully Symmetric (>{symmetric_threshold})')
    axes[0].axvline(x=asymmetric_threshold, color='red', linestyle='--', 
                    label=f'Asymmetric (<{asymmetric_threshold})')
    axes[0].legend()
    
    # Right plot: Box plot
    bp = axes[1].boxplot([values], tick_labels=[feature_name], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    axes[1].set_ylabel(feature_name, fontsize=10)
    axes[1].set_title(f'Box Plot of {feature_name}', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def visualize_protocol_comparison(
    df: pd.DataFrame,
    feature_name: str,
    protocol_col: str = 'Proto',
    figsize: Tuple[int, int] = (6, 4)
) -> None:
    """
    視覺化 UDP 和 TCP 流量的特徵分佈比較。
    
    用於協議切分分析的視覺化，使用英文標題。
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Proto': ['udp', 'tcp', 'udp', 'tcp'],
    ...     'TotBytes': [100, 200, 150, 300]
    ... })
    >>> visualize_protocol_comparison(df, 'TotBytes')
    
    Args:
        df: 包含特徵和協議欄位的 DataFrame
        feature_name: 要比較的特徵名稱
        protocol_col: 協議欄位名稱（預設為 'Proto'）
        figsize: 圖表大小
    """
    if feature_name not in df.columns:
        print(f"⚠️  Feature {feature_name} not found in DataFrame")
        return
    
    if protocol_col not in df.columns:
        print(f"⚠️  Protocol column {protocol_col} not found in DataFrame")
        return
    
    # 分離 UDP 和 TCP
    udp_df = df[df[protocol_col].str.lower() == 'udp']
    tcp_df = df[df[protocol_col].str.lower() == 'tcp']
    
    udp_values = udp_df[feature_name].dropna()
    tcp_values = tcp_df[feature_name].dropna()
    
    if len(udp_values) == 0 and len(tcp_values) == 0:
        print(f"⚠️  No valid values for {feature_name}")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Box plot comparison
    box_data = []
    labels = []
    
    if len(udp_values) > 0:
        box_data.append(udp_values)
        labels.append('UDP')
    
    if len(tcp_values) > 0:
        box_data.append(tcp_values)
        labels.append('TCP')
    
    if len(box_data) > 0:
        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
        if len(bp['boxes']) >= 1:
            bp['boxes'][0].set_facecolor('lightblue')
        if len(bp['boxes']) >= 2:
            bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_ylabel(feature_name, fontsize=10)
        ax.set_title(f'{feature_name} Distribution Comparison', fontsize=11, fontweight='bold')
        if feature_name in ['TotBytes', 'TotPkts', 'SrcBytes', 'DstBytes']:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
