"""
特徵轉換模組

針對長尾分佈特徵進行 Log-Transformation + RobustScaler 轉換。
用於優化 Unsupervised Model（如 Isolation Forest）的特徵分佈。

解決問題：
- Network Flow 數據具有 Power-law 分佈特性（少數連線產生極大流量）
- 關鍵特徵（Dur, TotBytes, SrcBytes 等）呈現嚴重的長尾分佈（Long-tail distribution）
- StandardScaler 對極端值無效：即便標準化後，極端值仍然把主體壓縮得看不見
- 這會影響 Unsupervised Model（如 Isolation Forest, K-Means）的距離計算

解決方案：
- Log1p 轉換：壓縮極端值的尺度，使分佈更接近正態分佈
- RobustScaler：使用四分位距（IQR）進行標準化，對極端值更穩健
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import RobustScaler, PowerTransformer
from scipy import stats


# 預設的長尾分佈特徵列表（基於 EDA 分析結果）
DEFAULT_SKEWED_FEATURES = [
    'Dur', 'TotBytes', 'TotPkts', 'SrcBytes', 'DstBytes',
    'flow_ratio', 'bytes_symmetry', 'bytes_per_second',
    'packets_per_second', 'time_since_last_flow', 
    'time_to_next_flow', 'flows_per_minute_by_src',
    'total_bytes_per_minute_by_src'
]


def apply_log_transformation(
    df: pd.DataFrame, 
    columns: List[str],
    prefix: str = 'log_'
) -> pd.DataFrame:
    """
    針對長尾分佈特徵進行 Log1p 轉換 (log(x+1))，避免 log(0) 錯誤並壓縮尺度。
    
    這個函數會為每個指定的欄位創建一個新的 log_ 前綴欄位（預設），
    或者直接替換原欄位（如果 prefix=''）。
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'value': [0, 100, 1000, 10000],
    ...     'other': [1, 2, 3, 4]
    ... })
    >>> result = apply_log_transformation(df, ['value'])
    >>> 'log_value' in result.columns
    True
    >>> result['log_value'].iloc[0] == 0.0  # log1p(0) = 0
    True
    >>> result['log_value'].iloc[3] > result['log_value'].iloc[2]  # 遞增
    True
    >>> result['other'].equals(df['other'])  # 其他欄位不變
    True
    
    Args:
        df: 輸入的 pandas DataFrame
        columns: 需要進行對數轉換的欄位名稱列表
        prefix: 新欄位的前綴，預設為 'log_'。如果設為空字串 ''，則直接替換原欄位
    
    Returns:
        包含 log_ 前綴新欄位（或替換原欄位）的 DataFrame
    """
    result_df = df.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
        
        # 確保值為非負數（log1p 要求 x >= -1，但我們確保 x >= 0）
        col_data = result_df[col]
        min_val = col_data.min()
        
        if pd.isna(min_val):
            # 如果欄位全為 NaN，創建一個全為 0 的 log 欄位
            new_col_name = f"{prefix}{col}" if prefix else col
            result_df[new_col_name] = 0.0
        elif min_val < 0:
            # 如果有負值，先平移使其為非負
            shifted_data = col_data - min_val
            new_col_name = f"{prefix}{col}" if prefix else col
            result_df[new_col_name] = np.log1p(shifted_data)
        else:
            # 直接使用 log1p
            new_col_name = f"{prefix}{col}" if prefix else col
            result_df[new_col_name] = np.log1p(col_data)
    
    return result_df


def apply_robust_scaling(
    df: pd.DataFrame,
    columns: List[str],
    with_centering: bool = True,
    with_scaling: bool = True
) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    使用 RobustScaler 對特徵進行標準化。
    
    RobustScaler 使用中位數和 IQR（四分位距），對極端值更穩健。
    公式：(x - median) / IQR
    
    與 StandardScaler 的差異：
    - StandardScaler: (x - mean) / std，對極端值敏感
    - RobustScaler: (x - median) / IQR，對極端值穩健
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],  # 有極端值
    ...     'feature2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    ... })
    >>> scaled_df, scaler = apply_robust_scaling(df, ['feature1', 'feature2'])
    >>> scaled_df.shape == df.shape
    True
    >>> isinstance(scaler, RobustScaler)
    True
    >>> # 檢查標準化後的資料範圍（應該更穩定）
    >>> scaled_df['feature1'].abs().max() < 10  # 極端值被壓縮
    True
    
    Args:
        df: 輸入的 pandas DataFrame
        columns: 需要標準化的欄位名稱列表
        with_centering: 是否使用中位數中心化，預設為 True
        with_scaling: 是否使用 IQR 縮放，預設為 True
    
    Returns:
        (轉換後的 DataFrame, 訓練好的 RobustScaler 模型)
    
    Raises:
        ValueError: 如果指定的欄位不存在於 DataFrame 中
    """
    # 檢查欄位是否存在
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下欄位不存在於 DataFrame 中: {missing_cols}")
    
    # 複製 DataFrame 避免修改原始資料
    result_df = df.copy()
    
    # 提取需要標準化的欄位
    X = result_df[columns].values
    
    # 創建並訓練 RobustScaler
    robust_scaler = RobustScaler(
        with_centering=with_centering,
        with_scaling=with_scaling
    )
    X_scaled = robust_scaler.fit_transform(X)
    
    # 將標準化後的結果放回 DataFrame
    for i, col in enumerate(columns):
        result_df[col] = X_scaled[:, i]
    
    return result_df, robust_scaler


def transform_features_for_unsupervised(
    df: pd.DataFrame,
    skewed_features: Optional[List[str]] = None,
    feature_columns: Optional[List[str]] = None,
    log_prefix: str = 'log_',
    replace_original: bool = False
) -> Tuple[pd.DataFrame, RobustScaler, List[str]]:
    """
    完整的特徵轉換流程：Log-Transformation + RobustScaler
    
    這是針對 Unsupervised Model（如 Isolation Forest）的優化流程：
    1. 對長尾分佈特徵進行 log1p 轉換
    2. 使用 RobustScaler 進行標準化（對極端值更穩健）
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'TotBytes': [100, 1000, 10000, 100000],
    ...     'SrcBytes': [50, 500, 5000, 50000],
    ...     'Dur': [1.0, 2.0, 3.0, 4.0],
    ...     'hour': [9, 10, 11, 12]  # 不需要轉換的特徵
    ... })
    >>> result_df, scaler, transformed_cols = transform_features_for_unsupervised(
    ...     df, 
    ...     skewed_features=['TotBytes', 'SrcBytes', 'Dur']
    ... )
    >>> 'log_TotBytes' in result_df.columns
    True
    >>> 'log_SrcBytes' in result_df.columns
    True
    >>> 'hour' in result_df.columns  # 其他特徵保留
    True
    >>> isinstance(scaler, RobustScaler)
    True
    >>> len(transformed_cols) > 0
    True
    
    Args:
        df: 輸入的 pandas DataFrame
        skewed_features: 需要進行對數轉換的長尾分佈特徵列表
                        如果為 None，則使用 DEFAULT_SKEWED_FEATURES
        feature_columns: 最終用於模型訓練的特徵欄位列表
                        如果為 None，則自動選擇：
                        - 所有 log_ 前綴欄位（如果 replace_original=False）
                        - 其他非 skewed 的數值欄位
        log_prefix: 對數轉換後新欄位的前綴，預設為 'log_'
        replace_original: 如果為 True，則直接替換原欄位（log_prefix 會被忽略）
                        如果為 False，則創建新欄位（使用 log_prefix）
    
    Returns:
        (轉換後的 DataFrame, 訓練好的 RobustScaler 模型, 被標準化的特徵欄位列表)
    """
    # 複製 DataFrame 避免修改原始資料
    result_df = df.copy()
    
    # 步驟 1: 決定需要對數轉換的特徵
    if skewed_features is None:
        skewed_features = DEFAULT_SKEWED_FEATURES.copy()
    
    # 只對存在的特徵進行轉換
    available_skewed_features = [col for col in skewed_features if col in result_df.columns]
    
    if available_skewed_features:
        # 應用對數轉換
        if replace_original:
            result_df = apply_log_transformation(result_df, available_skewed_features, prefix='')
        else:
            result_df = apply_log_transformation(result_df, available_skewed_features, prefix=log_prefix)
    
    # 步驟 2: 決定最終使用的特徵欄位
    if feature_columns is None:
        # 自動選擇：所有 log_ 前綴欄位 + 其他非 skewed 的數值欄位
        if replace_original:
            log_columns = available_skewed_features
        else:
            log_columns = [f"{log_prefix}{col}" for col in available_skewed_features 
                          if f"{log_prefix}{col}" in result_df.columns]
        
        # 獲取所有數值欄位
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除原始 skewed_features 和 log_ 欄位，保留其他數值欄位
        other_numeric_cols = [col for col in numeric_cols 
                             if col not in available_skewed_features 
                             and not col.startswith(log_prefix)]
        
        feature_columns = log_columns + other_numeric_cols
    else:
        # 如果提供了 feature_columns，確保 log_ 前綴的欄位存在
        if not replace_original:
            feature_columns = [
                f"{log_prefix}{col}" if col in available_skewed_features 
                and f"{log_prefix}{col}" in result_df.columns 
                else col 
                for col in feature_columns
            ]
        # 只保留存在的欄位
        feature_columns = [col for col in feature_columns if col in result_df.columns]
    
    if not feature_columns:
        raise ValueError("沒有可用的特徵欄位進行標準化")
    
    # 步驟 3: 使用 RobustScaler 進行標準化
    result_df, robust_scaler = apply_robust_scaling(
        result_df,
        columns=feature_columns,
        with_centering=True,
        with_scaling=True
    )
    
    return result_df, robust_scaler, feature_columns


def get_transformed_feature_names(
    original_features: List[str],
    skewed_features: Optional[List[str]] = None,
    log_prefix: str = 'log_',
    replace_original: bool = False
) -> List[str]:
    """
    獲取轉換後的特徵名稱列表（不實際執行轉換）。
    
    用於預先了解哪些特徵會被轉換，方便後續處理。
    
    >>> original = ['TotBytes', 'SrcBytes', 'Dur', 'hour']
    >>> skewed = ['TotBytes', 'SrcBytes', 'Dur']
    >>> transformed = get_transformed_feature_names(original, skewed)
    >>> 'log_TotBytes' in transformed
    True
    >>> 'log_SrcBytes' in transformed
    True
    >>> 'hour' in transformed  # 非 skewed 特徵保留
    True
    
    Args:
        original_features: 原始特徵名稱列表
        skewed_features: 需要對數轉換的特徵列表，如果為 None 則使用 DEFAULT_SKEWED_FEATURES
        log_prefix: 對數轉換後新欄位的前綴，預設為 'log_'
        replace_original: 如果為 True，則直接替換原欄位名稱
    
    Returns:
        轉換後的特徵名稱列表
    """
    if skewed_features is None:
        skewed_features = DEFAULT_SKEWED_FEATURES.copy()
    
    result = []
    for col in original_features:
        if col in skewed_features:
            if replace_original:
                result.append(col)
            else:
                result.append(f"{log_prefix}{col}")
        else:
            result.append(col)
    
    return result


def apply_sqrt_transformation(
    df: pd.DataFrame,
    columns: List[str],
    prefix: str = 'sqrt_'
) -> pd.DataFrame:
    """
    平方根轉換：比對數轉換溫和，適用於中等偏斜的分佈。
    
    >>> df = pd.DataFrame({'value': [0, 1, 4, 9, 16]})
    >>> result = apply_sqrt_transformation(df, ['value'])
    >>> 'sqrt_value' in result.columns
    True
    >>> np.allclose(result['sqrt_value'], [0, 1, 2, 3, 4])
    True
    
    Args:
        df: 輸入的 pandas DataFrame
        columns: 需要轉換的欄位名稱列表
        prefix: 新欄位的前綴，預設為 'sqrt_'
    
    Returns:
        包含轉換後欄位的 DataFrame
    """
    result_df = df.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
        
        col_data = result_df[col]
        min_val = col_data.min()
        
        if pd.isna(min_val):
            new_col_name = f"{prefix}{col}" if prefix else col
            result_df[new_col_name] = 0.0
        elif min_val < 0:
            # 如果有負值，先平移
            shifted_data = col_data - min_val
            new_col_name = f"{prefix}{col}" if prefix else col
            result_df[new_col_name] = np.sqrt(shifted_data)
        else:
            new_col_name = f"{prefix}{col}" if prefix else col
            result_df[new_col_name] = np.sqrt(col_data)
    
    return result_df


def apply_boxcox_transformation(
    df: pd.DataFrame,
    columns: List[str],
    prefix: str = 'boxcox_',
    method: str = 'box-cox'  # 'box-cox' 或 'yeo-johnson'
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Box-Cox 或 Yeo-Johnson 轉換：自動尋找最佳 lambda 參數。
    
    >>> df = pd.DataFrame({'value': [1, 10, 100, 1000, 10000]})
    >>> result, lambdas = apply_boxcox_transformation(df, ['value'], method='box-cox')
    >>> 'boxcox_value' in result.columns
    True
    >>> 'value' in lambdas
    True
    
    Args:
        df: 輸入的 pandas DataFrame
        columns: 需要轉換的欄位名稱列表
        prefix: 新欄位的前綴，預設為 'boxcox_'
        method: 'box-cox'（需要 x > 0）或 'yeo-johnson'（允許負值）
    
    Returns:
        (轉換後的 DataFrame, lambda 參數字典)
    """
    result_df = df.copy()
    lambdas = {}
    
    for col in columns:
        if col not in result_df.columns:
            continue
        
        col_data = result_df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        try:
            if method == 'box-cox':
                # Box-Cox 需要正數
                if col_data.min() <= 0:
                    # 平移使其為正數
                    shift = -col_data.min() + 1
                    shifted_data = col_data + shift
                else:
                    shifted_data = col_data
                    shift = 0
                
                # 尋找最佳 lambda
                transformed_data, lambda_param = stats.boxcox(shifted_data)
                lambdas[col] = lambda_param
                
                # 應用轉換到整個欄位
                new_col_name = f"{prefix}{col}" if prefix else col
                if shift > 0:
                    full_shifted = result_df[col] + shift
                    full_shifted = full_shifted.clip(lower=1e-10)
                    # 只對非 NaN 值進行轉換
                    valid_mask = ~result_df[col].isna()
                    new_col = result_df[col].copy()
                    if valid_mask.sum() > 0:
                        transformed_full, _ = stats.boxcox(full_shifted[valid_mask])
                        new_col.loc[valid_mask] = transformed_full
                    result_df[new_col_name] = new_col
                else:
                    valid_mask = ~result_df[col].isna()
                    new_col = result_df[col].copy()
                    if valid_mask.sum() > 0:
                        transformed_full, _ = stats.boxcox(result_df.loc[valid_mask, col])
                        new_col.loc[valid_mask] = transformed_full
                    result_df[new_col_name] = new_col
                
            else:  # yeo-johnson
                # 使用 sklearn 的 PowerTransformer
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                col_2d = col_data.values.reshape(-1, 1)
                pt.fit(col_2d)
                lambda_param = pt.lambdas_[0]
                lambdas[col] = lambda_param
                
                # 應用到整個欄位
                new_col_name = f"{prefix}{col}" if prefix else col
                full_col_2d = result_df[[col]].values
                transformed_full = pt.transform(full_col_2d).flatten()
                result_df[new_col_name] = transformed_full
            
        except Exception as e:
            print(f"   ⚠️  {col} 的 {method} 轉換失敗：{e}")
            continue
    
    return result_df, lambdas


def calculate_transformation_metrics(
    values: pd.Series,
    y_labels: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    計算轉換後的評估指標。
    
    Args:
        values: 轉換後的特徵值
        y_labels: 標籤（0=正常，1=異常），如果提供則計算分離度
    
    Returns:
        包含各項指標的字典
    """
    clean_values = values.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_values) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'cohens_d': 0.0,
            'is_normal_like': False
        }
    
    mean_val = clean_values.mean()
    std_val = clean_values.std()
    skewness = stats.skew(clean_values)
    kurtosis = stats.kurtosis(clean_values, fisher=False)  # fisher=False: 正態分佈的峰度為3
    
    # 判斷是否接近正態分佈
    is_normal_like = (abs(skewness) < 0.5 and abs(kurtosis - 3) < 1.0)
    
    # 計算 Cohen's d（如果有標籤）
    cohens_d = 0.0
    if y_labels is not None and len(y_labels) == len(values):
        normal_values = clean_values[y_labels == 0] if len(y_labels) == len(clean_values) else pd.Series()
        anomaly_values = clean_values[y_labels == 1] if len(y_labels) == len(clean_values) else pd.Series()
        
        if len(normal_values) > 0 and len(anomaly_values) > 0:
            normal_mean = normal_values.mean()
            anomaly_mean = anomaly_values.mean()
            normal_std = normal_values.std()
            anomaly_std = anomaly_values.std()
            
            pooled_std = np.sqrt((normal_std**2 + anomaly_std**2) / 2)
            if pooled_std > 0:
                cohens_d = abs(normal_mean - anomaly_mean) / pooled_std
    
    return {
        'mean': mean_val,
        'std': std_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'cohens_d': cohens_d,
        'is_normal_like': is_normal_like
    }


if __name__ == '__main__':
    # 簡單測試
    import doctest
    doctest.testmod(verbose=True)

