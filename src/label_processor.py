"""
標籤處理模組

負責將原始標籤轉換為模型可用的格式。
遵循單一職責原則，與特徵工程和特徵轉換分離。

使用範例：
    >>> import pandas as pd
    >>> from src.label_processor import convert_label_to_binary
    >>> 
    >>> df = pd.DataFrame({
    ...     'Label': ['From-Botnet-V50-1', 'Background-google', 'Normal', 'From-Botnet-V50-2']
    ... })
    >>> result = convert_label_to_binary(df)
    >>> result['label_binary'].tolist()
    [1, 0, 0, 1]
    >>> (result['label_binary'] == 1).sum()
    2
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple


def convert_label_to_binary(
    df: pd.DataFrame,
    label_column: str = 'Label',
    anomaly_keywords: Optional[List[str]] = None,
    output_column: str = 'label_binary',
    verbose: bool = False
) -> pd.DataFrame:
    """
    將標籤欄位轉換為二元標籤（0=正常, 1=異常）。
    
    支援多種標籤格式：
    - 字串標籤：檢查是否包含指定的異常關鍵字（如 'Botnet'）
    - 數值標籤：1 表示異常，0 表示正常
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Label': ['From-Botnet-V50-1', 'Background-google', 'Normal', 'From-Botnet-V50-2']
    ... })
    >>> result = convert_label_to_binary(df)
    >>> result['label_binary'].tolist()
    [1, 0, 0, 1]
    >>> (result['label_binary'] == 1).sum()
    2
    >>> 
    >>> # 測試數值標籤
    >>> df_numeric = pd.DataFrame({'Label': [0, 1, 0, 1, 0]})
    >>> result_numeric = convert_label_to_binary(df_numeric)
    >>> result_numeric['label_binary'].tolist()
    [0, 1, 0, 1, 0]
    >>> 
    >>> # 測試自訂關鍵字
    >>> df_custom = pd.DataFrame({
    ...     'Label': ['Malware-A', 'Normal-B', 'Attack-C', 'Safe-D']
    ... })
    >>> result_custom = convert_label_to_binary(
    ...     df_custom, 
    ...     anomaly_keywords=['Malware', 'Attack']
    ... )
    >>> result_custom['label_binary'].tolist()
    [1, 0, 1, 0]
    
    Args:
        df: 輸入的 DataFrame，必須包含 label_column 欄位
        label_column: 標籤欄位名稱，預設為 'Label'
        anomaly_keywords: 異常標籤關鍵字列表，預設為 ['Botnet', 'botnet', 'bot']
                        如果標籤包含這些關鍵字，則標記為異常 (1)
        output_column: 輸出欄位名稱，預設為 'label_binary'
        verbose: 是否顯示詳細資訊，預設為 False
    
    Returns:
        包含二元標籤的 DataFrame（新增 output_column 欄位）
    
    Raises:
        ValueError: 如果 label_column 不存在於 DataFrame 中
    """
    if label_column not in df.columns:
        raise ValueError(f"標籤欄位 '{label_column}' 不存在於 DataFrame 中")
    
    result_df = df.copy()
    
    # 預設異常關鍵字
    if anomaly_keywords is None:
        anomaly_keywords = ['Botnet', 'botnet', 'bot']
    
    # 獲取標籤欄位
    labels = result_df[label_column]
    
    # 根據標籤類型進行轉換
    if labels.dtype == 'object':
        # 字串標籤：檢查是否包含異常關鍵字
        # 使用 case=False 進行不區分大小寫的匹配
        # 使用 '|'.join() 來匹配任一關鍵字
        pattern = '|'.join(anomaly_keywords)
        is_anomaly = labels.str.contains(pattern, case=False, na=False)
        result_df[output_column] = is_anomaly.astype(int)
    else:
        # 數值標籤：1 表示異常，0 表示正常
        result_df[output_column] = (labels == 1).astype(int)
    
    # 顯示統計資訊（如果啟用）
    if verbose:
        normal_count = (result_df[output_column] == 0).sum()
        anomaly_count = (result_df[output_column] == 1).sum()
        total_count = len(result_df)
        
        print(f"\n[標籤轉換統計]")
        print(f"  原始標籤分布:")
        label_counts = result_df[label_column].value_counts()
        for label, count in label_counts.head(10).items():
            print(f"    {label}: {count:,} ({count/total_count*100:.2f}%)")
        print(f"  轉換後標籤分布:")
        print(f"    正常 (0): {normal_count:,} ({normal_count/total_count*100:.2f}%)")
        print(f"    異常 (1): {anomaly_count:,} ({anomaly_count/total_count*100:.2f}%)")
        
        if anomaly_count == 0:
            print(f"  ⚠️  警告：轉換後沒有異常樣本！")
            print(f"     請檢查標籤轉換邏輯或異常關鍵字設定")
        else:
            print(f"  ✅ 標籤轉換完成，異常樣本: {anomaly_count:,} 筆")
    
    return result_df


def get_label_statistics(
    df: pd.DataFrame,
    label_column: str = 'Label',
    binary_column: Optional[str] = 'label_binary'
) -> Dict[str, any]:
    """
    獲取標籤統計資訊。
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Label': ['From-Botnet-V50-1', 'Background-google', 'Normal', 'From-Botnet-V50-2'],
    ...     'label_binary': [1, 0, 0, 1]
    ... })
    >>> stats = get_label_statistics(df)
    >>> stats['total_count']
    4
    >>> stats['normal_count']
    2
    >>> stats['anomaly_count']
    2
    >>> stats['anomaly_ratio']
    0.5
    
    Args:
        df: 輸入的 DataFrame
        label_column: 原始標籤欄位名稱，預設為 'Label'
        binary_column: 二元標籤欄位名稱，預設為 'label_binary'
                        如果為 None，則只統計原始標籤
    
    Returns:
        包含標籤分布統計的字典，包含：
        - total_count: 總樣本數
        - normal_count: 正常樣本數（如果 binary_column 存在）
        - anomaly_count: 異常樣本數（如果 binary_column 存在）
        - anomaly_ratio: 異常比例（如果 binary_column 存在）
        - label_distribution: 原始標籤分布字典
    """
    stats = {
        'total_count': len(df),
        'label_distribution': {}
    }
    
    # 統計原始標籤分布
    if label_column in df.columns:
        label_counts = df[label_column].value_counts()
        stats['label_distribution'] = label_counts.to_dict()
    
    # 統計二元標籤分布（如果存在）
    if binary_column and binary_column in df.columns:
        normal_count = (df[binary_column] == 0).sum()
        anomaly_count = (df[binary_column] == 1).sum()
        total_count = len(df)
        
        stats['normal_count'] = int(normal_count)
        stats['anomaly_count'] = int(anomaly_count)
        stats['anomaly_ratio'] = float(anomaly_count / total_count) if total_count > 0 else 0.0
        stats['normal_ratio'] = float(normal_count / total_count) if total_count > 0 else 0.0
    
    return stats


def validate_labels(
    df: pd.DataFrame,
    label_column: str = 'Label',
    binary_column: Optional[str] = 'label_binary',
    min_anomaly_ratio: float = 0.0,
    max_anomaly_ratio: float = 1.0
) -> Tuple[bool, List[str]]:
    """
    驗證標籤是否符合預期要求。
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Label': ['From-Botnet-V50-1', 'Background-google', 'Normal', 'From-Botnet-V50-2'],
    ...     'label_binary': [1, 0, 0, 1]
    ... })
    >>> is_valid, messages = validate_labels(df, min_anomaly_ratio=0.1)
    >>> is_valid
    True
    >>> 
    >>> # 測試沒有異常樣本的情況
    >>> df_no_anomaly = pd.DataFrame({
    ...     'Label': ['Background-google', 'Normal'],
    ...     'label_binary': [0, 0]
    ... })
    >>> is_valid, messages = validate_labels(df_no_anomaly, min_anomaly_ratio=0.1)
    >>> is_valid
    False
    >>> '異常樣本比例過低' in messages[0]
    True
    
    Args:
        df: 輸入的 DataFrame
        label_column: 原始標籤欄位名稱，預設為 'Label'
        binary_column: 二元標籤欄位名稱，預設為 'label_binary'
        min_anomaly_ratio: 最小異常比例，預設為 0.0
        max_anomaly_ratio: 最大異常比例，預設為 1.0
    
    Returns:
        (is_valid, messages) 元組：
        - is_valid: 是否通過驗證
        - messages: 驗證訊息列表（如果有問題）
    """
    messages = []
    
    # 檢查標籤欄位是否存在
    if label_column not in df.columns:
        messages.append(f"缺少標籤欄位: {label_column}")
        return False, messages
    
    # 檢查二元標籤欄位（如果指定）
    if binary_column and binary_column not in df.columns:
        messages.append(f"缺少二元標籤欄位: {binary_column}")
        return False, messages
    
    # 如果提供了二元標籤，檢查異常比例
    if binary_column and binary_column in df.columns:
        anomaly_count = (df[binary_column] == 1).sum()
        total_count = len(df)
        anomaly_ratio = anomaly_count / total_count if total_count > 0 else 0.0
        
        if anomaly_ratio < min_anomaly_ratio:
            messages.append(
                f"異常樣本比例過低: {anomaly_ratio:.4f} < {min_anomaly_ratio:.4f} "
                f"({anomaly_count}/{total_count})"
            )
        
        if anomaly_ratio > max_anomaly_ratio:
            messages.append(
                f"異常樣本比例過高: {anomaly_ratio:.4f} > {max_anomaly_ratio:.4f} "
                f"({anomaly_count}/{total_count})"
            )
        
        if anomaly_count == 0:
            messages.append("沒有異常樣本，無法進行監督學習或分層抽樣")
    
    is_valid = len(messages) == 0
    return is_valid, messages


if __name__ == '__main__':
    # 簡單測試
    import doctest
    doctest.testmod(verbose=True)

