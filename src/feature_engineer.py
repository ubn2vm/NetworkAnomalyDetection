"""
統一的特徵工程模組

從 NetFlow 資料中提取特徵，支援雙向流資料格式。
根據 EDA 分析結果，提取關鍵特徵用於異常檢測。

時間特徵支援：
- 階段1：基本時間特徵（hour, day_of_week, is_weekend等）和週期性編碼
- 階段2：時間間隔特徵（time_since_last_flow等）
- 階段3：時間窗口聚合特徵（flows_per_minute_by_src等）
- 階段4：雙向流 Pair 聚合特徵（使用 Spark Window Function）
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from pathlib import Path


def _extract_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取基本時間特徵（階段1）。
    
    包括：
    - 基本時間特徵：hour, day_of_week, day_of_month, is_weekend, is_work_hour, is_night
    - 週期性編碼：sin_hour, cos_hour, sin_day_of_week, cos_day_of_week
    
    Args:
        df: 輸入的 DataFrame，必須包含 'StartTime' 欄位（datetime 類型）
    
    Returns:
        包含時間特徵的 DataFrame
    """
    features_df = df.copy()
    
    if 'StartTime' not in features_df.columns:
        return features_df
    
    # 確保 StartTime 是 datetime 類型
    if not pd.api.types.is_datetime64_any_dtype(features_df['StartTime']):
        features_df['StartTime'] = pd.to_datetime(features_df['StartTime'], errors='coerce')
    
    # 基本時間特徵
    features_df['hour'] = features_df['StartTime'].dt.hour
    features_df['day_of_week'] = features_df['StartTime'].dt.dayofweek  # 0=Monday, 6=Sunday
    features_df['day_of_month'] = features_df['StartTime'].dt.day
    features_df['is_weekend'] = (features_df['StartTime'].dt.dayofweek >= 5).astype(int)
    features_df['is_work_hour'] = ((features_df['StartTime'].dt.hour >= 9) & 
                                   (features_df['StartTime'].dt.hour < 17)).astype(int)
    features_df['is_night'] = ((features_df['StartTime'].dt.hour >= 22) | 
                              (features_df['StartTime'].dt.hour < 6)).astype(int)
    
    # 週期性編碼（將時間轉換為週期性特徵，有助於模型理解時間的循環性）
    # 小時的週期性編碼（24小時循環）
    features_df['sin_hour'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['cos_hour'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    
    # 星期的週期性編碼（7天循環）
    features_df['sin_day_of_week'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
    features_df['cos_day_of_week'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
    
    # 日期的週期性編碼（假設30天循環，用於識別月初/月底模式）
    features_df['sin_day_of_month'] = np.sin(2 * np.pi * features_df['day_of_month'] / 30)
    features_df['cos_day_of_month'] = np.cos(2 * np.pi * features_df['day_of_month'] / 30)
    
    return features_df


def _extract_time_interval_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取時間間隔特徵（階段2）。
    
    包括：
    - time_since_last_flow: 距離上一次流的時間間隔（秒）
    - time_to_next_flow: 距離下一次流的時間間隔（秒）
    
    注意：需要按源IP排序，計算時間間隔。
    
    Args:
        df: 輸入的 DataFrame，必須包含 'StartTime' 和 'SrcAddr' 欄位
    
    Returns:
        包含時間間隔特徵的 DataFrame
    """
    features_df = df.copy()
    
    if 'StartTime' not in features_df.columns or 'SrcAddr' not in features_df.columns:
        return features_df
    
    # 確保 StartTime 是 datetime 類型
    if not pd.api.types.is_datetime64_any_dtype(features_df['StartTime']):
        features_df['StartTime'] = pd.to_datetime(features_df['StartTime'], errors='coerce')
    
    # 按源IP和時間排序
    features_df = features_df.sort_values(['SrcAddr', 'StartTime']).reset_index(drop=True)
    
    # 計算距離上一次流的時間間隔（按源IP分組）
    features_df['time_since_last_flow'] = features_df.groupby('SrcAddr')['StartTime'].diff().dt.total_seconds()
    features_df['time_since_last_flow'] = features_df['time_since_last_flow'].fillna(0)
    
    # 計算距離下一次流的時間間隔
    features_df['time_to_next_flow'] = features_df.groupby('SrcAddr')['StartTime'].diff(-1).dt.total_seconds().abs()
    features_df['time_to_next_flow'] = features_df['time_to_next_flow'].fillna(0)
    
    return features_df


def _extract_time_window_features(df: pd.DataFrame, window_size: str = '1min') -> pd.DataFrame:
    """
    提取時間窗口聚合特徵（階段3）。
    
    包括：
    - flows_per_minute_by_src: 該源IP在該分鐘內的流數量
    - unique_dst_per_minute_by_src: 該源IP在該分鐘內連接的不同目標IP數量
    - unique_dport_per_minute_by_src: 該源IP在該分鐘內連接的不同目標端口數量
    - total_bytes_per_minute_by_src: 該源IP在該分鐘內的總位元組數
    
    注意：計算成本較高，需要分組聚合。
    
    Args:
        df: 輸入的 DataFrame，必須包含 'StartTime', 'SrcAddr', 'DstAddr', 'Dport', 'TotBytes' 欄位
        window_size: 時間窗口大小，預設為 '1min'（1分鐘）
    
    Returns:
        包含時間窗口聚合特徵的 DataFrame
    """
    features_df = df.copy()
    
    required_cols = ['StartTime', 'SrcAddr']
    if not all(col in features_df.columns for col in required_cols):
        return features_df
    
    # 確保 StartTime 是 datetime 類型
    if not pd.api.types.is_datetime64_any_dtype(features_df['StartTime']):
        features_df['StartTime'] = pd.to_datetime(features_df['StartTime'], errors='coerce')
    
    # 建立時間窗口
    features_df['time_window'] = features_df['StartTime'].dt.floor(window_size)
    
    # 計算該源IP在該分鐘內的流數量
    flows_per_min = features_df.groupby(['SrcAddr', 'time_window']).size()
    flows_per_min = flows_per_min.reset_index(name='flows_per_minute_by_src')
    features_df = features_df.merge(flows_per_min, on=['SrcAddr', 'time_window'], how='left')
    
    # 計算該源IP在該分鐘內連接的不同目標IP數量
    if 'DstAddr' in features_df.columns:
        unique_dst_per_min = features_df.groupby(['SrcAddr', 'time_window'])['DstAddr'].nunique()
        unique_dst_per_min = unique_dst_per_min.reset_index(name='unique_dst_per_minute_by_src')
        features_df = features_df.merge(unique_dst_per_min, on=['SrcAddr', 'time_window'], how='left')
    else:
        features_df['unique_dst_per_minute_by_src'] = 0
    
    # 計算該源IP在該分鐘內連接的不同目標端口數量
    if 'Dport' in features_df.columns:
        unique_dport_per_min = features_df.groupby(['SrcAddr', 'time_window'])['Dport'].nunique()
        unique_dport_per_min = unique_dport_per_min.reset_index(name='unique_dport_per_minute_by_src')
        features_df = features_df.merge(unique_dport_per_min, on=['SrcAddr', 'time_window'], how='left')
    else:
        features_df['unique_dport_per_minute_by_src'] = 0
    
    # 計算該源IP在該分鐘內的總位元組數
    if 'TotBytes' in features_df.columns:
        total_bytes_per_min = features_df.groupby(['SrcAddr', 'time_window'])['TotBytes'].sum()
        total_bytes_per_min = total_bytes_per_min.reset_index(name='total_bytes_per_minute_by_src')
        features_df = features_df.merge(total_bytes_per_min, on=['SrcAddr', 'time_window'], how='left')
    else:
        features_df['total_bytes_per_minute_by_src'] = 0
    
    # 移除臨時欄位
    features_df.drop(columns=['time_window'], errors='ignore', inplace=True)
    
    return features_df


def _get_or_create_spark_session():
    """
    取得或創建 SparkSession（用於雙向流特徵工程）。
    
    Returns:
        SparkSession 實例
    
    Raises:
        ImportError: 如果 PySpark 未安裝
    """
    try:
        from pyspark.sql import SparkSession
        import os
        import sys
        
        # 設定 Python 路徑（Windows 上避免 Python worker 連線問題）
        python_exe = sys.executable
        os.environ['PYSPARK_PYTHON'] = python_exe
        os.environ['PYSPARK_DRIVER_PYTHON'] = python_exe
        
        # 嘗試取得現有的 SparkSession
        spark = SparkSession.getActiveSession()
        if spark is not None:
            return spark
        
        # 創建新的 SparkSession
        from src.data_loader import get_project_root
        project_root = get_project_root()
        spark_temp_dir = project_root / "spark_temp"
        spark_temp_dir.mkdir(parents=True, exist_ok=True)
        
        os.environ['SPARK_LOCAL_DIRS'] = str(spark_temp_dir)
        # Windows 上設定 HADOOP_HOME（避免 winutils.exe 錯誤）
        os.environ['HADOOP_HOME'] = str(spark_temp_dir)
        os.environ['hadoop.home.dir'] = str(spark_temp_dir)
        
        # 根據系統記憶體自動調整 Spark 記憶體配置
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # 建議配置：使用可用記憶體的 50-60%，但至少 4g，最多 12g
            if available_memory_gb >= 20:
                driver_memory = "12g"
                executor_memory = "12g"
                shuffle_partitions = 400
            elif available_memory_gb >= 16:
                driver_memory = "8g"
                executor_memory = "8g"
                shuffle_partitions = 300
            elif available_memory_gb >= 12:
                driver_memory = "6g"
                executor_memory = "6g"
                shuffle_partitions = 250
            else:
                driver_memory = "4g"
                executor_memory = "4g"
                shuffle_partitions = 200
            
            print(f"   💾 系統記憶體：{total_memory_gb:.1f} GB（可用：{available_memory_gb:.1f} GB）")
            print(f"   ⚙️  Spark 記憶體配置：Driver={driver_memory}, Executor={executor_memory}")
        except ImportError:
            # 如果 psutil 未安裝，使用預設值
            driver_memory = "8g"
            executor_memory = "8g"
            shuffle_partitions = 300
            print(f"   ⚙️  Spark 記憶體配置：Driver={driver_memory}, Executor={executor_memory}（預設值）")
        
        # Windows 上使用 local[1] 避免 Python worker 連線問題
        # 如果需要並行處理，可以改回 local[*]，但需要確保 PYSPARK_PYTHON 設定正確
        import platform
        is_windows = platform.system() == 'Windows'
        master_url = "local[1]" if is_windows else "local[*]"
        
        spark = SparkSession.builder \
            .appName("BidirectionalFlowFeatures") \
            .master(master_url) \
            .config("spark.driver.memory", driver_memory) \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
            .config("spark.local.dir", str(spark_temp_dir)) \
            .config("spark.sql.warehouse.dir", str(spark_temp_dir)) \
            .config("spark.executor.tempDir", str(spark_temp_dir)) \
            .config("spark.driver.tempDir", str(spark_temp_dir)) \
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
            .config("spark.hadoop.fs.defaultFS", "file:///") \
            .config("spark.python.worker.reuse", "false") \
            .config("spark.python.worker.timeout", "600") \
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
            .config("spark.python.worker.faulthandler.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        return spark
    except ImportError:
        raise ImportError(
            "PySpark 未安裝。請執行: pip install pyspark\n"
            "雙向流特徵工程需要 PySpark 支援。"
        )


def _extract_bidirectional_pair_features_pandas(
    df: pd.DataFrame,
    window_size: str = '1min'
) -> pd.DataFrame:
    """
    使用純 Pandas 提取雙向流 Pair 聚合特徵（階段4，降級方案）。
    
    這是 PySpark 版本的替代實現，適用於 Windows 環境或 PySpark 不可用的情況。
    
    Args:
        df: 輸入的 DataFrame，必須包含 'StartTime', 'SrcAddr', 'DstAddr', 'SrcBytes', 'DstBytes', 'TotBytes', 'TotPkts' 欄位
        window_size: 時間窗口大小，預設為 '1min'（1分鐘）
    
    Returns:
        包含雙向流 Pair 聚合特徵的 DataFrame
    """
    features_df = df.copy()
    
    required_cols = ['StartTime', 'SrcAddr', 'DstAddr']
    if not all(col in features_df.columns for col in required_cols):
        return features_df
    
    # 確保 StartTime 是 datetime 類型
    if not pd.api.types.is_datetime64_any_dtype(features_df['StartTime']):
        features_df['StartTime'] = pd.to_datetime(features_df['StartTime'], errors='coerce')
    
    # 標準化 IP Pair：確保 (A, B) 和 (B, A) 被視為同一 Pair
    features_df['ip_pair_min'] = features_df[['SrcAddr', 'DstAddr']].min(axis=1)
    features_df['ip_pair_max'] = features_df[['SrcAddr', 'DstAddr']].max(axis=1)
    features_df['ip_pair'] = features_df['ip_pair_min'].astype(str) + '_' + features_df['ip_pair_max'].astype(str)
    
    # 建立時間窗口
    features_df['time_window_start'] = features_df['StartTime'].dt.floor(window_size)
    
    # 確保必要的數值欄位存在
    for col_name in ['TotBytes', 'TotPkts', 'SrcBytes', 'DstBytes']:
        if col_name not in features_df.columns:
            features_df[col_name] = 0
    
    # 使用 Pandas groupby 進行聚合
    group_cols = ['ip_pair', 'time_window_start']
    
    # 聚合特徵
    agg_dict = {
        'TotBytes': 'sum',
        'TotPkts': 'sum',
        'SrcBytes': 'sum',
        'DstBytes': 'sum'
    }
    
    if 'Dur' in features_df.columns:
        agg_dict['Dur'] = 'mean'
    
    # 計算聚合特徵
    bidirectional_features = features_df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # 重新命名聚合欄位
    bidirectional_features = bidirectional_features.rename(columns={
        'TotBytes': 'bidirectional_total_bytes',
        'TotPkts': 'bidirectional_total_packets',
        'SrcBytes': 'bidirectional_total_src_bytes',
        'DstBytes': 'bidirectional_total_dst_bytes',
        'Dur': 'bidirectional_avg_duration'
    })
    
    # 計算流數量
    flow_counts = features_df.groupby(group_cols).size().reset_index(name='bidirectional_flow_count')
    bidirectional_features = bidirectional_features.merge(flow_counts, on=group_cols, how='left')
    
    # 如果沒有 Dur 欄位，添加預設值
    if 'bidirectional_avg_duration' not in bidirectional_features.columns:
        bidirectional_features['bidirectional_avg_duration'] = 0.0
    
    # 計算雙向流量對稱性
    # symmetry = min(SrcBytes, DstBytes) / max(SrcBytes, DstBytes)
    max_bytes = bidirectional_features[['bidirectional_total_src_bytes', 'bidirectional_total_dst_bytes']].max(axis=1)
    min_bytes = bidirectional_features[['bidirectional_total_src_bytes', 'bidirectional_total_dst_bytes']].min(axis=1)
    bidirectional_features['bidirectional_symmetry'] = (min_bytes / max_bytes).fillna(0.0)
    bidirectional_features.loc[max_bytes == 0, 'bidirectional_symmetry'] = 0.0
    
    # 計算平均每個流的位元組數和封包數
    bidirectional_features['bidirectional_avg_bytes_per_flow'] = (
        bidirectional_features['bidirectional_total_bytes'] / bidirectional_features['bidirectional_flow_count']
    ).fillna(0.0)
    bidirectional_features['bidirectional_avg_packets_per_flow'] = (
        bidirectional_features['bidirectional_total_packets'] / bidirectional_features['bidirectional_flow_count']
    ).fillna(0.0)
    
    # 將雙向流特徵合併回原始 DataFrame
    merge_cols = ['ip_pair', 'time_window_start']
    features_df = features_df.merge(
        bidirectional_features[merge_cols + [
            'bidirectional_flow_count',
            'bidirectional_total_bytes',
            'bidirectional_total_packets',
            'bidirectional_total_src_bytes',
            'bidirectional_total_dst_bytes',
            'bidirectional_symmetry',
            'bidirectional_avg_bytes_per_flow',
            'bidirectional_avg_packets_per_flow',
            'bidirectional_avg_duration'
        ]],
        on=merge_cols,
        how='left'
    )
    
    # 填充缺失值，並確保所有必要的欄位都存在
    bidirectional_cols = [
        'bidirectional_flow_count',
        'bidirectional_total_bytes',
        'bidirectional_total_packets',
        'bidirectional_total_src_bytes',
        'bidirectional_total_dst_bytes',
        'bidirectional_symmetry',
        'bidirectional_avg_bytes_per_flow',
        'bidirectional_avg_packets_per_flow',
        'bidirectional_avg_duration'
    ]
    for col_name in bidirectional_cols:
        if col_name not in features_df.columns:
            # 如果欄位不存在，創建它並設為 0
            features_df[col_name] = 0.0
        else:
            # 如果欄位存在，填充 NaN 值
            features_df[col_name] = features_df[col_name].fillna(0)
    
    # 移除臨時欄位
    features_df.drop(columns=['ip_pair_min', 'ip_pair_max', 'ip_pair', 'time_window_start'], errors='ignore', inplace=True)
    
    return features_df


def _extract_bidirectional_pair_features_spark(
    df: pd.DataFrame, 
    window_size: str = '1min',
    spark_session=None
) -> pd.DataFrame:
    """
    使用 Spark Window Function 提取雙向流 Pair 聚合特徵（階段4）。
    
    將 Src -> Dst 和 Dst -> Src 的流量關聯起來，針對同一個 (SrcIP, DstIP) Pair
    在時間窗口內聚合數據，模擬 Session 行為。
    
    特徵包括：
    - bidirectional_flow_count: 該 IP Pair 在時間窗口內的流數量（雙向合計）
    - bidirectional_total_bytes: 該 IP Pair 在時間窗口內的總位元組數（雙向合計）
    - bidirectional_total_packets: 該 IP Pair 在時間窗口內的總封包數（雙向合計）
    - bidirectional_symmetry: 雙向流量對稱性（DstBytes / SrcBytes，範圍 0-1）
    - bidirectional_avg_bytes_per_flow: 平均每個流的位元組數
    - bidirectional_avg_packets_per_flow: 平均每個流的封包數
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'StartTime': pd.to_datetime(['2021-08-17 12:01:00', '2021-08-17 12:01:10', '2021-08-17 12:01:20']),
    ...     'SrcAddr': ['192.168.1.1', '10.0.0.1', '192.168.1.1'],
    ...     'DstAddr': ['10.0.0.1', '192.168.1.1', '10.0.0.1'],
    ...     'SrcBytes': [1000, 500, 200],
    ...     'DstBytes': [100, 200, 50],
    ...     'TotBytes': [1100, 700, 250],
    ...     'TotPkts': [10, 5, 2]
    ... })
    >>> features = _extract_bidirectional_pair_features_spark(df)
    >>> 'bidirectional_flow_count' in features.columns
    True
    >>> 'bidirectional_symmetry' in features.columns
    True
    
    Args:
        df: 輸入的 DataFrame，必須包含 'StartTime', 'SrcAddr', 'DstAddr', 'SrcBytes', 'DstBytes', 'TotBytes', 'TotPkts' 欄位
        window_size: 時間窗口大小，預設為 '1min'（1分鐘）
        spark_session: 可選的 SparkSession 實例。如果為 None，則自動創建
    
    Returns:
        包含雙向流 Pair 聚合特徵的 DataFrame
    """
    features_df = df.copy()
    
    required_cols = ['StartTime', 'SrcAddr', 'DstAddr']
    if not all(col in features_df.columns for col in required_cols):
        return features_df
    
    # 確保 StartTime 是 datetime 類型
    if not pd.api.types.is_datetime64_any_dtype(features_df['StartTime']):
        features_df['StartTime'] = pd.to_datetime(features_df['StartTime'], errors='coerce')
    
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import (
            col, window, count, sum as spark_sum, avg, 
            min as spark_min, max as spark_max, 
            when, concat, lit, least, greatest, expr
        )
        from pyspark.sql.types import TimestampType
        
        # 取得或創建 SparkSession
        if spark_session is None:
            spark = _get_or_create_spark_session()
        else:
            spark = spark_session
        
        # 準備轉換為 Spark DataFrame 的資料
        # 只選擇需要的欄位，並確保類型兼容
        required_spark_cols = ['StartTime', 'SrcAddr', 'DstAddr']
        optional_cols = ['TotBytes', 'TotPkts', 'SrcBytes', 'DstBytes', 'Dur']
        
        # 選擇存在的欄位
        cols_to_use = [col for col in required_spark_cols if col in features_df.columns]
        cols_to_use.extend([col for col in optional_cols if col in features_df.columns])
        
        # 創建一個乾淨的 DataFrame 用於轉換
        df_for_spark = features_df[cols_to_use].copy()
        
        # 處理 NaN 值（Spark 不喜歡某些類型的 NaN）
        # 注意：使用 col_name 而不是 col，避免覆蓋 PySpark 的 col 函數
        for col_name in df_for_spark.columns:
            if df_for_spark[col_name].dtype in ['object', 'string']:
                # 字串欄位：將 NaN 轉為空字串
                df_for_spark[col_name] = df_for_spark[col_name].fillna('')
            elif pd.api.types.is_numeric_dtype(df_for_spark[col_name]):
                # 數值欄位：將 NaN 轉為 0
                df_for_spark[col_name] = df_for_spark[col_name].fillna(0)
        
        # 確保 StartTime 是 datetime（不是 object）
        if 'StartTime' in df_for_spark.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_for_spark['StartTime']):
                df_for_spark['StartTime'] = pd.to_datetime(df_for_spark['StartTime'], errors='coerce')
            # 將 NaT 轉為 None（Spark 可以處理）
            df_for_spark['StartTime'] = df_for_spark['StartTime'].where(pd.notnull(df_for_spark['StartTime']), None)
        
        # 將 Pandas DataFrame 轉換為 Spark DataFrame
        # 使用 inferSchema=False 並手動指定類型，避免類型推斷問題
        try:
            spark_df = spark.createDataFrame(df_for_spark)
        except Exception as e:
            # 如果轉換失敗，嘗試只使用基本欄位
            print(f"   ⚠️  完整轉換失敗，嘗試基本欄位轉換: {e}")
            basic_cols = ['StartTime', 'SrcAddr', 'DstAddr']
            if all(col_name in df_for_spark.columns for col_name in basic_cols):
                spark_df = spark.createDataFrame(df_for_spark[basic_cols])
            else:
                raise
        
        # 標準化 IP Pair：確保 (A, B) 和 (B, A) 被視為同一 Pair
        # 使用 min 和 max 來標準化，確保 Pair 的順序一致
        spark_df = spark_df.withColumn(
            "ip_pair_min",
            least(col("SrcAddr"), col("DstAddr"))
        ).withColumn(
            "ip_pair_max",
            greatest(col("SrcAddr"), col("DstAddr"))
        ).withColumn(
            "ip_pair",
            concat(col("ip_pair_min"), lit("_"), col("ip_pair_max"))
        )
        
        # 確保 StartTime 是 TimestampType
        if spark_df.schema["StartTime"].dataType != TimestampType():
            spark_df = spark_df.withColumn(
                "StartTime",
                col("StartTime").cast(TimestampType())
            )
        
        # 使用 Window Function 按 (ip_pair, time_window) 聚合
        # 這比 Self-Join 更高效
        # 修正：PySpark window 函數使用字符串格式的時間間隔
        if window_size == '1min':
            window_expr = window(col("StartTime"), "1 minute").alias("time_window")
        elif window_size == '1hour' or window_size == '1h':
            window_expr = window(col("StartTime"), "1 hour").alias("time_window")
        else:
            # 嘗試直接使用（如果格式正確，如 "5 minutes", "30 seconds" 等）
            window_expr = window(col("StartTime"), window_size).alias("time_window")
        
        # 確保必要的數值欄位存在（如果不存在則添加預設值 0）
        for col_name in ['TotBytes', 'TotPkts', 'SrcBytes', 'DstBytes']:
            if col_name not in spark_df.columns:
                spark_df = spark_df.withColumn(col_name, lit(0))
        
        # 計算聚合特徵
        agg_exprs = [
            # 流數量（雙向合計）
            count("*").alias("bidirectional_flow_count"),
            
            # 總位元組數（雙向合計）
            spark_sum("TotBytes").alias("bidirectional_total_bytes"),
            
            # 總封包數（雙向合計）
            spark_sum("TotPkts").alias("bidirectional_total_packets"),
            
            # 總上行位元組數（SrcBytes 總和）
            spark_sum("SrcBytes").alias("bidirectional_total_src_bytes"),
            
            # 總下行位元組數（DstBytes 總和）
            spark_sum("DstBytes").alias("bidirectional_total_dst_bytes")
        ]
        
        # 平均持續時間（如果有 Dur 欄位）
        if "Dur" in spark_df.columns:
            agg_exprs.append(avg("Dur").alias("bidirectional_avg_duration"))
        else:
            agg_exprs.append(lit(0.0).alias("bidirectional_avg_duration"))
        
        bidirectional_features = spark_df.groupBy(
            "ip_pair",
            window_expr
        ).agg(*agg_exprs)
        
        # 計算雙向流量對稱性
        # symmetry = min(SrcBytes, DstBytes) / max(SrcBytes, DstBytes)
        # 範圍 0-1，1 表示完全對稱，0 表示完全不對稱
        # 注意：使用 greatest 而不是 spark_max（spark_max 是聚合函數，不能用於比較兩個欄位）
        bidirectional_features = bidirectional_features.withColumn(
            "bidirectional_symmetry",
            when(
                greatest(col("bidirectional_total_src_bytes"), col("bidirectional_total_dst_bytes")) > lit(0),
                least(col("bidirectional_total_src_bytes"), col("bidirectional_total_dst_bytes")) / 
                greatest(col("bidirectional_total_src_bytes"), col("bidirectional_total_dst_bytes"))
            ).otherwise(lit(0.0))
        )
        
        # 計算平均每個流的位元組數和封包數
        bidirectional_features = bidirectional_features.withColumn(
            "bidirectional_avg_bytes_per_flow",
            col("bidirectional_total_bytes") / col("bidirectional_flow_count")
        ).withColumn(
            "bidirectional_avg_packets_per_flow",
            col("bidirectional_total_packets") / col("bidirectional_flow_count")
        )
        
        # 提取 time_window 的 start 時間（用於後續 merge）
        # 在 PySpark 中，訪問結構欄位需要使用 expr() 或 col()["field"] 語法
        # 使用 expr("time_window.start") 來訪問 window 結構的 start 欄位
        bidirectional_features = bidirectional_features.withColumn(
            "time_window_start",
            expr("time_window.start")
        )
        
        # 轉換回 Pandas DataFrame
        bidirectional_features_pd = bidirectional_features.toPandas()
        
        # 處理 time_window 結構（如果需要的話）
        if 'time_window' in bidirectional_features_pd.columns:
            if len(bidirectional_features_pd) > 0 and isinstance(bidirectional_features_pd['time_window'].iloc[0], dict):
                bidirectional_features_pd['time_window_start'] = pd.to_datetime(
                    bidirectional_features_pd['time_window'].apply(lambda x: x['start'] if isinstance(x, dict) else x)
                )
        
        # 將雙向流特徵合併回原始 DataFrame
        # 需要為每筆原始記錄找到對應的 IP Pair 和時間窗口
        features_df['ip_pair_min'] = features_df[['SrcAddr', 'DstAddr']].min(axis=1)
        features_df['ip_pair_max'] = features_df[['SrcAddr', 'DstAddr']].max(axis=1)
        features_df['ip_pair'] = features_df['ip_pair_min'].astype(str) + '_' + features_df['ip_pair_max'].astype(str)
        features_df['time_window_start'] = features_df['StartTime'].dt.floor(window_size)
        
        # Merge 雙向流特徵
        # 確保時間欄位格式一致
        if 'time_window_start' not in bidirectional_features_pd.columns:
            # 如果沒有 time_window_start，嘗試從 time_window 結構中提取
            if 'time_window' in bidirectional_features_pd.columns:
                if len(bidirectional_features_pd) > 0:
                    if isinstance(bidirectional_features_pd['time_window'].iloc[0], dict):
                        bidirectional_features_pd['time_window_start'] = pd.to_datetime(
                            bidirectional_features_pd['time_window'].apply(lambda x: x.get('start', x) if isinstance(x, dict) else x)
                        )
                    else:
                        bidirectional_features_pd['time_window_start'] = pd.to_datetime(bidirectional_features_pd['time_window'])
        
        # 確保時間欄位格式一致
        if 'time_window_start' in bidirectional_features_pd.columns:
            bidirectional_features_pd['time_window_start'] = pd.to_datetime(bidirectional_features_pd['time_window_start'])
        
        merge_cols = ['ip_pair', 'time_window_start']
        
        # 確保 merge_cols 都存在
        if all(col in bidirectional_features_pd.columns for col in merge_cols):
            features_df = features_df.merge(
                bidirectional_features_pd[merge_cols + [
                    'bidirectional_flow_count',
                    'bidirectional_total_bytes',
                    'bidirectional_total_packets',
                    'bidirectional_total_src_bytes',
                    'bidirectional_total_dst_bytes',
                    'bidirectional_symmetry',
                    'bidirectional_avg_bytes_per_flow',
                    'bidirectional_avg_packets_per_flow',
                    'bidirectional_avg_duration'
                ]],
                on=merge_cols,
                how='left'
            )
        else:
            print(f"⚠️  無法 merge 雙向流特徵：缺少必要的 merge 欄位")
            print(f"   需要的欄位: {merge_cols}")
            print(f"   實際欄位: {list(bidirectional_features_pd.columns)}")
            # 即使 merge 失敗，也創建必要的欄位（設為 0），以便後續計算 bidirectional_window_flow_ratio
            print(f"   創建預設的雙向流特徵欄位（值為 0）")
        
        # 填充缺失值（如果某些 IP Pair 沒有對應的特徵）
        # 同時確保所有必要的欄位都存在
        bidirectional_cols = [
            'bidirectional_flow_count',
            'bidirectional_total_bytes',
            'bidirectional_total_packets',
            'bidirectional_total_src_bytes',
            'bidirectional_total_dst_bytes',
            'bidirectional_symmetry',
            'bidirectional_avg_bytes_per_flow',
            'bidirectional_avg_packets_per_flow',
            'bidirectional_avg_duration'
        ]
        for col_name in bidirectional_cols:
            if col_name not in features_df.columns:
                # 如果欄位不存在，創建它並設為 0
                features_df[col_name] = 0.0
            else:
                # 如果欄位存在，填充 NaN 值
                features_df[col_name] = features_df[col_name].fillna(0)
        
        # 移除臨時欄位
        features_df.drop(columns=['ip_pair_min', 'ip_pair_max', 'ip_pair', 'time_window_start'], errors='ignore', inplace=True)
        
        return features_df
        
    except ImportError:
        # 如果 PySpark 未安裝，使用 Pandas 降級方案
        print("⚠️  PySpark 未安裝，使用 Pandas 降級方案...")
        return _extract_bidirectional_pair_features_pandas(features_df, window_size)
    except Exception as e:
        # 如果 Spark 處理失敗，嘗試使用 Pandas 降級方案
        import platform
        is_windows = platform.system() == 'Windows'
        error_msg = str(e)
        
        # 檢查是否為 PySpark 相關錯誤（包括所有可能的錯誤類型）
        pyspark_errors = [
            'Python worker',
            'crashed',
            'collectToPython',
            'EOFException',
            'SparkException',
            'SocketTimeoutException',
            'WinError 10038',
            'not a socket'
        ]
        
        is_pyspark_error = any(err.lower() in error_msg.lower() for err in pyspark_errors)
        
        # Windows 上的常見錯誤，或任何 PySpark 相關錯誤，自動降級到 Pandas
        if is_windows or is_pyspark_error:
            error_preview = error_msg[:150].replace('\n', ' ') if len(error_msg) > 150 else error_msg
            print(f"⚠️  PySpark 執行失敗，使用 Pandas 降級方案...")
            print(f"   錯誤訊息：{error_preview}...")
            print("   💡 提示：Pandas 版本較慢但更穩定，適合 Windows 環境")
            try:
                return _extract_bidirectional_pair_features_pandas(features_df, window_size)
            except Exception as pandas_error:
                print(f"⚠️  Pandas 降級方案也失敗: {pandas_error}")
                print("   返回原始 DataFrame（不包含雙向流特徵）")
                return features_df
        else:
            # 其他錯誤，返回原始 DataFrame
            print(f"⚠️  雙向流特徵工程失敗: {e}")
            print("   返回原始 DataFrame（不包含雙向流特徵）")
            return features_df


def extract_features(
    df: pd.DataFrame,
    flow_type: str = 'auto',
    include_time_features: bool = True,
    time_feature_stage: int = 1,
    include_bidirectional_features: bool = False
) -> pd.DataFrame:
    """
    從 NetFlow 資料中提取特徵。
    
    根據 notebooks/bidirectional/01_EDA.ipynb 的分析結果，
    提取以下關鍵特徵：
    - DstBytes: 目的端位元組數
    - flow_ratio: 上下行流量比（檢測外洩行為）
    - bytes_symmetry: 行為對稱性（檢測掃描行為）
    - is_scanning: 掃描行為標記
    - src_ratio: 來源流量比例（%）
    - dst_ratio: 目的流量比例（%）
    - packet_size: 平均封包大小（bytes）
    - bytes_per_second: 位元組傳輸速率
    - packets_per_second: 封包傳輸速率
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'TotBytes': [1000, 2000, 3000],
    ...     'SrcBytes': [600, 1200, 1800],
    ...     'TotPkts': [10, 20, 30],
    ...     'Dur': [1.0, 2.0, 3.0],
    ...     'State': ['CON', 'RST', 'CON']
    ... })
    >>> features = extract_features(df)
    >>> 'flow_ratio' in features.columns
    True
    >>> 'bytes_symmetry' in features.columns
    True
    >>> 'is_scanning' in features.columns
    True
    >>> 'packet_size' in features.columns
    True
    >>> 'src_ratio' in features.columns
    True
    
    Args:
        df: 輸入的 NetFlow DataFrame
        flow_type: 流類型 ('auto', 'bidirectional')，目前僅支援 bidirectional
        include_time_features: 是否包含時間特徵
        time_feature_stage: 時間特徵階段（1-4）
            - 1: 基本時間特徵
            - 2: 時間間隔特徵
            - 3: 時間窗口聚合特徵（按 SrcAddr）
            - 4: 雙向流 Pair 聚合特徵（按 IP Pair，需要 PySpark）
        include_bidirectional_features: 是否包含雙向流特徵（已棄用，使用 time_feature_stage=4）
    
    Returns:
        包含特徵的 DataFrame
    """
    # 複製 DataFrame 避免修改原始資料
    features_df = df.copy()
    
    # 1. 計算目的端位元組數
    if 'DstBytes' not in features_df.columns:
        if 'TotBytes' in features_df.columns and 'SrcBytes' in features_df.columns:
            features_df['DstBytes'] = features_df['TotBytes'] - features_df['SrcBytes']
        else:
            raise ValueError("缺少必要欄位：需要 'TotBytes' 和 'SrcBytes' 或 'DstBytes'")
    
    # 2. 計算 flow_ratio（上下行流量比）
    # 用於檢測外洩行為：SrcBytes 遠大於 DstBytes 時，flow_ratio 會很大
    features_df['flow_ratio'] = features_df['SrcBytes'] / (features_df['DstBytes'] + 1)  # +1 避免除零
    
    # 3. 計算 bytes_symmetry（行為對稱性）
    # 用於檢測掃描行為：DstBytes 遠小於 SrcBytes 時，bytes_symmetry 接近 0
    features_df['bytes_symmetry'] = features_df['DstBytes'] / (features_df['SrcBytes'] + 1)  # +1 避免除零
    
    # 4. 計算流量方向比例（新增）
    # 用於分析流量方向平衡，識別單向流量異常
    if 'TotBytes' in features_df.columns:
        # 避免除零：將 TotBytes 為 0 的情況設為 NaN
        tot_bytes_safe = features_df['TotBytes'].replace(0, np.nan)
        features_df['src_ratio'] = (features_df['SrcBytes'] / tot_bytes_safe * 100).fillna(0)
        features_df['dst_ratio'] = (features_df['DstBytes'] / tot_bytes_safe * 100).fillna(0)
    
    # 5. 計算平均封包大小（新增）
    # 用於識別異常封包大小模式
    if 'TotPkts' in features_df.columns:
        features_df['packet_size'] = features_df['TotBytes'] / (features_df['TotPkts'] + 1)  # +1 避免除零
    
    # 6. 計算流量強度特徵（新增）
    # 用於檢測高強度流量（DDoS、掃描等）
    if 'Dur' in features_df.columns:
        features_df['bytes_per_second'] = features_df['TotBytes'] / (features_df['Dur'] + 1)  # +1 避免除零
        features_df['packets_per_second'] = features_df['TotPkts'] / (features_df['Dur'] + 1)  # +1 避免除零
    
    # 7. 計算 is_scanning（掃描行為標記）
    # 根據 State 欄位判斷是否為掃描行為
    if 'State' in features_df.columns:
        scanning_states = ['RST', 'REQ', 'S_', 'INT', 'URP']
        features_df['is_scanning'] = features_df['State'].isin(scanning_states).astype(int)
    else:
        # 如果沒有 State 欄位，使用 bytes_symmetry 作為替代指標
        features_df['is_scanning'] = (features_df['bytes_symmetry'] < 0.1).astype(int)
    
    # 8. 提取時間特徵（可選）
    if include_time_features and 'StartTime' in features_df.columns:
        if time_feature_stage >= 1:
            features_df = _extract_basic_time_features(features_df)
        if time_feature_stage >= 2:
            features_df = _extract_time_interval_features(features_df)
        if time_feature_stage >= 3:
            features_df = _extract_time_window_features(features_df)
        if time_feature_stage >= 4:
            # 階段4：雙向流 Pair 聚合特徵（使用 Spark Window Function）
            features_df = _extract_bidirectional_pair_features_spark(features_df)
            
            # 計算 bidirectional_window_flow_ratio（使用階段四已聚合的資料）
            # 在時間窗口內聚合後的上下行流量比，用於檢測持續性外洩行為
            # 確保必要的欄位存在，如果不存在則創建預設值
            if 'bidirectional_total_src_bytes' not in features_df.columns:
                features_df['bidirectional_total_src_bytes'] = 0.0
            if 'bidirectional_total_dst_bytes' not in features_df.columns:
                features_df['bidirectional_total_dst_bytes'] = 0.0
            
            # 計算 bidirectional_window_flow_ratio
            # 確保所有值都是數值類型，並處理 NaN 和無限值
            features_df['bidirectional_window_flow_ratio'] = (
                features_df['bidirectional_total_src_bytes'].astype(float) / 
                (features_df['bidirectional_total_dst_bytes'].astype(float) + 1)  # +1 避免除零
            ).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    # 9. 處理無限值和異常值
    # 將無限值替換為 NaN，然後用 0 填充
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    # 對於數值特徵，用 0 填充 NaN（或可以根據業務邏輯選擇其他策略）
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
    
    return features_df


def get_feature_columns(include_time_features: bool = True, time_feature_stage: int = 1) -> List[str]:
    """
    返回特徵工程產生的特徵欄位名稱列表。
    
    >>> cols = get_feature_columns()
    >>> 'flow_ratio' in cols
    True
    >>> 'bytes_symmetry' in cols
    True
    >>> 'packet_size' in cols
    True
    >>> 'src_ratio' in cols
    True
    
    Args:
        include_time_features: 是否包含時間特徵
        time_feature_stage: 時間特徵階段（1, 2, 或 3）
    
    Returns:
        特徵欄位名稱列表
    """
    base_features = [
        'DstBytes',
        'flow_ratio',
        'bytes_symmetry',
        'is_scanning',
        'src_ratio',
        'dst_ratio',
        'packet_size',
        'bytes_per_second',
        'packets_per_second'
    ]
    
    if not include_time_features:
        return base_features
    
    # 階段1：基本時間特徵
    time_features_stage1 = [
        'hour',
        'day_of_week',
        'day_of_month',
        'is_weekend',
        'is_work_hour',
        'is_night',
        'sin_hour',
        'cos_hour',
        'sin_day_of_week',
        'cos_day_of_week',
        'sin_day_of_month',
        'cos_day_of_month'
    ]
    
    # 階段2：時間間隔特徵
    time_features_stage2 = [
        'time_since_last_flow',
        'time_to_next_flow'
    ]
    
    # 階段3：時間窗口聚合特徵
    time_features_stage3 = [
        'flows_per_minute_by_src',
        'unique_dst_per_minute_by_src',
        'unique_dport_per_minute_by_src',
        'total_bytes_per_minute_by_src'
    ]
    
    # 階段4：雙向流 Pair 聚合特徵
    time_features_stage4 = [
        'bidirectional_flow_count',
        'bidirectional_total_bytes',
        'bidirectional_total_packets',
        'bidirectional_total_src_bytes',
        'bidirectional_total_dst_bytes',
        'bidirectional_symmetry',
        'bidirectional_avg_bytes_per_flow',
        'bidirectional_avg_packets_per_flow',
        'bidirectional_avg_duration',
        'bidirectional_window_flow_ratio'  # 時間窗口內聚合後的上下行流量比
    ]
    
    all_features = base_features + time_features_stage1
    
    if time_feature_stage >= 2:
        all_features.extend(time_features_stage2)
    
    if time_feature_stage >= 3:
        all_features.extend(time_features_stage3)
    
    if time_feature_stage >= 4:
        all_features.extend(time_features_stage4)
    
    return all_features


if __name__ == '__main__':
    # 簡單測試
    import doctest
    doctest.testmod(verbose=True)

