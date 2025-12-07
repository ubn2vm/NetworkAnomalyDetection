"""
統一的資料載入器模組

使用 Factory Pattern 支援不同資料來源和格式的載入。
提供雙向流和 API 資料來源的統一介面。
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod
from enum import Enum


class DataSourceType(Enum):
    """資料來源類型枚舉"""
    BIDIRECTIONAL_BINETFLOW = "bidirectional_binetflow"
    BIDIRECTIONAL_BINETFLOW_SPARK = "bidirectional_binetflow_spark"
    API = "api"


def get_project_root() -> Path:
    """
    取得專案根目錄路徑。

    >>> root = get_project_root()
    >>> root.exists()
    True
    >>> root.name == 'NetworkAnomalyDetection'
    True

    Returns:
        專案根目錄的 Path 物件。
    """
    current_file = Path(__file__)
    # 從 src/data_loader.py 往上兩層到專案根目錄
    return current_file.parent.parent


class BaseDataLoader(ABC):
    """資料載入器抽象基類
    
    定義所有資料載入器必須實作的統一介面。
    
    >>> from src.data_loader import BaseDataLoader
    >>> # BaseDataLoader 是抽象類別，不能直接實例化
    >>> # loader = BaseDataLoader()  # 這會失敗
    """
    
    @abstractmethod
    def load(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        載入原始資料。
        
        Args:
            file_path: 資料檔案路徑。如果為 None，則使用預設路徑。
        
        Returns:
            包含原始資料的 DataFrame。
        
        Raises:
            FileNotFoundError: 如果檔案不存在。
        """
        pass
    
    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗資料。
        
        Args:
            df: 原始資料 DataFrame。
        
        Returns:
            清洗後的 DataFrame。
        """
        pass
    
    def save_cleaned_data(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
        project_root: Optional[Path] = None
    ) -> Path:
        """
        儲存清洗後的資料為 Parquet 格式。

        >>> import pandas as pd
        >>> import tempfile
        >>> from pathlib import Path
        >>> loader = BidirectionalBinetflowLoader()
        >>> test_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     output = loader.save_cleaned_data(test_df, Path(tmpdir) / "test.parquet")
        ...     assert output.exists()
        ...     loaded = pd.read_parquet(output)
        ...     len(loaded) == 3
        True

        Args:
            df: 清洗後的 DataFrame。
            output_path: 輸出檔案路徑。如果為 None，則使用預設路徑。
            project_root: 專案根目錄。如果為 None，則自動偵測。

        Returns:
            輸出檔案的路徑。
        """
        if project_root is None:
            project_root = get_project_root()
        
        if output_path is None:
            output_dir = project_root / "data" / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "cleaned_data.parquet"
        
        df.to_parquet(
            output_path,
            engine='pyarrow',
            index=False
        )
        
        return output_path


class BidirectionalBinetflowLoader(BaseDataLoader):
    """雙向流 Binetflow 載入器
    
    讀取 .binetflow 格式（CSV）的雙向流資料。
    
    >>> from src.data_loader import BidirectionalBinetflowLoader
    >>> loader = BidirectionalBinetflowLoader()
    >>> # df = loader.load()  # 需要實際檔案
    >>> # cleaned = loader.clean(df)
    """
    
    def load(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        從原始 NetFlow 檔案讀取雙向流資料（.binetflow 格式）。

        >>> from pathlib import Path
        >>> loader = BidirectionalBinetflowLoader()
        >>> # 需要實際檔案才能測試
        >>> # df = loader.load()
        >>> # assert 'StartTime' in df.columns

        Args:
            file_path: 原始資料檔案路徑。如果為 None，則使用預設路徑。

        Returns:
            包含原始雙向流 NetFlow 資料的 DataFrame。

        Raises:
            FileNotFoundError: 如果檔案不存在。
        """
        if file_path is None:
            project_root = get_project_root()
            file_path = project_root / "data" / "raw" / "capture20110817.binetflow"
        
        if not file_path.exists():
            raise FileNotFoundError(f"找不到檔案: {file_path}")
        
        # 讀取 .binetflow 格式（CSV 格式）
        df = pd.read_csv(file_path)
        
        return df
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗與轉換雙向流 NetFlow 資料。

        >>> import pandas as pd
        >>> loader = BidirectionalBinetflowLoader()
        >>> test_df = pd.DataFrame({
        ...     'StartTime': ['2011-08-17 12:01:01.780', '2011-08-17 12:02:01.780'],
        ...     'Dur': [3.124, 5.456],
        ...     'Proto': ['TCP', 'UDP'],
        ...     'SrcAddr': ['192.168.1.1', '10.0.0.1'],
        ...     'Sport': ['80', '443'],
        ...     'DstAddr': ['172.16.0.1', '192.168.1.100'],
        ...     'Dport': ['8080', '22'],
        ...     'TotBytes': [1000, 2000],
        ...     'TotPkts': [10, 20],
        ...     'Label': ['Background', 'Botnet']
        ... })
        >>> cleaned = loader.clean(test_df)
        >>> pd.api.types.is_datetime64_any_dtype(cleaned['StartTime'])
        True
        >>> 'Dur' in cleaned.columns
        True

        Args:
            df: 原始雙向流 NetFlow DataFrame。

        Returns:
            清洗後的 DataFrame，包含正確的資料型別轉換。
        """
        df = df.copy()
        
        # 轉換 StartTime 為 datetime
        if 'StartTime' in df.columns:
            df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
        
        # 確保數值欄位為正確型別
        numeric_cols = ['Dur', 'sTos', 'dTos', 'TotPkts', 'TotBytes', 'SrcBytes', 'DstBytes']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 確保埠號為數值（使用 coerce 將無法轉換的值設為 NaN，避免 FutureWarning）
        port_cols = ['Sport', 'Dport']
        for col in port_cols:
            if col in df.columns:
                # 使用 coerce 替代 ignore，無法轉換的值會變成 NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df


class APIDataLoader(BaseDataLoader):
    """API 資料載入器（模擬）
    
    從 API 載入資料。目前為框架實作，待後續完善。
    
    >>> from src.data_loader import APIDataLoader
    >>> loader = APIDataLoader()
    >>> # 待實作
    """
    
    def load(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        從 API 載入資料（模擬實作）。

        >>> loader = APIDataLoader()
        >>> # 待實作
        >>> # df = loader.load()
        >>> # assert isinstance(df, pd.DataFrame)

        Args:
            file_path: API 端點 URL（可選）。

        Returns:
            包含 API 資料的 DataFrame。

        Raises:
            NotImplementedError: API 載入器待實作。
        """
        # TODO: 實作 API 載入邏輯
        # 範例實作方向：
        # 1. 使用 requests 或 httpx 發送 HTTP 請求
        # 2. 解析 JSON/CSV 回應
        # 3. 轉換為 DataFrame
        raise NotImplementedError(
            "API 載入器待實作。"
            "請實作 HTTP 請求邏輯，將 API 回應轉換為 DataFrame。"
        )
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗 API 資料。

        Args:
            df: 從 API 載入的原始 DataFrame。

        Returns:
            清洗後的 DataFrame。
        """
        # TODO: 實作 API 資料清洗邏輯
        # 根據實際 API 回應格式進行清洗
        return df.copy()


class BidirectionalBinetflowLoaderSpark(BaseDataLoader):
    """雙向流 Binetflow 載入器（PySpark 單機模式）
    
    使用 PySpark 單機模式讀取 Parquet 格式的雙向流資料。
    適合處理大檔案，自動利用多核心加速。
    
    >>> from src.data_loader import BidirectionalBinetflowLoaderSpark
    >>> loader = BidirectionalBinetflowLoaderSpark()
    >>> # df = loader.load()  # 需要實際檔案
    >>> # cleaned = loader.clean(df)
    """
    
    def __init__(self, spark_session=None):
        """
        初始化 PySpark 載入器。
        
        Args:
            spark_session: 可選的 SparkSession 實例。如果為 None，則自動創建單機模式 Session。
        """
        self._spark = spark_session
        self._spark_created = False
    
    @property
    def spark(self):
        """取得或創建 SparkSession（延遲初始化）"""
        if self._spark is None:
            try:
                from pyspark.sql import SparkSession
                import os
                from pathlib import Path
                
                # 設定 Spark 臨時目錄（避免 Windows 權限問題）
                project_root = get_project_root()
                spark_temp_dir = project_root / "spark_temp"
                spark_temp_dir.mkdir(parents=True, exist_ok=True)
                
                # 設定環境變數
                os.environ['SPARK_LOCAL_DIRS'] = str(spark_temp_dir)
                
                # 根據系統記憶體自動調整 Spark 記憶體配置
                try:
                    import psutil
                    total_memory_gb = psutil.virtual_memory().total / (1024**3)
                    available_memory_gb = psutil.virtual_memory().available / (1024**3)
                    
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
                except ImportError:
                    driver_memory = "8g"
                    executor_memory = "8g"
                    shuffle_partitions = 300
                
                # 創建 SparkSession（單機模式）
                self._spark = SparkSession.builder \
                    .appName("NetworkAnomalyDetection") \
                    .master("local[*]") \
                    .config("spark.driver.memory", driver_memory) \
                    .config("spark.executor.memory", executor_memory) \
                    .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
                    .config("spark.local.dir", str(spark_temp_dir)) \
                    .config("spark.sql.warehouse.dir", str(spark_temp_dir)) \
                    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
                    .config("spark.hadoop.fs.defaultFS", "file:///") \
                    .getOrCreate()
                
                self._spark_created = True
                print(f"✅ SparkSession 建立完成（單機模式，使用所有核心）")
            except ImportError:
                raise ImportError(
                    "PySpark 未安裝。請執行: pip install pyspark"
                )
        return self._spark
    
    def __del__(self):
        """清理 SparkSession（如果是由此類別創建的）"""
        if self._spark_created and self._spark is not None:
            try:
                self._spark.stop()
            except:
                pass
    
    def load(self, file_path: Optional[Path] = None, use_parquet: bool = True) -> pd.DataFrame:
        """
        從 Parquet 檔案快速載入資料（優先），或從原始 CSV 讀取。
        
        使用 PySpark 單機模式讀取 CSV，但使用 Pandas 寫入 Parquet 以避免 Windows 問題。
        對於已存在的 Parquet 檔案，直接使用 Pandas 讀取以提升速度。

        >>> from pathlib import Path
        >>> loader = BidirectionalBinetflowLoaderSpark()
        >>> # 需要實際檔案才能測試
        >>> # df = loader.load()
        >>> # assert 'StartTime' in df.columns

        Args:
            file_path: 資料檔案路徑。如果為 None，則使用預設路徑。
            use_parquet: 是否優先使用 Parquet 格式（預設 True）。

        Returns:
            包含原始雙向流 NetFlow 資料的 DataFrame。

        Raises:
            FileNotFoundError: 如果檔案不存在。
            ImportError: 如果 PySpark 未安裝。
        """
        project_root = get_project_root()
        
        # 優先使用 Parquet 格式（如果存在）
        if use_parquet:
            parquet_path = project_root / "data" / "processed" / "capture20110817_cleaned_spark.parquet"
            if parquet_path.exists():
                # 優化：直接用 Pandas 讀取 Parquet（比 Spark 快很多）
                print(f"✅ 使用 Pandas 讀取 Parquet 檔案: {parquet_path}")
                try:
                    pandas_df = pd.read_parquet(parquet_path, engine='pyarrow')
                    print(f"✅ 載入完成：{len(pandas_df):,} 筆資料")
                    return pandas_df
                except Exception as e:
                    print(f"⚠️ 讀取 Parquet 失敗: {e}")
                    print("   將從原始 CSV 重新載入...")
        
        # 如果沒有 Parquet 或讀取失敗，從原始 CSV 讀取（使用 Spark 加速）
        if file_path is None:
            file_path = project_root / "data" / "raw" / "capture20110817.binetflow"
        
        if not file_path.exists():
            raise FileNotFoundError(f"找不到檔案: {file_path}")
        
        print(f"📂 使用 PySpark 讀取原始 CSV 檔案: {file_path}")
        
        # 使用 Spark 讀取 CSV（比 Pandas 快，特別是大檔案）
        spark_df = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(str(file_path))
        
        # 轉換為 Pandas DataFrame
        print("正在轉換為 Pandas DataFrame...")
        pandas_df = spark_df.toPandas()
        
        # 自動儲存為 Parquet 以供下次使用（使用 Pandas 避免 Windows Hadoop 問題）
        if use_parquet:
            parquet_path = project_root / "data" / "processed" / "capture20110817_cleaned_spark.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用 Pandas 寫入 Parquet（避免 Windows 上的 Hadoop 問題）
            print(f"💾 正在儲存為 Parquet 格式: {parquet_path}")
            try:
                pandas_df.to_parquet(
                    parquet_path, 
                    engine='pyarrow', 
                    index=False
                )
                print(f"✅ Parquet 檔案已儲存，下次載入將更快")
            except Exception as e:
                print(f"⚠️ 儲存 Parquet 失敗: {e}")
                print("   將繼續使用原始資料，但下次仍需要重新載入")
        
        print(f"✅ 載入完成：{len(pandas_df):,} 筆資料")
        return pandas_df

    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗與轉換雙向流 NetFlow 資料。
        
        使用與 BidirectionalBinetflowLoader 相同的清洗邏輯。

        >>> import pandas as pd
        >>> loader = BidirectionalBinetflowLoaderSpark()
        >>> test_df = pd.DataFrame({
        ...     'StartTime': ['2011-08-17 12:01:01.780', '2011-08-17 12:02:01.780'],
        ...     'Dur': [3.124, 5.456],
        ...     'Proto': ['TCP', 'UDP'],
        ...     'SrcAddr': ['192.168.1.1', '10.0.0.1'],
        ...     'Sport': ['80', '443'],
        ...     'DstAddr': ['172.16.0.1', '192.168.1.100'],
        ...     'Dport': ['8080', '22'],
        ...     'TotBytes': [1000, 2000],
        ...     'TotPkts': [10, 20],
        ...     'Label': ['Background', 'Botnet']
        ... })
        >>> cleaned = loader.clean(test_df)
        >>> pd.api.types.is_datetime64_any_dtype(cleaned['StartTime'])
        True
        >>> 'Dur' in cleaned.columns
        True

        Args:
            df: 原始雙向流 NetFlow DataFrame。

        Returns:
            清洗後的 DataFrame，包含正確的資料型別轉換。
        """
        # 使用與 BidirectionalBinetflowLoader 相同的清洗邏輯
        df = df.copy()
        
        # 轉換 StartTime 為 datetime
        if 'StartTime' in df.columns:
            df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
        
        # 確保數值欄位為正確型別
        numeric_cols = ['Dur', 'sTos', 'dTos', 'TotPkts', 'TotBytes', 'SrcBytes', 'DstBytes']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 確保埠號為數值（使用 coerce 將無法轉換的值設為 NaN，避免 FutureWarning）
        port_cols = ['Sport', 'Dport']
        for col in port_cols:
            if col in df.columns:
                # 使用 coerce 替代 ignore，無法轉換的值會變成 NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df


class DataLoaderFactory:
    """資料載入器工廠
    
    根據資料來源類型創建對應的資料載入器。
    
    >>> from src.data_loader import DataLoaderFactory, DataSourceType
    >>> loader = DataLoaderFactory.create(DataSourceType.BIDIRECTIONAL_BINETFLOW)
    >>> isinstance(loader, BaseDataLoader)
    True
    >>> isinstance(loader, BidirectionalBinetflowLoader)
    True
    """
    
    _loaders = {
        DataSourceType.BIDIRECTIONAL_BINETFLOW: BidirectionalBinetflowLoader,
        DataSourceType.BIDIRECTIONAL_BINETFLOW_SPARK: BidirectionalBinetflowLoaderSpark,
        DataSourceType.API: APIDataLoader,
    }
    
    @classmethod
    def create(cls, source_type: DataSourceType) -> BaseDataLoader:
        """
        創建資料載入器。

        >>> from src.data_loader import DataLoaderFactory, DataSourceType
        >>> loader = DataLoaderFactory.create(DataSourceType.BIDIRECTIONAL_BINETFLOW)
        >>> isinstance(loader, BidirectionalBinetflowLoader)
        True

        Args:
            source_type: 資料來源類型（DataSourceType 枚舉）。

        Returns:
            對應的資料載入器實例（BaseDataLoader 的子類別）。

        Raises:
            ValueError: 如果資料來源類型不存在。
        """
        if source_type not in cls._loaders:
            available_types = [st.value for st in cls._loaders.keys()]
            raise ValueError(
                f"不支援的資料來源類型: {source_type.value}。"
                f"可用的類型: {available_types}"
            )
        return cls._loaders[source_type]()
    
    @classmethod
    def get_available_types(cls):
        """
        取得所有可用的資料來源類型。

        >>> from src.data_loader import DataLoaderFactory
        >>> types = DataLoaderFactory.get_available_types()
        >>> len(types) >= 1
        True
        >>> DataSourceType.BIDIRECTIONAL_BINETFLOW in types
        True

        Returns:
            可用的資料來源類型列表。
        """
        return list(cls._loaders.keys())
    
    @classmethod
    def is_supported(cls, source_type: DataSourceType) -> bool:
        """
        檢查資料來源類型是否被支援。

        >>> from src.data_loader import DataLoaderFactory, DataSourceType
        >>> DataLoaderFactory.is_supported(DataSourceType.BIDIRECTIONAL_BINETFLOW)
        True
        >>> DataLoaderFactory.is_supported(DataSourceType.API)
        True

        Args:
            source_type: 資料來源類型。

        Returns:
            如果支援則返回 True，否則返回 False。
        """
        return source_type in cls._loaders

