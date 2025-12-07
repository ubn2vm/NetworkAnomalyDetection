"""
測試 data_loader 模組

測試資料載入、清洗和儲存功能。
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from src.data_loader import (
    DataLoaderFactory,
    DataSourceType,
    BidirectionalBinetflowLoader,
    BaseDataLoader,
    get_project_root
)
from src.feature_engineer import extract_features


class TestDataLoaderFactory:
    """測試 DataLoaderFactory"""
    
    def test_create_bidirectional_loader(self):
        """測試創建雙向流載入器"""
        loader = DataLoaderFactory.create(DataSourceType.BIDIRECTIONAL_BINETFLOW)
        
        assert isinstance(loader, BaseDataLoader)
        assert isinstance(loader, BidirectionalBinetflowLoader)
    
    def test_create_invalid_type(self):
        """測試創建無效的載入器類型"""
        # 由於 Python Enum 的限制，無法直接創建不存在的枚舉值
        # 我們通過臨時修改 _loaders 字典來模擬不存在的類型
        # 這樣可以測試錯誤處理邏輯
        
        # 保存原始狀態
        original_loaders = DataLoaderFactory._loaders.copy()
        
        # 臨時移除一個類型來模擬不存在的類型
        test_type = DataSourceType.BIDIRECTIONAL_BINETFLOW
        if test_type in DataLoaderFactory._loaders:
            # 移除該類型
            del DataLoaderFactory._loaders[test_type]
            
            # 現在嘗試創建這個類型應該會失敗
            with pytest.raises(ValueError, match="不支援的資料來源類型"):
                DataLoaderFactory.create(test_type)
            
            # 恢復原始狀態
            DataLoaderFactory._loaders = original_loaders
        
        # 驗證恢復後仍然可以正常創建
        loader = DataLoaderFactory.create(DataSourceType.BIDIRECTIONAL_BINETFLOW)
        assert isinstance(loader, BaseDataLoader)
    
    def test_get_available_types(self):
        """測試取得可用類型"""
        types = DataLoaderFactory.get_available_types()
        
        assert len(types) >= 1
        assert DataSourceType.BIDIRECTIONAL_BINETFLOW in types
    
    def test_is_supported(self):
        """測試檢查類型是否被支援"""
        assert DataLoaderFactory.is_supported(DataSourceType.BIDIRECTIONAL_BINETFLOW) is True
        assert DataLoaderFactory.is_supported(DataSourceType.API) is True


class TestBidirectionalBinetflowLoader:
    """測試 BidirectionalBinetflowLoader"""
    
    def test_clean_datetime_conversion(self):
        """測試 StartTime 轉換為 datetime"""
        loader = BidirectionalBinetflowLoader()
        df = pd.DataFrame({
            'StartTime': ['2011-08-17 12:01:01.780', '2011-08-17 12:02:01.780'],
            'Dur': [3.124, 5.456],
            'Proto': ['TCP', 'UDP']
        })
        cleaned = loader.clean(df)
        
        assert pd.api.types.is_datetime64_any_dtype(cleaned['StartTime'])
        assert 'Dur' in cleaned.columns
    
    def test_clean_numeric_conversion(self):
        """測試數值欄位轉換"""
        loader = BidirectionalBinetflowLoader()
        df = pd.DataFrame({
            'Dur': ['3.124', '5.456'],  # 字串格式
            'TotPkts': ['10', '20'],
            'TotBytes': ['1000', '2000'],
            'SrcBytes': ['500', '1000']
        })
        cleaned = loader.clean(df)
        
        assert pd.api.types.is_numeric_dtype(cleaned['Dur'])
        assert pd.api.types.is_numeric_dtype(cleaned['TotPkts'])
        assert pd.api.types.is_numeric_dtype(cleaned['TotBytes'])
    
    def test_clean_port_conversion(self):
        """測試埠號轉換"""
        loader = BidirectionalBinetflowLoader()
        df = pd.DataFrame({
            'Sport': ['80', '443', 'invalid'],  # 包含無效值
            'Dport': ['8080', '22', '99999']
        })
        cleaned = loader.clean(df)
        
        assert pd.api.types.is_numeric_dtype(cleaned['Sport'])
        assert pd.api.types.is_numeric_dtype(cleaned['Dport'])
        # 無效值應該變成 NaN
        assert pd.isna(cleaned['Sport'].iloc[2])
    
    def test_clean_empty_dataframe(self):
        """測試空 DataFrame 的清洗"""
        loader = BidirectionalBinetflowLoader()
        df = pd.DataFrame()
        cleaned = loader.clean(df)
        
        assert len(cleaned) == 0
        assert isinstance(cleaned, pd.DataFrame)
    
    def test_clean_missing_columns(self):
        """測試缺少某些欄位的情況"""
        loader = BidirectionalBinetflowLoader()
        df = pd.DataFrame({
            'Proto': ['TCP', 'UDP'],
            'Label': ['Normal', 'Botnet']
        })
        cleaned = loader.clean(df)
        
        # 應該不會出錯，只是不轉換不存在的欄位
        assert 'Proto' in cleaned.columns
        assert 'Label' in cleaned.columns
    
    def test_clean_preserves_other_columns(self):
        """測試保留其他欄位"""
        loader = BidirectionalBinetflowLoader()
        df = pd.DataFrame({
            'StartTime': ['2011-08-17 12:01:01.780'],
            'Dur': [3.124],
            'CustomColumn': ['custom_value']
        })
        cleaned = loader.clean(df)
        
        assert 'CustomColumn' in cleaned.columns
        assert cleaned['CustomColumn'].iloc[0] == 'custom_value'
    
    def test_clean_with_nan_values(self):
        """測試包含 NaN 值的清洗"""
        loader = BidirectionalBinetflowLoader()
        df = pd.DataFrame({
            'StartTime': ['2011-08-17 12:01:01.780', None, '2011-08-17 12:03:01.780'],
            'Dur': [3.124, np.nan, 5.456],
            'TotPkts': [10, 20, np.nan]
        })
        cleaned = loader.clean(df)
        
        assert len(cleaned) == 3
        assert pd.isna(cleaned['StartTime'].iloc[1])
        assert pd.isna(cleaned['Dur'].iloc[1])


class TestSaveCleanedData:
    """測試 save_cleaned_data 方法"""
    
    def test_save_and_load_parquet(self, temp_dir):
        """測試儲存和載入 Parquet 檔案"""
        loader = BidirectionalBinetflowLoader()
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': ['a', 'b', 'c']
        })
        
        output_path = loader.save_cleaned_data(test_df, temp_dir / "test.parquet")
        
        assert output_path.exists()
        loaded_df = pd.read_parquet(output_path)
        
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ['col1', 'col2', 'col3']
        assert loaded_df['col1'].tolist() == [1, 2, 3]
    
    def test_save_with_default_path(self, temp_dir, monkeypatch):
        """測試使用預設路徑儲存"""
        loader = BidirectionalBinetflowLoader()
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # 暫時修改 get_project_root 以使用臨時目錄
        original_get_root = get_project_root
        
        def mock_get_root():
            return temp_dir
        
        monkeypatch.setattr('src.data_loader.get_project_root', mock_get_root)
        
        output_path = loader.save_cleaned_data(test_df)
        
        assert output_path.exists()
        assert 'cleaned_data.parquet' in str(output_path)
    
    def test_save_preserves_data_types(self, temp_dir):
        """測試儲存保留資料類型"""
        loader = BidirectionalBinetflowLoader()
        test_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        
        output_path = loader.save_cleaned_data(test_df, temp_dir / "test.parquet")
        loaded_df = pd.read_parquet(output_path)
        
        assert pd.api.types.is_integer_dtype(loaded_df['int_col'])
        assert pd.api.types.is_float_dtype(loaded_df['float_col'])
        assert loaded_df['str_col'].dtype == 'object'
        # Parquet 可能將 bool 轉為 int，這是正常的


class TestGetProjectRoot:
    """測試 get_project_root 函數"""
    
    def test_get_project_root_exists(self):
        """測試取得專案根目錄"""
        root = get_project_root()
        
        assert root.exists()
        assert root.is_dir()
        assert root.name == 'NetworkAnomalyDetection'


class TestDataCleaningEdgeCases:
    """測試資料清洗的邊界情況"""
    
    def test_clean_invalid_datetime(self):
        """測試無效的日期時間格式"""
        import warnings
        
        loader = BidirectionalBinetflowLoader()
        df = pd.DataFrame({
            'StartTime': ['invalid-date', '2011-08-17 12:01:01.780', 'another-invalid']
        })
        
        # 抑制 pandas 的日期解析警告（這是預期的行為，因為我們在測試無效日期）
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, 
                                   message='.*Could not infer format.*')
            cleaned = loader.clean(df)
        
        # 無效日期應該變成 NaT (Not a Time)
        assert pd.isna(cleaned['StartTime'].iloc[0])
        assert pd.isna(cleaned['StartTime'].iloc[2])
        assert pd.api.types.is_datetime64_any_dtype(cleaned['StartTime'])
    
    def test_clean_negative_values(self):
        """測試負值處理"""
        loader = BidirectionalBinetflowLoader()
        df = pd.DataFrame({
            'Dur': [-1.0, 0.0, 3.124],
            'TotPkts': [-5, 0, 10]
        })
        cleaned = loader.clean(df)
        
        # 負值應該被保留（因為可能是有效的）
        assert cleaned['Dur'].iloc[0] == -1.0
        assert cleaned['TotPkts'].iloc[0] == -5
    
    def test_clean_very_large_numbers(self):
        """測試非常大的數值"""
        loader = BidirectionalBinetflowLoader()
        df = pd.DataFrame({
            'TotBytes': [1e15, 2e15, 3e15],
            'Dur': [1.0, 2.0, 3.0]
        })
        cleaned = loader.clean(df)
        
        assert pd.api.types.is_numeric_dtype(cleaned['TotBytes'])
        assert cleaned['TotBytes'].iloc[0] == 1e15
    
    def test_clean_mixed_types_in_numeric_column(self):
        """測試數值欄位中的混合類型"""
        loader = BidirectionalBinetflowLoader()
        df = pd.DataFrame({
            'Dur': ['3.124', 5.456, 'invalid', 7.890]
        })
        cleaned = loader.clean(df)
        
        assert pd.api.types.is_numeric_dtype(cleaned['Dur'])
        # 無效值應該變成 NaN
        assert pd.isna(cleaned['Dur'].iloc[2])


class TestDataLoaderOutputSchema:
    """測試 Data Loader 輸出 Schema 驗證
    
    確保 data loader 清洗後的資料格式符合後續處理的需求。
    重點：如果資料格式改變，loader 是否還能正確處理並輸出正確格式？
    """
    
    def test_cleaned_data_has_correct_dtypes(self):
        """測試清洗後的資料型態是否正確"""
        loader = BidirectionalBinetflowLoader()
        
        # 模擬原始資料（可能是字串格式）
        df_raw = pd.DataFrame({
            'StartTime': ['2011-08-17 12:01:01.780', '2011-08-17 12:02:01.780'],
            'Dur': ['3.124', '5.456'],  # 字串格式
            'Sport': ['80', '443'],     # 字串格式
            'Dport': ['8080', '22'],    # 字串格式
            'TotBytes': ['1000', '2000'],  # 字串格式
            'TotPkts': ['10', '20'],    # 字串格式
            'SrcBytes': ['500', '1000'], # 字串格式
            'DstBytes': ['500', '1000'], # 字串格式
            'SrcAddr': ['192.168.1.1', '10.0.0.1'],
            'DstAddr': ['172.16.0.1', '192.168.1.100'],
            'Proto': ['TCP', 'UDP']
        })
        
        cleaned = loader.clean(df_raw)
        
        # 驗證型態轉換
        assert pd.api.types.is_datetime64_any_dtype(cleaned['StartTime']), \
            "StartTime 應該是 datetime 型態"
        assert pd.api.types.is_numeric_dtype(cleaned['Dur']), \
            "Dur 應該是數值型態"
        assert pd.api.types.is_numeric_dtype(cleaned['Sport']), \
            "Sport 應該是數值型態"
        assert pd.api.types.is_numeric_dtype(cleaned['Dport']), \
            "Dport 應該是數值型態"
        assert pd.api.types.is_numeric_dtype(cleaned['TotBytes']), \
            "TotBytes 應該是數值型態"
        assert pd.api.types.is_numeric_dtype(cleaned['TotPkts']), \
            "TotPkts 應該是數值型態"
        assert pd.api.types.is_numeric_dtype(cleaned['SrcBytes']), \
            "SrcBytes 應該是數值型態"
        assert pd.api.types.is_numeric_dtype(cleaned['DstBytes']), \
            "DstBytes 應該是數值型態"
        
        # 驗證字串欄位保留
        assert cleaned['SrcAddr'].dtype == 'object', \
            "SrcAddr 應該是字串型態"
        assert cleaned['DstAddr'].dtype == 'object', \
            "DstAddr 應該是字串型態"
        assert cleaned['Proto'].dtype == 'object', \
            "Proto 應該是字串型態"
    
    def test_cleaned_data_has_required_columns_for_feature_engineering(self):
        """測試清洗後的資料是否包含特徵工程所需的欄位"""
        loader = BidirectionalBinetflowLoader()
        
        df = pd.DataFrame({
            'StartTime': ['2011-08-17 12:01:01.780'],
            'Dur': [3.124],
            'Sport': [80],
            'Dport': [443],
            'TotBytes': [1000],
            'TotPkts': [10],
            'SrcBytes': [500],
            'DstBytes': [500],
            'SrcAddr': ['192.168.1.1'],
            'DstAddr': ['172.16.0.1'],
            'Proto': ['TCP']
        })
        
        cleaned = loader.clean(df)
        
        # 特徵工程需要的欄位
        required_for_features = [
            'StartTime',  # 時間特徵
            'Dur',        # 基礎統計特徵
            'TotBytes',   # 基礎統計特徵
            'TotPkts',    # 基礎統計特徵
            'SrcBytes',   # 基礎統計特徵
            'DstBytes',   # 基礎統計特徵
            'SrcAddr',    # 時間窗口聚合特徵（階段3）
            'DstAddr',    # 雙向流 Pair 特徵（階段4）
            'Sport',      # 可能用於特徵工程
            'Dport',      # 可能用於特徵工程
            'Proto'       # 可能用於特徵工程
        ]
        
        for col in required_for_features:
            assert col in cleaned.columns, \
                f"缺少特徵工程所需欄位: {col}"
    
    def test_cleaned_data_compatible_with_feature_engineering(self):
        """測試清洗後的資料是否可以直接用於特徵工程"""
        loader = BidirectionalBinetflowLoader()
        
        df = pd.DataFrame({
            'StartTime': ['2011-08-17 12:01:01.780', '2011-08-17 12:02:01.780'],
            'Dur': [3.124, 5.456],
            'Sport': [80, 443],
            'Dport': [8080, 22],
            'TotBytes': [1000, 2000],
            'TotPkts': [10, 20],
            'SrcBytes': [500, 1000],
            'DstBytes': [500, 1000],
            'SrcAddr': ['192.168.1.1', '10.0.0.1'],
            'DstAddr': ['172.16.0.1', '192.168.1.100'],
            'Proto': ['TCP', 'UDP']
        })
        
        cleaned = loader.clean(df)
        
        # 應該可以直接用於特徵工程，不會出錯
        try:
            features = extract_features(
                cleaned,
                include_time_features=True,
                time_feature_stage=1
            )
            assert isinstance(features, pd.DataFrame), \
                "特徵工程應該返回 DataFrame"
            assert len(features) == len(cleaned), \
                "特徵工程後的資料行數應該與輸入相同"
        except Exception as e:
            pytest.fail(f"清洗後的資料無法用於特徵工程: {e}")
    
    def test_data_loader_handles_format_changes(self):
        """測試如果輸入格式改變，loader 是否還能正確處理"""
        loader = BidirectionalBinetflowLoader()
        
        # 情境：數值欄位是字串格式（常見情況）
        df_string_numeric = pd.DataFrame({
            'StartTime': ['2011-08-17 12:01:01.780'],
            'Dur': ['3.124'],  # 字串
            'Sport': ['80'],    # 字串
            'Dport': ['443'],   # 字串
            'TotBytes': ['1000'],  # 字串
            'TotPkts': ['10'],  # 字串
            'SrcBytes': ['500']  # 字串
        })
        
        cleaned = loader.clean(df_string_numeric)
        
        # 應該正確轉換為數值型態
        assert pd.api.types.is_numeric_dtype(cleaned['Dur']), \
            "字串格式的 Dur 應該轉換為數值型態"
        assert pd.api.types.is_numeric_dtype(cleaned['Sport']), \
            "字串格式的 Sport 應該轉換為數值型態"
        assert pd.api.types.is_numeric_dtype(cleaned['Dport']), \
            "字串格式的 Dport 應該轉換為數值型態"
        assert pd.api.types.is_numeric_dtype(cleaned['TotBytes']), \
            "字串格式的 TotBytes 應該轉換為數值型態"
        assert pd.api.types.is_numeric_dtype(cleaned['TotPkts']), \
            "字串格式的 TotPkts 應該轉換為數值型態"
        assert pd.api.types.is_numeric_dtype(cleaned['SrcBytes']), \
            "字串格式的 SrcBytes 應該轉換為數值型態"
    
    def test_port_columns_remain_numeric_after_cleaning(self):
        """測試 Port 欄位清洗後仍然是數值型態（不會因為格式改變而變成字串）"""
        loader = BidirectionalBinetflowLoader()
        
        # 測試各種 Port 格式
        df = pd.DataFrame({
            'Sport': ['80', 443, '8080', 'invalid'],  # 混合格式
            'Dport': [22, '80', 443, 99999]
        })
        
        cleaned = loader.clean(df)
        
        # 所有 Port 欄位都應該是數值型態
        assert pd.api.types.is_numeric_dtype(cleaned['Sport']), \
            "Sport 應該是數值型態，即使輸入是混合格式"
        assert pd.api.types.is_numeric_dtype(cleaned['Dport']), \
            "Dport 應該是數值型態，即使輸入是混合格式"
        
        # 無效值應該變成 NaN，而不是字串
        assert pd.isna(cleaned['Sport'].iloc[3]), \
            "無效的 Port 值 'invalid' 應該變成 NaN"
    
    def test_cleaned_data_preserves_network_columns(self):
        """測試清洗後的資料保留網路相關欄位（用於白名單分析）"""
        loader = BidirectionalBinetflowLoader()
        
        df = pd.DataFrame({
            'StartTime': ['2011-08-17 12:01:01.780'],
            'Proto': ['TCP'],
            'Sport': [80],
            'Dport': [443],
            'SrcAddr': ['192.168.1.1'],
            'DstAddr': ['172.16.0.1']
        })
        
        cleaned = loader.clean(df)
        
        # 白名單分析需要的欄位
        whitelist_required = ['Proto', 'Dport', 'SrcAddr', 'DstAddr']
        for col in whitelist_required:
            assert col in cleaned.columns, \
                f"缺少白名單分析所需欄位: {col}"
    
    def test_consistent_output_schema_across_different_inputs(self):
        """測試不同輸入格式下，輸出 Schema 是否一致"""
        loader = BidirectionalBinetflowLoader()
        
        # 測試1：標準格式（數值型態）
        df1 = pd.DataFrame({
            'StartTime': ['2011-08-17 12:01:01.780'],
            'Dur': [3.124],
            'Sport': [80],
            'Dport': [443],
            'TotBytes': [1000],
            'TotPkts': [10]
        })
        cleaned1 = loader.clean(df1)
        
        # 測試2：字串格式
        df2 = pd.DataFrame({
            'StartTime': ['2011-08-17 12:01:01.780'],
            'Dur': ['3.124'],
            'Sport': ['80'],
            'Dport': ['443'],
            'TotBytes': ['1000'],
            'TotPkts': ['10']
        })
        cleaned2 = loader.clean(df2)
        
        # 輸出欄位應該相同
        assert set(cleaned1.columns) == set(cleaned2.columns), \
            "不同輸入格式下，輸出欄位應該相同"
        
        # 輸出型態應該相同
        for col in cleaned1.columns:
            if col in cleaned2.columns:
                assert cleaned1[col].dtype == cleaned2[col].dtype, \
                    f"欄位 {col} 的型態不一致: {cleaned1[col].dtype} vs {cleaned2[col].dtype}"
    
    def test_cleaned_data_schema_for_model_training(self):
        """測試清洗後的資料 Schema 是否適合模型訓練"""
        loader = BidirectionalBinetflowLoader()
        
        df = pd.DataFrame({
            'StartTime': ['2011-08-17 12:01:01.780', '2011-08-17 12:02:01.780'],
            'Dur': [3.124, 5.456],
            'TotBytes': [1000, 2000],
            'TotPkts': [10, 20],
            'SrcBytes': [500, 1000],
            'DstBytes': [500, 1000],
            'Sport': [80, 443],
            'Dport': [8080, 22],
            'SrcAddr': ['192.168.1.1', '10.0.0.1'],
            'DstAddr': ['172.16.0.1', '192.168.1.100'],
            'Proto': ['TCP', 'UDP'],
            'Label': ['Normal', 'Botnet']
        })
        
        cleaned = loader.clean(df)
        
        # 模型訓練需要的數值特徵應該是數值型態
        numeric_features = ['Dur', 'TotBytes', 'TotPkts', 'SrcBytes', 'DstBytes', 'Sport', 'Dport']
        for col in numeric_features:
            if col in cleaned.columns:
                assert pd.api.types.is_numeric_dtype(cleaned[col]), \
                    f"模型訓練需要的數值特徵 {col} 應該是數值型態"
        
        # 標籤欄位應該保留
        if 'Label' in df.columns:
            assert 'Label' in cleaned.columns, \
                "標籤欄位應該被保留"

