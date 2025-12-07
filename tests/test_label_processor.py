"""
測試 label_processor 模組

測試標籤轉換、統計和驗證功能。
"""
import pytest
import pandas as pd
import numpy as np
from src import convert_label_to_binary, get_label_statistics, validate_labels


class TestConvertLabelToBinary:
    """測試 convert_label_to_binary 函數"""
    
    def test_string_labels_with_botnet_keyword(self):
        """測試包含 Botnet 關鍵字的字串標籤轉換"""
        df = pd.DataFrame({
            'Label': ['From-Botnet-V50-1', 'Background-google', 'Normal', 'From-Botnet-V50-2']
        })
        result = convert_label_to_binary(df, verbose=False)
        
        assert result['label_binary'].tolist() == [1, 0, 0, 1]
        assert (result['label_binary'] == 1).sum() == 2
        assert (result['label_binary'] == 0).sum() == 2
    
    def test_numeric_labels(self):
        """測試數值標籤轉換"""
        df = pd.DataFrame({'Label': [0, 1, 0, 1, 0]})
        result = convert_label_to_binary(df)
        
        assert result['label_binary'].tolist() == [0, 1, 0, 1, 0]
        assert (result['label_binary'] == 1).sum() == 2
        assert (result['label_binary'] == 0).sum() == 3
    
    def test_custom_anomaly_keywords(self):
        """測試自訂異常關鍵字"""
        df = pd.DataFrame({
            'Label': ['Malware-A', 'Normal-B', 'Attack-C', 'Safe-D']
        })
        result = convert_label_to_binary(
            df,
            anomaly_keywords=['Malware', 'Attack']
        )
        
        assert result['label_binary'].tolist() == [1, 0, 1, 0]
    
    def test_empty_dataframe(self):
        """測試空 DataFrame"""
        df = pd.DataFrame({'Label': []})
        result = convert_label_to_binary(df)
        
        assert len(result) == 0
        assert 'label_binary' in result.columns
    
    def test_missing_label_column(self):
        """測試缺少標籤欄位的情況"""
        df = pd.DataFrame({'Other': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="標籤欄位 'Label' 不存在"):
            convert_label_to_binary(df)
    
    def test_custom_label_column_name(self):
        """測試自訂標籤欄位名稱"""
        df = pd.DataFrame({
            'CustomLabel': ['Botnet-A', 'Normal-B', 'Botnet-C']
        })
        result = convert_label_to_binary(
            df,
            label_column='CustomLabel',
            output_column='custom_binary'
        )
        
        assert 'custom_binary' in result.columns
        assert result['custom_binary'].tolist() == [1, 0, 1]
    
    def test_nan_labels(self):
        """測試包含 NaN 的標籤"""
        df = pd.DataFrame({
            'Label': ['Botnet-A', None, 'Normal-B', np.nan]
        })
        result = convert_label_to_binary(df)
        
        # NaN 應該被視為正常（0）
        assert result['label_binary'].tolist() == [1, 0, 0, 0]
    
    def test_case_insensitive_matching(self):
        """測試不區分大小寫的關鍵字匹配"""
        df = pd.DataFrame({
            'Label': ['BOTNET-A', 'botnet-B', 'BotNet-C', 'Normal-D']
        })
        result = convert_label_to_binary(df)
        
        assert result['label_binary'].tolist() == [1, 1, 1, 0]


class TestGetLabelStatistics:
    """測試 get_label_statistics 函數"""
    
    def test_basic_statistics(self):
        """測試基本統計資訊"""
        df = pd.DataFrame({
            'Label': ['From-Botnet-V50-1', 'Background-google', 'Normal', 'From-Botnet-V50-2'],
            'label_binary': [1, 0, 0, 1]
        })
        stats = get_label_statistics(df)
        
        assert stats['total_count'] == 4
        assert stats['normal_count'] == 2
        assert stats['anomaly_count'] == 2
        assert stats['anomaly_ratio'] == 0.5
        assert stats['normal_ratio'] == 0.5
        assert 'From-Botnet-V50-1' in stats['label_distribution']
    
    def test_statistics_without_binary_column(self):
        """測試沒有二元標籤欄位的情況"""
        df = pd.DataFrame({
            'Label': ['A', 'B', 'C']
        })
        stats = get_label_statistics(df, binary_column=None)
        
        assert stats['total_count'] == 3
        assert 'normal_count' not in stats
        assert 'anomaly_count' not in stats
        assert 'label_distribution' in stats
    
    def test_empty_dataframe_statistics(self):
        """測試空 DataFrame 的統計"""
        df = pd.DataFrame({'Label': [], 'label_binary': []})
        stats = get_label_statistics(df)
        
        assert stats['total_count'] == 0
        assert stats['normal_count'] == 0
        assert stats['anomaly_count'] == 0
        assert stats['anomaly_ratio'] == 0.0
    
    def test_all_anomaly_statistics(self):
        """測試全部為異常的情況"""
        df = pd.DataFrame({
            'Label': ['Botnet-A', 'Botnet-B', 'Botnet-C'],
            'label_binary': [1, 1, 1]
        })
        stats = get_label_statistics(df)
        
        assert stats['anomaly_count'] == 3
        assert stats['normal_count'] == 0
        assert stats['anomaly_ratio'] == 1.0
        assert stats['normal_ratio'] == 0.0


class TestValidateLabels:
    """測試 validate_labels 函數"""
    
    def test_valid_labels(self):
        """測試有效的標籤"""
        df = pd.DataFrame({
            'Label': ['From-Botnet-V50-1', 'Background-google', 'Normal', 'From-Botnet-V50-2'],
            'label_binary': [1, 0, 0, 1]
        })
        is_valid, messages = validate_labels(df, min_anomaly_ratio=0.1)
        
        assert is_valid is True
        assert len(messages) == 0
    
    def test_low_anomaly_ratio(self):
        """測試異常比例過低的情況"""
        df = pd.DataFrame({
            'Label': ['Background-google', 'Normal', 'Background-facebook'],
            'label_binary': [0, 0, 0]
        })
        is_valid, messages = validate_labels(df, min_anomaly_ratio=0.1)
        
        assert is_valid is False
        assert len(messages) > 0
        assert any('異常樣本比例過低' in msg for msg in messages)
        assert any('沒有異常樣本' in msg for msg in messages)
    
    def test_high_anomaly_ratio(self):
        """測試異常比例過高的情況"""
        df = pd.DataFrame({
            'Label': ['Botnet-A', 'Botnet-B', 'Botnet-C'],
            'label_binary': [1, 1, 1]
        })
        is_valid, messages = validate_labels(df, max_anomaly_ratio=0.5)
        
        assert is_valid is False
        assert any('異常樣本比例過高' in msg for msg in messages)
    
    def test_missing_label_column(self):
        """測試缺少標籤欄位"""
        df = pd.DataFrame({'Other': [1, 2, 3]})
        is_valid, messages = validate_labels(df)
        
        assert is_valid is False
        assert any('缺少標籤欄位' in msg for msg in messages)
    
    def test_missing_binary_column(self):
        """測試缺少二元標籤欄位"""
        df = pd.DataFrame({
            'Label': ['A', 'B', 'C']
        })
        is_valid, messages = validate_labels(df, binary_column='label_binary')
        
        assert is_valid is False
        assert any('缺少二元標籤欄位' in msg for msg in messages)
    
    def test_no_anomaly_samples(self):
        """測試沒有異常樣本的情況"""
        df = pd.DataFrame({
            'Label': ['Background-google', 'Normal'],
            'label_binary': [0, 0]
        })
        is_valid, messages = validate_labels(df)
        
        assert is_valid is False
        assert any('沒有異常樣本' in msg for msg in messages)
    
    def test_empty_dataframe_validation(self):
        """測試空 DataFrame 的驗證"""
        df = pd.DataFrame({'Label': [], 'label_binary': []})
        is_valid, messages = validate_labels(df, min_anomaly_ratio=0.1)
        
        assert is_valid is False
        assert any('異常樣本比例過低' in msg or '沒有異常樣本' in msg for msg in messages)
    
    @pytest.mark.parametrize("anomaly_count,total_count,min_ratio,expected_valid", [
        (10, 100, 0.05, True),   # 10% > 5%，有效
        (3, 100, 0.05, False),   # 3% < 5%，無效
        (50, 100, 0.3, False),   # 50% > 30%，無效
        (20, 100, 0.1, True),    # 20% 在範圍內，有效
    ])
    def test_anomaly_ratio_validation(self, anomaly_count, total_count, min_ratio, expected_valid):
        """參數化測試異常比例驗證"""
        normal_count = total_count - anomaly_count
        df = pd.DataFrame({
            'Label': ['A'] * total_count,
            'label_binary': [1] * anomaly_count + [0] * normal_count
        })
        is_valid, _ = validate_labels(df, min_anomaly_ratio=min_ratio, max_anomaly_ratio=0.3)
        
        assert is_valid == expected_valid

