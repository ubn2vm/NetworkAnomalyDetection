"""資料品質檢查模組的測試"""

import pytest
import pandas as pd
import numpy as np
from network_anomaly.data_quality import (
    check_missing_values,
    check_data_types,
    check_extreme_values,
    check_negative_values,
    generate_quality_report,
)


class TestDataQuality:
    """資料品質檢查功能的測試類別"""
    
    @pytest.fixture
    def sample_df(self):
        """建立測試用的 DataFrame（包含缺失值和負值）"""
        data = {
            "Dur": [1.5, None, 2.0, 3.0],  # 缺失值
            "TotPkts": [10, 20, -5, 30],  # 負值
            "TotBytes": [1024, 2048, 512, 4096],
            "SrcBytes": [512, 1024, 256, 2048],
            "Label": ["LEGIT", "Botnet", "LEGIT", "Normal"],
        }
        return pd.DataFrame(data)
    
    def test_check_missing_values(self, sample_df):
        """測試檢查缺失值"""
        missing = check_missing_values(sample_df)
        assert "Dur" in missing
        assert missing["Dur"] == 1  # 有一筆缺失值
        assert missing["TotPkts"] == 0  # 沒有缺失值
    
    def test_check_data_types(self, sample_df):
        """測試檢查資料型別"""
        types = check_data_types(sample_df)
        assert "Dur" in types
        assert "float" in types["Dur"].lower()
        assert "TotPkts" in types
        assert "int" in types["TotPkts"].lower()
    
    def test_check_extreme_values(self, sample_df):
        """測試檢查極端值"""
        numeric_cols = ["Dur", "TotPkts", "TotBytes", "SrcBytes"]
        extremes = check_extreme_values(sample_df, numeric_cols)
        
        assert "TotPkts" in extremes
        assert extremes["TotPkts"]["min"] == -5
        assert extremes["TotPkts"]["max"] == 30
    
    def test_check_negative_values(self, sample_df):
        """測試檢查負值"""
        numeric_cols = ["TotPkts", "TotBytes", "SrcBytes"]
        negatives = check_negative_values(sample_df, numeric_cols)
        
        assert "TotPkts" in negatives
        assert negatives["TotPkts"] == 1  # 有一筆負值
    
    def test_generate_quality_report(self, sample_df):
        """測試產生完整品質報告"""
        report = generate_quality_report(sample_df)
        
        assert "total_rows" in report
        assert report["total_rows"] == 4
        assert "missing_values" in report
        assert "extreme_values" in report
        assert "negative_values" in report
