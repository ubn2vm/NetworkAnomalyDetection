"""標籤處理模組的測試"""

import pytest
import pandas as pd
from network_anomaly.label_processor import binarize_labels, filter_ambiguous_labels


class TestLabelProcessor:
    """標籤處理功能的測試類別"""
    
    @pytest.fixture
    def sample_df(self):
        """建立測試用的 DataFrame"""
        data = {
            "SrcAddr": ["192.168.1.1", "192.168.1.2", "192.168.1.3", "192.168.1.4", "192.168.1.5"],
            "Label": ["LEGIT", "Background", "Botnet", "C&C", "Normal"],
        }
        return pd.DataFrame(data)
    
    def test_binarize_labels_legit(self, sample_df):
        """測試 LEGIT 標籤二元化為 0"""
        result = binarize_labels(sample_df)
        legit_row = result[result["SrcAddr"] == "192.168.1.1"]
        assert len(legit_row) > 0
        assert legit_row.iloc[0]["binary_label"] == 0
    
    def test_binarize_labels_botnet(self, sample_df):
        """測試 Botnet 標籤二元化為 1"""
        result = binarize_labels(sample_df)
        botnet_row = result[result["SrcAddr"] == "192.168.1.3"]
        assert len(botnet_row) > 0
        assert botnet_row.iloc[0]["binary_label"] == 1
    
    def test_binarize_labels_cc(self, sample_df):
        """測試 C&C 標籤二元化為 1"""
        result = binarize_labels(sample_df)
        cc_row = result[result["SrcAddr"] == "192.168.1.4"]
        assert len(cc_row) > 0
        assert cc_row.iloc[0]["binary_label"] == 1
    
    def test_binarize_labels_background(self, sample_df):
        """測試 Background 標籤二元化為 0"""
        result = binarize_labels(sample_df)
        bg_row = result[result["SrcAddr"] == "192.168.1.2"]
        assert len(bg_row) > 0
        assert bg_row.iloc[0]["binary_label"] == 0
    
    def test_filter_ambiguous_labels_keep(self, sample_df):
        """測試保留不明確標籤"""
        result = filter_ambiguous_labels(sample_df, keep_ambiguous=True)
        assert len(result) == len(sample_df)
    
    def test_filter_ambiguous_labels_remove(self, sample_df):
        """測試過濾不明確標籤"""
        result = filter_ambiguous_labels(sample_df, keep_ambiguous=False)
        # Background 應該被過濾掉
        assert len(result) < len(sample_df)
        # 檢查 Background 是否被過濾
        bg_rows = result[result["SrcAddr"] == "192.168.1.2"]
        assert len(bg_rows) == 0
