"""
pytest 配置和共享 fixture

提供測試中常用的 fixture 和配置。
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def sample_netflow_data():
    """提供範例 NetFlow 資料"""
    return pd.DataFrame({
        'StartTime': [
            '2011-08-17 12:01:01.780',
            '2011-08-17 12:02:01.780',
            '2011-08-17 12:03:01.780'
        ],
        'Dur': [3.124, 5.456, 1.234],
        'Proto': ['TCP', 'UDP', 'TCP'],
        'SrcAddr': ['192.168.1.1', '10.0.0.1', '172.16.0.1'],
        'Sport': ['80', '443', '22'],
        'DstAddr': ['172.16.0.1', '192.168.1.100', '10.0.0.2'],
        'Dport': ['8080', '22', '80'],
        'TotBytes': [1000, 2000, 500],
        'TotPkts': [10, 20, 5],
        'SrcBytes': [500, 1000, 250],
        'Label': ['Background-google', 'From-Botnet-V50-1', 'Normal']
    })


@pytest.fixture
def sample_labels():
    """提供範例標籤資料"""
    return pd.DataFrame({
        'Label': [
            'From-Botnet-V50-1',
            'Background-google',
            'Normal',
            'From-Botnet-V50-2',
            'Background-facebook'
        ]
    })


@pytest.fixture
def sample_features():
    """提供範例特徵資料（長尾分佈）"""
    return pd.DataFrame({
        'TotBytes': [100, 1000, 10000, 100000, 1000000],
        'SrcBytes': [50, 500, 5000, 50000, 500000],
        'Dur': [1.0, 2.0, 3.0, 4.0, 5.0],
        'TotPkts': [10, 20, 30, 40, 50],
        'hour': [9, 10, 11, 12, 13]  # 不需要轉換的特徵
    })


@pytest.fixture
def temp_dir():
    """提供臨時目錄"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

