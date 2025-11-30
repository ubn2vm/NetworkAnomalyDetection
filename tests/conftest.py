"""Pytest 設定檔 - 確保測試可以找到專案模組"""

import sys
from pathlib import Path

# 將專案根目錄的 src 資料夾加入 Python 路徑
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
