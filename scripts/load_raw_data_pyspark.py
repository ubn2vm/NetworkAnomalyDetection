import pandas as pd
from pathlib import Path
import sys

# 設定檔案路徑
script_dir = Path(__file__).parent
project_root = script_dir.parent
file_path = project_root / "data" / "raw" / "capture20110817.pcap.netflow.labeled"

# 1. 修改欄位定義：將 Date_Flow_Start 拆成 Date 和 Time，並加入 Arrow
column_names = [
    "Date",         # 2011-08-17
    "Time",         # 12:01:01.780
    "Duration",     # 3.124
    "Prot",         # UDP
    "Src_IP_Port",  # 188.75.133.98:16200
    "Arrow",        # -> 
    "Dst_IP_Port",  # 147.32.86.125:35248
    "Flags",        # INT
    "Tos",          # 0
    "Packets",      # 304
    "Bytes",        # 219158
    "Flows",        # 1
    "Label"         # Background
]

try:
    print(f"正在讀取檔案: {file_path}")
    
    # 2. 讀取資料
    # 使用 sep=r'\s+' 處理不規則的空白，但配合新的 column_names 讓它正確落位
    df = pd.read_csv(
        str(file_path),
        sep=r'\s+',          
        names=column_names,
        header=0,            
        index_col=False,     
        engine='python',
        on_bad_lines='skip'
    )

    # 3. 資料清洗與合併
    print("正在清洗資料...")

    # (A) 合併 Date 和 Time 成為 Date_Flow_Start
    # 確保是字串格式以免報錯
    df['Date'] = df['Date'].astype(str)
    df['Time'] = df['Time'].astype(str)
    
    # 合併字串
    df['Date_Flow_Start'] = df['Date'] + ' ' + df['Time']
    
    # 轉換為 datetime 物件
    df['Date_Flow_Start'] = pd.to_datetime(df['Date_Flow_Start'], errors='coerce')

    # (B) 刪除多餘欄位 (Date, Time, Arrow)
    cols_to_drop = ['Date', 'Time', 'Arrow']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # (C) 調整欄位順序 (把 Date_Flow_Start 放到最前面)
    cols = ['Date_Flow_Start'] + [c for c in df.columns if c != 'Date_Flow_Start']
    df = df[cols]

    # (D) 數值型別轉換
    numeric_cols = ['Duration', 'Tos', 'Packets', 'Bytes', 'Flows']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. 檢視結果
    print("資料讀取成功！前 5 筆資料：")
    # 設定顯示選項以免換行太亂
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head())
    print("-" * 30)
    print(f"資料維度: {df.shape}")
    print(f"欄位型態:\n{df.dtypes}")

    # 5. 儲存
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "capture20110817_cleaned_spark.parquet"
    
    # 在 load_raw_data.py 第 85 行附近，修改為：
    print(f"\n正在儲存至: {output_path}")

    # 使用 Spark 相容的時間戳記格式
    # 將 datetime 轉換為字串，讓 Spark 可以正確讀取
    df_to_save = df.copy()
    if 'Date_Flow_Start' in df_to_save.columns:
        # 轉換為字串格式（Spark 可以讀取）
        df_to_save['Date_Flow_Start'] = df_to_save['Date_Flow_Start'].astype(str)

    df_to_save.to_parquet(
        output_path, 
        engine='pyarrow', 
        index=False,
        coerce_timestamps='us'  # 使用微秒精度
    )
    print("✅ 資料儲存成功！")

except Exception as e:
    print(f"❌ 發生錯誤: {e}")
    import traceback
    traceback.print_exc()