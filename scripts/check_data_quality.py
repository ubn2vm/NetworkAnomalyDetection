import pandas as pd
from pathlib import Path

# 設定路徑
script_dir = Path(__file__).parent
project_root = script_dir.parent
file_path = project_root / "data" / "processed" / "capture20110817_cleaned_spark.parquet"

def check_data_quality():
    if not file_path.exists():
        print(f"❌ 找不到檔案: {file_path}")
        return

    print(f"🔍 正在讀取檔案: {file_path} ...")
    df = pd.read_parquet(file_path)
    
    print("-" * 50)
    print("【1. 基本資訊】")
    print(f"總筆數: {len(df)}")
    print(f"資料欄位: {df.columns.tolist()}")
    print("\n資料型別 (Dtypes):")
    print(df.dtypes)

    print("-" * 50)
    print("【2. 缺失值檢查 (Missing Values)】")
    # 計算每一欄有多少 NaN
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("✅ 完美！沒有任何缺失值。")
    else:
        print("⚠️ 警告：發現缺失值！")
        print(null_counts[null_counts > 0])
        
    print("-" * 50)
    print("【3. 關鍵欄位內容檢查 (Alignment Check)】")
    # 檢查 Label 欄位是否乾淨 (這是最容易發現錯位的地方)
    # 正常應該是 'Background', 'Legacy', 'Botnet' 等文字
    # 如果出現數字或 IP，代表前面的欄位又錯位了
    print("Label 欄位的前 10 種最常見值：")
    print(df['Label'].value_counts().head(10))
    
    print("\nProt (通訊協定) 分佈：")
    print(df['Prot'].value_counts().head(5))

    print("-" * 50)
    print("【4. 數值邏輯檢查 (Logic Check)】")
    
    # 檢查負數時間
    neg_duration = df[df['Duration'] < 0]
    print(f"Duration < 0 的筆數: {len(neg_duration)}")
    
    # 檢查 Bytes 或 Packets 為 0 (雖然有可能發生，但值得注意)
    zero_bytes = df[df['Bytes'] == 0]
    print(f"Bytes == 0 的筆數: {len(zero_bytes)}")
    
    # 檢查 IP 格式 (簡單抽樣)
    sample_src = df['Src_IP_Port'].iloc[0] if len(df) > 0 else "N/A"
    print(f"\n隨機抽樣 IP 欄位內容 (應包含 ':' ):")
    print(f"Src: {sample_src}")
    
    # 檢查時間範圍
    if 'Date_Flow_Start' in df.columns:
        print(f"\n時間範圍: {df['Date_Flow_Start'].min()} 到 {df['Date_Flow_Start'].max()}")

    print("-" * 50)

if __name__ == "__main__":
    check_data_quality()