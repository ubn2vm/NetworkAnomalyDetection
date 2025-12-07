"""
首次載入資料並生成 Parquet 檔案

此腳本用於首次載入資料，從 CSV 讀取並自動生成 Parquet 檔案以供後續快速載入。
適合在第一次使用時執行，或需要重新生成 Parquet 檔案時使用。
"""
import time
import sys
from pathlib import Path

# 將專案根目錄加入 Python 路徑
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DataLoaderFactory, DataSourceType, get_project_root


def format_time(seconds):
    """格式化時間顯示"""
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} 分 {secs:.2f} 秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} 小時 {minutes} 分 {secs:.2f} 秒"


def format_size(bytes_size):
    """格式化檔案大小顯示"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def main():
    print("=" * 70)
    print("首次載入資料並生成 Parquet 檔案")
    print("=" * 70)
    
    project_root = get_project_root()
    csv_path = project_root / "data" / "raw" / "capture20110817.binetflow"
    parquet_path = project_root / "data" / "processed" / "capture20110817_cleaned_spark.parquet"
    
    # 1. 檢查 CSV 檔案
    print("\n[步驟 1] 檢查原始 CSV 檔案...")
    if not csv_path.exists():
        print(f"❌ 錯誤：找不到 CSV 檔案: {csv_path}")
        print("   請確認檔案是否存在")
        return 1
    
    csv_size = csv_path.stat().st_size
    print(f"✅ CSV 檔案存在: {csv_path}")
    print(f"   檔案大小: {format_size(csv_size)}")
    
    # 2. 檢查 Parquet 檔案
    print("\n[步驟 2] 檢查 Parquet 檔案...")
    if parquet_path.exists():
        parquet_size = parquet_path.stat().st_size
        print(f"⚠️  Parquet 檔案已存在: {parquet_path}")
        print(f"   檔案大小: {format_size(parquet_size)}")
        
        # 檢查檔案是否為空或損壞
        if parquet_size == 0:
            print(f"   ⚠️  檔案大小為 0，可能是損壞的檔案")
            print(f"   將自動刪除並重新生成")
            try:
                if parquet_path.is_dir():
                    import shutil
                    shutil.rmtree(parquet_path)
                else:
                    parquet_path.unlink()
                print("✅ 已刪除損壞的 Parquet 檔案")
            except Exception as e:
                print(f"❌ 刪除失敗: {e}")
                return 1
        else:
            # 只有當檔案大小不為 0 時才計算壓縮比
            compression_ratio = csv_size / parquet_size if parquet_size > 0 else 0
            print(f"   壓縮比: {compression_ratio:.2f}:1")
            
            response = input("\n是否要重新生成 Parquet 檔案？(y/N): ").strip().lower()
            if response != 'y':
                print("✅ 取消操作，使用現有的 Parquet 檔案")
                print("   如需載入資料，請使用其他腳本或 notebook")
                return 0
            
            # 刪除現有的 Parquet 檔案
            print("\n🗑️  刪除現有的 Parquet 檔案...")
            try:
                if parquet_path.is_dir():
                    import shutil
                    shutil.rmtree(parquet_path)
                else:
                    parquet_path.unlink()
                print("✅ 已刪除舊的 Parquet 檔案")
            except Exception as e:
                print(f"❌ 刪除失敗: {e}")
                return 1
    else:
        print(f"ℹ️  Parquet 檔案不存在，將從 CSV 生成")
    
    # 3. 檢查依賴
    print("\n[步驟 3] 檢查依賴...")
    try:
        import pyarrow
        print(f"✅ pyarrow 已安裝，版本: {pyarrow.__version__}")
    except ImportError:
        print("❌ pyarrow 未安裝")
        print("   請執行: pip install pyarrow")
        return 1
    
    try:
        import pyspark
        print("✅ pyspark 已安裝")
    except ImportError:
        print("❌ pyspark 未安裝")
        print("   請執行: pip install pyspark")
        return 1
    
    # 4. 確保目錄存在
    print("\n[步驟 4] 準備目錄...")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"✅ 目錄已準備: {parquet_path.parent}")
    
    # 5. 載入資料（強制從 CSV 讀取）
    print("\n[步驟 5] 載入資料（從 CSV 讀取）...")
    print("   這可能需要一些時間，請耐心等待...")
    print("   使用 PySpark 多核心加速處理...")
    
    start_time = time.time()
    
    try:
        # 創建 Spark 載入器
        loader = DataLoaderFactory.create(DataSourceType.BIDIRECTIONAL_BINETFLOW_SPARK)
        
        # 強制從 CSV 讀取（即使 Parquet 存在也會重新生成）
        # 使用 use_parquet=False 可以跳過 Parquet 檢查，但我們已經刪除了
        # 所以這裡使用 use_parquet=True，它會自動從 CSV 讀取並生成 Parquet
        raw_df = loader.load(file_path=csv_path, use_parquet=True)
        
        load_time = time.time() - start_time
        
        print(f"\n✅ 資料載入完成")
        print(f"   資料筆數: {len(raw_df):,}")
        print(f"   資料欄位: {len(raw_df.columns)}")
        print(f"   載入時間: {format_time(load_time)}")
        
        # 顯示資料資訊
        print(f"\n   資料欄位列表:")
        for i, col in enumerate(raw_df.columns[:10], 1):
            print(f"     {i:2d}. {col}")
        if len(raw_df.columns) > 10:
            print(f"     ... 還有 {len(raw_df.columns) - 10} 個欄位")
        
    except Exception as e:
        print(f"\n❌ 載入失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 6. 驗證 Parquet 檔案
    print("\n[步驟 6] 驗證 Parquet 檔案...")
    if parquet_path.exists():
        parquet_size = parquet_path.stat().st_size
        
        if parquet_size == 0:
            print(f"⚠️  Parquet 檔案大小為 0，可能生成失敗")
            print(f"   請檢查錯誤訊息或重新執行腳本")
        else:
            compression_ratio = csv_size / parquet_size if parquet_size > 0 else 0
            space_saved = csv_size - parquet_size
            space_saved_pct = (1 - parquet_size/csv_size)*100 if csv_size > 0 else 0
            
            print(f"✅ Parquet 檔案已生成: {parquet_path}")
            print(f"   檔案大小: {format_size(parquet_size)}")
            if compression_ratio > 0:
                print(f"   壓縮比: {compression_ratio:.2f}:1")
                print(f"   節省空間: {format_size(space_saved)} ({space_saved_pct:.1f}%)")
            
            # 測試讀取
            try:
                import pandas as pd
                test_df = pd.read_parquet(parquet_path, engine='pyarrow')
                print(f"   驗證讀取: ✅ 成功 ({len(test_df):,} 筆資料)")
            except Exception as e:
                print(f"   驗證讀取: ❌ 失敗 - {e}")
                print(f"   檔案可能損壞，建議刪除後重新生成")
    else:
        print(f"⚠️  Parquet 檔案未生成，可能儲存失敗")
    
    # 7. 清洗資料（可選）
    print("\n[步驟 7] 清洗資料（可選）...")
    response = input("是否要清洗資料？(Y/n): ").strip().lower()
    if response != 'n':
        print("   正在清洗資料...")
        clean_start = time.time()
        cleaned_df = loader.clean(raw_df)
        clean_time = time.time() - clean_start
        
        print(f"✅ 清洗完成")
        print(f"   資料筆數: {len(cleaned_df):,}")
        print(f"   清洗時間: {format_time(clean_time)}")
        
        # 檢查 StartTime 是否已轉換為 datetime
        if 'StartTime' in cleaned_df.columns:
            if cleaned_df['StartTime'].dtype.name.startswith('datetime'):
                print(f"   ✅ StartTime 已轉換為 datetime 類型")
            else:
                print(f"   ⚠️  StartTime 類型: {cleaned_df['StartTime'].dtype}")
    else:
        print("   跳過清洗步驟")
    
    # 總結
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("✅ 首次載入完成！")
    print("=" * 70)
    print(f"總耗時: {format_time(total_time)}")
    print(f"CSV 檔案: {csv_path}")
    print(f"Parquet 檔案: {parquet_path}")
    print("\n💡 提示：")
    print("   - 下次載入時，Parquet 檔案會自動使用，速度會快很多（約 5-10 秒）")
    print("   - 可以在 notebook 或其他腳本中使用 Spark 載入器載入資料")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

