"""
Network Anomaly Detection - 主程式入口

整合所有腳本，提供統一的執行介面。
支援完整流程執行：資料載入 → 模型訓練 → 白名單後處理 → 報告生成。

使用方法：
    python main.py --pipeline full
    python main.py --pipeline full --force-reload
    python main.py --train unsupervised
    python main.py --postprocess
    python main.py --report
"""
import sys
import time
from pathlib import Path
from typing import Tuple

# 將專案根目錄加入 Python 路徑
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str, width: int = 70):
    """打印標題"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def check_parquet_exists() -> Tuple[bool, bool]:
    """
    檢查 Parquet 檔案是否存在且有效
    
    Returns:
        (exists, is_valid): (是否存在, 是否有效)
    """
    parquet_path = Path("data/processed/capture20110817_cleaned_spark.parquet")
    
    if not parquet_path.exists():
        return False, False
    
    # 檢查檔案大小（0 表示可能損壞）
    try:
        size = parquet_path.stat().st_size
        return True, size > 0
    except Exception:
        return True, False


def load_data_if_needed(force: bool = False) -> bool:
    """
    如果需要，載入資料
    
    Args:
        force: 是否強制重新載入
    
    Returns:
        bool: 是否成功
    """
    exists, is_valid = check_parquet_exists()
    
    if exists and is_valid and not force:
        print("   ✅ Parquet 檔案已存在，跳過資料載入")
        print(f"   📍 檔案位置：data/processed/capture20110817_cleaned_spark.parquet")
        return True
    
    if exists and not is_valid:
        print("   ⚠️  Parquet 檔案損壞（大小為 0），將重新載入...")
    elif force:
        print("   🔄 強制重新載入資料...")
    else:
        print("   ⚠️  Parquet 檔案不存在，執行資料載入...")
        print("   💡 這可能需要一些時間，請耐心等待...")
    
    try:
        from scripts.load_data_first_time import main as load_data
        result = load_data()
        return result == 0
    except KeyboardInterrupt:
        print("\n   ⚠️  使用者中斷資料載入")
        return False
    except Exception as e:
        print(f"\n   ❌ 資料載入失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="網路異常檢測系統 - 主程式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
執行範例：
  # 完整流程（自動檢查 Parquet，存在則跳過載入）
  python main.py --pipeline full
  
  # 強制重新載入資料
  python main.py --pipeline full --force-reload
  
  # 僅訓練無監督模型
  python main.py --train unsupervised
  
  # 僅執行白名單後處理
  python main.py --postprocess
  
  # 僅生成報告
  python main.py --report
        """
    )
    
    # 主要模式選項
    parser.add_argument(
        "--pipeline",
        choices=["full"],
        help="執行完整流程（資料載入 → 訓練 → 後處理 → 報告）"
    )
    
    # 訓練選項
    parser.add_argument(
        "--train",
        choices=["unsupervised", "supervised", "both"],
        help="訓練模式：無監督、監督、或兩者"
    )
    
    # 其他選項
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="執行白名單後處理（需要先執行訓練）"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="生成 HTML 報告"
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="強制重新載入資料（即使 Parquet 存在）"
    )
    
    args = parser.parse_args()
    
    # 如果沒有任何參數，顯示幫助
    if not any([args.pipeline, args.train, args.postprocess, args.report]):
        parser.print_help()
        return 1
    
    total_start = time.time()
    
    try:
        # 模式 1: 完整流程
        if args.pipeline == "full":
            print_header("完整流程執行")
            
            # 1. 資料準備（自動檢查，不會重複處理）
            print("\n[階段 1] 資料準備...")
            stage1_start = time.time()
            if not load_data_if_needed(force=args.force_reload):
                print("\n❌ 資料載入失敗，無法繼續執行")
                return 1
            stage1_time = time.time() - stage1_start
            print(f"   ⏱️  階段 1 耗時：{stage1_time:.2f} 秒")
            
            # 2. 訓練無監督模型
            print("\n[階段 2] 無監督模型訓練...")
            stage2_start = time.time()
            try:
                from scripts.train_unsupervised import main as train_unsupervised
                train_unsupervised()
            except KeyboardInterrupt:
                print("\n   ⚠️  使用者中斷訓練")
                return 1
            except Exception as e:
                print(f"\n   ❌ 模型訓練失敗：{e}")
                import traceback
                traceback.print_exc()
                return 1
            stage2_time = time.time() - stage2_start
            print(f"   ⏱️  階段 2 耗時：{stage2_time:.2f} 秒 ({stage2_time/60:.2f} 分鐘)")
            
            # 3. 白名單後處理
            print("\n[階段 3] 白名單後處理...")
            stage3_start = time.time()
            try:
                from scripts.postprocess_with_whitelist import main as postprocess
                postprocess()
            except KeyboardInterrupt:
                print("\n   ⚠️  使用者中斷後處理")
                return 1
            except Exception as e:
                print(f"\n   ❌ 白名單後處理失敗：{e}")
                import traceback
                traceback.print_exc()
                return 1
            stage3_time = time.time() - stage3_start
            print(f"   ⏱️  階段 3 耗時：{stage3_time:.2f} 秒")
            
            # 4. 生成報告
            print("\n[階段 4] 生成報告...")
            stage4_start = time.time()
            try:
                # 保存原始 sys.argv，然後修改為 generate_report.py 需要的參數
                original_argv = sys.argv.copy()
                sys.argv = ['generate_report.py']  # 只保留腳本名稱，移除其他參數
                
                from scripts.generate_report import main as generate_report
                generate_report()
                
                # 恢復原始 sys.argv
                sys.argv = original_argv
            except KeyboardInterrupt:
                print("\n   ⚠️  使用者中斷報告生成")
                sys.argv = original_argv if 'original_argv' in locals() else sys.argv
                return 1
            except Exception as e:
                print(f"\n   ❌ 報告生成失敗：{e}")
                sys.argv = original_argv if 'original_argv' in locals() else sys.argv
                import traceback
                traceback.print_exc()
                return 1
            stage4_time = time.time() - stage4_start
            print(f"   ⏱️  階段 4 耗時：{stage4_time:.2f} 秒")
        
        # 模式 2: 僅訓練
        elif args.train:
            print_header(f"模型訓練：{args.train}")
            
            # 自動檢查 Parquet（不會重複處理）
            exists, is_valid = check_parquet_exists()
            if not exists or not is_valid:
                print("\n⚠️  Parquet 檔案不存在或損壞，請先執行資料載入")
                print("   執行: python main.py --pipeline full --force-reload")
                return 1
            
            if args.train in ["unsupervised", "both"]:
                print("\n[訓練] 無監督模型（Isolation Forest）...")
                try:
                    from scripts.train_unsupervised import main as train_unsupervised
                    train_unsupervised()
                except KeyboardInterrupt:
                    print("\n   ⚠️  使用者中斷訓練")
                    return 1
                except Exception as e:
                    print(f"\n   ❌ 訓練失敗：{e}")
                    import traceback
                    traceback.print_exc()
                    return 1
            
            if args.train in ["supervised", "both"]:
                print("\n[訓練] 監督模型（XGBoost）...")
                try:
                    from scripts.train_supervised import main as train_supervised
                    train_supervised()
                except KeyboardInterrupt:
                    print("\n   ⚠️  使用者中斷訓練")
                    return 1
                except Exception as e:
                    print(f"\n   ❌ 訓練失敗：{e}")
                    import traceback
                    traceback.print_exc()
                    return 1
        
        # 模式 3: 僅後處理
        elif args.postprocess:
            print_header("白名單後處理")
            
            # 檢查訓練結果是否存在
            training_dir = Path("data/models/unsupervised_training")
            if not training_dir.exists():
                print("\n❌ 找不到訓練結果目錄")
                print(f"   預期位置：{training_dir}")
                print("   請先執行訓練：python main.py --train unsupervised")
                return 1
            
            try:
                from scripts.postprocess_with_whitelist import main as postprocess
                postprocess()
            except KeyboardInterrupt:
                print("\n   ⚠️  使用者中斷後處理")
                return 1
            except Exception as e:
                print(f"\n   ❌ 後處理失敗：{e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # 模式 4: 僅報告
        elif args.report:
            print_header("生成報告")
            try:
                # 保存原始 sys.argv，然後修改為 generate_report.py 需要的參數
                original_argv = sys.argv.copy()
                sys.argv = ['generate_report.py']  # 只保留腳本名稱，移除其他參數
                
                from scripts.generate_report import main as generate_report
                generate_report()
                
                # 恢復原始 sys.argv
                sys.argv = original_argv
            except KeyboardInterrupt:
                print("\n   ⚠️  使用者中斷報告生成")
                sys.argv = original_argv if 'original_argv' in locals() else sys.argv
                return 1
            except Exception as e:
                print(f"\n   ❌ 報告生成失敗：{e}")
                sys.argv = original_argv if 'original_argv' in locals() else sys.argv
                import traceback
                traceback.print_exc()
                return 1
        
        # 總結
        total_time = time.time() - total_start
        print_header("執行完成")
        print(f"總執行時間：{total_time:.2f} 秒 ({total_time/60:.2f} 分鐘)")
        
        if args.pipeline == "full":
            print("\n📊 各階段耗時統計：")
            if 'stage1_time' in locals():
                print(f"   階段 1（資料準備）：{stage1_time:.2f} 秒")
            if 'stage2_time' in locals():
                print(f"   階段 2（模型訓練）：{stage2_time:.2f} 秒 ({stage2_time/60:.2f} 分鐘)")
            if 'stage3_time' in locals():
                print(f"   階段 3（白名單後處理）：{stage3_time:.2f} 秒")
            if 'stage4_time' in locals():
                print(f"   階段 4（報告生成）：{stage4_time:.2f} 秒")
        
        print("\n💡 提示：")
        print("   - 訓練結果保存在：data/models/unsupervised_training/")
        print("   - 報告位置：output/reports/ 或 output/visualizations/")
        print("   - 如需重新執行，使用：python main.py --pipeline full")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷執行")
        return 1
    except Exception as e:
        print(f"\n\n❌ 執行失敗：{e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

