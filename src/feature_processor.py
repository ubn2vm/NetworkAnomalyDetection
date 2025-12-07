"""
特徵處理器模組

使用類似 BaseDataLoader 的設計模式，統一管理特徵工程和特徵轉換的流程。
支援特徵的儲存和載入，避免重複計算 PySpark 特徵。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from abc import ABC, abstractmethod
import pickle
import json
from datetime import datetime

from src.feature_engineer import extract_features
from src.feature_transformer import (
    transform_features_for_unsupervised,
    DEFAULT_SKEWED_FEATURES
)
from src.feature_selector import FeatureSelector, FeatureSelectionStrategy, prepare_feature_set
from src.label_processor import convert_label_to_binary
from src.data_loader import get_project_root


class BaseFeatureProcessor(ABC):
    """
    特徵處理器抽象基類
    
    定義所有特徵處理器必須實作的統一介面。
    遵循與 BaseDataLoader 相同的設計模式。
    
    >>> from src.feature_processor import BaseFeatureProcessor
    >>> # BaseFeatureProcessor 是抽象類別，不能直接實例化
    """
    
    @abstractmethod
    def extract(
        self,
        df: pd.DataFrame,
        include_time_features: bool = True,
        time_feature_stage: int = 1
    ) -> pd.DataFrame:
        """
        提取特徵。
        
        >>> import pandas as pd
        >>> processor = StandardFeatureProcessor()
        >>> df = pd.DataFrame({
        ...     'TotBytes': [100, 200, 300],
        ...     'SrcBytes': [50, 100, 150],
        ...     'StartTime': pd.to_datetime(['2021-08-17 12:00:00', '2021-08-17 12:01:00', '2021-08-17 12:02:00'])
        ... })
        >>> features = processor.extract(df, time_feature_stage=1)
        >>> 'flow_ratio' in features.columns
        True
        
        Args:
            df: 清洗後的 DataFrame
            include_time_features: 是否包含時間特徵
            time_feature_stage: 時間特徵階段（1-4）
        
        Returns:
            包含特徵的 DataFrame
        """
        pass
    
    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        skewed_features: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Any, List[str]]:
        """
        轉換特徵。
        
        >>> import pandas as pd
        >>> processor = StandardFeatureProcessor()
        >>> df = pd.DataFrame({
        ...     'TotBytes': [100, 200, 300],
        ...     'hour': [9, 10, 11]
        ... })
        >>> transformed, scaler, cols = processor.transform(df, ['TotBytes'])
        >>> 'log_TotBytes' in transformed.columns
        True
        
        Args:
            df: 包含特徵的 DataFrame
            skewed_features: 需要對數轉換的特徵列表
        
        Returns:
            (轉換後的 DataFrame, scaler 物件, 轉換的特徵欄位列表)
        """
        pass
    
    def save_features(
        self,
        features_df: pd.DataFrame,
        output_path: Optional[Path] = None,
        project_root: Optional[Path] = None,
        stage: Optional[int] = None
    ) -> Path:
        """
        儲存特徵工程結果為 Parquet 格式。
        
        >>> import pandas as pd
        >>> import tempfile
        >>> from pathlib import Path
        >>> processor = StandardFeatureProcessor()
        >>> test_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     output = processor.save_features(test_df, Path(tmpdir) / "features.parquet")
        ...     assert output.exists()
        ...     loaded = pd.read_parquet(output)
        ...     len(loaded) == 3
        True
        
        Args:
            features_df: 特徵工程後的 DataFrame
            output_path: 輸出檔案路徑。如果為 None，則使用預設路徑
            project_root: 專案根目錄。如果為 None，則自動偵測
            stage: 特徵階段（3 或 4），用於決定檔案名稱。如果為 None，則使用預設
        
        Returns:
            輸出檔案的路徑
        """
        if project_root is None:
            project_root = get_project_root()
        
        if output_path is None:
            output_dir = project_root / "data" / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            if stage is not None:
                output_path = output_dir / f"features_stage{stage}.parquet"
            else:
                output_path = output_dir / "features_stage4.parquet"
        
        features_df.to_parquet(
            output_path,
            engine='pyarrow',
            index=False
        )
        
        return output_path
    
    def load_features(
        self,
        input_path: Optional[Path] = None,
        project_root: Optional[Path] = None,
        stage: Optional[int] = None
    ) -> pd.DataFrame:
        """
        載入已處理的特徵 Parquet 檔案。
        
        >>> import pandas as pd
        >>> import tempfile
        >>> from pathlib import Path
        >>> processor = StandardFeatureProcessor()
        >>> test_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     output = processor.save_features(test_df, Path(tmpdir) / "features.parquet")
        ...     loaded = processor.load_features(Path(tmpdir) / "features.parquet")
        ...     len(loaded) == 3
        True
        
        Args:
            input_path: 輸入檔案路徑。如果為 None，則使用預設路徑
            project_root: 專案根目錄。如果為 None，則自動偵測
            stage: 特徵階段（3 或 4），用於決定檔案名稱。如果為 None，則優先載入階段4
        
        Returns:
            包含特徵的 DataFrame
        
        Raises:
            FileNotFoundError: 如果檔案不存在
        """
        if project_root is None:
            project_root = get_project_root()
        
        if input_path is None:
            if stage is not None:
                input_path = project_root / "data" / "processed" / f"features_stage{stage}.parquet"
            else:
                # 優先載入階段4，如果沒有則載入階段3
                stage4_path = project_root / "data" / "processed" / "features_stage4.parquet"
                stage3_path = project_root / "data" / "processed" / "features_stage3.parquet"
                if stage4_path.exists():
                    input_path = stage4_path
                elif stage3_path.exists():
                    input_path = stage3_path
                else:
                    input_path = stage4_path  # 預設使用階段4路徑（會觸發錯誤）
        
        if not input_path.exists():
            raise FileNotFoundError(
                f"找不到特徵檔案: {input_path}\n"
                f"請先執行特徵工程生成特徵檔案。"
            )
        
        return pd.read_parquet(input_path, engine='pyarrow')
    
    def save_transformed_features(
        self,
        transformed_df: pd.DataFrame,
        scaler: Any,
        transformed_columns: List[str],
        output_path: Optional[Path] = None,
        project_root: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """
        儲存轉換後的特徵和 scaler 物件。
        
        Args:
            transformed_df: 轉換後的特徵 DataFrame
            scaler: 訓練好的 scaler 物件（如 RobustScaler）
            transformed_columns: 被轉換的特徵欄位列表
            output_path: 輸出檔案路徑（不含副檔名）。如果為 None，則使用預設路徑
            project_root: 專案根目錄。如果為 None，則自動偵測
        
        Returns:
            (特徵檔案路徑, scaler 檔案路徑)
        """
        if project_root is None:
            project_root = get_project_root()
        
        if output_path is None:
            output_dir = project_root / "data" / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            base_path = output_dir / "features_transformed"
        else:
            base_path = output_path
        
        # 儲存轉換後的特徵
        features_path = base_path.with_suffix('.parquet')
        transformed_df.to_parquet(
            features_path,
            engine='pyarrow',
            index=False
        )
        
        # 儲存 scaler 物件
        scaler_path = base_path.with_suffix('.scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # 儲存轉換資訊（JSON）
        info_path = base_path.with_suffix('.info.json')
        info = {
            'transformed_columns': transformed_columns,
            'timestamp': datetime.now().isoformat(),
            'feature_count': len(transformed_columns),
            'data_shape': list(transformed_df.shape)
        }
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        return features_path, scaler_path
    
    def load_transformed_features(
        self,
        input_path: Optional[Path] = None,
        project_root: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, Any, List[str]]:
        """
        載入轉換後的特徵和 scaler 物件。
        
        Args:
            input_path: 輸入檔案路徑（不含副檔名）。如果為 None，則使用預設路徑
            project_root: 專案根目錄。如果為 None，則自動偵測
        
        Returns:
            (轉換後的特徵 DataFrame, scaler 物件, 轉換的特徵欄位列表)
        
        Raises:
            FileNotFoundError: 如果檔案不存在
        """
        if project_root is None:
            project_root = get_project_root()
        
        if input_path is None:
            base_path = project_root / "data" / "processed" / "features_transformed"
        else:
            base_path = input_path
        
        features_path = base_path.with_suffix('.parquet')
        scaler_path = base_path.with_suffix('.scaler.pkl')
        info_path = base_path.with_suffix('.info.json')
        
        if not features_path.exists():
            raise FileNotFoundError(f"找不到特徵檔案: {features_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"找不到 scaler 檔案: {scaler_path}")
        
        # 載入特徵
        transformed_df = pd.read_parquet(features_path, engine='pyarrow')
        
        # 載入 scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # 載入轉換資訊
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            transformed_columns = info.get('transformed_columns', [])
        else:
            # 如果沒有 info 檔案，從 DataFrame 推斷
            transformed_columns = list(transformed_df.columns)
        
        return transformed_df, scaler, transformed_columns


class StandardFeatureProcessor(BaseFeatureProcessor):
    """
    標準特徵處理器
    
    實作完整的特徵處理流程：
    1. 特徵提取（包含 PySpark 階段4特徵）
    2. 特徵選擇
    3. 特徵轉換（Log + RobustScaler）
    """
    
    def __init__(
        self,
        time_feature_stage: int = 4,
        use_feature_selection: bool = True,
        feature_selection_strategies: Optional[List[FeatureSelectionStrategy]] = None
    ):
        """
        初始化標準特徵處理器。
        
        Args:
            time_feature_stage: 時間特徵階段（1-4），預設為 4（最完整）
            use_feature_selection: 是否使用特徵選擇
            feature_selection_strategies: 特徵選擇策略列表
        """
        self.time_feature_stage = time_feature_stage
        self.use_feature_selection = use_feature_selection
        self.feature_selection_strategies = (
            feature_selection_strategies 
            if feature_selection_strategies is not None 
            else [FeatureSelectionStrategy.ALL]
        )
    
    def extract(
        self,
        df: pd.DataFrame,
        include_time_features: bool = True,
        time_feature_stage: Optional[int] = None
    ) -> pd.DataFrame:
        """
        提取特徵（包含階段4 PySpark 特徵）。
        
        Args:
            df: 清洗後的 DataFrame
            include_time_features: 是否包含時間特徵
            time_feature_stage: 時間特徵階段，如果為 None 則使用初始化時的設定
        
        Returns:
            包含特徵的 DataFrame
        """
        if time_feature_stage is None:
            time_feature_stage = self.time_feature_stage
        
        return extract_features(
            df,
            include_time_features=include_time_features,
            time_feature_stage=time_feature_stage
        )
    
    def transform(
        self,
        df: pd.DataFrame,
        skewed_features: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Any, List[str]]:
        """
        轉換特徵（Log + RobustScaler）。
        
        Args:
            df: 包含特徵的 DataFrame
            skewed_features: 需要對數轉換的特徵列表
            feature_columns: 最終使用的特徵欄位列表
        
        Returns:
            (轉換後的 DataFrame, scaler 物件, 轉換的特徵欄位列表)
        """
        if skewed_features is None:
            skewed_features = DEFAULT_SKEWED_FEATURES.copy()
        
        # 確保只保留數值欄位（移除 Timestamp、字串等）
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        
        # 如果指定了 feature_columns，只保留這些欄位
        if feature_columns is not None:
            available_cols = [col for col in feature_columns if col in numeric_df.columns]
            if not available_cols:
                raise ValueError("指定的特徵欄位中沒有數值型別欄位")
            numeric_df = numeric_df[available_cols]
            feature_columns = available_cols
        
        return transform_features_for_unsupervised(
            numeric_df,
            skewed_features=skewed_features,
            feature_columns=list(numeric_df.columns) if feature_columns is None else feature_columns
        )
    
    def _test_stage4_with_sample(
        self,
        features_df: pd.DataFrame,
        sample_size: int = 5000
    ) -> bool:
        """
        使用小批量資料測試階段4特徵工程。
        
        Args:
            features_df: 階段3特徵 DataFrame
            sample_size: 測試樣本大小，預設 5000 筆
        
        Returns:
            True 如果測試成功，False 如果失敗
        """
        print(f"   🧪 使用 {sample_size:,} 筆樣本測試階段4特徵工程...")
        
        try:
            # 抽取樣本
            if len(features_df) > sample_size:
                test_df = features_df.sample(n=sample_size, random_state=42).copy()
            else:
                test_df = features_df.copy()
                print(f"   ⚠️  資料量 ({len(features_df):,} 筆) 小於測試樣本大小，使用全部資料測試")
            
            # 執行階段4特徵工程
            from src.feature_engineer import _extract_bidirectional_pair_features_spark
            test_result = _extract_bidirectional_pair_features_spark(test_df)
            
            # 檢查是否有階段4特徵
            stage4_features = [
                'bidirectional_flow_count',
                'bidirectional_total_bytes',
                'bidirectional_symmetry'
            ]
            has_stage4_features = any(col in test_result.columns for col in stage4_features)
            
            if has_stage4_features:
                print(f"   ✅ 小批量測試成功：階段4特徵工程正常運作")
                print(f"   📊 測試樣本產生 {len(test_result.columns)} 個特徵（原始 {len(test_df.columns)} 個）")
                return True
            else:
                print(f"   ❌ 小批量測試失敗：未產生階段4特徵")
                print(f"   📊 測試結果特徵數：{len(test_result.columns)} 個（預期應增加階段4特徵）")
                return False
                
        except Exception as e:
            print(f"   ❌ 小批量測試失敗：{e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process(
        self,
        cleaned_df: pd.DataFrame,
        save_features: bool = True,
        save_transformed: bool = True,
        project_root: Optional[Path] = None,
        incremental: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any, List[str]]:
        """
        完整的特徵處理流程：提取 -> 選擇 -> 轉換。
        
        Args:
            cleaned_df: 清洗後的 DataFrame
            save_features: 是否儲存特徵工程結果
            save_transformed: 是否儲存轉換後的特徵
            project_root: 專案根目錄
            incremental: 如果為 True，且 time_feature_stage=4，則先載入階段3特徵，再執行階段4
        
        Returns:
            (原始特徵 DataFrame, 轉換後的特徵 DataFrame, scaler 物件, 轉換的特徵欄位列表)
        """
        # 1. 特徵提取
        actual_stage = self.time_feature_stage  # 實際使用的階段
        
        if incremental and self.time_feature_stage == 4:
            # 增量模式：先載入階段3特徵
            stage3_path = (project_root or get_project_root()) / "data" / "processed" / "features_stage3.parquet"
            if stage3_path.exists():
                print("   📂 載入階段3特徵作為基礎...")
                features_df_before = self.load_features(stage=3, project_root=project_root)
                
                # 先進行小批量測試（避免長時間執行後才發現錯誤）
                print("   🧪 先進行小批量測試（避免長時間執行後才發現錯誤）...")
                test_success = self._test_stage4_with_sample(features_df_before, sample_size=5000)
                
                if not test_success:
                    print("   ⚠️  小批量測試失敗，跳過階段4特徵工程，使用階段3特徵繼續...")
                    features_df = features_df_before
                    actual_stage = 3
                    save_features_current = False
                else:
                    # 測試成功，執行完整階段4
                    print("   🔄 小批量測試通過，執行完整階段4特徵工程（PySpark）...")
                    print("   ⏱️  預計需要 30-60 分鐘，請耐心等待...")
                    from src.feature_engineer import _extract_bidirectional_pair_features_spark
                    import numpy as np
                    features_df = _extract_bidirectional_pair_features_spark(features_df_before)
                    
                    # 計算 bidirectional_window_flow_ratio（使用階段四已聚合的資料）
                    if ('bidirectional_total_src_bytes' in features_df.columns and 
                        'bidirectional_total_dst_bytes' in features_df.columns):
                        if 'bidirectional_window_flow_ratio' not in features_df.columns:
                            features_df['bidirectional_window_flow_ratio'] = (
                                features_df['bidirectional_total_src_bytes'].astype(float) / 
                                (features_df['bidirectional_total_dst_bytes'].astype(float) + 1)
                            ).fillna(0.0).replace([np.inf, -np.inf], 0.0)
                    
                    # 檢測階段4是否成功（檢查是否有階段4特徵）
                    stage4_features = [
                        'bidirectional_flow_count',
                        'bidirectional_total_bytes',
                        'bidirectional_symmetry'
                    ]
                    stage4_success = any(col in features_df.columns for col in stage4_features)
                    
                    if not stage4_success:
                        print("   ⚠️  階段4特徵工程失敗，使用階段3特徵繼續...")
                        features_df = features_df_before
                        actual_stage = 3  # 使用階段3進行後續處理
                        # 階段4失敗時，不儲存特徵檔案（因為沒有新的特徵，避免覆蓋階段3檔案）
                        save_features_current = False
                    else:
                        print("   ✅ 階段4特徵工程成功")
                        actual_stage = 4
                        save_features_current = save_features
            else:
                # 如果沒有階段3，執行完整流程
                print("   ⚠️  未找到階段3特徵，執行完整流程...")
                features_df = self.extract(cleaned_df)
                save_features_current = save_features
        else:
            features_df = self.extract(cleaned_df)
            save_features_current = save_features
        
        if save_features_current:
            self.save_features(features_df, project_root=project_root, stage=actual_stage)
        
        # 2. 特徵選擇（如果需要）
        if self.use_feature_selection:
            X = prepare_feature_set(
                features_df,
                include_base_features=True,
                include_time_features=True,
                time_feature_stage=actual_stage  # 使用實際階段
            )
            
            selector = FeatureSelector(
                remove_constant=True,
                remove_low_variance=True,
                remove_high_correlation=True,
                remove_inf=True,
                remove_high_missing=True,
                correlation_threshold=0.98
            )
            X, _ = selector.select_features(
                X,
                features_df=features_df,
                strategies=self.feature_selection_strategies,
                verbose=False
            )
            feature_columns = list(X.columns)
        else:
            feature_columns = None
        
        # 3. 特徵轉換
        transformed_df, scaler, transformed_columns = self.transform(
            features_df,
            feature_columns=feature_columns
        )
        
        if save_transformed:
            self.save_transformed_features(
                transformed_df,
                scaler,
                transformed_columns,
                project_root=project_root
            )
        
        return features_df, transformed_df, scaler, transformed_columns


# 工廠函數（可選，未來可以擴展為 Factory Pattern）
def create_feature_processor(
    processor_type: str = "standard",
    **kwargs
) -> BaseFeatureProcessor:
    """
    創建特徵處理器（工廠函數）。
    
    >>> processor = create_feature_processor("standard")
    >>> isinstance(processor, StandardFeatureProcessor)
    True
    
    Args:
        processor_type: 處理器類型，目前僅支援 "standard"
        **kwargs: 傳遞給處理器的額外參數
    
    Returns:
        特徵處理器實例
    """
    if processor_type == "standard":
        return StandardFeatureProcessor(**kwargs)
    else:
        raise ValueError(f"未知的處理器類型: {processor_type}")


if __name__ == '__main__':
    # 簡單測試
    import doctest
    doctest.testmod(verbose=True)

