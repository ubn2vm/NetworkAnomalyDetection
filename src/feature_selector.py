"""
特徵選擇模組

負責統一管理特徵選擇邏輯，遵循單一職責原則。
提供多種特徵選擇策略（品質檢查、相關性分析、重要性選擇等）。

使用範例：
    >>> import pandas as pd
    >>> from src.feature_selector import FeatureSelector, prepare_feature_set
    >>> 
    >>> # 準備特徵
    >>> X = prepare_feature_set(features_df, include_time_features=True)
    >>> 
    >>> # 執行特徵選擇
    >>> selector = FeatureSelector()
    >>> X_selected, removed = selector.select_features(X, verbose=True)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from enum import Enum

# 基礎統計特徵（統一管理）
BASE_STATISTICAL_FEATURES = [
    'Dur',
    'TotBytes', 
    'TotPkts',
    'SrcBytes'
]


class FeatureSelectionStrategy(Enum):
    """特徵選擇策略"""
    QUALITY_CHECK = "quality_check"  # 品質檢查
    CORRELATION = "correlation"       # 相關性分析
    IMPORTANCE = "importance"         # 基於重要性
    ALL = "all"                       # 全部策略


class FeatureSelector:
    """
    特徵選擇器
    
    使用 Strategy Pattern 支援多種特徵選擇策略。
    遵循單一職責原則，專門負責特徵選擇邏輯。
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # 創建測試資料
    >>> X = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4, 5],
    ...     'feature2': [1, 1, 1, 1, 1],  # 常數特徵
    ...     'feature3': [10, 20, 30, 40, 50]
    ... })
    >>> 
    >>> selector = FeatureSelector(remove_constant=True)
    >>> X_selected, removed = selector.select_features(X, verbose=False)
    >>> len(X_selected.columns)
    2
    >>> 'feature2' in removed['constant']
    True
    """
    
    def __init__(
        self,
        remove_constant: bool = True,
        remove_low_variance: bool = True,
        variance_threshold: float = 1e-6,
        remove_inf: bool = True,
        inf_ratio_threshold: float = 0.1,
        remove_high_missing: bool = True,
        missing_ratio_threshold: float = 0.5,
        remove_high_correlation: bool = True,
        correlation_threshold: float = 0.98
    ):
        """
        初始化特徵選擇器
        
        Args:
            remove_constant: 是否移除常數特徵
            remove_low_variance: 是否移除低變異數特徵
            variance_threshold: 變異數閾值
            remove_inf: 是否移除無限值比例過高的特徵
            inf_ratio_threshold: 無限值比例閾值
            remove_high_missing: 是否移除高缺失值特徵
            missing_ratio_threshold: 缺失值比例閾值
            remove_high_correlation: 是否移除高度相關的特徵
            correlation_threshold: 相關性閾值
        """
        self.remove_constant = remove_constant
        self.remove_low_variance = remove_low_variance
        self.variance_threshold = variance_threshold
        self.remove_inf = remove_inf
        self.inf_ratio_threshold = inf_ratio_threshold
        self.remove_high_missing = remove_high_missing
        self.missing_ratio_threshold = missing_ratio_threshold
        self.remove_high_correlation = remove_high_correlation
        self.correlation_threshold = correlation_threshold
    
    def select_features(
        self,
        X: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
        strategies: Optional[List[FeatureSelectionStrategy]] = None,
        verbose: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        執行特徵選擇
        
        Args:
            X: 特徵 DataFrame
            features_df: 包含標籤的完整 DataFrame（可選，用於重要性選擇）
            strategies: 選擇策略列表，預設為全部
            verbose: 是否顯示詳細資訊
        
        Returns:
            (選擇後的特徵 DataFrame, 移除的特徵字典)
        """
        if strategies is None:
            strategies = [FeatureSelectionStrategy.ALL]
        
        removed_features = {
            'constant': [],
            'low_variance': [],
            'inf': [],
            'high_missing': [],
            'high_correlation': [],
            'low_importance': []
        }
        
        X_selected = X.copy()
        initial_count = len(X_selected.columns)
        
        # 策略1：品質檢查
        if (FeatureSelectionStrategy.QUALITY_CHECK in strategies or 
            FeatureSelectionStrategy.ALL in strategies):
            X_selected, removed = self._quality_check(X_selected, verbose)
            removed_features.update(removed)
        
        # 策略2：相關性分析
        if (FeatureSelectionStrategy.CORRELATION in strategies or 
            FeatureSelectionStrategy.ALL in strategies):
            X_selected, removed = self._correlation_analysis(X_selected, verbose)
            removed_features['high_correlation'] = removed
        
        # 策略3：重要性選擇（如果有標籤）
        if (FeatureSelectionStrategy.IMPORTANCE in strategies or 
            FeatureSelectionStrategy.ALL in strategies):
            if features_df is not None and 'Label' in features_df.columns:
                X_selected, removed = self._importance_selection(
                    X_selected, features_df, verbose
                )
                removed_features['low_importance'] = removed
        
        final_count = len(X_selected.columns)
        if verbose:
            print(f"✅ 特徵選擇完成：{initial_count} → {final_count} 個特徵")
            total_removed = sum(len(v) for v in removed_features.values())
            if total_removed > 0:
                print(f"   移除：{total_removed} 個特徵")
                for key, features in removed_features.items():
                    if features:
                        print(f"     - {key}: {len(features)} 個")
        
        return X_selected, removed_features
    
    def _quality_check(
        self, 
        X: pd.DataFrame, 
        verbose: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """品質檢查：移除低品質特徵"""
        removed = {
            'constant': [],
            'low_variance': [],
            'inf': [],
            'high_missing': []
        }
        
        # 檢查常數特徵
        if self.remove_constant:
            constant_features = [
                col for col in X.columns 
                if X[col].nunique() <= 1
            ]
            removed['constant'] = constant_features
            if constant_features and verbose:
                print(f"   發現 {len(constant_features)} 個常數特徵：{constant_features[:5]}{'...' if len(constant_features) > 5 else ''}")
        
        # 檢查低變異數特徵
        if self.remove_low_variance:
            low_variance_features = []
            for col in X.select_dtypes(include=[np.number]).columns:
                if col in X.columns:
                    var_value = X[col].var()
                    if pd.notna(var_value) and var_value < self.variance_threshold:
                        low_variance_features.append(col)
            removed['low_variance'] = low_variance_features
            if low_variance_features and verbose:
                print(f"   發現 {len(low_variance_features)} 個低變異數特徵")
        
        # 檢查無限值特徵
        if self.remove_inf:
            inf_features = []
            for col in X.select_dtypes(include=[np.number]).columns:
                if col in X.columns:
                    inf_count = np.isinf(X[col]).sum()
                    inf_ratio = inf_count / len(X) if len(X) > 0 else 0
                    if inf_ratio > self.inf_ratio_threshold:
                        inf_features.append(col)
            removed['inf'] = inf_features
            if inf_features and verbose:
                print(f"   發現 {len(inf_features)} 個無限值特徵（比例 > {self.inf_ratio_threshold*100:.1f}%）")
        
        # 檢查高缺失值特徵
        if self.remove_high_missing:
            missing_features = []
            for col in X.columns:
                missing_count = X[col].isna().sum()
                missing_ratio = missing_count / len(X) if len(X) > 0 else 0
                if missing_ratio > self.missing_ratio_threshold:
                    missing_features.append(col)
            removed['high_missing'] = missing_features
            if missing_features and verbose:
                print(f"   發現 {len(missing_features)} 個高缺失值特徵（比例 > {self.missing_ratio_threshold*100:.1f}%）")
        
        # 合併所有要移除的特徵
        all_removed = set()
        for feature_list in removed.values():
            all_removed.update(feature_list)
        
        if all_removed and verbose:
            print(f"   ✅ 移除 {len(all_removed)} 個低品質特徵")
        
        X_cleaned = X[[col for col in X.columns if col not in all_removed]]
        return X_cleaned, removed
    
    def _correlation_analysis(
        self, 
        X: pd.DataFrame, 
        verbose: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        """相關性分析：移除高度相關的特徵"""
        if not self.remove_high_correlation or len(X.columns) <= 1:
            return X, []
        
        # 清理資料以便計算相關性
        X_clean = X.copy()
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_clean.loc[:, numeric_cols] = X_clean.loc[:, numeric_cols].replace(
                [np.inf, -np.inf], np.nan
            )
            median_values = X_clean.loc[:, numeric_cols].median()
            X_clean.loc[:, numeric_cols] = X_clean.loc[:, numeric_cols].fillna(
                median_values.fillna(0)
            )
        
        # 計算相關性
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) <= 1:
            return X, []
        
        corr_matrix = X_clean[numeric_cols].corr().abs()
        features_to_remove = set()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if pd.notna(corr_value) and corr_value > self.correlation_threshold:
                    high_corr_pairs.append((col_i, col_j, corr_value))
                    features_to_remove.add(col_j)  # 保留第一個，移除第二個
        
        removed = list(features_to_remove)
        if removed and verbose:
            print(f"   發現 {len(high_corr_pairs)} 對高度相關的特徵（相關性 > {self.correlation_threshold:.2f}）")
            if len(high_corr_pairs) <= 10:
                for pair in high_corr_pairs:
                    print(f"     {pair[0]} <-> {pair[1]}: {pair[2]:.4f}")
            else:
                for pair in high_corr_pairs[:10]:
                    print(f"     {pair[0]} <-> {pair[1]}: {pair[2]:.4f}")
                print(f"     ... 還有 {len(high_corr_pairs) - 10} 對")
            print(f"   ✅ 移除 {len(removed)} 個冗餘特徵")
        
        X_cleaned = X[[col for col in X.columns if col not in features_to_remove]]
        return X_cleaned, removed
    
    def _importance_selection(
        self,
        X: pd.DataFrame,
        features_df: pd.DataFrame,
        verbose: bool = False,
        min_features: int = 15,
        max_features: int = 25,
        importance_threshold: float = 0.98
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        基於 XGBoost 特徵重要性選擇
        
        Args:
            X: 特徵 DataFrame
            features_df: 包含標籤的完整 DataFrame
            verbose: 是否顯示詳細資訊
            min_features: 最少保留特徵數
            max_features: 最多保留特徵數
            importance_threshold: 累積重要性閾值
        
        Returns:
            (選擇後的特徵 DataFrame, 移除的特徵列表)
        """
        try:
            from src.models import ModelFactory, ModelType
            from src.label_processor import convert_label_to_binary
            
            # 轉換標籤
            if 'label_binary' not in features_df.columns:
                features_df = convert_label_to_binary(features_df, verbose=False)
            y = features_df['label_binary']
            
            # 確保索引對齊
            if len(X) != len(y):
                common_idx = X.index.intersection(y.index)
                X = X.loc[common_idx]
                y = y.loc[common_idx]
            
            # 使用小樣本快速訓練
            sample_size = min(100000, len(X))
            if sample_size < len(X):
                # 使用 RandomState 確保可重現性（兼容舊版 NumPy）
                rng = np.random.RandomState(42)
                sample_idx = rng.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[sample_idx]
                y_sample = y.iloc[sample_idx]
            else:
                X_sample = X
                y_sample = y
            
            if verbose:
                print(f"   使用 {len(X_sample):,} 筆樣本快速訓練 XGBoost...")
            
            # 訓練模型獲取重要性
            xgb_model = ModelFactory.create(ModelType.XGBOOST)
            xgb_model.train(
                X_sample, y_sample,
                test_size=0.2, random_state=42,
                n_estimators=50, max_depth=4, learning_rate=0.1,
                verbose=False
            )
            
            feature_importance = xgb_model.get_feature_importance()
            sorted_importance = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # 選擇累積重要性達到閾值的特徵
            total_importance = sum(imp for _, imp in sorted_importance)
            cumulative_importance = 0
            important_features = []
            
            for feature, importance in sorted_importance:
                cumulative_importance += importance
                important_features.append(feature)
                if ((cumulative_importance / total_importance >= importance_threshold 
                     and len(important_features) >= min_features) or 
                    len(important_features) >= max_features):
                    break
            
            if len(important_features) < min_features:
                important_features = [f[0] for f in sorted_importance[:min_features]]
                cumulative_importance = sum(imp for _, imp in sorted_importance[:min_features])
            
            removed = [col for col in X.columns if col not in important_features]
            
            if verbose:
                print(f"   Top {len(important_features)} 最重要特徵（累積重要性 {cumulative_importance/total_importance*100:.1f}%）：")
                for i, (feature, importance) in enumerate(sorted_importance[:len(important_features)], 1):
                    print(f"     {i:2d}. {feature:30s}: {importance:.4f} ({importance/total_importance*100:.2f}%)")
                print(f"   ✅ 基於特徵重要性，從 {len(X.columns)} 個特徵減少到 {len(important_features)} 個特徵")
            
            X_selected = X[[col for col in X.columns if col in important_features]]
            return X_selected, removed
            
        except Exception as e:
            if verbose:
                print(f"   ⚠️  特徵重要性分析失敗：{e}")
            return X, []
    
    def check_time_feature_bias(
        self,
        X: pd.DataFrame,
        features_df: pd.DataFrame,
        time_features: Optional[List[str]] = None,
        importance_threshold: float = 0.05,
        sample_size: int = 10000,
        verbose: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        檢查時間特徵是否過於重要（避免時間偏差）
        
        攻擊可能發生在任何時間，過度依賴時間特徵可能導致：
        - 在特定時間段誤報率較高
        - 在特定時間段漏報率較高
        - 模型泛化能力下降
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> from src.label_processor import convert_label_to_binary
        >>> 
        >>> # 創建測試資料
        >>> features_df = pd.DataFrame({
        ...     'hour': [1, 2, 3, 4, 5],
        ...     'cos_hour': [0.5, 0.6, 0.7, 0.8, 0.9],
        ...     'Label': ['Normal', 'Botnet', 'Normal', 'Botnet', 'Normal']
        ... })
        >>> X = pd.DataFrame({
        ...     'hour': [1, 2, 3, 4, 5],
        ...     'cos_hour': [0.5, 0.6, 0.7, 0.8, 0.9],
        ...     'feature1': [10, 20, 30, 40, 50]
        ... })
        >>> 
        >>> selector = FeatureSelector()
        >>> X_checked, importance_dict = selector.check_time_feature_bias(
        ...     X, features_df, verbose=False
        ... )
        >>> isinstance(X_checked, pd.DataFrame)
        True
        
        Args:
            X: 特徵 DataFrame
            features_df: 包含標籤的完整 DataFrame
            time_features: 要檢查的時間特徵列表。如果為 None，則使用預設列表：
                ['hour', 'cos_hour', 'sin_hour']
            importance_threshold: 時間特徵總重要性閾值（預設 0.05，即 5%）
            sample_size: 用於快速檢查的樣本數（預設 10000）
            verbose: 是否顯示詳細資訊
        
        Returns:
            (處理後的特徵 DataFrame, 時間特徵重要性字典)
        """
        if 'Label' not in features_df.columns:
            if verbose:
                print("   ⚠️  無標籤，無法檢查時間特徵重要性，保留所有時間特徵")
            return X, {}
        
        # 預設時間特徵列表
        if time_features is None:
            time_features = ['hour', 'cos_hour', 'sin_hour']
        
        # 檢查是否有任何時間特徵存在
        existing_time_features = [f for f in time_features if f in X.columns]
        if not existing_time_features:
            if verbose:
                print("   ⚠️  未找到時間特徵，跳過時間偏差檢查")
            return X, {}
        
        try:
            from src.models import ModelFactory, ModelType
            from src.label_processor import convert_label_to_binary
            
            # 準備標籤
            if 'label_binary' not in features_df.columns:
                features_df_temp = convert_label_to_binary(features_df, verbose=False)
            else:
                features_df_temp = features_df.copy()
            y_temp = features_df_temp['label_binary']
            
            # 確保索引對齊
            common_idx = X.index.intersection(y_temp.index)
            if len(common_idx) == 0:
                if verbose:
                    print("   ⚠️  索引不匹配，跳過時間偏差檢查")
                return X, {}
            
            X_temp = X.loc[common_idx].copy()
            y_temp = y_temp.loc[common_idx]
            
            # 使用小樣本快速檢查時間特徵重要性
            actual_sample_size = min(sample_size, len(X_temp))
            if actual_sample_size < len(X_temp):
                # 使用 RandomState 確保可重現性（兼容舊版 NumPy）
                rng = np.random.RandomState(42)
                sample_idx = rng.choice(len(X_temp), actual_sample_size, replace=False)
                X_sample = X_temp.iloc[sample_idx]
                y_sample = y_temp.iloc[sample_idx]
            else:
                X_sample = X_temp
                y_sample = y_temp
            
            if verbose:
                print(f"   使用 {len(X_sample):,} 筆樣本快速檢查時間特徵重要性...")
            
            # 訓練模型獲取重要性
            time_check_model = ModelFactory.create(ModelType.XGBOOST)
            
            # 計算不平衡權重
            negative_count = (y_sample == 0).sum()
            positive_count = (y_sample == 1).sum()
            scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
            
            time_check_model.train(
                X_sample, y_sample,
                test_size=0.2, random_state=42,
                n_estimators=50, max_depth=4, learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                verbose=False
            )
            
            feature_importance = time_check_model.get_feature_importance()
            
            # 計算時間特徵重要性
            time_importance_dict = {}
            total_time_importance = 0.0
            
            for time_feat in existing_time_features:
                importance = feature_importance.get(time_feat, 0.0)
                time_importance_dict[time_feat] = importance
                total_time_importance += importance
            
            if verbose:
                print(f"   📊 時間特徵重要性分析：")
                for time_feat in existing_time_features:
                    imp = time_importance_dict.get(time_feat, 0.0)
                    print(f"     {time_feat}: {imp:.4f} ({imp*100:.2f}%)")
                print(f"     時間特徵總重要性: {total_time_importance:.4f} ({total_time_importance*100:.2f}%)")
            
            # 決定是否移除時間特徵
            if total_time_importance > importance_threshold:
                if verbose:
                    print(f"\n   ⚠️  警告：時間特徵總重要性 ({total_time_importance:.4f}) 超過閾值 ({importance_threshold:.4f})")
                    print(f"   💡 攻擊可能發生在任何時間，過度依賴時間特徵可能導致：")
                    print(f"      - 在特定時間段誤報率較高")
                    print(f"      - 在特定時間段漏報率較高")
                    print(f"      - 模型泛化能力下降")
                    print(f"\n   🔧 建議移除時間特徵，只保留行為特徵...")
                
                # 移除所有時間相關特徵（但保留時間窗口聚合特徵，因為它們是行為特徵）
                time_features_to_remove = [
                    'hour', 'day_of_week', 'day_of_month', 
                    'is_weekend', 'is_work_hour', 'is_night',
                    'sin_hour', 'cos_hour', 
                    'sin_day_of_week', 'cos_day_of_week',
                    'sin_day_of_month', 'cos_day_of_month'
                ]
                # 只移除實際存在的特徵
                time_features_to_remove = [f for f in time_features_to_remove if f in X.columns]
                
                if time_features_to_remove:
                    X = X[[col for col in X.columns if col not in time_features_to_remove]]
                    if verbose:
                        print(f"   ✅ 已移除 {len(time_features_to_remove)} 個時間特徵：{time_features_to_remove}")
                        print(f"   ✅ 保留時間窗口聚合特徵（如 flows_per_minute_by_src），因為它們反映行為模式")
            else:
                if verbose:
                    print(f"   ✅ 時間特徵重要性在可接受範圍內，保留所有特徵")
                    print(f"   💡 如果擔心時間偏差，可以手動移除時間特徵")
            
            return X, time_importance_dict
            
        except Exception as e:
            if verbose:
                print(f"   ⚠️  無法檢查時間特徵重要性：{e}")
                import traceback
                traceback.print_exc()
            return X, {}


def prepare_feature_set(
    features_df: pd.DataFrame,
    include_base_features: bool = True,
    include_time_features: bool = True,
    time_feature_stage: int = 1
) -> pd.DataFrame:
    """
    準備完整的特徵集合（統一接口）
    
    結合基礎統計特徵和工程特徵，提供統一的特徵準備接口。
    
    >>> import pandas as pd
    >>> from src.feature_engineer import extract_features
    >>> 
    >>> # 假設已經有 features_df
    >>> # features_df = extract_features(cleaned_df)
    >>> # X = prepare_feature_set(features_df, include_time_features=True)
    >>> # 'Dur' in X.columns  # 基礎特徵
    >>> # 'flow_ratio' in X.columns  # 工程特徵
    
    Args:
        features_df: 經過特徵工程的 DataFrame
        include_base_features: 是否包含基礎統計特徵
        include_time_features: 是否包含時間特徵
        time_feature_stage: 時間特徵階段（1, 2, 3, 或 4）
            - 1: 基本時間特徵
            - 2: 時間間隔特徵
            - 3: 時間窗口聚合特徵（按 SrcAddr）
            - 4: 雙向流 Pair 聚合特徵（按 IP Pair，需要 PySpark）
    
    Returns:
        包含所有特徵的 DataFrame
    """
    from src.feature_engineer import get_feature_columns
    
    feature_cols = []
    
    # 基礎統計特徵
    if include_base_features:
        feature_cols.extend(BASE_STATISTICAL_FEATURES)
    
    # 工程特徵
    engineered_features = get_feature_columns(
        include_time_features=include_time_features,
        time_feature_stage=time_feature_stage
    )
    feature_cols.extend(engineered_features)
    
    # 只選擇存在的欄位
    available_cols = [col for col in feature_cols if col in features_df.columns]
    X = features_df[available_cols].copy()
    
    return X


def get_base_statistical_features() -> List[str]:
    """
    獲取基礎統計特徵列表
    
    >>> features = get_base_statistical_features()
    >>> 'Dur' in features
    True
    >>> 'TotBytes' in features
    True
    
    Returns:
        基礎統計特徵列表的副本
    """
    return BASE_STATISTICAL_FEATURES.copy()


if __name__ == '__main__':
    # 簡單測試
    import doctest
    doctest.testmod(verbose=True)

