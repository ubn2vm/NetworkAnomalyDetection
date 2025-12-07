"""
統一的模型訓練與預測模組

使用 Factory Pattern 支援不同模型的訓練與預測。
提供 Isolation Forest（無監督學習）和 XGBoost（監督學習）的統一介面。
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from abc import ABC, abstractmethod
from enum import Enum
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb


class ModelType(Enum):
    """模型類型枚舉"""
    ISOLATION_FOREST = "isolation_forest"
    XGBOOST = "xgboost"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"


class BaseModel(ABC):
    """模型抽象基類
    
    定義所有模型必須實作的統一介面。
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> # BaseModel 是抽象類別，不能直接實例化
    >>> # model = BaseModel()  # 這會失敗
    """
    
    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs
    ) -> Any:
        """
        訓練模型。
        
        Args:
            X: 特徵 DataFrame。
            y: 標籤 Series（可選，視模型類型而定）。
            **kwargs: 其他模型特定參數。
        
        Returns:
            訓練結果（視模型類型而定）。
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用訓練好的模型進行預測。
        
        Args:
            X: 特徵 DataFrame。
        
        Returns:
            預測結果陣列。
        
        Raises:
            ValueError: 如果模型尚未訓練。
        """
        pass


class IsolationForestModel(BaseModel):
    """Isolation Forest 模型包裝器（無監督學習）
    
    用於異常偵測，不需要標籤即可訓練。
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> model = IsolationForestModel()
    >>> X = pd.DataFrame({
    ...     'feature1': np.random.randn(100),
    ...     'feature2': np.random.randn(100)
    ... })
    >>> trained_model, scaler = model.train(X, contamination=0.1)
    >>> predictions = model.predict(X)
    >>> len(predictions) == len(X)
    True
    >>> isinstance(predictions, np.ndarray)
    True
    """
    
    def __init__(self):
        """初始化 Isolation Forest 模型"""
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[Union[StandardScaler, RobustScaler]] = None
        self.use_external_scaler: bool = False  # 標記是否使用外部 scaler
    
    def train(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        contamination: float = 0.1,
        random_state: int = 42,
        use_external_scaler: bool = False,
        external_scaler: Optional[Union[StandardScaler, RobustScaler]] = None,
        **kwargs
    ) -> Tuple[IsolationForest, Union[StandardScaler, RobustScaler]]:
        """
        訓練 Isolation Forest 模型（無監督學習）。
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> model = IsolationForestModel()
        >>> X = pd.DataFrame({
        ...     'feature1': np.random.randn(100),
        ...     'feature2': np.random.randn(100)
        ... })
        >>> trained_model, scaler = model.train(X, contamination=0.1)
        >>> isinstance(trained_model, IsolationForest)
        True
        >>> isinstance(scaler, StandardScaler)
        True

        Args:
            X: 特徵 DataFrame。如果 use_external_scaler=True，X 應該已經被標準化過。
            y: 標籤 Series（Isolation Forest 不使用，但保留介面一致性）。
            contamination: 預期的異常比例（0.0 到 0.5 之間）。
            random_state: 隨機種子。
            use_external_scaler: 如果為 True，使用外部提供的 scaler（跳過內部標準化）。
            external_scaler: 外部提供的 scaler（例如 RobustScaler）。如果為 None 且 use_external_scaler=True，則假設 X 已經標準化。
            **kwargs: 傳遞給 IsolationForest 的其他參數。

        Returns:
            (訓練好的模型, 標準化器)。

        Raises:
            ValueError: 如果 contamination 不在有效範圍內。
        """
        if not 0.0 <= contamination <= 0.5:
            raise ValueError(f"contamination 必須在 0.0 到 0.5 之間，目前為 {contamination}")
        
        # 標準化特徵（Isolation Forest 對尺度敏感）
        if use_external_scaler:
            self.use_external_scaler = True
            if external_scaler is not None:
                # 使用外部提供的 scaler（例如 RobustScaler）
                # 注意：X 應該已經被這個 scaler 標準化過，這裡只是保存 scaler 供預測時使用
                self.scaler = external_scaler
                # X 已經標準化，直接轉換為 numpy array 供模型使用
                X_scaled = X.values if isinstance(X, pd.DataFrame) else X
            else:
                # 假設 X 已經標準化，不需要再次標準化
                # 設置 scaler 為 None，表示不需要額外轉換
                self.scaler = None
                X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        else:
            self.use_external_scaler = False
            # 使用內部 StandardScaler（原有邏輯）
            self.scaler = StandardScaler()
            # 轉換為 numpy array 以避免 feature names 警告
            X_for_fit = X.values if isinstance(X, pd.DataFrame) else X
            X_scaled = self.scaler.fit_transform(X_for_fit)
        
        # 訓練模型且參數合併
        model_params = {
            'contamination': contamination,
            'random_state': random_state,
            'n_estimators': 100,
            'n_jobs': -1,  # 使用所有 CPU 核心加速
        }
        model_params.update(kwargs)
        
        self.model = IsolationForest(**model_params)
        self.model.fit(X_scaled)
        
        return self.model, self.scaler
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用訓練好的 Isolation Forest 模型進行預測。
        
        返回異常分數（負值表示異常，值越小越異常）。
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> model = IsolationForestModel()
        >>> X_train = pd.DataFrame({
        ...     'feature1': np.random.randn(100),
        ...     'feature2': np.random.randn(100)
        ... })
        >>> model.train(X_train, contamination=0.1)
        >>> X_test = pd.DataFrame({
        ...     'feature1': np.random.randn(10),
        ...     'feature2': np.random.randn(10)
        ... })
        >>> scores = model.predict(X_test)
        >>> len(scores) == len(X_test)
        True
        >>> scores.dtype == np.float64
        True

        Args:
            X: 特徵 DataFrame。如果訓練時使用了 use_external_scaler=True，X 應該已經被標準化過。

        Returns:
            異常分數陣列（負值表示異常，值越小越異常）。

        Raises:
            ValueError: 如果模型尚未訓練。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        
        # 如果使用外部 scaler，X 應該已經被標準化過，不需要再次標準化
        if self.use_external_scaler:
            # X 已經標準化，直接使用
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        elif self.scaler is None:
            # 如果 scaler 為 None，假設 X 已經標準化
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        else:
            # 使用保存的 scaler，使用 transform() 而非 fit_transform()，避免重新擬合
            # 將 DataFrame 轉換為 numpy array 以避免 feature names 警告
            X_for_transform = X.values if isinstance(X, pd.DataFrame) else X
            X_scaled = self.scaler.transform(X_for_transform)
        anomaly_scores = self.model.score_samples(X_scaled)
        return anomaly_scores
    
    def predict_labels(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測異常標籤（-1 表示異常，1 表示正常）。
        
        Args:
            X: 特徵 DataFrame。如果訓練時使用了 use_external_scaler=True，X 應該已經被標準化過。
        
        Returns:
            預測標籤陣列（-1 表示異常，1 表示正常）。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        
        # 如果使用外部 scaler，X 應該已經被標準化過，不需要再次標準化
        if self.use_external_scaler:
            # X 已經標準化，直接使用
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        elif self.scaler is None:
            # 如果 scaler 為 None，表示訓練時使用了 use_external_scaler=True 且 external_scaler=None
            # 這意味著 X 在訓練時已經被標準化過，預測時也應該已經標準化
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        else:
            # 使用保存的 scaler 進行標準化
            # 將 DataFrame 轉換為 numpy array 以避免 feature names 警告
            X_for_transform = X.values if isinstance(X, pd.DataFrame) else X
            X_scaled = self.scaler.transform(X_for_transform)
        return self.model.predict(X_scaled)
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測異常分數（負值表示異常，值越小越異常）。
        轉換為正數分數（值越大越異常）以便理解。
        
        Args:
            X: 特徵 DataFrame。如果訓練時使用了 use_external_scaler=True，X 應該已經被標準化過。
        
        Returns:
            異常分數陣列（正數，值越大越異常）。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        
        # 獲取原始異常分數（負值表示異常）
        anomaly_scores = self.predict(X)
        # 轉換為正數（值越大越異常）
        anomaly_scores_normalized = -anomaly_scores
        return anomaly_scores_normalized


class XGBoostModel(BaseModel):
    """XGBoost 模型包裝器（監督學習）
    
    用於分類任務，需要標籤進行訓練。
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> model = XGBoostModel()
    >>> X = pd.DataFrame({
    ...     'feature1': np.random.randn(100),
    ...     'feature2': np.random.randn(100)
    ... })
    >>> y = pd.Series(np.random.randint(0, 2, 100))
    >>> trained_model, metrics = model.train(X, y, test_size=0.2)
    >>> 'accuracy' in metrics
    True
    >>> predictions = model.predict(X)
    >>> len(predictions) == len(X)
    True
    """
    
    def __init__(self):
        """初始化 XGBoost 模型"""
        self.model: Optional[xgb.XGBClassifier] = None
        self.metrics: Optional[Dict[str, Any]] = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs
    ) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """
        訓練 XGBoost 模型（監督學習）。
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> model = XGBoostModel()
        >>> X = pd.DataFrame({
        ...     'feature1': np.random.randn(100),
        ...     'feature2': np.random.randn(100)
        ... })
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> trained_model, metrics = model.train(X, y, test_size=0.2)
        >>> isinstance(trained_model, xgb.XGBClassifier)
        True
        >>> 'accuracy' in metrics
        True
        >>> 'classification_report' in metrics
        True

        Args:
            X: 特徵 DataFrame。
            y: 標籤 Series（必須提供）。
            test_size: 測試集比例（0.0 到 1.0 之間）。
            random_state: 隨機種子。
            **kwargs: 傳遞給 XGBClassifier 的其他參數。

        Returns:
            (訓練好的模型, 評估指標字典)。

        Raises:
            ValueError: 如果 y 為 None 或 test_size 不在有效範圍內。
        """
        if y is None:
            raise ValueError("XGBoost 是監督學習模型，必須提供標籤 y")
        
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size 必須在 0.0 到 1.0 之間，目前為 {test_size}")
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # 分割資料
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 訓練模型
        default_params = {
            'random_state': random_state,
            'eval_metric': 'logloss',
            'n_estimators': 100,
        }
        
        # 先提取 early_stopping_rounds（避免傳給構造函數）
        early_stopping_rounds = kwargs.pop('early_stopping_rounds', 10)
        
        default_params.update(kwargs)
        
        self.model = xgb.XGBClassifier(**default_params)
        
        # 使用 eval_set 和 early_stopping 防止過擬合
        eval_set = [(X_train, y_train), (X_test, y_test)]
        
        # 根據 XGBoost 版本選擇合適的 API
        # XGBoost 2.0+ 必須使用 callbacks，1.7-1.9 可以使用 callbacks 或 early_stopping_rounds，< 1.7 使用 early_stopping_rounds
        try:
            # 檢查 XGBoost 版本
            xgb_version = xgb.__version__
            version_parts = [int(x) for x in xgb_version.split('.')[:2]]
            major_version = version_parts[0]
            minor_version = version_parts[1] if len(version_parts) > 1 else 0
            
            is_xgb_2_0_plus = major_version >= 2
            is_xgb_1_7_plus = major_version > 1 or (major_version == 1 and minor_version >= 7)
            
            if is_xgb_2_0_plus:
                # XGBoost 2.0+ 必須使用 callbacks，不支持 early_stopping_rounds
                try:
                    from xgboost.callback import EarlyStopping
                    # 使用新的 callbacks API
                    early_stop_callback = EarlyStopping(
                        rounds=early_stopping_rounds,
                        metric_name='logloss',
                        data_name='validation_1',  # eval_set 中的第二個數據集（測試集）
                        save_best=True
                    )
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        callbacks=[early_stop_callback],
                        verbose=False
                    )
                except (TypeError, AttributeError, ValueError, ImportError) as e:
                    # 如果 callbacks 失敗，不使用 early stopping（2.0+ 不支持 early_stopping_rounds）
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=False
                    )
            elif is_xgb_1_7_plus:
                # XGBoost 1.7-1.9 可以嘗試 callbacks，失敗則使用 early_stopping_rounds
                try:
                    from xgboost.callback import EarlyStopping
                    early_stop_callback = EarlyStopping(
                        rounds=early_stopping_rounds,
                        metric_name='logloss',
                        data_name='validation_1',
                        save_best=True
                    )
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        callbacks=[early_stop_callback],
                        verbose=False
                    )
                except (TypeError, AttributeError, ValueError, ImportError):
                    # 回退到 early_stopping_rounds
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
            else:
                # XGBoost < 1.7 使用 early_stopping_rounds
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
        except (ImportError, AttributeError, ValueError) as e:
            # 如果版本檢查失敗，嘗試最安全的方法：不使用 early stopping
            try:
                # 先嘗試 callbacks（適用於新版本）
                try:
                    from xgboost.callback import EarlyStopping
                    early_stop_callback = EarlyStopping(
                        rounds=early_stopping_rounds,
                        metric_name='logloss',
                        data_name='validation_1',
                        save_best=True
                    )
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        callbacks=[early_stop_callback],
                        verbose=False
                    )
                except (TypeError, AttributeError, ValueError, ImportError):
                    # 嘗試 early_stopping_rounds（適用於舊版本）
                    try:
                        self.model.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose=False
                        )
                    except TypeError:
                        # 如果都不支持，不使用 early stopping
                        self.model.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            verbose=False
                        )
            except Exception:
                # 最後的安全網：不使用 early stopping
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
        
        # 評估訓練集和測試集性能（用於過擬合診斷）
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # 計算訓練集和測試集的詳細指標
        train_report = classification_report(y_train, y_pred_train, output_dict=True, zero_division=0)
        test_report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        
        # 計算過擬合指標
        accuracy_gap = train_accuracy - test_accuracy
        overfitting_risk = 'low' if accuracy_gap < 0.01 else ('medium' if accuracy_gap < 0.05 else 'high')
        
        self.metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'accuracy_gap': accuracy_gap,
            'overfitting_risk': overfitting_risk,
            'accuracy': test_accuracy,  # 保持向後兼容
            'classification_report': test_report,  # 保持向後兼容
            'train_classification_report': train_report,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'best_iteration': getattr(self.model, 'best_iteration', default_params.get('n_estimators', 100)),
        }
        
        return self.model, self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用訓練好的 XGBoost 模型進行預測。
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> model = XGBoostModel()
        >>> X_train = pd.DataFrame({
        ...     'feature1': np.random.randn(100),
        ...     'feature2': np.random.randn(100)
        ... })
        >>> y_train = pd.Series(np.random.randint(0, 2, 100))
        >>> model.train(X_train, y_train, test_size=0.2)
        >>> X_test = pd.DataFrame({
        ...     'feature1': np.random.randn(10),
        ...     'feature2': np.random.randn(10)
        ... })
        >>> predictions = model.predict(X_test)
        >>> len(predictions) == len(X_test)
        True
        >>> predictions.dtype in [np.int32, np.int64]
        True

        Args:
            X: 特徵 DataFrame。

        Returns:
            預測結果陣列（類別標籤）。

        Raises:
            ValueError: 如果模型尚未訓練。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測類別概率。
        
        Args:
            X: 特徵 DataFrame。
        
        Returns:
            類別概率陣列。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        取得特徵重要性。
        
        Returns:
            特徵名稱與重要性的字典。
        
        Raises:
            ValueError: 如果模型尚未訓練。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        
        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        # 將 NumPy 類型轉換為 Python 原生 float，以確保 JSON 序列化相容性
        return {name: float(imp) for name, imp in zip(feature_names, importances)}


class LOFModel(BaseModel):
    """Local Outlier Factor (LOF) 模型包裝器（無監督學習）
    
    基於局部密度的異常檢測，可與 Isolation Forest 互補。
    適合檢測局部異常，能識別 Isolation Forest 可能遺漏的異常模式。
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> model = LOFModel()
    >>> X = pd.DataFrame({
    ...     'feature1': np.random.randn(100),
    ...     'feature2': np.random.randn(100)
    ... })
    >>> trained_model, scaler = model.train(X, n_neighbors=20, contamination=0.1)
    >>> predictions = model.predict(X)
    >>> len(predictions) == len(X)
    True
    >>> isinstance(predictions, np.ndarray)
    True
    """
    
    def __init__(self):
        """初始化 LOF 模型"""
        self.model: Optional[Any] = None  # LocalOutlierFactor
        self.scaler: Optional[Union[StandardScaler, RobustScaler]] = None
        self.use_external_scaler: bool = False
        self.training_data: Optional[np.ndarray] = None  # 儲存訓練資料供預測使用
    
    def train(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        use_external_scaler: bool = False,
        external_scaler: Optional[Union[StandardScaler, RobustScaler]] = None,
        **kwargs
    ) -> Tuple[Any, Union[StandardScaler, RobustScaler]]:
        """
        訓練 LOF 模型（無監督學習）。
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> model = LOFModel()
        >>> X = pd.DataFrame({
        ...     'feature1': np.random.randn(100),
        ...     'feature2': np.random.randn(100)
        ... })
        >>> trained_model, scaler = model.train(X, n_neighbors=20, contamination=0.1)
        >>> trained_model is not None
        True
        >>> isinstance(scaler, StandardScaler)
        True
        
        Args:
            X: 特徵 DataFrame。
            y: 標籤 Series（LOF 不使用，但保留介面一致性）。
            n_neighbors: 鄰居數量（預設 20，建議範圍 10-50）。
            contamination: 預期的異常比例（0.0 到 0.5 之間）。
            use_external_scaler: 如果為 True，使用外部提供的 scaler。
            external_scaler: 外部提供的 scaler。
            **kwargs: 傳遞給 LocalOutlierFactor 的其他參數。
        
        Returns:
            (訓練好的模型, 標準化器)。
        
        Raises:
            ValueError: 如果 contamination 不在有效範圍內。
        """
        from sklearn.neighbors import LocalOutlierFactor
        
        if not 0.0 <= contamination <= 0.5:
            raise ValueError(f"contamination 必須在 0.0 到 0.5 之間，目前為 {contamination}")
        
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors 必須大於 0，目前為 {n_neighbors}")
        
        # 標準化特徵（LOF 對尺度敏感）
        if use_external_scaler:
            self.use_external_scaler = True
            if external_scaler is not None:
                self.scaler = external_scaler
                X_scaled = X.values if isinstance(X, pd.DataFrame) else X
            else:
                self.scaler = None
                X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        else:
            self.use_external_scaler = False
            self.scaler = StandardScaler()
            X_for_fit = X.values if isinstance(X, pd.DataFrame) else X
            X_scaled = self.scaler.fit_transform(X_for_fit)
        
        # 儲存標準化後的訓練資料（供預測時使用）
        self.training_data = X_scaled.copy()
        
        # 訓練模型（novelty=False 用於訓練階段）
        model_params = {
            'n_neighbors': min(n_neighbors, len(X_scaled) - 1),  # 確保不超過樣本數
            'contamination': contamination,
            'novelty': False,  # 訓練階段使用 False
            'n_jobs': -1,
        }
        model_params.update(kwargs)
        
        self.model = LocalOutlierFactor(**model_params)
        self.model.fit(X_scaled)
        
        return self.model, self.scaler
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用訓練好的 LOF 模型進行預測。
        
        返回異常分數（負值表示異常，值越小越異常）。
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> model = LOFModel()
        >>> X_train = pd.DataFrame({
        ...     'feature1': np.random.randn(100),
        ...     'feature2': np.random.randn(100)
        ... })
        >>> model.train(X_train, n_neighbors=20, contamination=0.1)
        >>> X_test = pd.DataFrame({
        ...     'feature1': np.random.randn(10),
        ...     'feature2': np.random.randn(10)
        ... })
        >>> scores = model.predict(X_test)
        >>> len(scores) == len(X_test)
        True
        >>> scores.dtype == np.float64
        True
        
        Args:
            X: 特徵 DataFrame。如果訓練時使用了 use_external_scaler=True，X 應該已經被標準化過。
        
        Returns:
            異常分數陣列（負值表示異常，值越小越異常）。
        
        Raises:
            ValueError: 如果模型尚未訓練。
        """
        from sklearn.neighbors import LocalOutlierFactor
        
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        
        if self.training_data is None:
            raise ValueError("訓練資料未儲存，無法進行預測")
        
        # 標準化
        if self.use_external_scaler:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        elif self.scaler is None:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        else:
            X_for_transform = X.values if isinstance(X, pd.DataFrame) else X
            X_scaled = self.scaler.transform(X_for_transform)
        
        # LOF 需要 novelty=True 的模型來預測新資料
        # 建立新的預測模型
        prediction_model = LocalOutlierFactor(
            n_neighbors=self.model.n_neighbors,
            contamination=self.model.contamination,
            novelty=True,
            n_jobs=-1
        )
        # 使用原始訓練資料擬合
        prediction_model.fit(self.training_data)
        # 計算異常分數（負值表示異常）
        anomaly_scores = prediction_model.score_samples(X_scaled)
        
        return anomaly_scores
    
    def predict_labels(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測異常標籤（-1 表示異常，1 表示正常）。
        
        Args:
            X: 特徵 DataFrame。如果訓練時使用了 use_external_scaler=True，X 應該已經被標準化過。
        
        Returns:
            預測標籤陣列（-1 表示異常，1 表示正常）。
        """
        from sklearn.neighbors import LocalOutlierFactor
        
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        
        if self.training_data is None:
            raise ValueError("訓練資料未儲存，無法進行預測")
        
        # 標準化
        if self.use_external_scaler:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        elif self.scaler is None:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        else:
            X_for_transform = X.values if isinstance(X, pd.DataFrame) else X
            X_scaled = self.scaler.transform(X_for_transform)
        
        # 建立 novelty=True 的模型用於預測
        prediction_model = LocalOutlierFactor(
            n_neighbors=self.model.n_neighbors,
            contamination=self.model.contamination,
            novelty=True,
            n_jobs=-1
        )
        prediction_model.fit(self.training_data)
        return prediction_model.predict(X_scaled)
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測異常分數（負值表示異常，值越小越異常）。
        轉換為正數分數（值越大越異常）以便理解。
        
        Args:
            X: 特徵 DataFrame。如果訓練時使用了 use_external_scaler=True，X 應該已經被標準化過。
        
        Returns:
            異常分數陣列（正數，值越大越異常）。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        
        # 獲取原始異常分數（負值表示異常）
        anomaly_scores = self.predict(X)
        # 轉換為正數（值越大越異常）
        anomaly_scores_normalized = -anomaly_scores
        return anomaly_scores_normalized


class OneClassSVMModel(BaseModel):
    """One-Class SVM 模型包裝器（無監督學習）
    
    適合高維特徵空間，對非線性邊界處理較好。
    使用核函數（如 RBF）可以捕捉複雜的異常模式。
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> model = OneClassSVMModel()
    >>> X = pd.DataFrame({
    ...     'feature1': np.random.randn(100),
    ...     'feature2': np.random.randn(100)
    ... })
    >>> trained_model, scaler = model.train(X, nu=0.1, kernel='rbf')
    >>> predictions = model.predict(X)
    >>> len(predictions) == len(X)
    True
    >>> isinstance(predictions, np.ndarray)
    True
    """
    
    def __init__(self):
        """初始化 One-Class SVM 模型"""
        self.model: Optional[Any] = None  # OneClassSVM
        self.scaler: Optional[Union[StandardScaler, RobustScaler]] = None
        self.use_external_scaler: bool = False
    
    def train(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        nu: float = 0.1,
        kernel: str = 'rbf',
        gamma: Optional[Union[str, float]] = 'scale',
        use_external_scaler: bool = False,
        external_scaler: Optional[Union[StandardScaler, RobustScaler]] = None,
        **kwargs
    ) -> Tuple[Any, Union[StandardScaler, RobustScaler]]:
        """
        訓練 One-Class SVM 模型（無監督學習）。
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> model = OneClassSVMModel()
        >>> X = pd.DataFrame({
        ...     'feature1': np.random.randn(100),
        ...     'feature2': np.random.randn(100)
        ... })
        >>> trained_model, scaler = model.train(X, nu=0.1, kernel='rbf')
        >>> trained_model is not None
        True
        >>> isinstance(scaler, StandardScaler)
        True
        
        Args:
            X: 特徵 DataFrame。
            y: 標籤 Series（One-Class SVM 不使用，但保留介面一致性）。
            nu: 異常比例的上限（0.0 到 1.0 之間，類似 contamination）。
            kernel: 核函數類型（'rbf', 'poly', 'sigmoid', 'linear'）。
            gamma: 核函數係數（'scale', 'auto' 或浮點數）。
            use_external_scaler: 如果為 True，使用外部提供的 scaler。
            external_scaler: 外部提供的 scaler。
            **kwargs: 傳遞給 OneClassSVM 的其他參數。如果包含 'contamination'，會自動轉換為 'nu'。
        
        Returns:
            (訓練好的模型, 標準化器)。
        
        Raises:
            ValueError: 如果 nu 不在有效範圍內。
        """
        from sklearn.svm import OneClassSVM
        
        # 處理 contamination 參數（為了與其他模型介面一致）
        if 'contamination' in kwargs:
            contamination = kwargs.pop('contamination')
            # 如果 nu 使用預設值，則使用 contamination 的值
            if nu == 0.1:  # 預設值
                nu = contamination
            # 確保 contamination 在有效範圍內
            if not 0.0 < contamination <= 1.0:
                raise ValueError(f"contamination 必須在 0.0 到 1.0 之間，目前為 {contamination}")
        
        if not 0.0 < nu <= 1.0:
            raise ValueError(f"nu 必須在 0.0 到 1.0 之間，目前為 {nu}")
        
        # 標準化特徵（SVM 對尺度敏感）
        if use_external_scaler:
            self.use_external_scaler = True
            if external_scaler is not None:
                self.scaler = external_scaler
                X_scaled = X.values if isinstance(X, pd.DataFrame) else X
            else:
                self.scaler = None
                X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        else:
            self.use_external_scaler = False
            self.scaler = StandardScaler()
            X_for_fit = X.values if isinstance(X, pd.DataFrame) else X
            X_scaled = self.scaler.fit_transform(X_for_fit)
        
        # 訓練模型
        model_params = {
            'nu': nu,
            'kernel': kernel,
            'gamma': gamma,
        }
        model_params.update(kwargs)
        
        self.model = OneClassSVM(**model_params)
        self.model.fit(X_scaled)
        
        return self.model, self.scaler
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用訓練好的 One-Class SVM 模型進行預測。
        
        返回決策函數值（負值表示異常，值越小越異常）。
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> model = OneClassSVMModel()
        >>> X_train = pd.DataFrame({
        ...     'feature1': np.random.randn(100),
        ...     'feature2': np.random.randn(100)
        ... })
        >>> model.train(X_train, nu=0.1, kernel='rbf')
        >>> X_test = pd.DataFrame({
        ...     'feature1': np.random.randn(10),
        ...     'feature2': np.random.randn(10)
        ... })
        >>> scores = model.predict(X_test)
        >>> len(scores) == len(X_test)
        True
        >>> scores.dtype == np.float64
        True
        
        Args:
            X: 特徵 DataFrame。如果訓練時使用了 use_external_scaler=True，X 應該已經被標準化過。
        
        Returns:
            決策函數值陣列（負值表示異常，值越小越異常）。
        
        Raises:
            ValueError: 如果模型尚未訓練。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        
        # 標準化
        if self.use_external_scaler:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        elif self.scaler is None:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        else:
            X_for_transform = X.values if isinstance(X, pd.DataFrame) else X
            X_scaled = self.scaler.transform(X_for_transform)
        
        # decision_function 返回負值（值越小越異常）
        decision_scores = self.model.decision_function(X_scaled)
        return decision_scores
    
    def predict_labels(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測異常標籤（-1 表示異常，1 表示正常）。
        
        Args:
            X: 特徵 DataFrame。如果訓練時使用了 use_external_scaler=True，X 應該已經被標準化過。
        
        Returns:
            預測標籤陣列（-1 表示異常，1 表示正常）。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        
        # 標準化
        if self.use_external_scaler:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        elif self.scaler is None:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        else:
            X_for_transform = X.values if isinstance(X, pd.DataFrame) else X
            X_scaled = self.scaler.transform(X_for_transform)
        
        return self.model.predict(X_scaled)
    
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測異常分數（負值表示異常，值越小越異常）。
        轉換為正數分數（值越大越異常）以便理解。
        
        Args:
            X: 特徵 DataFrame。如果訓練時使用了 use_external_scaler=True，X 應該已經被標準化過。
        
        Returns:
            異常分數陣列（正數，值越大越異常）。
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先呼叫 train() 方法")
        
        # 獲取原始決策函數值（負值表示異常）
        decision_scores = self.predict(X)
        # 轉換為正數（值越大越異常）
        anomaly_scores_normalized = -decision_scores
        return anomaly_scores_normalized


class ModelFactory:
    """模型工廠
    
    根據模型類型創建對應的模型實例。
    
    >>> from src.models import ModelFactory, ModelType
    >>> model = ModelFactory.create(ModelType.ISOLATION_FOREST)
    >>> isinstance(model, BaseModel)
    True
    >>> model = ModelFactory.create(ModelType.XGBOOST)
    >>> isinstance(model, BaseModel)
    True
    """
    
    _models = {
        ModelType.ISOLATION_FOREST: IsolationForestModel,
        ModelType.XGBOOST: XGBoostModel,
        ModelType.LOCAL_OUTLIER_FACTOR: LOFModel,
        ModelType.ONE_CLASS_SVM: OneClassSVMModel,
    }
    
    @classmethod
    def create(cls, model_type: ModelType) -> BaseModel:
        """
        創建模型實例。
        
        >>> from src.models import ModelFactory, ModelType
        >>> model = ModelFactory.create(ModelType.ISOLATION_FOREST)
        >>> isinstance(model, IsolationForestModel)
        True
        >>> model = ModelFactory.create(ModelType.XGBOOST)
        >>> isinstance(model, XGBoostModel)
        True

        Args:
            model_type: 模型類型（ModelType 枚舉）。

        Returns:
            對應的模型實例（BaseModel 的子類別）。

        Raises:
            ValueError: 如果模型類型不存在。
        """
        if model_type not in cls._models:
            available_types = [mt.value for mt in cls._models.keys()]
            raise ValueError(
                f"不支援的模型類型: {model_type.value}。"
                f"可用的類型: {available_types}"
            )
        return cls._models[model_type]()
    
    @classmethod
    def get_available_types(cls) -> List[ModelType]:
        """
        取得所有可用的模型類型。
        
        >>> from src.models import ModelFactory
        >>> types = ModelFactory.get_available_types()
        >>> len(types) >= 2
        True
        >>> ModelType.ISOLATION_FOREST in types
        True
        >>> ModelType.XGBOOST in types
        True
        >>> ModelType.LOCAL_OUTLIER_FACTOR in types
        True
        >>> ModelType.ONE_CLASS_SVM in types
        True

        Returns:
            可用的模型類型列表。
        """
        return list(cls._models.keys())
    
    @classmethod
    def is_supported(cls, model_type: ModelType) -> bool:
        """
        檢查模型類型是否被支援。
        
        >>> from src.models import ModelFactory, ModelType
        >>> ModelFactory.is_supported(ModelType.ISOLATION_FOREST)
        True
        >>> ModelFactory.is_supported(ModelType.XGBOOST)
        True
        >>> ModelFactory.is_supported(ModelType.LOCAL_OUTLIER_FACTOR)
        True
        >>> ModelFactory.is_supported(ModelType.ONE_CLASS_SVM)
        True

        Args:
            model_type: 模型類型。

        Returns:
            如果支援則返回 True，否則返回 False。
        """
        return model_type in cls._models

