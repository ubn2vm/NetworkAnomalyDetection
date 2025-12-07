"""
訓練相關工具函數

提供模型訓練過程中常用的工具函數，包括：
- contamination 參數計算
- 協議分組訓練
- 單一模型訓練
"""
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np

from src.label_processor import convert_label_to_binary
from src.models import ModelFactory, ModelType, BaseModel


def calculate_contamination(
    features_df: pd.DataFrame,
    multiplier: float = 1.3,
    max_contamination: float = 0.2,
    high_threshold: float = 0.15,
    min_contamination: float = 0.01,
    default: float = 0.1,
    verbose: bool = True
) -> Tuple[float, Optional[pd.Series]]:
    """
    計算 Isolation Forest 的 contamination 參數
    
    策略：
    1. 如果有標籤，使用實際異常比例的 multiplier 倍，但不超過 max_contamination
    2. 如果實際比例很高（>high_threshold），直接使用實際比例
    3. 確保 contamination 不小於 min_contamination
    4. 如果沒有標籤，使用 default 值
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Label': ['Normal', 'Botnet', 'Normal', 'Normal'],
    ...     'feature1': [1, 2, 3, 4]
    ... })
    >>> contamination, y_true = calculate_contamination(df, verbose=False)
    >>> 0.0 < contamination <= 0.2
    True
    >>> y_true is not None
    True
    
    Args:
        features_df: 包含標籤的 DataFrame
        multiplier: 實際比例的倍數（預設 1.3）
        max_contamination: 最大 contamination（預設 0.2）
        high_threshold: 高異常比例閾值（預設 0.15）
        min_contamination: 最小 contamination（預設 0.01）
        default: 無標籤時的預設值（預設 0.1）
        verbose: 是否輸出詳細信息
    
    Returns:
        (contamination 值, y_true Series 或 None)
    """
    if 'Label' in features_df.columns:
        # 顯示標籤分布
        if verbose:
            print("\n   標籤分布統計：")
            label_counts = features_df['Label'].value_counts()
            print(label_counts.head(10))
        
        # 使用統一的標籤轉換函數
        if 'label_binary' not in features_df.columns:
            features_df = convert_label_to_binary(features_df, verbose=False)
        y_true = features_df['label_binary']
        
        # 顯示二元化後的分布
        if verbose:
            print(f"\n   二元化標籤分布：")
            print(f"     正常 (0): {(y_true == 0).sum():,} ({(y_true == 0).sum()/len(y_true)*100:.2f}%)")
            print(f"     異常 (1): {(y_true == 1).sum():,} ({(y_true == 1).sum()/len(y_true)*100:.2f}%)")
        
        actual_contamination = y_true.sum() / len(y_true)
        if verbose:
            print(f"   實際異常比例：{actual_contamination:.4f} ({actual_contamination*100:.2f}%)")
        
        # 改進的 contamination 策略
        # 使用實際比例的 multiplier 倍，但不超過 max_contamination
        contamination = min(actual_contamination * multiplier, max_contamination)
        # 如果實際比例很高（>high_threshold），直接使用實際比例
        if actual_contamination > high_threshold:
            contamination = actual_contamination
        # 確保 contamination 不小於最小值（IsolationForest 要求 > 0.0）
        contamination = max(contamination, min_contamination)
        
        if verbose:
            print(f"   設定 contamination：{contamination:.4f} ({contamination*100:.2f}%)")
        
        return contamination, y_true
    else:
        # 無標籤時，使用預設值
        contamination = default
        if verbose:
            print(f"   使用預設 contamination 參數：{contamination:.4f} (無標籤環境)")
        return contamination, None


def train_single_model(
    X: pd.DataFrame,
    contamination: float,
    feature_robust_scaler: Optional[Any] = None,
    random_state: int = 42,
    n_estimators: int = 300,
    max_samples: str = 'auto',
    max_features: Optional[float] = None,
    bootstrap: bool = True,
    verbose: bool = True
) -> Tuple[BaseModel, Any]:
    """
    訓練單一 Isolation Forest 模型
    
    Args:
        X: 特徵 DataFrame（應該已經被標準化過）
        contamination: 異常比例
        feature_robust_scaler: 外部 RobustScaler（如果使用）
        random_state: 隨機種子
        n_estimators: 樹的數量
        max_samples: 每棵樹使用的樣本數
        max_features: 每棵樹使用的特徵比例
        bootstrap: 是否使用 bootstrap
        verbose: 是否輸出詳細信息
    
    Returns:
        (訓練好的模型, scaler)
    """
    model = ModelFactory.create(ModelType.ISOLATION_FOREST)
    
    # 計算 max_features（如果未指定）
    n_features = len(X.columns)
    if max_features is None:
        if n_features <= 5:
            max_features = 1.0
        elif n_features <= 10:
            max_features = 0.8
        else:
            max_features = 0.6
    
    if verbose:
        print(f"   特徵數量：{n_features}，使用 max_features={max_features}")
    
    trained_model, scaler = model.train(
        X,
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        use_external_scaler=(feature_robust_scaler is not None),
        external_scaler=feature_robust_scaler if feature_robust_scaler is not None else None
    )
    
    if verbose:
        print("✅ 模型訓練完成")
        if feature_robust_scaler is not None:
            print("   ✅ 已使用 Log-Transformation + RobustScaler 優化特徵分佈")
            print("   ✅ 模型跳過內部標準化（避免雙重標準化）")
        else:
            print("   ✅ 已使用 StandardScaler 進行 Z-score 標準化（均值=0，標準差=1）")
    
    return model, scaler


def train_protocol_grouped_models(
    X: pd.DataFrame,
    features_df: pd.DataFrame,
    contamination: float,
    feature_robust_scaler: Optional[Any] = None,
    main_protocols: List[str] = None,
    min_samples: int = 1000,
    random_state: int = 42,
    n_estimators: int = 300,
    max_samples: str = 'auto',
    bootstrap: bool = True,
    verbose: bool = True
) -> Tuple[Dict[str, BaseModel], Dict[str, Any], Dict[str, float], bool]:
    """
    按協議分組訓練 Isolation Forest 模型
    
    Args:
        X: 特徵 DataFrame（應該已經被標準化過）
        features_df: 包含協議信息的完整 DataFrame
        contamination: 全局 contamination 參數
        feature_robust_scaler: 外部 RobustScaler（如果使用）
        main_protocols: 主要協議列表（預設 ['tcp', 'udp']）
        min_samples: 最小樣本數（低於此值跳過）
        random_state: 隨機種子
        n_estimators: 樹的數量
        max_samples: 每棵樹使用的樣本數
        bootstrap: 是否使用 bootstrap
        verbose: 是否輸出詳細信息
    
    Returns:
        (protocol_models, protocol_scalers, protocol_contaminations, success)
    """
    if main_protocols is None:
        main_protocols = ['tcp', 'udp']
    
    protocol_models = {}
    protocol_scalers = {}
    protocol_contaminations = {}
    
    # 只對主要協議（TCP/UDP）進行分組，其他協議合併處理
    all_protocols = features_df['Proto'].unique()
    other_protocols = [p for p in all_protocols if pd.notna(p) and str(p).lower() not in main_protocols]
    
    if verbose:
        print(f"   發現協議：{list(all_protocols)}")
        print(f"   主要協議（分組訓練）：{main_protocols}")
        print(f"   其他協議（合併處理）：{len(other_protocols)} 個")
    
    # 計算 max_features（所有協議共用）
    n_features = len(X.columns)
    if n_features <= 5:
        max_features = 1.0
    elif n_features <= 10:
        max_features = 0.8
    else:
        max_features = 0.6
    
    # 為主要協議（TCP/UDP）訓練模型
    for protocol in main_protocols:
        # 過濾該協議的數據
        proto_mask = features_df['Proto'].str.lower() == protocol.lower()
        X_proto = X[proto_mask]
        features_df_proto = features_df[proto_mask]
        
        if len(X_proto) < min_samples:
            if verbose:
                print(f"   ⚠️  {protocol.upper()} 協議樣本數不足 ({len(X_proto)} < {min_samples})，跳過")
            continue
        
        if verbose:
            print(f"\n   訓練 {protocol.upper()} 協議模型...")
            print(f"   {protocol.upper()} 流量：{len(X_proto):,} 筆 ({len(X_proto)/len(X)*100:.2f}%)")
        
        # 計算該協議的 contamination
        contamination_proto, _ = calculate_contamination(
            features_df_proto,
            multiplier=1.3,
            max_contamination=0.2,
            high_threshold=0.15,
            min_contamination=0.01,
            default=contamination,
            verbose=verbose
        )
        
        protocol_contaminations[protocol] = contamination_proto
        
        # 訓練該協議的模型
        model_proto = ModelFactory.create(ModelType.ISOLATION_FOREST)
        trained_model_proto, scaler_proto = model_proto.train(
            X_proto,
            contamination=contamination_proto,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            use_external_scaler=(feature_robust_scaler is not None),
            external_scaler=feature_robust_scaler if feature_robust_scaler is not None else None
        )
        
        protocol_models[protocol] = model_proto
        protocol_scalers[protocol] = scaler_proto
        
        if verbose:
            print(f"   ✅ {protocol.upper()} 協議模型訓練完成")
    
    # 處理其他協議（合併為一個模型）
    if other_protocols:
        other_mask = features_df['Proto'].isin(other_protocols)
        X_other = X[other_mask]
        features_df_other = features_df[other_mask]
        
        if len(X_other) >= min_samples:
            if verbose:
                print(f"\n   訓練其他協議模型（合併 {len(other_protocols)} 個協議）...")
                print(f"   其他協議流量：{len(X_other):,} 筆 ({len(X_other)/len(X)*100:.2f}%)")
            
            # 計算其他協議的 contamination
            contamination_other, _ = calculate_contamination(
                features_df_other,
                multiplier=1.3,
                max_contamination=0.2,
                high_threshold=0.15,
                min_contamination=0.01,
                default=contamination,
                verbose=verbose
            )
            
            # 訓練其他協議的模型
            model_other = ModelFactory.create(ModelType.ISOLATION_FOREST)
            trained_model_other, scaler_other = model_other.train(
                X_other,
                contamination=contamination_other,
                random_state=random_state,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                bootstrap=bootstrap,
                use_external_scaler=(feature_robust_scaler is not None),
                external_scaler=feature_robust_scaler if feature_robust_scaler is not None else None
            )
            
            protocol_models['other'] = model_other
            protocol_scalers['other'] = scaler_other
            protocol_contaminations['other'] = contamination_other
            
            if verbose:
                print(f"   ✅ 其他協議模型訓練完成")
        else:
            if verbose:
                print(f"   ⚠️  其他協議樣本數不足 ({len(X_other)} < {min_samples})，跳過")
    
    success = len(protocol_models) > 0
    
    if verbose:
        if success:
            print(f"\n   ✅ 協議分組訓練完成，共 {len(protocol_models)} 個協議模型")
        else:
            print("   ⚠️  沒有足夠樣本的協議，回退到單一模型訓練")
    
    return protocol_models, protocol_scalers, protocol_contaminations, success

