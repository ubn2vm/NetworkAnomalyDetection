"""
白名單分析與應用模組

提供 False Positive 模式分析和白名單規則應用功能。
使用類別封裝，遵循單一職責原則，可獨立使用。

使用範例：
    >>> import pandas as pd
    >>> import numpy as np
    >>> from src.whitelist import WhitelistAnalyzer, WhitelistApplier
    >>> 
    >>> # 準備資料（只需要包含必要欄位的 DataFrame）
    >>> features_df = pd.DataFrame({
    ...     'Proto': ['TCP', 'UDP', 'TCP'],
    ...     'Dport': [80, 53, 443],
    ...     'DstAddr': ['192.168.1.1', '8.8.8.8', '10.0.0.1']
    ... })
    >>> y_pred = np.array([1, 1, 0])  # 預測結果
    >>> y_true = np.array([0, 0, 0])  # 真實標籤（可選）
    >>> 
    >>> # 分析 FP 模式
    >>> analyzer = WhitelistAnalyzer(verbose=False)
    >>> rules = analyzer.analyze_fp_patterns(features_df, y_pred, y_true)
    >>> isinstance(rules, list)
    True
    >>> 
    >>> # 應用規則
    >>> applier = WhitelistApplier(verbose=False)
    >>> y_filtered, stats = applier.apply_rules(y_pred, features_df, rules)
    >>> len(y_filtered) == len(y_pred)
    True
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import ipaddress
import json
from pathlib import Path
from enum import Enum


class WhitelistRuleType(Enum):
    """白名單規則類型"""
    PROTO_PORT = "proto_port"
    PROTO_PORT_BEHAVIORAL = "proto_port_behavioral"
    PROTO_IP = "proto_ip"
    PROTO_PORT_IP = "proto_port_ip"
    PROTO_PORT_RANGE = "proto_port_range"
    PORT = "port"
    PORT_BEHAVIORAL = "port_behavioral"


class WhitelistAnalyzer:
    """
    白名單規則分析器
    
    分析訓練集上的 False Positives 模式，歸納白名單規則。
    結合協議、端口、IP 等網路層資訊。
    可獨立使用，只需提供包含必要欄位的 DataFrame 和預測結果。
    
    必要欄位：
        - Proto: 協議（如 'TCP', 'UDP'）
        - Dport: 目標端口
        - DstAddr: 目標 IP（可選，用於 IP 相關規則）
        - SrcAddr: 來源 IP（可選）
    
    可選欄位（用於行為特徵分析）：
        - TotBytes, TotPkts, SrcBytes, DstBytes, Dur 等
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # 創建測試資料
    >>> df = pd.DataFrame({
    ...     'Proto': ['TCP', 'UDP', 'TCP'],
    ...     'Dport': [80, 53, 443]
    ... })
    >>> y_pred = np.array([1, 1, 0])
    >>> y_true = np.array([0, 0, 0])
    >>> 
    >>> analyzer = WhitelistAnalyzer(verbose=False)
    >>> rules = analyzer.analyze_fp_patterns(df, y_pred, y_true)
    >>> isinstance(rules, list)
    True
    """
    
    def __init__(
        self,
        fp_ratio_threshold: float = 0.05,
        normal_ratio_threshold: float = 0.01,
        attack_ratio_threshold: float = 0.05,
        anomaly_score_threshold: Optional[float] = None,
        use_scoring_method: bool = False,  # 🔧 新增：是否使用評分方法（預設 False，保持向後兼容）
        top_n_combos: int = 20,  # 🔧 新增：使用評分方法時，選擇 Top-N 個組合
        min_combo_samples: int = 50,  # 🔧 新增：最小樣本量要求
        score_threshold: Optional[float] = None,  # 🔧 新增：評分閾值（可選，與 top_n_combos 二選一）
        verbose: bool = True
    ):
        """
        初始化分析器
        
        Args:
            fp_ratio_threshold: FP 佔比閾值（使用閾值方法時，超過此值才考慮加入白名單）
            normal_ratio_threshold: 正常流量佔比閾值（在正常流量中也常見才加入白名單）
            attack_ratio_threshold: 攻擊者佔比閾值（規則匹配的流量中，攻擊者比例不能超過此值）
            anomaly_score_threshold: 異常分數閾值（可選，只對低分數流量應用白名單）
            use_scoring_method: 是否使用評分方法（True）或固定閾值方法（False，預設）
            top_n_combos: 使用評分方法時，選擇 Top-N 個組合（預設 20）
            min_combo_samples: 最小樣本量要求（確保統計可靠性，預設 50）
            score_threshold: 評分閾值（可選，如果設置則使用評分閾值而非 Top-N）
            verbose: 是否輸出詳細信息
        """
        self.fp_ratio_threshold = fp_ratio_threshold
        self.normal_ratio_threshold = normal_ratio_threshold
        self.attack_ratio_threshold = attack_ratio_threshold
        self.anomaly_score_threshold = anomaly_score_threshold
        self.use_scoring_method = use_scoring_method
        self.top_n_combos = top_n_combos
        self.min_combo_samples = min_combo_samples
        self.score_threshold = score_threshold
        self.verbose = verbose
    
    def analyze_fp_patterns(
        self,
        features_df: pd.DataFrame,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        anomaly_scores: Optional[np.ndarray] = None,
        cleaned_df: Optional[pd.DataFrame] = None,
        train_idx: Optional[pd.Index] = None
    ) -> List[Tuple[str, dict, float, float, float]]:
        """
        分析 False Positives 模式，歸納白名單規則
        
        可獨立使用，只需提供 DataFrame 和預測結果。
        
        Args:
            features_df: 特徵 DataFrame（必須包含 Proto, Dport 等欄位）
            y_pred: 預測結果（1=異常, 0=正常）
            y_true: 真實標籤（1=異常, 0=正常）
            anomaly_scores: 異常分數（可選，用於更精確的分析）
            cleaned_df: 清洗後的原始 DataFrame（可選，用於獲取缺失欄位）
            train_idx: 訓練集索引（可選，用於從 cleaned_df 獲取資料）
        
        Returns:
            白名單規則列表，每個規則為：
            (規則名稱, 規則參數字典, FP佔比, 正常流量佔比, 攻擊者佔比)
        """
        # 驗證輸入
        self._validate_inputs(features_df, y_pred, y_true)
        
        # 識別 False Positives
        fp_mask = (y_pred == 1) & (y_true == 0)
        num_fp = fp_mask.sum()
        
        if num_fp == 0:
            if self.verbose:
                print("   ⚠️  沒有 False Positives，無法歸納規則")
            return []
        
        if self.verbose:
            print(f"   📊 False Positives：{num_fp:,} 筆")
        
        # 獲取原始資訊
        original_info = features_df.copy()
        
        # 檢查必要欄位，如果缺少則從 cleaned_df 取得
        required_cols = ['Proto', 'DstAddr', 'Dport', 'SrcAddr']
        missing_cols = [col for col in required_cols if col not in original_info.columns]
        
        if missing_cols and cleaned_df is not None and train_idx is not None:
            cleaned_train = cleaned_df.loc[train_idx]
            for col in missing_cols:
                if col in cleaned_train.columns:
                    original_info[col] = cleaned_train[col]
                    if self.verbose:
                        print(f"      ✅ 從 cleaned_df 取得 {col}")
        
        # 分析 FP 的模式
        fp_df = original_info[fp_mask].copy()
        normal_mask = (y_true == 0)
        normal_df = original_info[normal_mask].copy()
        attack_mask = (y_true == 1)
        attack_df = original_info[attack_mask].copy() if len(original_info[attack_mask]) > 0 else pd.DataFrame()
        
        # 🔧 新增：識別 False Negatives（用於雙向驗證）
        fn_mask = (y_pred == 0) & (y_true == 1)
        fn_df = original_info[fn_mask].copy() if fn_mask.sum() > 0 else pd.DataFrame()
        num_fn = fn_mask.sum()
        
        if self.verbose and num_fn > 0:
            print(f"   📊 False Negatives：{num_fn:,} 筆（用於雙向驗證）")
        
        if self.verbose:
            print(f"\n   📈 False Positive 模式分析：")
            
            # 1. 協議分布
            if 'Proto' in fp_df.columns:
                proto_counts = fp_df['Proto'].value_counts()
                print(f"      - 協議分布：")
                for proto, count in proto_counts.head(5).items():
                    pct = count / num_fp * 100
                    print(f"        {proto}: {count:,} ({pct:.1f}%)")
            
            # 2. 端口分布
            if 'Dport' in fp_df.columns:
                port_counts = fp_df['Dport'].value_counts()
                print(f"      - 常見端口（Top 10）：")
                for port, count in port_counts.head(10).items():
                    pct = count / num_fp * 100
                    print(f"        Port {port}: {count:,} ({pct:.1f}%)")
            
            # 3. 協議+端口組合
            if 'Proto' in fp_df.columns and 'Dport' in fp_df.columns:
                proto_port_counts = fp_df.groupby(['Proto', 'Dport']).size().sort_values(ascending=False)
                print(f"      - 協議+端口組合（Top 10）：")
                for (proto, port), count in proto_port_counts.head(10).items():
                    pct = count / num_fp * 100
                    print(f"        {proto.upper()}/{port}: {count:,} ({pct:.1f}%)")
        
        # 定義可用於區分正常和攻擊的行為特徵
        behavioral_features = [
            'TotBytes', 'TotPkts', 'SrcBytes', 'DstBytes', 'Dur',
            'flow_ratio', 'bytes_symmetry', 'packet_size', 
            'bytes_per_second', 'packets_per_second',
            'unique_dst_per_minute_by_src', 'unique_dport_per_minute_by_src',
            'flows_per_minute_by_src', 'total_bytes_per_minute_by_src'
        ]
        
        # 檢查哪些行為特徵可用
        available_behavioral_features = []
        for feat in behavioral_features:
            if feat in original_info.columns:
                available_behavioral_features.append(feat)
            elif f'log_{feat}' in original_info.columns:
                available_behavioral_features.append(f'log_{feat}')
        
        if self.verbose:
            print(f"   📊 可用於行為分析的特徵：{len(available_behavioral_features)} 個")
            print(f"\n   🔍 歸納白名單規則（檢查攻擊者比例 + 行為特徵差異）...")
        
        whitelist_rules = []
        
        # 規則 1: 高頻協議+端口組合
        if 'Proto' in fp_df.columns and 'Dport' in fp_df.columns:
            proto_port_counts = fp_df.groupby(['Proto', 'Dport']).size().sort_values(ascending=False)
            proto_port_fp_ratio = proto_port_counts / num_fp
            
            if self.use_scoring_method:
                # 🔧 方法 2：使用綜合評分 + Top-N 或評分閾值
                if self.verbose:
                    print(f"   📊 使用評分方法分析協議+端口組合...")
                
                # 計算每個組合的綜合重要性評分
                combo_scores = []
                for (proto, port), fp_count in proto_port_counts.items():
                    port_float = float(port)
                    proto_lower = proto.lower()
                    fp_ratio = proto_port_fp_ratio[(proto, port)]
                    
                    # 檢查在正常和攻擊流量中的比例
                    normal_combo_count = ((normal_df['Proto'].str.lower() == proto_lower) & 
                                         (normal_df['Dport'].astype(float) == port_float)).sum()
                    normal_combo_ratio = normal_combo_count / len(normal_df) if len(normal_df) > 0 else 0.0
                    
                    attack_combo_count = 0
                    attack_combo_ratio = 0.0
                    if len(attack_df) > 0 and 'Proto' in attack_df.columns and 'Dport' in attack_df.columns:
                        attack_combo_count = ((attack_df['Proto'].str.lower() == proto_lower) & 
                                             (attack_df['Dport'].astype(float) == port_float)).sum()
                        attack_combo_ratio = attack_combo_count / len(attack_df) if len(attack_df) > 0 else 0.0
                    
                    # 計算綜合重要性評分
                    importance_score = self._calculate_combo_importance_score(
                        fp_ratio=fp_ratio,
                        normal_ratio=normal_combo_ratio,
                        attack_ratio=attack_combo_ratio,
                        fp_count=int(fp_count),
                        normal_count=int(normal_combo_count)
                    )
                    
                    combo_scores.append({
                        'proto': proto,
                        'port': port,
                        'score': importance_score,
                        'fp_ratio': fp_ratio,
                        'normal_ratio': normal_combo_ratio,
                        'attack_ratio': attack_combo_ratio,
                        'fp_count': int(fp_count),
                        'normal_count': int(normal_combo_count)
                    })
                
                # 按評分排序
                combo_scores_df = pd.DataFrame(combo_scores)
                combo_scores_df = combo_scores_df.sort_values('score', ascending=False)
                
                # 選擇 Top-N 或評分閾值
                if self.score_threshold is not None:
                    # 使用評分閾值
                    top_combos = combo_scores_df[combo_scores_df['score'] > self.score_threshold]
                    if self.verbose:
                        if len(top_combos) > 0:
                            print(f"   🎯 使用評分閾值 {self.score_threshold:.3f}，選中 {len(top_combos)} 個組合（評分範圍：{top_combos['score'].min():.3f} - {top_combos['score'].max():.3f}）")
                        else:
                            print(f"   ⚠️  使用評分閾值 {self.score_threshold:.3f}，沒有組合符合條件")
                else:
                    # 使用 Top-N
                    top_combos = combo_scores_df.head(self.top_n_combos)
                    if self.verbose:
                        if len(top_combos) > 0:
                            print(f"   🎯 選擇 Top {len(top_combos)} 個組合（評分範圍：{top_combos['score'].min():.3f} - {top_combos['score'].max():.3f}）")
                        else:
                            print(f"   ⚠️  沒有組合可選")
                
                # 對選中的組合進行詳細分析
                for _, combo_info in top_combos.iterrows():
                    proto = combo_info['proto']
                    port = combo_info['port']
                    ratio = combo_info['fp_ratio']
                    normal_combo_ratio = combo_info['normal_ratio']
                    attack_combo_ratio = combo_info['attack_ratio']
                    score = combo_info['score']
                    
                    port_float = float(port)
                    proto_lower = proto.lower()
                    
                    if self.verbose:
                        print(f"      🔍 檢查 {proto.upper()}/{port}: 評分={score:.3f}, FP佔比={ratio*100:.1f}%, 正常={normal_combo_ratio*100:.2f}%, 攻擊={attack_combo_ratio*100:.2f}%")
                    
                    # 後續處理邏輯（生成規則、行為特徵分析、雙向驗證等）
                    # 🔧 方案一：完全移除攻擊者佔比限制，只要正常流量中常見就嘗試生成規則
                    # 讓雙向驗證（FN 檢查）來決定是否安全
                    if normal_combo_ratio > self.normal_ratio_threshold:
                        if self.verbose:
                            print(f"         → 正常流量佔比 {normal_combo_ratio*100:.2f}% > {self.normal_ratio_threshold*100:.1f}%，嘗試生成規則（攻擊者佔比: {attack_combo_ratio*100:.2f}%）")
                        
                        # 先嘗試添加行為特徵條件
                        behavioral_conditions = self._analyze_behavioral_differences(
                            normal_df, attack_df, proto_lower, port_float, available_behavioral_features,
                            fn_df=fn_df if len(fn_df) > 0 else None,
                            max_features=5
                        )
                        
                        if behavioral_conditions:
                            # 找到行為特徵差異，生成帶行為特徵的規則
                            rule_name = f"{proto.upper()}/{port} (行為特徵過濾)"
                            rule_params = {
                                'type': 'proto_port_behavioral',
                                'proto': proto_lower,
                                'port': port_float,
                                'anomaly_score_threshold': self.anomaly_score_threshold,
                                'behavioral_conditions': behavioral_conditions
                            }
                            if self.verbose:
                                print(f"         → 找到 {len(behavioral_conditions)} 個行為特徵差異，生成帶行為特徵的規則")
                        else:
                            # 沒找到行為特徵差異，生成簡單規則
                            rule_name = f"{proto.upper()}/{port}"
                            rule_params = {
                                'type': 'proto_port',
                                'proto': proto_lower,
                                'port': port_float,
                                'anomaly_score_threshold': self.anomaly_score_threshold
                            }
                            if self.verbose:
                                print(f"         → 未找到行為特徵差異，生成簡單規則")
                        
                        # 🔧 關鍵：只做雙向驗證（FN 檢查），讓驗證來決定是否安全
                        if self._validate_rule_against_attacks(rule_params, original_info, attack_mask, fn_mask):
                            whitelist_rules.append((rule_name, rule_params, ratio, normal_combo_ratio, attack_combo_ratio))
                            if self.verbose:
                                print(f"      ✅ 規則：{rule_name} (FP佔比: {ratio*100:.1f}%, 正常: {normal_combo_ratio*100:.1f}%, 攻擊: {attack_combo_ratio*100:.1f}%)")
                                if behavioral_conditions:
                                    for feat, cond in behavioral_conditions.items():
                                        if 'max' in cond:
                                            print(f"         - {feat} < {cond['max']:.2f}")
                        elif self.verbose:
                            print(f"      ⚠️  規則 {rule_name} 可能誤殺真實攻擊，已跳過（雙向驗證失敗）")
                    else:
                        # 正常流量中不常見，跳過
                        if self.verbose:
                            print(f"         → 跳過：正常流量佔比 {normal_combo_ratio*100:.2f}% <= {self.normal_ratio_threshold*100:.1f}%")
            
            else:
                # 🔧 方法 1：使用固定閾值（原有方法，保持向後兼容）
                high_freq_combos = proto_port_fp_ratio[proto_port_fp_ratio > self.fp_ratio_threshold]
                
                if self.verbose:
                    print(f"   📊 使用閾值方法（fp_ratio_threshold={self.fp_ratio_threshold*100:.1f}%），找到 {len(high_freq_combos)} 個組合")
                
                for (proto, port), ratio in high_freq_combos.items():
                    port_float = float(port)
                    proto_lower = proto.lower()
                    
                    # 檢查在正常和攻擊流量中的比例
                    normal_combo_count = ((normal_df['Proto'].str.lower() == proto_lower) & 
                                         (normal_df['Dport'].astype(float) == port_float)).sum()
                    normal_combo_ratio = normal_combo_count / len(normal_df) if len(normal_df) > 0 else 0.0
                    
                    attack_combo_count = 0
                    attack_combo_ratio = 0.0
                    if len(attack_df) > 0 and 'Proto' in attack_df.columns and 'Dport' in attack_df.columns:
                        attack_combo_count = ((attack_df['Proto'].str.lower() == proto_lower) & 
                                             (attack_df['Dport'].astype(float) == port_float)).sum()
                        attack_combo_ratio = attack_combo_count / len(attack_df) if len(attack_df) > 0 else 0.0
                    
                    # 🔧 調試輸出：顯示每個組合的詳細信息
                    if self.verbose:
                        print(f"      🔍 檢查 {proto.upper()}/{port}: FP佔比={ratio*100:.1f}%, 正常={normal_combo_ratio*100:.2f}%, 攻擊={attack_combo_ratio*100:.2f}%")
                        print(f"         條件檢查: normal_combo_ratio > {self.normal_ratio_threshold*100:.1f}%? {normal_combo_ratio > self.normal_ratio_threshold}")
                        print(f"         條件檢查: attack_combo_ratio < {self.attack_ratio_threshold*100:.1f}%? {attack_combo_ratio < self.attack_ratio_threshold}")
                        print(f"         條件檢查: attack_combo_ratio < {self.attack_ratio_threshold*2*100:.1f}%? {attack_combo_ratio < self.attack_ratio_threshold * 2}")
                    
                    # 🔧 方案一：完全移除攻擊者佔比限制，只要正常流量中常見就嘗試生成規則
                    # 讓雙向驗證（FN 檢查）來決定是否安全
                    if normal_combo_ratio > self.normal_ratio_threshold:
                        if self.verbose:
                            print(f"         → 正常流量佔比 {normal_combo_ratio*100:.2f}% > {self.normal_ratio_threshold*100:.1f}%，嘗試生成規則（攻擊者佔比: {attack_combo_ratio*100:.2f}%）")
                        
                        # 先嘗試添加行為特徵條件
                        behavioral_conditions = self._analyze_behavioral_differences(
                            normal_df, attack_df, proto_lower, port_float, available_behavioral_features,
                            fn_df=fn_df if len(fn_df) > 0 else None,
                            max_features=5
                        )
                        
                        if behavioral_conditions:
                            # 找到行為特徵差異，生成帶行為特徵的規則
                            rule_name = f"{proto.upper()}/{port} (行為特徵過濾)"
                            rule_params = {
                                'type': 'proto_port_behavioral',
                                'proto': proto_lower,
                                'port': port_float,
                                'anomaly_score_threshold': self.anomaly_score_threshold,
                                'behavioral_conditions': behavioral_conditions
                            }
                            if self.verbose:
                                print(f"         → 找到 {len(behavioral_conditions)} 個行為特徵差異，生成帶行為特徵的規則")
                        else:
                            # 沒找到行為特徵差異，生成簡單規則
                            rule_name = f"{proto.upper()}/{port}"
                            rule_params = {
                                'type': 'proto_port',
                                'proto': proto_lower,
                                'port': port_float,
                                'anomaly_score_threshold': self.anomaly_score_threshold
                            }
                            if self.verbose:
                                print(f"         → 未找到行為特徵差異，生成簡單規則")
                        
                        # 🔧 關鍵：只做雙向驗證（FN 檢查），讓驗證來決定是否安全
                        if self._validate_rule_against_attacks(rule_params, original_info, attack_mask, fn_mask):
                            whitelist_rules.append((rule_name, rule_params, ratio, normal_combo_ratio, attack_combo_ratio))
                            if self.verbose:
                                print(f"      ✅ 規則：{rule_name} (FP佔比: {ratio*100:.1f}%, 正常: {normal_combo_ratio*100:.1f}%, 攻擊: {attack_combo_ratio*100:.1f}%)")
                                if behavioral_conditions:
                                    for feat, cond in behavioral_conditions.items():
                                        if 'max' in cond:
                                            print(f"         - {feat} < {cond['max']:.2f}")
                        elif self.verbose:
                            print(f"      ⚠️  規則 {rule_name} 可能誤殺真實攻擊，已跳過（雙向驗證失敗）")
                    else:
                        # 正常流量中不常見，跳過
                        if self.verbose:
                            print(f"         → 跳過：正常流量佔比 {normal_combo_ratio*100:.2f}% <= {self.normal_ratio_threshold*100:.1f}%")
        
        # 規則 2: 常見服務端口
        common_service_ports = {
            53: 'DNS',
            123: 'NTP',
            67: 'DHCP',
            68: 'DHCP',
            161: 'SNMP',
            5353: 'mDNS',
            80: 'HTTP',
            443: 'HTTPS',
            22: 'SSH',
            25: 'SMTP'
        }
        
        if 'Dport' in fp_df.columns:
            for port, service_name in common_service_ports.items():
                port_float = float(port)
                port_fp_count = (fp_df['Dport'].astype(float) == port_float).sum()
                port_fp_ratio = port_fp_count / num_fp
                
                if port_fp_ratio > self.fp_ratio_threshold:
                    normal_port_count = (normal_df['Dport'].astype(float) == port_float).sum()
                    normal_port_ratio = normal_port_count / len(normal_df) if len(normal_df) > 0 else 0.0
                    
                    attack_port_count = 0
                    attack_port_ratio = 0.0
                    if len(attack_df) > 0 and 'Dport' in attack_df.columns:
                        attack_port_count = (attack_df['Dport'].astype(float) == port_float).sum()
                        attack_port_ratio = attack_port_count / len(attack_df) if len(attack_df) > 0 else 0.0
                    
                    if normal_port_ratio > self.normal_ratio_threshold and attack_port_ratio < self.attack_ratio_threshold:
                        rule_name = f"{service_name} (Port {port})"
                        rule_params = {
                            'type': 'port',
                            'port': port_float,
                            'anomaly_score_threshold': self.anomaly_score_threshold
                        }
                        # 🔧 修正：在加入前進行驗證
                        if self._validate_rule_against_attacks(rule_params, original_info, attack_mask, fn_mask):
                            whitelist_rules.append((rule_name, rule_params, port_fp_ratio, normal_port_ratio, attack_port_ratio))
                            if self.verbose:
                                print(f"      ✅ 規則：{rule_name} (FP佔比: {port_fp_ratio*100:.1f}%, 正常: {normal_port_ratio*100:.1f}%, 攻擊: {attack_port_ratio*100:.1f}%)")
                        elif self.verbose:
                            print(f"      ⚠️  規則 {rule_name} 可能誤殺真實攻擊，已跳過")
                    elif normal_port_ratio > self.normal_ratio_threshold and attack_port_ratio < self.attack_ratio_threshold * 2:
                        # 🔧 改進：對於攻擊者佔比稍高的情況（< 2倍閾值），也嘗試添加行為特徵條件
                        # 需要先獲取協議資訊（從 FP 資料中）
                        if 'Proto' in fp_df.columns:
                            # 找到該端口最常見的協議
                            port_fp_df = fp_df[fp_df['Dport'].astype(float) == port_float]
                            if len(port_fp_df) > 0:
                                most_common_proto = port_fp_df['Proto'].mode()
                                if len(most_common_proto) > 0:
                                    proto_lower = most_common_proto[0].lower()
                                    behavioral_conditions = self._analyze_behavioral_differences(
                                        normal_df, attack_df, proto_lower, port_float, available_behavioral_features,
                                        fn_df=fn_df if len(fn_df) > 0 else None,
                                        max_features=5  # 🔧 改進：允許最多 5 個行為特徵條件
                                    )
                                    
                                    if behavioral_conditions:
                                        rule_name = f"{service_name} (Port {port}, 行為特徵過濾)"
                                        rule_params = {
                                            'type': 'port_behavioral',
                                            'port': port_float,
                                            'anomaly_score_threshold': self.anomaly_score_threshold,
                                            'behavioral_conditions': behavioral_conditions
                                        }
                                        # 🔧 新增：雙向驗證 - 檢查規則是否會誤殺真實攻擊
                                        if self._validate_rule_against_attacks(rule_params, original_info, attack_mask, fn_mask):
                                            whitelist_rules.append((rule_name, rule_params, port_fp_ratio, normal_port_ratio, attack_port_ratio))
                                            if self.verbose:
                                                print(f"      ✅ 規則（含行為特徵）：{rule_name}")
                                                for feat, cond in behavioral_conditions.items():
                                                    if 'max' in cond:
                                                        print(f"         - {feat} < {cond['max']:.2f}")
                                        elif self.verbose:
                                            print(f"      ⚠️  規則 {rule_name} 可能誤殺真實攻擊，已跳過")
                                    else:
                                        # 🔧 修正：即使沒有找到行為特徵差異，如果攻擊者佔比仍然較低（< 1.5倍閾值），也生成簡單規則
                                        if attack_port_ratio < self.attack_ratio_threshold * 1.5:
                                            rule_name = f"{service_name} (Port {port})"
                                            rule_params = {
                                                'type': 'port',
                                                'port': port_float,
                                                'anomaly_score_threshold': self.anomaly_score_threshold
                                            }
                                            # 驗證後加入
                                            if self._validate_rule_against_attacks(rule_params, original_info, attack_mask, fn_mask):
                                                whitelist_rules.append((rule_name, rule_params, port_fp_ratio, normal_port_ratio, attack_port_ratio))
                                                if self.verbose:
                                                    print(f"      ✅ 規則（攻擊者佔比低，無行為特徵差異）：{rule_name} (FP佔比: {port_fp_ratio*100:.1f}%, 正常: {normal_port_ratio*100:.1f}%, 攻擊: {attack_port_ratio*100:.1f}%)")
                                            elif self.verbose:
                                                print(f"      ⚠️  規則 {rule_name} 可能誤殺真實攻擊，已跳過")
        
        # 🔧 修正：規則在生成時已經驗證過了，不需要重複驗證
        # 直接返回已驗證的規則
        if self.verbose:
            print(f"\n   ✅ 歸納出 {len(whitelist_rules)} 個白名單規則（經過雙向驗證）")
        
        return whitelist_rules
    
    def _validate_inputs(
        self,
        features_df: pd.DataFrame,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ):
        """驗證輸入資料格式"""
        required_cols = ['Proto', 'Dport']
        missing_cols = [col for col in required_cols if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要欄位：{missing_cols}")
        
        if len(y_pred) != len(features_df):
            raise ValueError(f"y_pred 長度 ({len(y_pred)}) 與 features_df 長度 ({len(features_df)}) 不一致")
        
        if len(y_true) != len(features_df):
            raise ValueError(f"y_true 長度 ({len(y_true)}) 與 features_df 長度 ({len(features_df)}) 不一致")
    
    def _analyze_behavioral_differences(
        self,
        normal_df: pd.DataFrame,
        attack_df: pd.DataFrame,
        proto: str,
        port: float,
        available_features: List[str],
        fn_df: Optional[pd.DataFrame] = None,
        max_features: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        分析正常和攻擊流量的行為特徵差異
        
        使用正常流量的眾數作為典型值，並納入攻擊流量的 P10 來避免誤殺真實攻擊。
        
        Args:
            max_features: 最多添加的行為特徵條件數量（預設 5）
        """
        behavioral_conditions = {}
        
        # 檢查必要欄位是否存在（防禦性編程）
        # 當 attack_df 是空的 DataFrame（沒有列）時，需要提前返回
        required_cols = ['Proto', 'Dport']
        if len(normal_df) == 0 or not all(col in normal_df.columns for col in required_cols):
            return behavioral_conditions
        if len(attack_df) == 0 or not all(col in attack_df.columns for col in required_cols):
            return behavioral_conditions
        
        # 提取該協議+端口組合的正常和攻擊流量
        normal_mask = ((normal_df['Proto'].str.lower() == proto) & 
                      (normal_df['Dport'].astype(float) == port))
        attack_mask = ((attack_df['Proto'].str.lower() == proto) & 
                      (attack_df['Dport'].astype(float) == port))
        
        normal_flows = normal_df[normal_mask] if normal_mask.sum() > 0 else pd.DataFrame()
        attack_flows = attack_df[attack_mask] if attack_mask.sum() > 0 else pd.DataFrame()
        
        # 如果有 FN 資料，也提取
        fn_flows = pd.DataFrame()
        if fn_df is not None and len(fn_df) > 0 and all(col in fn_df.columns for col in required_cols):
            fn_mask = ((fn_df['Proto'].str.lower() == proto) & 
                      (fn_df['Dport'].astype(float) == port))
            fn_flows = fn_df[fn_mask] if fn_mask.sum() > 0 else pd.DataFrame()
        
        if len(normal_flows) == 0 or len(attack_flows) == 0:
            return behavioral_conditions
        
        for feat in available_features:
            if feat in normal_flows.columns and feat in attack_flows.columns:
                normal_values = normal_flows[feat].dropna()
                attack_values = attack_flows[feat].dropna()
                
                if len(normal_values) > 10 and len(attack_values) > 10:
                    # 正常流量的統計值
                    normal_p75 = normal_values.quantile(0.75)
                    normal_p95 = normal_values.quantile(0.95)  # 保持 P95 標準
                    # 計算眾數（最常見的值）
                    normal_mode = normal_values.mode()
                    normal_mode_value = normal_mode[0] if len(normal_mode) > 0 else normal_p75
                    
                    # 攻擊流量的統計值（使用更保守的 P5 而非 P10）
                    attack_p5 = attack_values.quantile(0.05)  # 攻擊的最小值（P5，更保守）
                    attack_p10 = attack_values.quantile(0.10)  # 保留 P10 作為備用
                    attack_p50 = attack_values.quantile(0.50)
                    
                    # 如果有 FN 資料，也計算
                    fn_p5 = None
                    fn_p10 = None
                    if len(fn_flows) > 0 and feat in fn_flows.columns:
                        fn_values = fn_flows[feat].dropna()
                        if len(fn_values) > 5:
                            fn_p5 = fn_values.quantile(0.05)  # FN 的 P5
                            fn_p10 = fn_values.quantile(0.10)
                    
                    # 🔧 改進：使用多種條件來檢測行為特徵差異，更容易找到差異
                    # 條件 1：攻擊的中位數明顯高於正常的 P75（放寬到 1.2 倍）
                    condition_1 = attack_p50 > normal_p75 * 1.2
                    # 條件 2：攻擊的 25% 分位數高於正常的 90% 分位數
                    attack_p25 = attack_values.quantile(0.25)
                    normal_p90 = normal_values.quantile(0.90)
                    condition_2 = attack_p25 > normal_p90
                    # 條件 3：攻擊的 10% 分位數高於正常的 95% 分位數
                    condition_3 = attack_p10 > normal_p95
                    
                    # 如果滿足任一條件，且尚未達到最大特徵數量
                    if (condition_1 or condition_2 or condition_3) and len(behavioral_conditions) < max_features:
                        # 使用更保守的閾值：優先使用 attack_p5（更嚴格）
                        # 確保不會誤殺攻擊流量
                        if attack_p5 > 0 and not np.isnan(attack_p5):
                            max_threshold = min(normal_p95, attack_p5)
                        else:
                            max_threshold = normal_p95
                        
                        
                        # 如果有 FN 資料，進一步收緊（使用 FN 的 P5 作為參考）
                        if fn_p5 is not None and fn_p5 < max_threshold:
                            max_threshold = min(max_threshold, fn_p5 * 0.9)  # 再保守 10%
                        elif fn_p10 is not None and fn_p10 < max_threshold:
                            max_threshold = min(max_threshold, fn_p10 * 0.9)  # 再保守 10%
                        
                        # 確保閾值有意義（不能小於正常流量的眾數）
                        if max_threshold > normal_mode_value:
                            behavioral_conditions[feat] = {'max': float(max_threshold)}
                            if self.verbose:
                                condition_desc = []
                                if condition_1:
                                    condition_desc.append("中位數>P75×1.2")
                                if condition_2:
                                    condition_desc.append("P25>P90")
                                if condition_3:
                                    condition_desc.append("P10>P95")
                                
                                info_parts = [
                                    f"條件: {', '.join(condition_desc)}",
                                    f"正常眾數: {normal_mode_value:.2f}",
                                    f"正常 P95: {normal_p95:.2f}",
                                    f"攻擊 P5: {attack_p5:.2f}",
                                    f"攻擊 P10: {attack_p10:.2f}",
                                    f"攻擊中位數: {attack_p50:.2f}"
                                ]
                                if fn_p5 is not None:
                                    info_parts.append(f"FN P5: {fn_p5:.2f}")
                                elif fn_p10 is not None:
                                    info_parts.append(f"FN P10: {fn_p10:.2f}")
                                info_parts.append(f"使用閾值: {max_threshold:.2f}")
                                print(f"         💡 發現差異：{feat} ({', '.join(info_parts)})")
        
        return behavioral_conditions
    
    def _calculate_combo_importance_score(
        self,
        fp_ratio: float,
        normal_ratio: float,
        attack_ratio: float,
        fp_count: int,
        normal_count: int
    ) -> float:
        """
        計算協議+端口組合的重要性評分
        
        綜合考慮：
        1. FP 佔比（越高越重要）
        2. 正常流量佔比（越高越好，表示是常見的正常流量）
        3. 攻擊流量佔比（越低越好，表示不是攻擊）
        4. 絕對數量（確保統計可靠性）
        
        Args:
            fp_ratio: FP 中該組合的佔比
            normal_ratio: 正常流量中該組合的佔比
            attack_ratio: 攻擊流量中該組合的佔比
            fp_count: FP 中該組合的絕對數量
            normal_count: 正常流量中該組合的絕對數量
        
        Returns:
            重要性評分（0-1 之間，越高越重要）
        """
        # 1. 樣本量檢查：如果樣本太少，降低評分
        sample_penalty = 1.0
        if fp_count < self.min_combo_samples:
            sample_penalty = fp_count / self.min_combo_samples  # 線性懲罰
        
        # 2. FP 佔比（權重 0.4）：越高越好
        fp_score = min(fp_ratio * 10, 1.0)  # 假設 10% 以上為滿分
        
        # 3. 正常流量佔比（權重 0.3）：越高越好，表示是常見的正常流量
        normal_score = min(normal_ratio * 20, 1.0)  # 假設 5% 以上為滿分
        
        # 4. 攻擊流量佔比（權重 0.3）：越低越好（反向評分）
        attack_score = max(0, 1.0 - attack_ratio * 10)  # 假設 10% 以下為滿分
        
        # 5. 綜合評分（加權平均）
        importance_score = (
            fp_score * 0.4 +
            normal_score * 0.3 +
            attack_score * 0.3
        ) * sample_penalty
        
        return importance_score
    
    def _validate_rule_against_attacks(
        self,
        rule_params: dict,
        original_info: pd.DataFrame,
        attack_mask: np.ndarray,
        fn_mask: np.ndarray
    ) -> bool:
        """
        雙向驗證規則：檢查規則是否會誤殺真實攻擊
        
        🔧 方案一：只檢查 FN，不檢查攻擊流量（讓規則能生成，再根據結果調整策略）
        
        Args:
            rule_params: 規則參數字典
            original_info: 原始特徵 DataFrame
            attack_mask: 攻擊流量的遮罩
            fn_mask: False Negatives 的遮罩
        
        Returns:
            True 如果規則安全（不會誤殺太多攻擊），False 否則
        """
        # 創建規則遮罩
        rule_mask = self._create_rule_mask_for_validation(rule_params, original_info)
        
        if rule_mask.sum() == 0:
            return True  # 如果規則不匹配任何流量，視為安全
        
        # 🔧 方案一：只檢查 FN，不檢查攻擊流量
        # 檢查該規則匹配的 FN 流量（這些是已經被誤判為正常的攻擊）
        matched_fn = (rule_mask & fn_mask).sum() if fn_mask.sum() > 0 else 0
        total_fn = fn_mask.sum() if fn_mask.sum() > 0 else 0
        
        # 只檢查 FN：如果匹配超過 5% 的 FN，視為不安全
        if total_fn > 0:
            fn_ratio = matched_fn / total_fn
            if fn_ratio > 0.10:
                if self.verbose:
                    print(f"         ⚠️  規則匹配 {fn_ratio*100:.2f}% 的 FN 流量 ({matched_fn:,}/{total_fn:,})")
                return False
        
        # 🔧 暫時移除攻擊流量檢查，讓規則能生成
        # 後續可以根據實際效果再調整策略
        # if total_attacks > 0:
        #     attack_ratio = matched_attacks / total_attacks
        #     if attack_ratio > 0.01:
        #         if self.verbose:
        #             print(f"         ⚠️  規則可能誤殺 {attack_ratio*100:.2f}% 的攻擊流量 ({matched_attacks:,}/{total_attacks:,})")
        #         return False
        
        return True
    
    def _create_rule_mask_for_validation(
        self,
        rule_params: dict,
        original_info: pd.DataFrame
    ) -> np.ndarray:
        """
        為驗證創建規則遮罩（類似 _create_rule_mask，但用於驗證階段）
        
        Args:
            rule_params: 規則參數字典
            original_info: 特徵 DataFrame
        
        Returns:
            布林遮罩陣列
        """
        rule_type = rule_params.get('type')
        
        if rule_type == 'proto_port' or rule_type == 'proto_port_behavioral':
            proto = rule_params.get('proto')
            port = rule_params.get('port')
            if 'Proto' in original_info.columns and 'Dport' in original_info.columns:
                rule_mask = (
                    (original_info['Proto'].str.lower() == proto).values & 
                    (original_info['Dport'].astype(float) == port).values
                )
            else:
                rule_mask = np.zeros(len(original_info), dtype=bool)
        elif rule_type == 'port' or rule_type == 'port_behavioral':
            port = rule_params.get('port')
            if 'Dport' in original_info.columns:
                rule_mask = (original_info['Dport'].astype(float) == port).values
            else:
                rule_mask = np.zeros(len(original_info), dtype=bool)
        else:
            # 其他規則類型暫時不處理
            rule_mask = np.zeros(len(original_info), dtype=bool)
        
        # 如果有行為特徵條件，也應用
        behavioral_conditions = rule_params.get('behavioral_conditions', {})
        if behavioral_conditions:
            for feat, cond in behavioral_conditions.items():
                feat_name = None
                if feat in original_info.columns:
                    feat_name = feat
                elif f'log_{feat}' in original_info.columns:
                    feat_name = f'log_{feat}'
                
                if feat_name:
                    feat_values = original_info[feat_name].values
                    if 'max' in cond:
                        feat_mask = (feat_values < cond['max']) | np.isnan(feat_values)
                        rule_mask = rule_mask & feat_mask
                    elif 'min' in cond:
                        feat_mask = (feat_values > cond['min']) | np.isnan(feat_values)
                        rule_mask = rule_mask & feat_mask
        
        return rule_mask
    
    def get_statistics(
        self,
        features_df: pd.DataFrame,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        whitelist_rules: List[Tuple[str, dict, float, float, float]]
    ) -> Dict[str, Any]:
        """
        獲取白名單規則的統計分析
        
        可獨立使用，用於分析規則的效果。
        
        Args:
            features_df: 特徵 DataFrame
            y_pred: 預測結果
            y_true: 真實標籤
            whitelist_rules: 白名單規則列表
        
        Returns:
            統計字典，包含：
            - total_rules: 規則總數
            - rules_by_type: 按類型分組的規則數量
            - estimated_fp_reduction: 估計的 FP 減少數量
            - coverage: 規則覆蓋的流量比例
        """
        stats = {
            'total_rules': len(whitelist_rules),
            'rules_by_type': {},
            'estimated_fp_reduction': 0,
            'coverage': 0.0
        }
        
        # 統計規則類型
        for rule_name, rule_params, fp_ratio, normal_ratio, attack_ratio in whitelist_rules:
            rule_type = rule_params.get('type', 'unknown')
            stats['rules_by_type'][rule_type] = stats['rules_by_type'].get(rule_type, 0) + 1
        
        # 計算估計的 FP 減少（基於 FP 佔比）
        total_fp = ((y_pred == 1) & (y_true == 0)).sum()
        estimated_reduction = sum(fp_ratio * total_fp for _, _, fp_ratio, _, _ in whitelist_rules)
        stats['estimated_fp_reduction'] = int(estimated_reduction)
        
        return stats
    
    def save_rules(
        self,
        rules: List[Tuple[str, dict, float, float, float]],
        filepath: Union[str, Path]
    ):
        """
        保存白名單規則到 JSON 檔案
        
        可獨立使用，方便規則的保存和分享。
        
        Args:
            rules: 白名單規則列表
            filepath: 保存路徑（.json 檔案）
        
        範例：
            >>> rules = [("TCP/80", {'type': 'proto_port', 'proto': 'tcp', 'port': 80.0}, 0.1, 0.05, 0.01)]
            >>> analyzer = WhitelistAnalyzer(verbose=False)
            >>> analyzer.save_rules(rules, 'data/models/whitelist_rules/my_rules.json')
        """
        filepath = Path(filepath)
        if not filepath.suffix == '.json':
            filepath = filepath.with_suffix('.json')
        
        # 轉換為可序列化的格式
        serializable_rules = []
        for rule_name, rule_params, fp_ratio, normal_ratio, attack_ratio in rules:
            serializable_rules.append({
                'name': rule_name,
                'params': rule_params,
                'fp_ratio': float(fp_ratio),
                'normal_ratio': float(normal_ratio),
                'attack_ratio': float(attack_ratio)
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_rules, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"   💾 已保存 {len(rules)} 個規則到 {filepath}")
    
    def load_rules(
        self,
        filepath: Union[str, Path]
    ) -> List[Tuple[str, dict, float, float, float]]:
        """
        從 JSON 檔案載入白名單規則
        
        可獨立使用，方便規則的載入和重用。
        
        Args:
            filepath: JSON 檔案路徑
        
        Returns:
            白名單規則列表
        
        範例：
            >>> analyzer = WhitelistAnalyzer(verbose=False)
            >>> rules = analyzer.load_rules('data/models/whitelist_rules/my_rules.json')
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"規則檔案不存在：{filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            serializable_rules = json.load(f)
        
        # 轉換回原始格式
        rules = []
        for rule_data in serializable_rules:
            rules.append((
                rule_data['name'],
                rule_data['params'],
                rule_data['fp_ratio'],
                rule_data['normal_ratio'],
                rule_data['attack_ratio']
            ))
        
        if self.verbose:
            print(f"   📂 已載入 {len(rules)} 個規則從 {filepath}")
        
        return rules
    
    @staticmethod
    def is_private_ip(ip_str: str) -> bool:
        """
        判斷是否為內網 IP（靜態方法，可獨立使用）
        
        >>> WhitelistAnalyzer.is_private_ip('192.168.1.1')
        True
        >>> WhitelistAnalyzer.is_private_ip('8.8.8.8')
        False
        """
        try:
            ip = ipaddress.ip_address(str(ip_str))
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except:
            return False


class WhitelistApplier:
    """
    白名單規則應用器
    
    將白名單規則應用到預測結果上，可修正 False Positives。
    可獨立使用，只需提供 DataFrame 和規則列表即可。
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> df = pd.DataFrame({'Proto': ['TCP'], 'Dport': [80]})
    >>> y_pred = np.array([1])
    >>> rules = [("TCP/80", {'type': 'proto_port', 'proto': 'tcp', 'port': 80.0}, 0.1, 0.05, 0.01)]
    >>> applier = WhitelistApplier(verbose=False)
    >>> y_filtered, stats = applier.apply_rules(y_pred, df, rules)
    >>> len(y_filtered) == len(y_pred)
    True
    """
    
    def __init__(
        self,
        verbose: bool = True,
        use_anomaly_score_filter: bool = True,
        anomaly_score_percentile: float = 75.0
    ):
        """
        初始化應用器
        
        Args:
            verbose: 是否輸出詳細信息
            use_anomaly_score_filter: 是否使用異常分數過濾（預設 True，但可以設為 False 完全移除限制）
            anomaly_score_percentile: 異常分數分位數（預設 75.0，即 75% 分位數）
        """
        self.verbose = verbose
        self.use_anomaly_score_filter = use_anomaly_score_filter
        self.anomaly_score_percentile = anomaly_score_percentile
    
    def apply_rules(
        self,
        y_pred: np.ndarray,
        features_df: pd.DataFrame,
        whitelist_rules: List[Tuple[str, dict, float, float, float]],
        anomaly_scores: Optional[np.ndarray] = None,
        cleaned_df: Optional[pd.DataFrame] = None,
        test_idx: Optional[pd.Index] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        應用白名單規則到預測結果
        
        可獨立使用，只需提供預測結果、特徵和規則。
        主要用於修正 False Positives（將被誤判為異常的正常流量標記為正常）。
        
        Args:
            y_pred: 原始預測結果（1=異常, 0=正常）
            features_df: 特徵 DataFrame
            whitelist_rules: 白名單規則列表
            anomaly_scores: 異常分數（可選，用於更精確的白名單應用）
            cleaned_df: 清洗後的原始 DataFrame（可選，用於獲取缺失欄位）
            test_idx: 測試集索引（可選，用於從 cleaned_df 獲取資料）
            y_true: 真實標籤（可選，用於計算效果）
        
        Returns:
            (修正後的預測結果, 效果統計字典)
        """
        if not whitelist_rules:
            return y_pred, {}
        
        if self.verbose:
            print("\n[應用白名單規則]...")
        
        # 驗證輸入
        if len(y_pred) != len(features_df):
            raise ValueError(f"y_pred 長度 ({len(y_pred)}) 與 features_df 長度 ({len(features_df)}) 不一致")
        
        # 獲取原始資訊（重置索引為位置索引，避免索引對齊問題）
        original_info = features_df.reset_index(drop=True).copy()
        
        # 檢查必要欄位
        required_cols = ['Proto', 'Dport']
        missing_cols = [col for col in required_cols if col not in original_info.columns]
        
        if missing_cols and cleaned_df is not None and test_idx is not None:
            try:
                if isinstance(test_idx, pd.RangeIndex) or (hasattr(test_idx, 'is_monotonic_increasing') and test_idx.is_monotonic_increasing and len(test_idx) == len(original_info)):
                    cleaned_test = cleaned_df.iloc[test_idx].reset_index(drop=True)
                else:
                    cleaned_test = cleaned_df.loc[test_idx].reset_index(drop=True)
                
                for col in missing_cols:
                    if col in cleaned_test.columns:
                        original_info[col] = cleaned_test[col].values
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  從 cleaned_df 獲取欄位時出錯：{e}")
        
        # 合併所有白名單遮罩
        whitelist_mask = np.zeros(len(original_info), dtype=bool)
        
        if self.verbose:
            print(f"   📋 應用 {len(whitelist_rules)} 個白名單規則：")
        
        # 計算異常分數閾值（用於過濾高置信度的異常預測）
        # 策略：只對分數接近閾值的異常預測應用白名單（這些是邊緣案例，更可能是 FP）
        if self.use_anomaly_score_filter and anomaly_scores is not None:
            predicted_anomaly_mask = (y_pred == 1)
            if predicted_anomaly_mask.sum() > 0:
                # 計算被預測為異常的流量的分數分佈
                anomaly_scores_only = anomaly_scores[predicted_anomaly_mask]
                # 使用指定分位數作為上限（只對分數較低的異常預測應用白名單）
                # 這些是「不太確定」的異常預測，更可能是 FP
                score_threshold = np.percentile(anomaly_scores_only, self.anomaly_score_percentile)
                if self.verbose:
                    print(f"   💡 異常分數閾值：{score_threshold:.4f} (只對被預測為異常且分數 < {score_threshold:.4f} 的流量應用白名單)")
                    low_score_anomalies = ((anomaly_scores < score_threshold) & predicted_anomaly_mask).sum()
                    total_anomalies = predicted_anomaly_mask.sum()
                    print(f"   📊 被預測為異常且分數 < 閾值的流量：{low_score_anomalies:,} / {total_anomalies:,} ({low_score_anomalies/total_anomalies*100:.1f}%)")
            else:
                score_threshold = None
                if self.verbose:
                    print(f"   ⚠️  沒有被預測為異常的流量，跳過異常分數檢查")
        elif not self.use_anomaly_score_filter:
            score_threshold = None
            if self.verbose:
                print(f"   💡 不使用異常分數過濾（基於規則匹配即可）")
        else:
            score_threshold = None
            if self.verbose:
                print(f"   ⚠️  未提供異常分數，不進行分數過濾")
        
        # 應用每個規則
        for rule_name, rule_params, fp_ratio, normal_ratio, attack_ratio in whitelist_rules:
            rule_mask = self._create_rule_mask(rule_params, original_info)
            
            # 應用行為特徵條件
            behavioral_conditions = rule_params.get('behavioral_conditions', {})
            if behavioral_conditions:
                initial_count = rule_mask.sum()
                for feat, cond in behavioral_conditions.items():
                    feat_name = None
                    if feat in original_info.columns:
                        feat_name = feat
                    elif f'log_{feat}' in original_info.columns:
                        feat_name = f'log_{feat}'
                    
                    if feat_name:
                        feat_values = original_info[feat_name].values
                        if 'max' in cond:
                            feat_mask = (feat_values < cond['max']) | np.isnan(feat_values)
                            rule_mask = rule_mask & feat_mask
                        elif 'min' in cond:
                            feat_mask = (feat_values > cond['min']) | np.isnan(feat_values)
                            rule_mask = rule_mask & feat_mask
                
                if self.verbose and initial_count > rule_mask.sum():
                    behavioral_filtered = initial_count - rule_mask.sum()
                    print(f"         (行為特徵過濾：{initial_count:,} → {rule_mask.sum():,}, 過濾 {behavioral_filtered:,} 筆)")
            
            # 應用異常分數閾值（只對被預測為異常的流量進行分數過濾）
            rule_anomaly_threshold = rule_params.get('anomaly_score_threshold', score_threshold)
            if self.use_anomaly_score_filter and anomaly_scores is not None and rule_anomaly_threshold is not None:
                if len(anomaly_scores) == len(rule_mask):
                    # 只對被預測為異常的流量應用分數過濾
                    predicted_anomaly_mask = (y_pred == 1)
                    # 對於被預測為異常的流量，只保留分數 < 閾值的（邊緣案例）
                    # 對於被預測為正常的流量，不進行分數過濾
                    low_score_mask = (
                        (~predicted_anomaly_mask) |  # 正常預測：不進行分數過濾
                        (predicted_anomaly_mask & (anomaly_scores < rule_anomaly_threshold))  # 異常預測：只保留低分數的
                    )
                    total_matched = rule_mask.sum()
                    rule_mask = rule_mask & low_score_mask
                    filtered_count = rule_mask.sum()
                    if self.verbose and total_matched > filtered_count:
                        score_filtered = total_matched - filtered_count
                        print(f"         (異常分數過濾：{total_matched:,} → {filtered_count:,}, 過濾 {score_filtered:,} 筆)")
                    elif self.verbose and not behavioral_conditions:
                        print(f"      - {rule_name}: {filtered_count:,} 筆流量")
                elif self.verbose:
                    print(f"      ⚠️  異常分數長度不匹配：{len(anomaly_scores)} vs {len(rule_mask)}，跳過異常分數檢查")
                    print(f"      - {rule_name}: {rule_mask.sum():,} 筆流量")
            elif self.verbose and not behavioral_conditions:
                print(f"      - {rule_name}: {rule_mask.sum():,} 筆流量")
            
            whitelist_mask = whitelist_mask | rule_mask
        
        num_whitelisted = whitelist_mask.sum()
        if self.verbose:
            print(f"   🛡️  符合白名單規則的流量總數：{num_whitelisted:,}")
        
        # 🔧 關鍵修正：只對被預測為異常的流量應用白名單
        # 白名單的目的是修正 FP（將誤判為異常的正常流量改為正常）
        # 所以只應用到 y_pred == 1 的流量
        predicted_anomaly_mask = (y_pred == 1)
        whitelist_mask = whitelist_mask & predicted_anomaly_mask
        
        num_whitelisted_anomalies = whitelist_mask.sum()
        if self.verbose:
            print(f"   🔍 其中被預測為異常的流量：{num_whitelisted_anomalies:,} 筆")
            if num_whitelisted > 0:
                anomaly_ratio = num_whitelisted_anomalies / num_whitelisted * 100
                print(f"   📊 符合規則的流量中被預測為異常的比例：{anomaly_ratio:.1f}%")
        
        # 應用白名單（將符合規則的異常預測改為正常）
        y_pred_original = y_pred.copy()
        y_pred_filtered = y_pred.copy()
        y_pred_filtered[whitelist_mask] = 0
        
        # 計算效果統計
        stats = {
            'original_anomalies': int(y_pred_original.sum()),
            'filtered_anomalies': int(y_pred_filtered.sum()),
            'reduced_anomalies': int(y_pred_original.sum() - y_pred_filtered.sum()),
            'whitelisted_count': int(num_whitelisted)
        }
        
        if y_true is not None:
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            elif not isinstance(y_true, np.ndarray):
                y_true = np.array(y_true)
            
            if len(y_true) == len(y_pred_original):
                rescued_fp = ((y_pred_original == 1) & (y_true == 0) & whitelist_mask).sum()
                wrongly_whitelisted = ((y_pred_original == 1) & (y_true == 1) & whitelist_mask).sum()
                stats['rescued_fp'] = int(rescued_fp)
                stats['wrongly_whitelisted'] = int(wrongly_whitelisted)
                
                if self.verbose:
                    print(f"   📉 成功消除的 False Positives：{rescued_fp:,}")
                    if wrongly_whitelisted > 0:
                        print(f"   ⚠️  誤將攻擊者放入白名單：{wrongly_whitelisted:,}")
                    else:
                        print(f"   ✅ 未誤殺任何真實攻擊")
        
        if self.verbose:
            print(f"   📊 修正後的預測異常數量：{y_pred_filtered.sum():,} (原：{y_pred_original.sum():,})")
            print(f"   📉 減少異常預測：{stats['reduced_anomalies']:,} 筆")
        
        return y_pred_filtered, stats
    
    def _create_rule_mask(
        self,
        rule_params: dict,
        original_info: pd.DataFrame
    ) -> np.ndarray:
        """
        根據規則參數創建遮罩
        
        Args:
            rule_params: 規則參數字典
            original_info: 特徵 DataFrame
        
        Returns:
            布林遮罩陣列
        """
        rule_type = rule_params.get('type')
        
        if rule_type == 'proto_port' or rule_type == 'proto_port_behavioral':
            proto = rule_params.get('proto')
            port = rule_params.get('port')
            if 'Proto' in original_info.columns and 'Dport' in original_info.columns:
                rule_mask = (
                    (original_info['Proto'].str.lower() == proto).values & 
                    (original_info['Dport'].astype(float) == port).values
                )
            else:
                rule_mask = np.zeros(len(original_info), dtype=bool)
        
        elif rule_type == 'port' or rule_type == 'port_behavioral':
            port = rule_params.get('port')
            if 'Dport' in original_info.columns:
                rule_mask = (original_info['Dport'].astype(float) == port).values
            else:
                rule_mask = np.zeros(len(original_info), dtype=bool)
        
        elif rule_type == 'proto_ip':
            proto = rule_params.get('proto')
            ip = rule_params.get('ip')
            if 'Proto' in original_info.columns and 'DstAddr' in original_info.columns:
                rule_mask = (
                    (original_info['Proto'].str.lower() == proto).values & 
                    (original_info['DstAddr'] == ip).values
                )
            else:
                rule_mask = np.zeros(len(original_info), dtype=bool)
        
        elif rule_type == 'proto_port_ip':
            proto = rule_params.get('proto')
            port = rule_params.get('port')
            ip = rule_params.get('ip')
            if 'Proto' in original_info.columns and 'Dport' in original_info.columns and 'DstAddr' in original_info.columns:
                rule_mask = (
                    (original_info['Proto'].str.lower() == proto).values & 
                    (original_info['Dport'].astype(float) == port).values &
                    (original_info['DstAddr'] == ip).values
                )
            else:
                rule_mask = np.zeros(len(original_info), dtype=bool)
        
        elif rule_type == 'proto_port_range':
            proto = rule_params.get('proto')
            port_min = rule_params.get('port_min')
            port_max = rule_params.get('port_max')
            if 'Proto' in original_info.columns and 'Dport' in original_info.columns:
                rule_mask = (
                    (original_info['Proto'].str.lower() == proto).values & 
                    (original_info['Dport'].astype(float) >= port_min).values &
                    (original_info['Dport'].astype(float) <= port_max).values
                )
            else:
                rule_mask = np.zeros(len(original_info), dtype=bool)
        
        else:
            rule_mask = np.zeros(len(original_info), dtype=bool)
        
        return rule_mask


if __name__ == '__main__':
    # 簡單測試
    import doctest
    import sys
    import os
    
    # 將專案根目錄加入 Python 路徑，以便正確匯入模組
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    doctest.testmod(verbose=True)

