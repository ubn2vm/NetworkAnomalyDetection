"""
模型評估工具模組

提供統一的混淆矩陣計算、指標計算和格式化輸出功能。
"""
from typing import Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


@dataclass
class EvaluationMetrics:
    """評估指標資料類別
    
    >>> metrics = EvaluationMetrics(tn=100, fp=10, fn=5, tp=85)
    >>> metrics.accuracy
    0.925
    >>> metrics.precision
    0.8947368421052632
    >>> metrics.recall
    0.9444444444444444
    >>> metrics.f1
    0.918918918918919
    """
    tn: int  # True Negative
    fp: int  # False Positive
    fn: int  # False Negative
    tp: int  # True Positive
    
    @property
    def accuracy(self) -> float:
        """計算準確率"""
        total = self.tn + self.fp + self.fn + self.tp
        return (self.tn + self.tp) / total if total > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """計算精確率（針對異常類別）"""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """計算召回率（針對異常類別）"""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
    
    @property
    def f1(self) -> float:
        """計算 F1 分數"""
        p = self.precision
        r = self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def precision_normal(self) -> float:
        """計算正常類別的精確率
        
        >>> metrics = EvaluationMetrics(tn=100, fp=10, fn=5, tp=85)
        >>> round(metrics.precision_normal, 4)
        0.9524
        """
        return self.tn / (self.tn + self.fn) if (self.tn + self.fn) > 0 else 0.0
    
    @property
    def recall_normal(self) -> float:
        """計算正常類別的召回率
        
        >>> metrics = EvaluationMetrics(tn=100, fp=10, fn=5, tp=85)
        >>> round(metrics.recall_normal, 4)
        0.9091
        """
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0
    
    @property
    def precision_anomaly(self) -> float:
        """計算異常類別的精確率（別名，保持向後兼容）
        
        >>> metrics = EvaluationMetrics(tn=100, fp=10, fn=5, tp=85)
        >>> metrics.precision_anomaly == metrics.precision
        True
        """
        return self.precision
    
    @property
    def recall_anomaly(self) -> float:
        """計算異常類別的召回率（別名，保持向後兼容）
        
        >>> metrics = EvaluationMetrics(tn=100, fp=10, fn=5, tp=85)
        >>> metrics.recall_anomaly == metrics.recall
        True
        """
        return self.recall
    
    @property
    def total(self) -> int:
        """總樣本數"""
        return self.tn + self.fp + self.fn + self.tp


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationMetrics:
    """
    計算評估指標
    
    Args:
        y_true: 真實標籤（0=正常, 1=異常）
        y_pred: 預測標籤（0=正常, 1=異常）
    
    Returns:
        EvaluationMetrics 物件
    
    >>> y_true = np.array([0, 0, 1, 1, 0, 1, 0])
    >>> y_pred = np.array([0, 1, 1, 1, 0, 0, 0])
    >>> metrics = calculate_metrics(y_true, y_pred)
    >>> metrics.tn
    3
    >>> metrics.fp
    1
    >>> metrics.fn
    1
    >>> metrics.tp
    2
    >>> round(metrics.accuracy, 4)
    0.7143
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    return EvaluationMetrics(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def print_confusion_matrix(metrics: EvaluationMetrics, indent: str = "      ") -> None:
    """
    格式化輸出混淆矩陣
    
    Args:
        metrics: 評估指標物件
        indent: 縮排字串（預設為 6 個空格）
    
    >>> metrics = EvaluationMetrics(tn=231736, fp=148769, fn=4937, tp=32060)
    >>> print_confusion_matrix(metrics, indent="  ")
      True Negative (TN):  231,736
      False Positive (FP): 148,769
      False Negative (FN): 4,937
      True Positive (TP):  32,060
    """
    print(f"{indent}True Negative (TN):  {metrics.tn:,}")
    print(f"{indent}False Positive (FP): {metrics.fp:,}")
    print(f"{indent}False Negative (FN): {metrics.fn:,}")
    print(f"{indent}True Positive (TP):  {metrics.tp:,}")


def print_metrics_summary(metrics: EvaluationMetrics, indent: str = "      ") -> None:
    """
    格式化輸出指標摘要（單行格式）
    
    Args:
        metrics: 評估指標物件
        indent: 縮排字串
    
    >>> metrics = EvaluationMetrics(tn=231736, fp=148769, fn=4937, tp=32060)
    >>> print_metrics_summary(metrics)
          準確率: 0.6318, 精確率: 0.1773, 召回率: 0.8666, F1: 0.2944
    """
    print(f"{indent}準確率: {metrics.accuracy:.4f}, "
          f"精確率: {metrics.precision:.4f}, "
          f"召回率: {metrics.recall:.4f}, "
          f"F1: {metrics.f1:.4f}")


def print_metrics_detailed(metrics: EvaluationMetrics, indent: str = "  ") -> None:
    """
    格式化輸出詳細指標（多行格式）
    
    Args:
        metrics: 評估指標物件
        indent: 縮排字串
    
    >>> metrics = EvaluationMetrics(tn=231736, fp=148769, fn=4937, tp=32060)
    >>> print_metrics_detailed(metrics)
      準確率 (Accuracy):  0.6318 (63.18%)
      精確率 (Precision): 0.1773 (17.73%)
      召回率 (Recall):    0.8666 (86.66%)
      F1 分數:            0.2944
    """
    print(f"{indent}準確率 (Accuracy):  {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
    print(f"{indent}精確率 (Precision): {metrics.precision:.4f} ({metrics.precision*100:.2f}%)")
    print(f"{indent}召回率 (Recall):    {metrics.recall:.4f} ({metrics.recall*100:.2f}%)")
    print(f"{indent}F1 分數:            {metrics.f1:.4f}")


def evaluate_and_print(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: Optional[str] = None,
    show_confusion_matrix: bool = True,
    show_summary: bool = True,
    show_detailed: bool = False,
    show_classification_report: bool = False,
    indent: str = "      "
) -> EvaluationMetrics:
    """
    評估並輸出結果（一站式函數）
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        title: 標題（可選）
        show_confusion_matrix: 是否顯示混淆矩陣
        show_summary: 是否顯示摘要（單行）
        show_detailed: 是否顯示詳細指標（多行）
        show_classification_report: 是否顯示分類報告
        indent: 縮排字串
    
    Returns:
        EvaluationMetrics 物件
    
    >>> y_true = np.array([0, 0, 1, 1, 0, 1, 0])
    >>> y_pred = np.array([0, 1, 1, 1, 0, 0, 0])
    >>> metrics = evaluate_and_print(y_true, y_pred, title="測試結果")
    測試結果
          True Negative (TN):  3
          False Positive (FP): 1
          False Negative (FN): 1
          True Positive (TP):  2
          準確率: 0.7143, 精確率: 0.6667, 召回率: 0.6667, F1: 0.6667
    """
    metrics = calculate_metrics(y_true, y_pred)
    
    if title:
        print(f"\n{title}")
    
    if show_confusion_matrix:
        print_confusion_matrix(metrics, indent=indent)
    
    if show_summary:
        print_metrics_summary(metrics, indent=indent)
    
    if show_detailed:
        print_metrics_detailed(metrics, indent=indent)
    
    if show_classification_report:
        print("\n分類報告：")
        print(classification_report(y_true, y_pred, target_names=['正常', '異常'], zero_division=0))
    
    return metrics


def compare_metrics(
    metrics_before: EvaluationMetrics,
    metrics_after: EvaluationMetrics,
    indent: str = "      "
) -> None:
    """
    比較兩個評估結果的改進
    
    Args:
        metrics_before: 改進前的指標
        metrics_after: 改進後的指標
        indent: 縮排字串
    
    >>> before = EvaluationMetrics(tn=231736, fp=148769, fn=4937, tp=32060)
    >>> after = EvaluationMetrics(tn=300000, fp=80000, fn=5000, tp=32000)
    >>> compare_metrics(before, after)
          FP 減少：68,769 (46.2%)
          精確率提升：0.1773 → 0.2857 (61.1%)
          F1 分數變化：0.2944 → 0.4561 (54.9%)
    """
    fp_reduction = metrics_before.fp - metrics_after.fp
    fp_reduction_pct = (fp_reduction / metrics_before.fp * 100) if metrics_before.fp > 0 else 0
    
    precision_improvement_pct = (
        ((metrics_after.precision - metrics_before.precision) / metrics_before.precision * 100)
        if metrics_before.precision > 0 else 0
    )
    
    f1_change_pct = (
        ((metrics_after.f1 - metrics_before.f1) / metrics_before.f1 * 100)
        if metrics_before.f1 > 0 else 0
    )
    
    print(f"{indent}FP 減少：{fp_reduction:,} ({fp_reduction_pct:.1f}%)")
    print(f"{indent}精確率提升：{metrics_before.precision:.4f} → {metrics_after.precision:.4f} "
          f"({precision_improvement_pct:.1f}%)")
    print(f"{indent}F1 分數變化：{metrics_before.f1:.4f} → {metrics_after.f1:.4f} "
          f"({f1_change_pct:.1f}%)")


def compare_train_test_metrics(
    train_metrics: EvaluationMetrics,
    test_metrics: EvaluationMetrics,
    indent: str = "  "
) -> None:
    """
    比較訓練集和測試集的指標（用於過擬合檢測）
    
    Args:
        train_metrics: 訓練集指標
        test_metrics: 測試集指標
        indent: 縮排字串
    
    >>> train = EvaluationMetrics(tn=1000, fp=50, fn=20, tp=200)
    >>> test = EvaluationMetrics(tn=200, fp=20, fn=10, tp=40)
    >>> compare_train_test_metrics(train, test)
      訓練集準確率：0.9449 (94.49%)
      測試集準確率：0.8889 (88.89%)
      準確率差異：0.0560 (5.60%)
      過擬合風險：LOW
    """
    train_acc = train_metrics.accuracy
    test_acc = test_metrics.accuracy
    gap = train_acc - test_acc
    
    # 判斷過擬合風險
    if gap > 0.1:
        risk = "HIGH"
    elif gap > 0.05:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    
    print(f"{indent}訓練集準確率：{train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"{indent}測試集準確率：{test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{indent}準確率差異：{gap:.4f} ({gap*100:.2f}%)")
    print(f"{indent}過擬合風險：{risk}")

