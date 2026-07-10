import logging
from typing import Dict

import numpy as np

log = logging.getLogger("ml-platform.cv_evaluator")


class CVEvaluator:
    @staticmethod
    def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-7) -> float:
        y_true_f = y_true.flatten().astype(np.float32)
        y_pred_f = y_pred.flatten().astype(np.float32)
        intersection = np.sum(y_true_f * y_pred_f)
        return float((2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))

    @staticmethod
    def iou_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-7) -> float:
        y_true_f = y_true.flatten().astype(np.float32)
        y_pred_f = y_pred.flatten().astype(np.float32)
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        return float((intersection + smooth) / (union + smooth))

    @staticmethod
    def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_f = y_true.flatten().astype(bool)
        y_pred_f = y_pred.flatten().astype(bool)
        tp = np.sum(y_true_f & y_pred_f)
        fn = np.sum(y_true_f & ~y_pred_f)
        return float(tp / (tp + fn + 1e-7))

    @staticmethod
    def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_f = y_true.flatten().astype(bool)
        y_pred_f = y_pred.flatten().astype(bool)
        tn = np.sum(~y_true_f & ~y_pred_f)
        fp = np.sum(~y_true_f & y_pred_f)
        return float(tn / (tn + fp + 1e-7))

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        y_true_f = y_true.flatten().astype(bool)
        y_pred_f = y_pred.flatten().astype(bool)
        tp = np.sum(y_true_f & y_pred_f)
        tn = np.sum(~y_true_f & ~y_pred_f)
        fp = np.sum(~y_true_f & y_pred_f)
        fn = np.sum(y_true_f & ~y_pred_f)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

    @staticmethod
    def evaluate_segmentation(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        y_pred_bin = (y_pred > 0.5).astype(np.float32)
        return {
            "dice_coefficient": round(CVEvaluator.dice_coefficient(y_true, y_pred_bin), 4),
            "iou_score": round(CVEvaluator.iou_score(y_true, y_pred_bin), 4),
            "sensitivity": round(CVEvaluator.sensitivity(y_true, y_pred_bin), 4),
            "specificity": round(CVEvaluator.specificity(y_true, y_pred_bin), 4),
            "pixel_accuracy": round(float(np.mean((y_pred_bin == y_true).astype(float))), 4),
        }

    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        from sklearn.metrics import roc_auc_score
        y_pred_bin = (y_pred_proba > 0.5).astype(np.float32).flatten()
        y_true_f = y_true.flatten().astype(np.float32)
        metrics = CVEvaluator.classification_metrics(y_true_f, y_pred_bin)
        try:
            metrics["auc_roc"] = round(float(roc_auc_score(y_true_f, y_pred_proba.flatten())), 4)
        except Exception:
            metrics["auc_roc"] = None
        return metrics
