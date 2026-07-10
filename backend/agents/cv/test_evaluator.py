import sys; sys.path.insert(0, 'backend')
import numpy as np
from agents.cv.evaluator import CVEvaluator

ev = CVEvaluator()

# Test segmentation metrics
y_true = (np.random.rand(100, 100) > 0.5).astype(np.float32)
y_pred = (np.random.rand(100, 100) > 0.4).astype(np.float32)

seg = ev.evaluate_segmentation(y_true, y_pred)
print(f'Segmentation: dice={seg["dice_coefficient"]}, iou={seg["iou_score"]}')

# Test classification metrics
y_cls_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
y_cls_pred = np.random.rand(8, 1)
cls = ev.evaluate_classification(y_cls_true, y_cls_pred)
print(f'Classification: acc={cls["accuracy"]}, f1={cls["f1_score"]}, auc={cls["auc_roc"]}')

print('All evaluator tests PASSED')
