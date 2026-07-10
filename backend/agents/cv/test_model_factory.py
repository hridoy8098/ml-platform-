import sys; sys.path.insert(0, 'backend')
import numpy as np
from agents.cv.model_factory import CVModelFactory
from config import settings
from pathlib import Path

factory = CVModelFactory(settings.CV_MODELS_DIR, settings.CV_UPLOADS_DIR)

# Test build classifier
clf = factory.build_classifier(input_shape=(224, 224, 3), num_classes=2)
clf.summary()
print(f'Classifier built: {clf.count_params()} params')

# Test build U-Net
unet = factory.build_segmentation_unet(input_shape=(224, 224, 3))
print(f'U-Net built: {unet.count_params()} params')

# Quick forward pass test
X_dummy = np.random.rand(2, 224, 224, 3).astype(np.float32)
out_clf = clf.predict(X_dummy, verbose=0)
print(f'Classifier output shape: {out_clf.shape}, values: {out_clf[0]}')

out_unet = unet.predict(X_dummy, verbose=0)
print(f'U-Net output shape: {out_unet.shape}')

print('All model_factory tests PASSED')
