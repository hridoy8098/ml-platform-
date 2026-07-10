import sys; sys.path.insert(0, 'backend')
from agents.cv.data_engine import CVDataEngine
from config import settings
import numpy as np
import tempfile, os
from PIL import Image

engine = CVDataEngine(settings.CV_UPLOADS_DIR)

# Create a test image
tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
img = Image.new('RGB', (100, 100), color='red')
img.save(tmp.name)
tmp.close()

# Test load_image
result = engine.load_image(tmp.name)
assert result is not None, 'load_image failed'
print(f'load_image OK: {result["filename"]}, shape={result["image"].shape}')

# Test preprocess
X, meta = engine.preprocess([result])
assert X.shape == (1, 224, 224, 3), f'Bad shape: {X.shape}'
assert X.min() >= 0 and X.max() <= 1, f'Bad range: [{X.min()}, {X.max()}]'
print(f'preprocess OK: X.shape={X.shape}')

# Test augmentation
X_aug, _ = engine.augment(X, factor=2)
assert X_aug.shape[0] == 2, f'Augmentation failed: {X_aug.shape}'
print(f'augment OK: X_aug.shape={X_aug.shape}')

os.unlink(tmp.name)
print('All data_engine tests PASSED')
