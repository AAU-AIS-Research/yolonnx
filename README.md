# You Only Look ONNX
This repository is a light weight library to ease the use of ONNX models exported by the Ultralytics YOLOv8 framework.

## Example Detector Usage
```python
from pathlib import Path

from onnxruntime import InferenceSession
from PIL import Image

from yolonnx.services import Detector
from yolonnx.to_tensor_strategies import PillowToTensorContainStrategy

model = Path("path/to/file.onnx")
session = InferenceSession(
    model.as_posix(),
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)

predictor = Detector(session, PillowToTensorContainStrategy())
img = Image.open("path/to/image.jpg")
print(predictor.run(img))
```

## Example Classifier Usage
```python
from pathlib import Path

from onnxruntime import InferenceSession
from PIL import Image

from yolonnx.services import Classifier
from yolonnx.to_tensor_strategies import PillowToTensorContainStrategy

model = Path("path/to/file.onnx")
session = InferenceSession(
    model.as_posix(),
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)

predictor = Classifier(session, PillowToTensorContainStrategy())
img = Image.open("path/to/image.jpg")
print(predictor.run(img))

```