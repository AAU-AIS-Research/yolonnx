from pathlib import Path

from onnxruntime import InferenceSession
from PIL import Image

from yolonnx.services import Detector
from yolonnx.to_tensor_strategies import PillowToTensorContainStrategy

model = Path(r"C:\Users\Kasper Fromm\Desktop\test_images\best.onnx")
session = InferenceSession(
    model.as_posix(),
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)

predictor = Detector(session, PillowToTensorContainStrategy())
img = Image.open(
    r"C:\Users\Kasper Fromm\Desktop\test_images\2018-10-12_02\GRMN0002.JPG"
)

print(predictor.run(img))
