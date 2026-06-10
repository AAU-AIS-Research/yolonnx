import os
from pathlib import Path

import pytest
from aau_label.io import Pascal
from aau_label.model import AAULabelImage
from onnxruntime import InferenceSession
from PIL import Image, ImageOps

from yolonnx.services import Detector, Yolo26ModelOutputParser
from yolonnx.to_tensor_strategies import PillowToTensorContainStrategy


@pytest.fixture
def detector():
    env_var = "PYTEST_DETECTION_MODEL"
    model_path = os.environ.get(env_var)
    if not model_path:
        raise ValueError(f"Could not {env_var} in environment variables")
    session = InferenceSession(model_path)
    return Detector(
        session,
        PillowToTensorContainStrategy(),
        output_parser=Yolo26ModelOutputParser(),
    )


@pytest.fixture
def image_dir():
    env_var = "PYTEST_IMAGE_DIR"
    img_dir = os.environ.get(env_var)
    if not img_dir:
        raise ValueError(f"Could not {env_var} in environment variables")
    return Path(img_dir)


@pytest.fixture
def images(image_dir: Path) -> list[Path]:
    image_formats = {".png", ".jpg", ".jpeg"}
    image_files = [img for img in image_dir.rglob("*") if img.suffix in image_formats]

    if len(image_files) < 1:
        raise ValueError("Image directory is empty")
    return image_files


def test_detect_dir(detector: Detector, image_dir: Path, images: list[Path]) -> None:
    label_dir = image_dir.joinpath("pytest_labels")

    label_dir.mkdir(exist_ok=True)
    pascal = Pascal(label_dir)

    for img_path in sorted(images, key=str):
        img = Image.open(img_path)
        ImageOps.exif_transpose(img)

        results = detector.run(img)
        width, height = img.size
        label_image = AAULabelImage(img_path, width, height, results)

        if len(label_image.labels) < 1:
            continue
        label_path = label_dir.joinpath(img_path.with_suffix(".xml").name)
        label_path.write_text(pascal.serialize(label_image))
