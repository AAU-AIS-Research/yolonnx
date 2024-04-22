from __future__ import annotations

from dataclasses import dataclass

from aau_label.model import Label
from numpy.typing import NDArray


@dataclass
class Size:
    width: float
    height: float


@dataclass
class ImgTensor:
    scale: Size
    data: NDArray


@dataclass
class ClassifierResult:
    classifier: str
    score: float


@dataclass
class DetectorResult(ClassifierResult, Label): ...
