from __future__ import annotations

from dataclasses import dataclass

from aau_label.model import AAULabel
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class Size:
    width: float
    height: float


@dataclass(slots=True, frozen=True)
class ImgTensor:
    scale: Size
    data: NDArray


@dataclass(slots=True)
class ClassifierResult:
    name: str
    score: float


@dataclass(slots=True)
class DetectorResult(ClassifierResult, AAULabel): ...
