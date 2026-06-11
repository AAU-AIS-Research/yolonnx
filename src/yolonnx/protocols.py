from __future__ import annotations

from typing import Protocol, TypeVar

from .model import ImgTensor

T = TypeVar("T", contravariant=True)


class ToTensorStrategyProtocol(Protocol[T]):
    def __call__(self, img: T, tensor_width: int, tensor_height: int) -> ImgTensor: ...
