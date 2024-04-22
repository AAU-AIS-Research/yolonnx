from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from numpy import int64
from numpy.typing import NDArray

from .model import ImgTensor

T = TypeVar("T", contravariant=True)


class PNodeArg(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def shape(self) -> Any: ...

    @property
    def type(self) -> str: ...


@runtime_checkable
class SparseTensorProtocol(Protocol):
    def dense_shape(self) -> NDArray[int64]: ...
    def values(self) -> NDArray: ...
    def device_name(self) -> str: ...


class ModelMetadataProtocol(Protocol):
    @property
    def custom_metadata_map(self) -> dict: ...


class InferenceSessionProtocol(Protocol):
    def run(
        self, output_names, input_feed: dict[str, Any], run_options=None
    ) -> list[NDArray] | list[list] | list[dict] | list[SparseTensorProtocol]: ...

    def run_async(
        self,
        output_names,
        input_feed: dict[str, Any],
        callback: Callable,
        user_data: Any,
        run_options=None,
    ): ...

    def get_inputs(self) -> list[PNodeArg]: ...

    def get_modelmeta(self) -> ModelMetadataProtocol: ...


class ToTensorStrategyProtocol(Protocol[T]):
    def __call__(self, img: T, tensor_width: int, tensor_height: int) -> ImgTensor: ...
