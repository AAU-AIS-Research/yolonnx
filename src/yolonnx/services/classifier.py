import ast
import logging
from typing import Generic, Sequence, TypeVar

import numpy
from numpy import float32
from numpy.typing import NDArray

from ..model import ClassifierResult
from ..protocols import (
    InferenceSessionProtocol,
    SparseTensorProtocol,
    ToTensorStrategyProtocol,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class Classifier(Generic[T]):
    def __init__(
        self,
        session: InferenceSessionProtocol,
        to_tensor_strategy: ToTensorStrategyProtocol[T],
        threshold: float = 0.1,
    ) -> None:
        self.__session = session
        self.__to_tensor_strategy = to_tensor_strategy
        self.__threshold = threshold
        meta = self.__session.get_modelmeta()
        self.__names: dict[int, str] = ast.literal_eval(
            meta.custom_metadata_map["names"]
        )

    @property
    def threshold(self) -> float:
        return self.__threshold

    @property
    def shape(self) -> tuple[int, int]:
        return self.__session.get_inputs()[0].shape[2:]

    @property
    def names(self) -> dict[int, str]:
        return self.__names

    def warmup(self) -> None:
        tensor = numpy.zeros((1, 3, self.shape[0], self.shape[1]), float32)
        self.__session.run(None, {"images": tensor})

    def __result_handler(self, results: NDArray) -> Sequence[ClassifierResult]:
        labels_idx = numpy.argwhere(results > self.threshold)
        scores: NDArray[float32] = results[labels_idx]

        rv: list[ClassifierResult] = []
        for i in range(len(labels_idx)):
            id = labels_idx[i][0]
            label_name = self.names[id]
            rv.append(ClassifierResult(name=label_name, score=scores[i]))

        rv.sort(key=lambda x: x.score, reverse=True)
        return rv

    def run(self, img: T) -> Sequence[ClassifierResult]:
        tensor = self.__to_tensor_strategy(img, *self.shape)
        results = self.__session.run(None, {"images": tensor})[0]

        if isinstance(results, SparseTensorProtocol):
            results = results.values()

        return self.__result_handler(results[0])
