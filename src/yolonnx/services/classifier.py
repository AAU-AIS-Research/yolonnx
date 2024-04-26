import ast
import logging
from typing import Callable, Generic, Sequence, TypeVar

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
        name_converter: Callable[[str], str] | None = None,
        threshold: float = 0.1,
    ) -> None:
        """
        Initializes the Classifier object with the provided session, to_tensor_strategy, name_converter, none_cls_name, and threshold.

        Parameters:
            session (InferenceSessionProtocol): The inference session for the classifier.
            to_tensor_strategy (ToTensorStrategyProtocol[T]): The strategy for converting input data to tensor.
            name_converter (Callable[[str], str] | None, optional): A function to convert class names. Defaults to None.
            threshold (float, optional): The threshold value. Defaults to 0.1.

        Returns:
            None
        """
        self.__session = session
        self.__to_tensor_strategy = to_tensor_strategy
        self.__threshold = threshold
        meta = self.__session.get_modelmeta()
        self.__names: dict[int, str] = ast.literal_eval(
            meta.custom_metadata_map["names"]
        )
        if name_converter:
            self.__names = {k: name_converter(v) for k, v in self.__names.items()}

    @property
    def threshold(self) -> float:
        """
        Returns the threshold value of the classifier.

        Returns:
            float: The threshold value.
        """
        return self.__threshold

    @property
    def shape(self) -> tuple[int, int]:
        """
        Returns the shape of the input tensor for the model.

        Returns:
            tuple[int, int]: A tuple representing the shape of the input tensor. The tuple contains two integers representing the height and width of the input tensor.
        """
        return self.__session.get_inputs()[0].shape[2:]

    @property
    def names(self) -> dict[int, str]:
        """
        Returns the id to name mapping of the classifier.

        Returns:
            dict[int, str]: The id to name mapping.
        """
        return self.__names

    def warmup(self) -> None:
        """
        Warms up the classifier by running an empty tensor through the model.
        """
        tensor = numpy.zeros((1, 3, self.shape[0], self.shape[1]), float32)
        self.__session.run(None, {"images": tensor})

    def __result_handler(self, results: NDArray) -> Sequence[ClassifierResult]:
        labels_idx = numpy.argwhere(results >= self.threshold)
        scores: NDArray[float32] = results[labels_idx]

        rv: list[ClassifierResult] = []
        for i in range(len(labels_idx)):
            id = labels_idx[i][0]
            label_name = self.names[id]
            rv.append(ClassifierResult(name=label_name, score=scores[i][0]))

        rv.sort(key=lambda x: x.score, reverse=True)
        return rv

    def run(self, img: T) -> Sequence[ClassifierResult]:
        """
        Run the classifier on the given image and return a sequence of ClassifierResult objects.

        Args:
            img (T): The input image to be classified.

        Returns:
            Sequence[ClassifierResult]: A sequence of ClassifierResult objects representing the classification results.
            If no result sattisfies the threshold, an empty sequence is returned.
        """
        tensor = self.__to_tensor_strategy(img, *self.shape)
        results = self.__session.run(None, {"images": tensor.data})[0]

        if isinstance(results, SparseTensorProtocol):
            results = results.values()

        return self.__result_handler(results[0])
