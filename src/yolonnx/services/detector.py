import ast
import asyncio
import logging
from asyncio import Future
from typing import Generic, Sequence, TypeVar, cast

import numpy
from numpy.typing import NDArray

from .. import utils
from ..model import DetectorResult, ImgTensor
from ..protocols import InferenceSessionProtocol, ToTensorStrategyProtocol

T = TypeVar("T")
logger = logging.getLogger(__name__)


class Detector(Generic[T]):
    def __init__(
        self,
        session: InferenceSessionProtocol,
        to_tensor_strategy: ToTensorStrategyProtocol[T],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> None:
        self.__session = session
        self.__to_tensor_strategy = to_tensor_strategy
        self.__conf_threshold = conf_threshold
        self.__iou_threshold = iou_threshold

        meta = self.__session.get_modelmeta()
        self.__names: dict[int, str] = ast.literal_eval(
            meta.custom_metadata_map["names"]
        )

    @property
    def conf_threshold(self) -> float:
        return self.__conf_threshold

    @property
    def iou_threshold(self) -> float:
        return self.__iou_threshold

    @property
    def shape(self) -> tuple[int, int]:
        return self.__session.get_inputs()[0].shape[2:]

    @property
    def names(self) -> dict[int, str]:
        return self.__names

    def __result_handler(
        self, results: NDArray, tensor: ImgTensor
    ) -> Sequence[DetectorResult]:
        predictions = numpy.squeeze(results[0]).T

        scores = numpy.max(predictions[:, 4:], axis=1)
        keep = scores > self.__conf_threshold
        predictions = predictions[keep, :]
        scores = scores[keep]
        class_ids = numpy.argmax(predictions[:, 4:], axis=1)

        boxes = predictions[:, :4]
        # Make x0, y0 left upper corner instead of box center
        boxes[:, 0:2] -= boxes[:, 2:4] / 2
        boxes /= numpy.array(
            [
                tensor.scale.width,
                tensor.scale.height,
                tensor.scale.width,
                tensor.scale.height,
            ],
            dtype=numpy.float32,
        )
        boxes = boxes.astype(numpy.int32)

        keep = utils.nms(boxes, scores, self.__iou_threshold)
        rv = []
        for bbox, label, score in zip(boxes[keep], class_ids[keep], scores[keep]):
            rv.append(
                DetectorResult(
                    x=bbox[0].item(),
                    y=bbox[1].item(),
                    width=bbox[2].item(),
                    height=bbox[3].item(),
                    classifier=self.__names[label],
                    score=score.item(),
                )
            )

        return rv

    def run(self, img: T) -> Sequence[DetectorResult]:
        tensor = self.__to_tensor_strategy(img, *self.shape)
        results = cast(list[NDArray], self.__session.run(None, {"images": tensor.data}))
        return self.__result_handler(results[0], tensor)
