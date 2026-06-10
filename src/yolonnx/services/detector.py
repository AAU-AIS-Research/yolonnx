import ast
import logging
from collections.abc import Callable
from typing import Any, Generic, Protocol, Sequence, TypeVar, cast

import numpy
from numpy.typing import NDArray

from .. import utils
from ..model import DetectorResult, ImgTensor
from ..protocols import InferenceSessionProtocol, ToTensorStrategyProtocol

logger = logging.getLogger(__name__)


T = TypeVar("T")


class ParserOption(Protocol):
    @property
    def conf_threshold(self) -> float: ...

    @property
    def iou_threshold(self) -> float: ...

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def names(self) -> dict[int, str]: ...

    @property
    def nms(self) -> bool: ...

    @property
    def end2end(self) -> bool: ...


ModelOutputParser = Callable[[NDArray, ImgTensor, ParserOption], list[DetectorResult]]


class Yolo8ModelOutputParser:
    def __call__(
        self, results: NDArray, tensor: ImgTensor, options: ParserOption
    ) -> list[DetectorResult]:
        predictions = numpy.squeeze(results[0]).T

        scores = numpy.max(predictions[:, 4:], axis=1)
        keep = scores > options.conf_threshold
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

        keep = utils.nms(boxes, scores, options.iou_threshold)
        rv: list[DetectorResult] = []
        for bbox, label, score in zip(boxes[keep], class_ids[keep], scores[keep]):
            rv.append(
                DetectorResult(
                    x=bbox[0].item(),
                    y=bbox[1].item(),
                    width=bbox[2].item(),
                    height=bbox[3].item(),
                    name=options.names[label],
                    score=score.item(),
                )
            )

        return rv


class Yolo26ModelOutputParser:
    def __call__(
        self, results: NDArray, tensor: ImgTensor, options: ParserOption
    ) -> list[DetectorResult]:
        print(tensor.scale)
        predictions = numpy.squeeze(results[0])

        # If batched, handle that:
        if predictions.ndim == 3:
            predictions = predictions[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        class_ids = predictions[:, 5].astype(numpy.int32)

        keep = scores > options.conf_threshold
        scores = scores[keep]
        class_ids = class_ids[keep]
        boxes = (
            boxes[keep]
            / numpy.array(
                [
                    tensor.scale.width,
                    tensor.scale.height,
                    tensor.scale.width,
                    tensor.scale.height,
                ],
                dtype=numpy.float32,
            )
        ).astype(numpy.int32)

        if options.nms or options.end2end:
            data = zip(boxes, class_ids, scores)
        else:
            keep = utils.nms(boxes, scores, options.iou_threshold)
            data = zip(boxes[keep], class_ids[keep], scores[keep])

        rv: list[DetectorResult] = []
        for bbox, label, score in data:
            x0 = bbox[0].item()
            y0 = bbox[1].item()
            x1 = bbox[2].item()
            y1 = bbox[3].item()

            rv.append(
                DetectorResult(
                    x=x0,
                    y=y0,
                    width=x1 - x0,
                    height=y1 - y0,
                    name=options.names[label],
                    score=score.item(),
                )
            )

        return rv


class Detector(Generic[T]):
    def __init__(
        self,
        session: InferenceSessionProtocol,
        to_tensor_strategy: ToTensorStrategyProtocol[T],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        output_parser: ModelOutputParser = Yolo8ModelOutputParser(),
    ) -> None:
        self.__session = session
        self.__to_tensor_strategy = to_tensor_strategy
        self.__conf_threshold = conf_threshold
        self.__iou_threshold = iou_threshold
        self.__output_parser = output_parser

        meta = self.__session.get_modelmeta()
        self.__names: dict[int, str] = ast.literal_eval(
            meta.custom_metadata_map["names"]
        )
        self.__end2end = bool(meta.custom_metadata_map.get("end2end", False))
        args: dict[str, Any] = ast.literal_eval(meta.custom_metadata_map["args"])
        self.__nms = args.get("nms", False)

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

    @property
    def nms(self) -> bool:
        return self.__nms

    @property
    def end2end(self) -> bool:
        return self.__end2end

    def run(self, img: T) -> Sequence[DetectorResult]:
        tensor = self.__to_tensor_strategy(img, *self.shape)
        results = cast(list[NDArray], self.__session.run(None, {"images": tensor.data}))
        return self.__output_parser(results[0], tensor, self)
