import numpy
from PIL import ImageOps
from PIL.Image import Image as ImageType
from PIL.Image import Resampling

from .model import ImgTensor, Size
from .protocols import ToTensorStrategyProtocol


class PillowToTensorContainStrategy(ToTensorStrategyProtocol[ImageType]):
    """
    Takes a Pillow image and scales it such it fits inside the tensor. The image aspect ratio is preserved.

    The image is then padded with the color (114, 114, 114) to make the image fit inside the tensor.
    """

    def __call__(
        self, img: ImageType, tensor_width: int, tensor_height: int
    ) -> ImgTensor:
        tensor_dims = (tensor_width, tensor_height)
        original_width, original_height = img.size

        img = ImageOps.contain(img, tensor_dims, Resampling.BILINEAR)
        new_width, new_height = img.size

        img = ImageOps.pad(
            img,
            tensor_dims,
            Resampling.BILINEAR,
            (114, 114, 114),
            (0, 0),
        )
        data = numpy.array(img)

        data = data / 255.0
        data = data.transpose(2, 0, 1)
        tensor = data[numpy.newaxis, :, :, :].astype(numpy.float32)

        scale = Size(
            width=new_width / original_width,
            height=new_height / original_height,
        )

        return ImgTensor(scale, tensor)
