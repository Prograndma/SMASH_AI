from typing import Optional, Union

import numpy as np
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import resize
from transformers.image_utils import (
    ImageInput,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    PILImageResampling
)
from transformers.utils import TensorType


class CustomImageProcessor(BaseImageProcessor):
    r"""
    Constructs a sick nasty, custy image processor.
    """

    model_input_names = ["pixel_values"]

    def __init__(self, make_smaller=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.make_smaller = make_smaller

    @staticmethod
    def resize(
        image: np.ndarray,
        size,
        data_format,
        input_data_format,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        return resize(
            image,
            size=size,
            resample=PILImageResampling.BILINEAR,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(self, images: ImageInput, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, don't.
            make_smaller (`MakeSmaller`):
                Should this function make the images smaller? Default is yes.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        """
        images = make_list_of_images(images)
        # All transformations expect numpy arrays.
        if self.make_smaller:
            size = (292, 240)
            images = [
                self.resize(image=image, size=size, input_data_format="channels_last", data_format="channels_last")
                for image in images
            ]
        images = [to_numpy_array(image) for image in images]
        images = [
            self.rescale(image=image, scale=255, input_data_format="channels_last", data_format="channels_first")
            for image in images
        ]
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
