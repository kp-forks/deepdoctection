# -*- coding: utf-8 -*-
# File: viz.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Some visualisation utils. Copied and pasted from

<https://github.com/tensorpack/tensorpack/blob/master/tensorpack/utils/viz.py>

and

<https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/colormap.py>
"""

from io import BytesIO
import base64
import os
import sys
from typing import List, Optional, Tuple, no_type_check, Sequence, Any

import numpy as np
import numpy.typing as npt
from numpy import float32, uint8

from .detection_types import ImageType
from .file_utils import get_opencv_requirement, get_pillow_requirement, opencv_available, pillow_available

if opencv_available():
    import cv2

if pillow_available():
    from PIL import Image, ImageDraw


__all__ = ["draw_boxes", "interactive_imshow", "viz_handler"]

_COLORS = (
    np.array(
        [
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.857,
            0.857,
            0.857,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def random_color(rgb: bool = True, maximum: int = 255) -> Tuple[int, int, int]:
    """
    :param rgb: Whether to return RGB colors or BGR colors.
    :param maximum: either 255 or 1
    :return:
    """

    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return tuple(int(x) for x in ret)  # type: ignore



def draw_boxes(
    np_image: ImageType,
    boxes: npt.NDArray[float32],
    category_names_list: Optional[List[Optional[str]]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    font_scale: float = 1.0,
    rectangle_thickness: int = 4,
    box_color_by_category: bool = True,
) -> ImageType:
    """
    Dray bounding boxes with category names into image.

    :param np_image: Image as np.ndarray
    :param boxes: A numpy array of shape Nx4 where each row is [x1, y1, x2, y2].
    :param category_names_list: List of N category names.
    :param color: A 3-tuple BGR color (in range [0, 255])
    :param font_scale: Font scale of text box
    :param rectangle_thickness: Thickness of bounding box
    :param box_color_by_category:
    :return: A new image np.ndarray
    """
    if color is not None:
        box_color_by_category = False

    category_to_color = {}
    if box_color_by_category:
        category_names = set(category_names_list)  # type: ignore
        category_to_color = {category: random_color() for category in category_names}

    boxes = np.array(boxes, dtype="int32")
    if category_names_list is not None:
        assert len(category_names_list) == len(boxes), f"{len(category_names_list)} != {len(boxes)}"
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    sorted_inds = np.argsort(-areas)  # draw large ones first
    assert areas.min() > 0, areas.min()

    # allow equal, because we are not very strict about rounding error here
    assert (
        boxes[:, 0].min() >= 0
        and boxes[:, 1].min() >= 0
        and boxes[:, 2].max() <= np_image.shape[1]
        and boxes[:, 3].max() <= np_image.shape[0]
    ), f"Image shape: {str(np_image.shape)}\n Boxes:\n{str(boxes)}"

    np_image = np_image.copy()

    if np_image.ndim == 2 or (np_image.ndim == 3 and np_image.shape[2] == 1):
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)
    for i in sorted_inds:
        box = boxes[i, :]
        if category_names_list is not None:
            choose_color = category_to_color.get(category_names_list[i]) if category_to_color is not None else color
            if choose_color is None:
                choose_color = random_color()
            if category_names_list[i] is not None:
                np_image = viz_handler.draw_text(np_image,(box[0], box[1]) , category_names_list[i], color=choose_color, font_scale=font_scale)
            np_image= viz_handler.draw_rectangle(np_image,(box[0],box[1],box[2],box[3]),choose_color,rectangle_thickness)

    # draw a (very ugly) color palette
    y_0 = np_image.shape[0]
    for category, col in category_to_color.items():
        if category is not None:
            viz_handler.draw_text(np_image, (np_image.shape[1], y_0), category, color=col,
                                  font_scale=font_scale)
            _, text_h = viz_handler.get_text_size(category, 2)
            y_0 = y_0 - int(10 * text_h)

    return np_image


@no_type_check
def interactive_imshow(img: ImageType) -> None:
    """
    Display an image in a pop-up window

    :param img: An image (expect BGR) to show.
    """
    viz_handler.interactive_imshow(img)


class VizPackageHandler:

    PACKAGE_FUNCS = {"cv2": {"read_image": "_cv2_read_image",
                             "write_image": "_cv2_write_image",
                             "convert_np_to_b64": "_cv2_convert_np_to_b64",
                             "convert_b64_to_np": "_cv2_convert_b64_to_np",
                             "resize": "_cv2_resize",
                             "get_text_size": "_cv2_get_text_size",
                             "draw_rectangle": "_cv2_draw_rectangle",
                             "draw_text": "_cv2_draw_text",
                             "interactive_imshow": "_cv2_interactive_imshow",
                             "encode": "_cv2_encode"},
                     "pillow": {"read_image": "_pillow_read_image",
                                "write_image": "_pillow_write_image",
                                "convert_np_to_b64": "_pillow_convert_np_to_b64",
                                "convert_b64_to_np": "_pillow_convert_b64_to_np",
                                "resize": "_pillow_resize",
                                "get_text_size": "_pillow_get_text_size",
                                "draw_rectangle": "_pillow_draw_rectangle",
                                "draw_text": "_pillow_draw_text",
                                "interactive_imshow": "_pillow_interactive_imshow",
                                "encode": "_pillow_encode"}}

    def __init__(self):
        package = self._select_package()
        self.pkg_func_dict=self.PACKAGE_FUNCS[package]
        if package == "pillow":
            image = Image.fromarray(np.uint8(np.ones((1, 1, 3))))
            self.font = ImageDraw.ImageDraw(image).getfont()
        else:
            self.font = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def _select_package():
        """
        USE_OPENCV has priority and will enforce to use OpenCV
        Otherwise it will use Pillow as default package, if it is installed.
        If Pillow is not installed it will try to load OpenCV again
        :return:
        """
        if os.environ.get("USE_OPENCV") is not None:
            requirements =  get_opencv_requirement()
            if not requirements[1]:
                raise ImportError(requirements[2])
            return "cv2"
        requirements = get_pillow_requirement()
        if os.environ["USE_PILLOW"]:
            if not requirements[1]:
               raise ImportError(requirements[2])
            return "pillow"
        requirements = get_opencv_requirement()
        if not requirements[1]:
            raise ImportError(requirements[2])
        return "cv2"

    def read_image(self, path: str) -> ImageType:
        return getattr(self,self.pkg_func_dict["read_image"])(path)

    @staticmethod
    def _cv2_read_image(path: str) -> ImageType:
        return cv2.imread(path, cv2.IMREAD_COLOR)

    @staticmethod
    def _pillow_read_image(path: str) -> ImageType:
        with Image.open(path) as im:
            np_image = np.array(im)[:, :, ::-1]
        return np_image

    def write_image(self, path: str, image: ImageType)->None:
        return getattr(self, self.pkg_func_dict["write_image"])(path, image)

    @staticmethod
    def _cv2_write_image(path: str, image: ImageType) ->None:
        cv2.imwrite(path, image)

    @staticmethod
    def _pillow_write_image(path: str, image: ImageType) -> None:
        pil_image = Image.fromarray(np.uint8(image[:, :, ::-1]))
        pil_image.save(path)

    def convert_np_to_b64(self, image: ImageType) -> str:
        return getattr(self, self.pkg_func_dict["convert_np_to_b64"])(image)

    def encode(self, np_image: ImageType):
        return getattr(self, self.pkg_func_dict["encode"])(np_image)

    @staticmethod
    def _cv2_encode(np_image: ImageType):
        np_encode = cv2.imencode(".png", np_image)
        b_image = np_encode[1].tobytes()
        return b_image

    @staticmethod
    def _pillow_encode(np_image: ImageType):
        buffered = BytesIO()
        pil_image = Image.fromarray(np_image[:, :, ::-1])
        pil_image.save(buffered, format="PNG")
        return buffered.getvalue()

    @staticmethod
    def _cv2_convert_np_to_b64(image: ImageType) -> str:
        np_encode = cv2.imencode(".png", image)
        return base64.b64encode(np_encode[1]).decode("utf-8")  # type: ignore

    @staticmethod
    def _pillow_convert_np_to_b64(image: ImageType) -> str:
        buffered = BytesIO()
        pil_image = Image.fromarray(image[:, :, ::-1])
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")  # type: ignore

    def convert_b64_to_np(self, image: str) -> ImageType:
        return getattr(self, self.pkg_func_dict["convert_b64_to_np"])(image)

    @staticmethod
    def _cv2_convert_b64_to_np(image: str) -> ImageType:
        np_array = np.fromstring(base64.b64decode(image), np.uint8)  # type: ignore
        np_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR).astype(np.float32)
        return np_array.astype(uint8)

    @staticmethod
    def _pillow_convert_b64_to_np(image: str) -> ImageType:
        array = base64.b64decode(image)  # type: ignore
        im_file = BytesIO(array)
        pil_image = Image.open(im_file)
        return np.array(pil_image)[:, :, ::-1]

    def resize(self, image: ImageType, width: int, height: int, interpolation: str) -> ImageType:
        return getattr(self, self.pkg_func_dict["resize"])(image, width, height, interpolation)

    @staticmethod
    def _cv2_resize(image: ImageType, width: int, height: int, interpolation: str) -> ImageType:
        intpol_method_dict = { "INTER_NEAREST": cv2.INTER_NEAREST, "INTER_LINEAR": cv2.INTER_LINEAR,
                           "INTER_AREA": cv2.INTER_AREA, "VIZ": cv2.INTER_LINEAR}
        return cv2.resize(image, (width, height), interpolation=intpol_method_dict[interpolation])

    @staticmethod
    def _pillow_resize(image: ImageType, width: int, height: int, interpolation: str) -> ImageType:

        intpol_method_dict = { "NEAREST": Image.Resampling.NEAREST,
                               "BOX": Image.Resampling.BOX,
                               "BILINEAR": Image.Resampling.BILINEAR,
                               "BICUBIC": Image.Resampling.BICUBIC,
                               "VIZ": Image.Resampling.BILINEAR}
        pil_image = Image.fromarray(np.uint8(image[:, :, ::-1]))
        pil_image_resized =pil_image.resize(size=(width, height), resample=intpol_method_dict[interpolation], box=None, reducing_gap=None)
        return np.array(pil_image_resized)[:, :, ::-1]


    def get_text_size(self, text: str, font_scale: float) -> Tuple[float, float]:
        return getattr(self, self.pkg_func_dict["get_text_size"])(text, font_scale)


    def _cv2_get_text_size(self, text: str, font_scale: float) -> Tuple[float, float]:
        ((width, height), _) = cv2.getTextSize(text, self.font, font_scale, 1)
        return width, height

    def _pillow_get_text_size(self, text: str, font_scale: float) -> Tuple[float, float]:
        _,_, width, height = self.font.getbbox(text)
        return width, height

    def draw_rectangle(self, np_image: ImageType, box:  Tuple[Any], color: Tuple[int, int, int], thickness: int)-> ImageType:
        return getattr(self, self.pkg_func_dict["draw_rectangle"])(np_image, box, color, thickness)

    @staticmethod
    def _cv2_draw_rectangle(np_image: ImageType, box: Tuple[Any], color: Sequence[int], thickness: int):
        cv2.rectangle(np_image, (box[0], box[1]), (box[2], box[3]), color=color, thickness=thickness)
        return np_image

    @staticmethod
    def _pillow_draw_rectangle(np_image: ImageType, box: Tuple[Any], color: Sequence[int], thickness: int)-> ImageType:
        pil_image = Image.fromarray(np.uint8(np_image[:, :, ::-1]))
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle(box,outline=color,width=thickness)
        np_image= np.array(pil_image)[:, :, ::-1]
        return np_image

    def draw_text(self, np_image: ImageType, pos: Tuple[Any], text: str, color: Tuple[int, int, int], font_scale: float)-> ImageType:
        return getattr(self, self.pkg_func_dict["draw_text"])(np_image, pos, text, color, font_scale)

    def _cv2_draw_text(self, np_image: ImageType, pos: Tuple[Any], text: str, color: Tuple[int, int, int], font_scale: float)-> ImageType:
        """
        Draw text on an image.

        :param np_image: image as np.ndarray
        :param pos: x, y; the position of the text
        :param text: text string to draw
        :param color: a 3-tuple BGR color in [0, 255]
        :param font_scale: float
        :return: numpy array
        """

        np_image = np_image.astype(np.uint8)
        x_0, y_0 = int(pos[0]), int(pos[1])
        # Compute text size.
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_w, text_h = viz_handler.get_text_size(text, font_scale)
        # Place text background.
        if x_0 + text_w > np_image.shape[1]:
            x_0 = np_image.shape[1] - text_w
        if y_0 - int(1.15 * text_h) < 0:
            y_0 = int(1.15 * text_h)
        back_top_left = x_0, y_0 - int(1.3 * text_h)
        back_bottom_right = x_0 + text_w, y_0
        np_image = self.draw_rectangle(np_image, (
        back_top_left[0], back_top_left[1], back_bottom_right[0], back_bottom_right[1]), color, 1)
        # Show text.
        text_bottomleft = x_0, y_0 - int(0.25 * text_h)
        cv2.putText(np_image, text, text_bottomleft, font, font_scale, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        return np_image

    def _pillow_draw_text(self, np_image: ImageType, pos: Tuple[Any], text: str, color: Tuple[int, int, int], font_scale: float)-> ImageType:
        # using PIL default font size that does not scale to larger image sizes.
        # Compare with https://github.com/python-pillow/Pillow/issues/6622
        pil_image = Image.fromarray(np.uint8(np_image[:, :, ::-1]))
        draw = ImageDraw.Draw(pil_image)
        draw.text(pos,text, fill=(0, 0, 0),anchor="lb")
        return np.array(pil_image)[:, :, ::-1]

    def interactive_imshow(self, np_image):
        return getattr(self, self.pkg_func_dict["interactive_imshow"])(np_image)

    def _cv2_interactive_imshow(self, np_image):
        """
        Display an image in a pop-up window

        :param img: An image (expect BGR) to show.
        """
        name = "q, x: quit / s: save"
        cv2.imshow(name, np_image)

        key = cv2.waitKey(-1)
        while key >= 128:
            key = cv2.waitKey(-1)
        key = chr(key & 0xFF)

        if key == "q":
            cv2.destroyWindow(name)
        elif key == "x":
            sys.exit()
        elif key == "s":
            cv2.imwrite("out.png", np_image)
        elif key in ["+", "="]:
            np_image = cv2.resize(np_image, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
            self._cv2_interactive_imshow(np_image)
        elif key == "-":
            np_image = cv2.resize(np_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
            self._cv2_interactive_imshow(np_image)

    @staticmethod
    def _pillow_interactive_imshow(np_image):
        name = "q, x: quit / s: save"
        pil_image = Image.fromarray(np.uint8(np_image[:, :, ::-1]))
        pil_image.show(name)


# default image package
os.environ["USE_PILLOW"] = "True"

viz_handler = VizPackageHandler()
