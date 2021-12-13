# -*- coding: utf-8 -*-
# File: test_tessocr.py

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
Testing module extern.tessocr
"""
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from deep_doctection.extern.base import DetectionResult
from deep_doctection.extern.tessocr import TesseractOcrDetector
from deep_doctection.utils.detection_types import ImageType
from tests.data import Annotations


def get_mock_word_results(
    np_img: ImageType, supported_languages: str, config: str  # pylint: disable=W0613
) -> List[DetectionResult]:
    """
    Returns WordResults attr: word_results_list
    """
    return Annotations().get_word_detect_results()


class TestTesseractOcrDetector:
    """
    Test TesseractOcrDetector
    """

    @staticmethod
    @patch(
        "deep_doctection.extern.tessocr.get_py_tesseract_requirement",
        MagicMock(return_value=("pytesseract", False, "pytesseract not available")),
    )
    @patch(
        "deep_doctection.extern.tessocr.get_tesseract_requirement",
        MagicMock(return_value=("tesseract", True, "tesseract available")),
    )
    def test_tesseract_ocr_raises_import_error_when_dependencies_not_satisfied(path_to_tesseract_yaml: str) -> None:
        """
        This tests shows that the dependency implementation of the base class and is representative for
        the dependency logic of all derived classes.
        """

        # Act and Assert
        with pytest.raises(ImportError):
            TesseractOcrDetector(path_yaml=path_to_tesseract_yaml)

    @staticmethod
    @patch(
        "deep_doctection.extern.tessocr.get_py_tesseract_requirement", MagicMock(return_value=("pytesseract", True, ""))
    )
    @patch("deep_doctection.extern.tessocr.get_tesseract_requirement", MagicMock(return_value=("tesseract", True, "")))
    @patch("deep_doctection.extern.tessocr.predict_text", MagicMock(side_effect=get_mock_word_results))
    def test_tesseract_ocr_predicts_image(path_to_tesseract_yaml: str, np_image: ImageType) -> None:
        """
        Detector calls predict_text
        """

        # Arrange
        tess = TesseractOcrDetector(path_yaml=path_to_tesseract_yaml)

        # Act
        results = tess.predict(np_image)

        # Assert
        assert len(results) == 2