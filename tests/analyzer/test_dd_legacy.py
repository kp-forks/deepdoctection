# -*- coding: utf-8 -*-
# File: test_dd_legacy.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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
Testing module analyzer.dd_legacy. This test case requires a GPU and should be considered as integration test
"""

from pytest import mark

from deepdoctection.analyzer import get_dd_analyzer
from deepdoctection.datapoint import Page

from ..test_utils import collect_datapoint_from_dataflow, get_integration_test_path


@mark.pt_legacy
def test_legacy_dd_analyzer_builds_and_process_image_layout_correctly() -> None:
    """
    Analyzer integration test with setting USE_TABLE_SEGMENTATION = False and USE_OCR = False
    """

    # Arrange
    analyzer = get_dd_analyzer(
        config_overwrite=[
            "USE_TABLE_SEGMENTATION=False",
            "USE_OCR=False",
            "PT.LAYOUT.WEIGHTS=layout/d2_model_0829999_layout_inf_only.pt",
            "PT.ITEM.WEIGHTS=item/d2_model_1639999_item_inf_only.pt",
            "PT.ITEM.FILTER=None",
            "PT.LAYOUT.PAD.TOP=60",
            "PT.LAYOUT.PAD.RIGHT=60",
            "PT.LAYOUT.PAD.BOTTOM=60",
            "PT.LAYOUT.PAD.LEFT=60",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS=0.001",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS=0.001",
            "WORD_MATCHING.THRESHOLD=0.6",
            "WORD_MATCHING.PARENTAL_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.TEXT_BLOCK_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES=['text','title','list','figure']",
            "TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER=False",
            "USE_LAYOUT_LINK=False",
            "LAYOUT_LINK.PARENTAL_CATEGORIES=[]",
            "LAYOUT_LINK.CHILD_CATEGORIES=[]",
            "OCR.USE_DOCTR=False",
            "OCR.USE_TESSERACT=True",
            "USE_LAYOUT_NMS=False",
            "USE_TABLE_REFINEMENT=True",
            "USE_LINE_MATCHER=False",
            "LAYOUT_NMS_PAIRS.COMBINATIONS=None",
            "LAYOUT_NMS_PAIRS.THRESHOLDS=None",
            "LAYOUT_NMS_PAIRS.PRIORITY=None",
            "OCR.WEIGHTS.DOCTR_RECOGNITION.PT=doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt",
        ]
    )

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert len(page.layouts) in {9, 10, 12, 13, 16}
    assert len(page.tables) == 1
    assert page.height == 2339
    assert page.width == 1654


@mark.pt_legacy
def test_legacy_dd_analyzer_builds_and_process_image_layout_and_tables_correctly() -> None:
    """
    Analyzer integration test with setting USE_OCR = False
    """

    # Arrange
    analyzer = get_dd_analyzer(
        config_overwrite=[
            "USE_TABLE_SEGMENTATION=True",
            "USE_OCR=False",
            "PT.LAYOUT.WEIGHTS=layout/d2_model_0829999_layout_inf_only.pt",
            "PT.ITEM.WEIGHTS=item/d2_model_1639999_item_inf_only.pt",
            "PT.ITEM.FILTER=None",
            "PT.LAYOUT.PAD.TOP=60",
            "PT.LAYOUT.PAD.RIGHT=60",
            "PT.LAYOUT.PAD.BOTTOM=60",
            "PT.LAYOUT.PAD.LEFT=60",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS=0.001",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS=0.001",
            "WORD_MATCHING.THRESHOLD=0.6",
            "WORD_MATCHING.PARENTAL_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.TEXT_BLOCK_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES=['text','title','list','figure']",
            "TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER=False",
            "USE_LAYOUT_LINK=False",
            "LAYOUT_LINK.PARENTAL_CATEGORIES=[]",
            "LAYOUT_LINK.CHILD_CATEGORIES=[]",
            "OCR.USE_DOCTR=False",
            "OCR.USE_TESSERACT=True",
            "USE_LAYOUT_NMS=False",
            "USE_TABLE_REFINEMENT=True",
            "USE_LINE_MATCHER=False",
            "LAYOUT_NMS_PAIRS.COMBINATIONS=None",
            "LAYOUT_NMS_PAIRS.THRESHOLDS=None",
            "LAYOUT_NMS_PAIRS.PRIORITY=None",
            "OCR.WEIGHTS.DOCTR_RECOGNITION.PT=doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt",
        ]
    )

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert len(page.layouts) in {9, 10, 12, 13, 16}
    assert len(page.tables) in {1, 2}
    # 15 cells for d2 and 16 for tp model
    assert len(page.tables[0].cells) in {14, 15, 16}  # type: ignore
    # first html for tp model, second for d2 model
    assert page.tables[0].html in {
        "<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td>"
        "</tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td>"
        "</tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>",
        "<table><tr><td></td><td rowspan=2></td></tr><tr><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>"
        "</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td>"
        "<td></td></tr></table>",
        "<table><tr><td></td><td></td></tr><tr><td rowspan=2></td><td></td></tr><tr><td></td></tr><tr><td></td><td>"
        "</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>"
        "</td></tr></table>",
    }
    assert page.height == 2339
    assert page.width == 1654


@mark.pt_legacy
def test_legacy_dd_analyzer_builds_and_process_image_correctly() -> None:
    """
    Analyzer integration test with setting USE_TABLE_SEGMENTATION = True and USE_OCR = True
    """

    # Arrange
    analyzer = get_dd_analyzer(
        config_overwrite=[
            "USE_TABLE_SEGMENTATION=True",
            "USE_OCR=True",
            "PT.LAYOUT.WEIGHTS=layout/d2_model_0829999_layout_inf_only.pt",
            "PT.ITEM.WEIGHTS=item/d2_model_1639999_item_inf_only.pt",
            "PT.ITEM.FILTER=None",
            "PT.LAYOUT.PAD.TOP=60",
            "PT.LAYOUT.PAD.RIGHT=60",
            "PT.LAYOUT.PAD.BOTTOM=60",
            "PT.LAYOUT.PAD.LEFT=60",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS=0.001",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS=0.001",
            "WORD_MATCHING.THRESHOLD=0.6",
            "WORD_MATCHING.PARENTAL_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.TEXT_BLOCK_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES=['text','title','list','figure']",
            "TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER=False",
            "USE_LAYOUT_LINK=False",
            "LAYOUT_LINK.PARENTAL_CATEGORIES=[]",
            "LAYOUT_LINK.CHILD_CATEGORIES=[]",
            "OCR.USE_DOCTR=False",
            "OCR.USE_TESSERACT=True",
            "USE_LAYOUT_NMS=False",
            "USE_TABLE_REFINEMENT=True",
            "USE_LINE_MATCHER=False",
            "LAYOUT_NMS_PAIRS.COMBINATIONS=None",
            "LAYOUT_NMS_PAIRS.THRESHOLDS=None",
            "LAYOUT_NMS_PAIRS.PRIORITY=None",
            "OCR.WEIGHTS.DOCTR_RECOGNITION.PT=doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt",
        ]
    )

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert len(page.layouts) in {9, 10, 12, 13, 16}
    assert len(page.tables) == 1
    # 15 cells for d2 and 16 for tp model
    assert len(page.tables[0].cells) in {13, 15, 16}  # type: ignore
    assert page.height == 2339
    assert page.width == 1654
    # first number for tp model, second for pt model
    assert len(page.text) in {4343, 4345, 4346, 5042}
    text_ = page.text_
    assert text_["text"] == page._make_text(line_break=False)  # pylint: disable=W0212
    assert len(text_["words"]) in {555, 631}
    assert len(text_["ann_ids"]) in {555, 631}


@mark.deprecated
def test_dd_tf_analyzer_builds_and_process_image_layout_and_tables_correctly() -> None:
    """
    Analyzer integration test with setting USE_OCR = False
    """

    # Arrange
    analyzer = get_dd_analyzer(
        config_overwrite=[
            "USE_OCR=False",
            "PT.LAYOUT.WEIGHTS=layout/d2_model_0829999_layout_inf_only.pt",
            "PT.ITEM.WEIGHTS=item/d2_model_1639999_item_inf_only.pt",
            "PT.ITEM.FILTER=None",
            "PT.LAYOUT.PAD.TOP=60",
            "PT.LAYOUT.PAD.RIGHT=60",
            "PT.LAYOUT.PAD.BOTTOM=60",
            "PT.LAYOUT.PAD.LEFT=60",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS=0.001",
            "SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS=0.001",
            "WORD_MATCHING.THRESHOLD=0.6",
            "WORD_MATCHING.PARENTAL_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.TEXT_BLOCK_CATEGORIES=['text','title','list','figure','cell','spanning']",
            "TEXT_ORDERING.FLOATING_TEXT_BLOCK_CATEGORIES=['text','title','list','figure']",
            "TEXT_ORDERING.INCLUDE_RESIDUAL_TEXT_CONTAINER=False",
            "USE_LAYOUT_LINK=False",
            "LAYOUT_LINK.PARENTAL_CATEGORIES=[]",
            "LAYOUT_LINK.CHILD_CATEGORIES=[]",
            "OCR.USE_DOCTR=False",
            "OCR.USE_TESSERACT=True",
            "USE_LAYOUT_NMS=False",
            "USE_TABLE_REFINEMENT=True",
            "USE_LINE_MATCHER=False",
            "LAYOUT_NMS_PAIRS.COMBINATIONS=None",
            "LAYOUT_NMS_PAIRS.THRESHOLDS=None",
            "LAYOUT_NMS_PAIRS.PRIORITY=None",
            "OCR.WEIGHTS.DOCTR_RECOGNITION.PT=doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt",
        ]
    )

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert len(page.layouts) in {9, 10, 12, 16}
    assert len(page.tables) == 1
    # 15 cells for d2 and 16 for tp model
    assert len(page.tables[0].cells) in {15, 16}  # type: ignore
    # first html for tp model, second for d2 model
    print(page.tables[0].html)
    assert page.tables[0].html in {
        "<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td>"
        "</tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td>"
        "</tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>",
        "<table><tr><td></td><td rowspan=2></td></tr><tr><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>"
        "</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td>"
        "<td></td></tr></table>",
    }
    assert page.height == 2339
    assert page.width == 1654


@mark.deprecated
def test_dd_tf_analyzer_builds_and_process_image_correctly() -> None:
    """
    Analyzer integration test with setting USE_TABLE_SEGMENTATION = True and USE_OCR = True
    """

    # Arrange
    analyzer = get_dd_analyzer()

    # Act
    df = analyzer.analyze(path=get_integration_test_path())
    output = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(output) == 1
    page = output[0]
    assert isinstance(page, Page)
    # 9 for d2 and 10 for tp model
    assert len(page.layouts) in {9, 10, 12, 16}
    assert len(page.tables) == 1
    # 15 cells for d2 and 16 for tp model
    assert len(page.tables[0].cells) in {15, 16}  # type: ignore
    # first html for tp model, second for d2 model
    assert page.tables[0].html in {
        "<table><tr><td>Jahresdurchschnitt der Mitarbeiterzahl</td><td>139</td></tr><tr>"
        "<td>Gesamtvergiitung ?</td><td>EUR 15.315.952</td></tr><tr><td>Fixe Vergiitung</td>"
        "<td>EUR 13.151.856</td></tr><tr><td>Variable Vergiitung</td><td>EUR 2.164.096</td>"
        "</tr><tr><td>davon: Carried Interest</td><td>EURO</td></tr><tr><td>Gesamtvergiitung"
        " fiir Senior Management °</td><td>EUR 1.468.434</td></tr><tr><td>Gesamtvergiitung"
        " fiir sonstige Risikotrager</td><td>EUR 324.229</td></tr><tr><td>Gesamtvergiitung"
        " fir Mitarbeiter mit Kontrollfunktionen</td><td>EUR 554.046</td></tr></table>",
        "<table><tr><td></td><td rowspan=2>EUR 15.315.952</td></tr><tr><td>Gesamtvergiitung ?</td></tr><tr><td>Fixe"
        " Vergiitung</td><td>EUR 13.151.856</td></tr><tr><td>Variable Vergiitung</td><td>EUR 2.164.096</td></tr><tr>"
        "<td>davon: Carried Interest</td><td>EURO</td></tr><tr><td>Gesamtvergiitung fiir Senior Management °</td><td>"
        "EUR 1.468.434</td></tr><tr><td>Gesamtvergiitung fuir sonstige Risikotrager</td><td>EUR 324.229</td></tr><tr>"
        "<td>Gesamtvergiitung fir Mitarbeiter mit Kontrollfunktionen</td><td>EUR 554.046</td></tr></table>",
        "<table><tr><td></td><td>139</td></tr><tr><td>Gesamtvergiitung ?</td><td>EUR 15.315.952</td></tr><tr><td>Fixe "
        "Vergiitung</td><td>EUR 13.151.856</td></tr><tr><td>Variable Vergiitung</td><td>EUR "
        "2.164.096</td></tr><tr><td>davon: Carried Interest</td><td>EURO</td></tr><tr><td>Gesamtvergiitung fiir "
        "Senior Management °</td><td>EUR 1.468.434</td></tr><tr><td>Gesamtvergiitung fiir sonstige Risikotrager"
        "</td><td>EUR 324.229</td></tr><tr><td></td><td></td></tr></table>",
    }
    assert page.height == 2339
    assert page.width == 1654
    # first number for tp model, second for pt model
    assert len(page.text) in {4290, 5042, 5043, 5044, 5045, 5153}
    text_ = page.text_
    assert text_["text"] == page._make_text(line_break=False)  # pylint: disable=W0212
    assert len(text_["words"]) in {529}
    assert len(text_["ann_ids"]) in {529}
