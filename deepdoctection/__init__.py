# -*- coding: utf-8 -*-
# File: __init__.py

"""
Init file for deepdoctection package
"""

import sys
import os
from typing import TYPE_CHECKING
from packaging import version

from .utils.file_utils import get_tf_version, pytorch_available, tf_available, _LazyModule
from .utils.logger import logger

__version__ = 0.17

_IMPORT_STRUCTURE = {
    "analyzer": ["get_dd_analyzer",
                 "build_analyzer",
                 "load_page",
                 "load_document"],
    "configs": [],
    "dataflow": ["DataFlowTerminated",
                 "DataFlowResetStateNotCalled",
                 "DataFlowReentrantGuard",
                 "DataFlow",
                 "RNGDataFlow",
                 "ProxyDataFlow",
                 "TestDataSpeed",
                 "FlattenData",
                 "MapData",
                 "MapDataComponent",
                 "RepeatedData",
                 "ConcatData",
                 "JoinData",
                 "CacheData",
                 "CustomDataFromList",
                 "CustomDataFromIterable",
                 "SerializerJsonlines",
                 "SerializerTabsepFiles",
                 "SerializerFiles",
                 "CocoParser",
                 "SerializerCoco",
                 "SerializerPdfDoc",
                 "MultiThreadMapData",
                 "MultiProcessMapData"
                 "DataFromList",
                 "DataFromIterable",
                 "FakeData",
                 "PickleSerializer",
                 "MeanFromDataFlow",
                 "StdFromDataFlow",
                 ],
    "datapoint": ["Annotation",
                  "CategoryAnnotation",
                  "ImageAnnotation",
                  "SummaryAnnotation",
                  "ContainerAnnotation",
                  "coco_iou",
                  "area",
                  "intersection",
                  "np_iou",
                  "iou",
                  "BoundingBoxError",
                  "BoundingBox",
                  "intersection_box",
                  "crop_box_from_image",
                  "local_to_global_coords",
                  "global_to_local_coords",
                  "merge_boxes",
                  "rescale_coords",
                  "convert_b64_to_np_array",
                  "convert_np_array_to_b64",
                  "convert_np_array_to_b64_b",
                  "convert_pdf_bytes_to_np_array_v2",
                  "box_to_point4",
                  "point4_to_box",
                  "as_dict",
                  "Image",
                  "Word",
                  "Layout",
                  "Cell",
                  "Table",
                  "Page",
                  ],
    "datasets" : ["DatasetAdapter",
                  "DatasetBase",
                  "MergeDataset",
                  "DataFlowBaseBuilder",
                  "DatasetInfo",
                  "DatasetCategories",
                  "get_merged_categories",
                  "dataset_registry",
                  "get_dataset",
                  "print_dataset_infos",
                  "dataflow_to_json"
                  ],
    "datasets.instances":["DocLayNet",
                          "DocLayNetSeq",
                          "Fintabnet",
                          "Funsd",
                          "IIITar13K",
                          "LayoutTest",
                          "Publaynet",
                          "Pubtables1M",
                          "Pubtabnet",
                          "Rvlcdip",
                          "Xfund"],
    "datasets.instances.xsl": [],
    "eval": ["AccuracyMetric",
             "ConfusionMetric",
             "PrecisionMetric",
             "RecallMetric",
             "F1Metric",
             "PrecisionMetricMicro",
             "RecallMetricMicro",
             "F1MetricMicro",
             "MetricBase",
             "CocoMetric",
             "Evaluator",
             "metric_registry",
             "get_metric",
             "TableTree",
             "CustomConfig",
             "TEDS",
             "TedsMetric",
             "EvalCallback"
             ],
    "extern": ["PredictorBase",
               "DetectionResult",
               "ObjectDetector",
               "PdfMiner",
               "TextRecognizer",
               "TokenClassResult",
               "SequenceClassResult",
               "LMTokenClassifier",
               "LMSequenceClassifier",
               "LanguageDetector",
               "InferenceResize",
               "D2FrcnnDetector",
               "DoctrTextlineDetector",
               "DoctrTextRecognizer",
               "FasttextLangDetector",
               "HFLayoutLmTokenClassifier",
               "HFLayoutLmSequenceClassifier",
               "ModelProfile",
               "ModelCatalog",
               "print_model_infos",
               "ModelDownloadManager",
               "PdfPlumberTextDetector",
               "TesseractOcrDetector",
               "TextractOcrDetector",
               "TPFrcnnDetector",
               ],
    "extern.pt": ["set_torch_auto_device",
                  "get_num_gpu"],
    "extern.tp": ["disable_tfv2",
                  "ModelDescWithConfig",
                  "TensorpackPredictor"
                  ],
    "extern.tp.tpfrcnn": [ "CustomResize",
                           "anchors_and_labels",
                           "augment"],
    "extern.tp.tpfrcnn.utils" : ["area",
                                 "pairwise_intersection",
                                 "pairwise_iou"],
    "extern.tp.tpfrcnn.config" : ["model_frcnn_config",
                                  "train_frcnn_config"],
    "extern.tp.tpfrcnn.modeling" : ["ResNetFPNModel"],
    "mapper" : ["cat_to_sub_cat",
                "re_assign_cat_ids",
                "filter_cat",
                "filter_summary",
                "image_to_cat_id",
                "remove_cats",
                "coco_to_image",
                "image_to_coco",
                "image_to_d2_frcnn_training",
                "image_to_layoutlm",
                "image_to_raw_layoutlm_features",
                "raw_features_to_layoutlm_features",
                "LayoutLMDataCollator",
                "image_to_layoutlm_features",
                "DataCollator",
                "LayoutLMFeatures",
                "MappingContextManager",
                "DefaultMapper",
                "maybe_get_fake_score",
                "LabelSummarizer",
                "curry",
                "match_anns_by_intersection",
                "to_image",
                "maybe_load_image",
                "maybe_remove_image",
                "maybe_remove_image_from_category",
                "image_ann_to_image",
                "maybe_ann_to_sub_image",
                "xml_to_dict",
                "to_page",
                "page_dict_to_page",
                "pascal_voc_dict_to_image",
                "prodigy_to_image",
                "image_to_prodigy",
                "pub_to_image_uncur",
                "pub_to_image",
                "image_to_tp_frcnn_training",
                "xfund_to_image",
                ],
    "pipe": ["DatapointManager",
             "PipelineComponent",
             "PredictorPipelineComponent",
             "LanguageModelPipelineComponent",
             "Pipeline",
             "DetectResultGenerator",
             "SubImageLayoutService",
             "ImageCroppingService",
             "MatchingService",
             "PageParsingService",
             "MultiThreadPipelineComponent",
             "DoctectionPipe",
             "LanguageDetectionService",
             "ImageLayoutService",
             "LMTokenClassifierService",
             "LMSequenceClassifierService",
             "TableSegmentationRefinementService",
             "pipeline_component_registry",
             "TableSegmentationService",
             "SegmentationResult",
             "TextExtractionService",
             "TextOrderService",
             ],
    "train": ["D2Trainer",
              "train_d2_faster_rcnn",
              "LayoutLMTrainer",
              "train_hf_layoutlm",
              "train_faster_rcnn"],
    "utils": ["timeout_manager",
              "save_tmp_file",
              "timed_operation",
              "get_tensorflow_requirement",
              "tf_addons_available",
              "get_tf_addons_requirements",
              "tensorpack_available",
              "get_tensorpack_requirement",
              "get_pytorch_requirement",
              "lxml_available",
              "get_lxml_requirement",
              "apted_available",
              "get_apted_requirement",
              "distance_available",
              "get_distance_requirement",
              "transformers_available",
              "get_transformers_requirement",
              "detectron2_available",
              "get_detectron2_requirement",
              "tesseract_available",
              "get_tesseract_version",
              "get_tesseract_requirement",
              "pdf_to_ppm_available",
              "pdf_to_cairo_available",
              "get_poppler_requirement",
              "pdfplumber_available",
              "get_pdfplumber_requirement",
              "cocotools_available",
              "get_cocotools_requirement",
              "scipy_available",
              "sklearn_available",
              "get_sklearn_requirement",
              "qpdf_available",
              "boto3_available",
              "get_boto3_requirement",
              "aws_available",
              "get_aws_requirement",
              "doctr_available",
              "get_doctr_requirement",
              "fasttext_available",
              "get_fasttext_requirement",
              "load_image_from_file",
              "load_bytes_from_pdf_file",
              "get_load_image_func",
              "maybe_path_or_pdf",
              "download",
              "mkdir_p",
              "is_file_extension",
              "load_json",
              "FileExtensionError",
              "is_uuid_like",
              "get_uuid_from_str",
              "get_uuid",
              "logger",
              "set_logger_dir",
              "auto_set_dir",
              "get_logger_dir",
              "AttrDict",
              "set_config_by_yaml",
              "save_config_to_yaml",
              "config_to_cli_str",
              "decrypt_pdf_document",
              "get_pdf_file_reader",
              "get_pdf_file_writer",
              "PDFStreamer",
              "pdf_to_np_array",
              "ObjectTypes",
              "TypeOrStr",
              "object_types_registry",
              "DefaultType",
              "PageType",
              "SummaryType",
              "DocumentType",
              "LayoutType",
              "TableType",
              "CellType",
              "WordType",
              "TokenClasses",
              "BioTag",
              "TokenClassWithTag",
              "Relationships",
              "Languages",
              "DatasetType",
              "get_type",
              "sub_path",
              "get_package_path",
              "get_configs_dir_path",
              "get_weights_dir_path",
              "get_dataset_dir_path",
              "get_tqdm",
              "get_tqdm_default_kwargs",
              "ResizeTransform",
              "InferenceResize",
              "delete_keys_from_dict",
              "split_string",
              "string_to_dict",
              "to_bool",
              "call_only_once",
              "get_rng",
              "FileExtensionError",
              "is_file_extension",
              "draw_text",
              "draw_boxes",
              "interactive_imshow",
              ]




}

if not tf_available() and not pytorch_available():
    logger.info(
        "Neither Tensorflow or Pytorch are available. You will not be able to use any Deep Learning model from"
        "the library."
    )

# disable TF warnings for versions > 2.4.1
if tf_available():
    if version.parse(get_tf_version()) > version.parse("2.4.1"):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        import tensorflow.python.util.deprecation as deprecation  # type: ignore # pylint: disable=E0401,R0402

        deprecation._PRINT_DEPRECATION_WARNINGS = False  # pylint: disable=W0212
    except Exception:  # pylint: disable=W0703
        try:
            from tensorflow.python.util import deprecation  # type: ignore # pylint: disable=E0401

            deprecation._PRINT_DEPRECATION_WARNINGS = False  # pylint: disable=W0212
        except Exception:  # pylint: disable=W0703
            pass

# Direct imports for type-checking
if TYPE_CHECKING:
    # pylint: disable=wrong-import-position
    from .analyzer import *
    from .dataflow import *
    from .datapoint import *
    from .datasets import *
    from .eval import *
    from .extern import *
    from .mapper import *
    from .pipe import *
    from .train import *
    from .utils import *
    # pylint: enable=wrong-import-position
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _IMPORT_STRUCTURE,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
