# -*- coding: utf-8 -*-
# File: test_cont_ann.py

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
Unit tests for the functionality of the ContainerAnnotation class.

This module provides test cases to validate the functionality of
the `ContainerAnnotation` class, including its ability to handle
type-aware value assignment, type coercion, and validations. It
ensures the integrity of data encapsulated in the annotation
objects and helps enforce correct behavior under various conditions.
"""

import json
import re

import pytest
from pydantic import ValidationError

from dd_core.datapoint.annotation import (
    AnnotationRef,
    CategoryAnnotation,
    ContainerAnnotation,
    ReferencePayload,
    build_container_annotation,
    container_annotation_registry,
)
from dd_core.utils.object_types import get_type


class TestContainerAnnotation:
    """Tests for ContainerAnnotation class with type-aware value handling"""

    def test_container_annotation_creation_basic(self) -> None:
        """Test basic ContainerAnnotation creation"""
        container = ContainerAnnotation(category_name="test_cat_1", value="Hello World")
        assert container.value == "Hello World"
        assert isinstance(container.value, str)
        assert container.value_type == "str"

    def test_container_annotation_value_coercion_int_untyped(self) -> None:
        """Test that ContainerAnnotation coerces int value to int"""
        container = ContainerAnnotation(category_name="test_cat_2", value=42)
        assert container.value == 42
        assert isinstance(container.value, int)
        assert container.value_type == "int"

    def test_container_annotation_set_type_str(self) -> None:
        """Test that ContainerAnnotation can be set to a new type"""
        container = ContainerAnnotation(category_name="test_cat_1")
        container.set_type("str")
        container.value = "test"
        assert container.value == "test"

    def test_container_annotation_set_type_int_assignment_preserved(self) -> None:
        """Test that ContainerAnnotation preserves assignment type when set_type() is called"""
        container = ContainerAnnotation(category_name="test_cat_2")
        container.set_type("int")
        container.value = 42
        assert container.value == 42

    def test_container_annotation_set_type_float(self) -> None:
        """Test that ContainerAnnotation can be set to a new type"""
        container = ContainerAnnotation(category_name="test_cat_2")
        container.set_type("float")
        container.value = 3.14
        assert container.value == 3.14

    def test_container_annotation_set_type_list_str(self) -> None:
        """Test that ContainerAnnotation can be set to a new type"""
        container = ContainerAnnotation(category_name="test_cat_1", value=["a", "b"])
        assert container.value == ["a", "b"]
        container.set_type("list[str]")

    def test_container_annotation_value_dict_infers_type(self) -> None:
        """Test that dict values infer value_type='dict[str,Any]'."""
        payload = {"a": 1, "b": "x", "c": {"nested": True}}
        container = ContainerAnnotation(category_name="test_cat_1", value=payload)
        assert container.value == payload
        assert isinstance(container.value, dict)
        assert container.value_type == "dict[str,Any]"

    def test_container_annotation_set_type_dict_rejects_non_dict(self) -> None:
        """Test that setting type to dict[str,Any] rejects non-dict values."""
        container = ContainerAnnotation(category_name="test_cat_1", value="not_a_dict")
        msg = "value must be dict[str,Any] or JSON object string when type='dict[str,Any]'"
        with pytest.raises(TypeError, match=re.escape(msg)):
            container.set_type("dict[str,Any]")

    def test_container_annotation_list_str_rejects_non_str(self) -> None:
        """Test that ContainerAnnotation rejects non-str values in list[str]"""
        with pytest.raises(ValidationError):
            _ = ContainerAnnotation(category_name="test_cat_1", value=["a", 2])

    def test_container_annotation_set_type_none_disables_validation_and_coerces(self) -> None:
        """Test that ContainerAnnotation can be set to None"""
        container = ContainerAnnotation(category_name="test_cat_2", value=42)
        container.value = 42
        container.set_type("str")
        assert container.value == "42"

    def test_container_annotation_set_type_invalid_raises_error(self) -> None:
        """Test that set_type raises error for invalid type"""
        container = ContainerAnnotation(category_name="test_cat_1")
        with pytest.raises(ValueError):
            container.set_type("invalid_type")  # type: ignore

    def test_container_validates_and_converts_value(self) -> None:
        """Test that ContainerAnnotation validates and converts value on assignment"""
        container = ContainerAnnotation(category_name="test_cat_1")
        container.set_type("str")
        container.value = 456
        assert container.value == "456"

    def test_container_value_type_raises_value_error_when_None(self) -> None:
        """Test that value_type raises ValueError when value is None"""
        container = ContainerAnnotation(category_name="test_cat_2", value=5)
        with pytest.raises(ValueError, match="type cannot be None"):
            container.set_type(None)  # type: ignore


class TestContainerAnnotationAdvanced:
    """Advanced tests for ContainerAnnotation"""

    def test_container_annotation_from_dict_with_value_field(self) -> None:
        """Test that ContainerAnnotation is correctly identified from dict with value field"""
        data = {"category_name": "test_cat_1", "category_id": 1, "value": "test_text"}
        parent = CategoryAnnotation(category_name="test_cat_3", external_id="parent_id")
        parent.dump_sub_category(get_type("sub_cat_1"), ContainerAnnotation(**data))
        retrieved = parent.get_sub_category(get_type("sub_cat_1"))
        assert isinstance(retrieved, ContainerAnnotation)
        assert retrieved.value == "test_text"

    def test_container_annotation_set_type_validates_existing_value(self) -> None:
        """Test that set_type validates existing value immediately"""
        container = ContainerAnnotation(category_name="test_cat_1", value="string_value")
        container.set_type("str")

        container2 = ContainerAnnotation(category_name="test_cat_1", value="string_value")
        with pytest.raises(TypeError):
            container2.set_type("int")


class TestContainerAnnotationReferencePayloadSerialization:
    """Tests for serialization/round-trip of ReferencePayload/AnnotationRef container values."""

    def test_reference_payload_value_serializes_with_ref_type_markers(self) -> None:
        """as_dict must emit ``_ref_type`` markers for ReferencePayload/AnnotationRef leaves."""
        payload = ReferencePayload(
            content={"h": {"num": [AnnotationRef(image_id="img1", annotation_id="ann1")]}}
        )
        container = ContainerAnnotation(category_name="test_cat_1", value=payload)
        assert container.value_type == "reference_payload"

        data = container.as_dict()
        assert "_ref_type" in json.dumps(data["value"], default=str)
        assert data["value"] == {
            "_ref_type": "reference_payload",
            "content": {"h": {"num": [{"_ref_type": "annotation_ref", "image_id": "img1", "annotation_id": "ann1"}]}},
        }

    def test_reference_payload_value_round_trips(self) -> None:
        """Construct -> as_dict -> build_container_annotation must recover ReferencePayload + AnnotationRef leaves."""
        payload = ReferencePayload(
            content={"h": {"num": [AnnotationRef(image_id="img1", annotation_id="ann1")]}}
        )
        container = ContainerAnnotation(category_name="test_cat_1", value=payload)

        reloaded = build_container_annotation(container.as_dict())

        assert isinstance(reloaded.value, ReferencePayload)
        assert reloaded.value_type == "reference_payload"
        leaf = reloaded.value.content["h"]["num"][0]
        assert isinstance(leaf, AnnotationRef)
        assert leaf == AnnotationRef(image_id="img1", annotation_id="ann1")

    def test_llm_container_reference_payload_round_trips_and_keeps_markers(self) -> None:
        """LLMContainerAnnotation must keep its ``_container_type``/``_annotation_id`` and round-trip the payload."""
        llm_cls = container_annotation_registry.get("llm")
        payload = ReferencePayload(
            content={"h": {"num": [AnnotationRef(image_id="img1", annotation_id="ann1")]}}
        )
        container = llm_cls(
            category_name="test_cat_1",
            value=payload,
            task_id="t",
            prompt_id="p",
            output_format_id="o",
            model_id="m",
        )

        data = container.as_dict()
        assert data["_container_type"] == "llm"
        assert "_annotation_id" in data
        assert "_ref_type" in json.dumps(data["value"], default=str)

        reloaded = build_container_annotation(data)
        assert type(reloaded).__name__ == "LLMContainerAnnotation"
        assert isinstance(reloaded.value, ReferencePayload)
        assert isinstance(reloaded.value.content["h"]["num"][0], AnnotationRef)

    @pytest.mark.parametrize(
        "value, expected_type",
        [
            ("hello", "str"),
            (42, "int"),
            (3.14, "float"),
            (["a", "b"], "list[str]"),
            ({"a": 1, "b": {"nested": True}}, "dict[str,Any]"),
        ],
    )
    def test_plain_values_round_trip_unchanged(self, value: object, expected_type: str) -> None:
        """Plain values must serialize unchanged (to_json_compatible is a no-op) and round-trip exactly."""
        container = ContainerAnnotation(category_name="test_cat_1", value=value)
        assert container.value_type == expected_type

        data = container.as_dict()
        assert data["value"] == value

        reloaded = build_container_annotation(data)
        assert reloaded.value == value
        assert reloaded.value_type == expected_type

    def test_legacy_dict_without_markers_loads_without_error(self) -> None:
        """Backward compat: a value dict written before the fix (no markers) loads as plain dict, no exception."""
        legacy = {
            "category_name": "test_cat_1",
            "value": {"h": {"num": [{"image_id": "img1", "annotation_id": "ann1"}]}},
        }
        reloaded = build_container_annotation(legacy)
        assert isinstance(reloaded.value, dict)
        assert reloaded.value_type == "dict[str,Any]"
        assert reloaded.value == {"h": {"num": [{"image_id": "img1", "annotation_id": "ann1"}]}}
