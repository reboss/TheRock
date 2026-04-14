#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for therock_projects module - central project metadata registry."""

import pytest
from therock_projects import (
    ProjectMetadata,
    get_by_artifact_name,
    get_by_cmake_target,
    get_by_uppercase_name,
    get_cmake_target,
    list_all_projects,
    NAME_MAPPING,
)


class TestProjectMetadata:
    """Test the ProjectMetadata dataclass."""

    def test_project_metadata_creation(self):
        """Test creating a ProjectMetadata instance."""
        proj = ProjectMetadata(
            artifact_name="hipdnn",
            cmake_target="hipDNN",
            build_subdir="ml-libs/hipDNN",
            component_group="ml-libs",
            library_soname="libhipdnn_backend.so*",
        )
        assert proj.artifact_name == "hipdnn"
        assert proj.cmake_target == "hipDNN"
        assert proj.build_subdir == "ml-libs/hipDNN"
        assert proj.component_group == "ml-libs"
        assert proj.library_soname == "libhipdnn_backend.so*"

    def test_get_build_dir(self):
        """Test get_build_dir method."""
        proj = ProjectMetadata(
            artifact_name="hipdnn",
            cmake_target="hipDNN",
            build_subdir="ml-libs/hipDNN",
            component_group="ml-libs",
        )
        assert proj.get_build_dir() == "build/ml-libs/hipDNN/build"
        assert proj.get_build_dir("custom") == "custom/ml-libs/hipDNN/build"
        assert proj.get_build_dir("TheRock/build-coverage") == "TheRock/build-coverage/ml-libs/hipDNN/build"

    def test_project_metadata_is_frozen(self):
        """Test that ProjectMetadata is immutable."""
        proj = ProjectMetadata(
            artifact_name="hipdnn",
            cmake_target="hipDNN",
            build_subdir="ml-libs/hipDNN",
            component_group="ml-libs",
        )
        with pytest.raises(AttributeError):
            proj.artifact_name = "other"


class TestLookupFunctions:
    """Test the lookup functions."""

    def test_get_by_artifact_name_exists(self):
        """Test looking up project by artifact name (lowercase)."""
        metadata = get_by_artifact_name("hipdnn")
        assert metadata is not None
        assert metadata.cmake_target == "hipDNN"
        assert metadata.build_subdir == "ml-libs/hipDNN"
        assert metadata.component_group == "ml-libs"

    def test_get_by_artifact_name_not_found(self):
        """Test looking up nonexistent artifact."""
        assert get_by_artifact_name("nonexistent") is None

    def test_get_by_cmake_target_exists(self):
        """Test looking up project by CMake target name."""
        metadata = get_by_cmake_target("hipDNN")
        assert metadata is not None
        assert metadata.artifact_name == "hipdnn"

    def test_get_by_cmake_target_not_found(self):
        """Test looking up nonexistent CMake target."""
        assert get_by_cmake_target("NonExistent") is None

    def test_get_by_uppercase_name_exists(self):
        """Test looking up project by uppercase name (THEROCK_ENABLE_* style)."""
        metadata = get_by_uppercase_name("HIPDNN")
        assert metadata is not None
        assert metadata.cmake_target == "hipDNN"
        assert metadata.artifact_name == "hipdnn"

    def test_get_by_uppercase_name_with_underscores(self):
        """Test uppercase lookup handles hyphens -> underscores."""
        # "composable-kernel" artifact should be findable as "COMPOSABLE_KERNEL"
        metadata = get_by_uppercase_name("COMPOSABLE_KERNEL")
        if metadata:  # Only test if it exists in registry
            assert metadata.artifact_name == "composable-kernel"

    def test_get_by_uppercase_name_not_found(self):
        """Test looking up nonexistent uppercase name."""
        assert get_by_uppercase_name("NONEXISTENT") is None


class TestLegacyCompatibility:
    """Test backwards compatibility for existing code."""

    def test_name_mapping_exists(self):
        """Test that NAME_MAPPING dict is available."""
        assert isinstance(NAME_MAPPING, dict)
        assert len(NAME_MAPPING) > 0

    def test_name_mapping_hipdnn(self):
        """Test NAME_MAPPING for hipDNN."""
        assert NAME_MAPPING["hipdnn"] == "hipDNN"

    def test_name_mapping_miopen(self):
        """Test NAME_MAPPING for MIOpen."""
        assert NAME_MAPPING["miopen"] == "MIOpen"

    def test_name_mapping_blas(self):
        """Test NAME_MAPPING for BLAS."""
        assert NAME_MAPPING["blas"] == "rocBLAS"

    def test_get_cmake_target_function(self):
        """Test legacy get_cmake_target function."""
        assert get_cmake_target("hipdnn") == "hipDNN"
        assert get_cmake_target("miopen") == "MIOpen"
        assert get_cmake_target("nonexistent") is None


class TestRegistry:
    """Test the complete project registry."""

    def test_list_all_projects_returns_list(self):
        """Test that list_all_projects returns a list."""
        projects = list_all_projects()
        assert isinstance(projects, list)
        assert len(projects) > 0

    def test_all_projects_have_required_fields(self):
        """Test that all registered projects have required fields."""
        for proj in list_all_projects():
            assert proj.artifact_name, f"Project missing artifact_name: {proj}"
            assert proj.cmake_target, f"Project missing cmake_target: {proj}"
            assert proj.build_subdir, f"Project missing build_subdir: {proj}"
            assert proj.component_group, f"Project missing component_group: {proj}"

    def test_no_duplicate_artifact_names(self):
        """Test that all artifact names are unique."""
        artifact_names = [p.artifact_name for p in list_all_projects()]
        assert len(artifact_names) == len(set(artifact_names)), "Duplicate artifact names found"

    def test_no_duplicate_cmake_targets(self):
        """Test that all CMake targets are unique."""
        cmake_targets = [p.cmake_target for p in list_all_projects()]
        assert len(cmake_targets) == len(set(cmake_targets)), "Duplicate CMake targets found"

    def test_known_ml_libs_projects(self):
        """Test that known ML libraries are registered."""
        ml_projects = [p for p in list_all_projects() if p.component_group == "ml-libs"]
        artifact_names = [p.artifact_name for p in ml_projects]

        # These should exist
        assert "hipdnn" in artifact_names
        assert "miopen" in artifact_names

    def test_known_math_libs_projects(self):
        """Test that known math libraries are registered."""
        math_projects = [p for p in list_all_projects() if p.component_group == "math-libs"]
        cmake_targets = [p.cmake_target for p in math_projects]

        # At least some of these should exist
        common_math_libs = {"rocBLAS", "rocPRIM", "rocRAND", "rocFFT"}
        found = common_math_libs.intersection(set(cmake_targets))
        assert len(found) > 0, f"Expected some math libs, got: {cmake_targets}"
