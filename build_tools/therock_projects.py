#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Central registry of TheRock project metadata.

This module provides a single source of truth for project metadata including:
- Artifact names (lowercase, used in artifact files and S3 paths)
- CMake target names (mixed-case, used in CMakeLists.txt)
- Build subdirectories (relative paths where projects are built)
- Component groups (ml-libs, math-libs, etc.)
- LibrarySONAMEs (shared library file patterns)

Usage:
    from therock_projects import get_by_artifact_name, get_by_cmake_target

    # Lookup by artifact name
    metadata = get_by_artifact_name("hipdnn")
    print(metadata.cmake_target)  # "hipDNN"

    # Lookup by CMake target
    metadata = get_by_cmake_target("MIOpen")
    print(metadata.artifact_name)  # "miopen"

    # Get build directory
    build_dir = metadata.get_build_dir()  # "build/ml-libs/MIOpen/build"
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProjectMetadata:
    """Metadata for a TheRock project/component.

    Attributes:
        artifact_name: Lowercase name used in artifacts and S3 paths (e.g., "hipdnn")
        cmake_target: CMake target name, may be mixed-case (e.g., "hipDNN")
        build_subdir: Relative path to build directory (e.g., "ml-libs/hipDNN")
        component_group: Component group name (e.g., "ml-libs", "math-libs")
        library_soname: Optional shared library filename pattern (e.g., "libhipdnn_backend.so*")
    """

    artifact_name: str
    cmake_target: str
    build_subdir: str
    component_group: str
    library_soname: Optional[str] = None

    def get_build_dir(self, base_dir: str = "build") -> str:
        """Get the full build directory path for this project.

        Args:
            base_dir: Base build directory (default: "build")

        Returns:
            Full path to project build directory, e.g., "build/ml-libs/hipDNN/build"
        """
        return f"{base_dir}/{self.build_subdir}/build"


# Central project registry
_PROJECT_REGISTRY: list[ProjectMetadata] = [
    # ML Libraries
    ProjectMetadata(
        artifact_name="hipdnn",
        cmake_target="hipDNN",
        build_subdir="ml-libs/hipDNN",
        component_group="ml-libs",
        library_soname="libhipdnn_backend.so*",
    ),
    ProjectMetadata(
        artifact_name="miopen",
        cmake_target="MIOpen",
        build_subdir="ml-libs/MIOpen",
        component_group="ml-libs",
        library_soname="libMIOpen.so*",
    ),
    ProjectMetadata(
        artifact_name="composable-kernel",
        cmake_target="composable_kernel",
        build_subdir="ml-libs/composable_kernel",
        component_group="ml-libs",
    ),
    # Math Libraries
    ProjectMetadata(
        artifact_name="blas",
        cmake_target="rocBLAS",
        build_subdir="math-libs/BLAS",
        component_group="math-libs",
        library_soname="librocblas.so*",
    ),
    ProjectMetadata(
        artifact_name="prim",
        cmake_target="rocPRIM",
        build_subdir="math-libs/PRIM",
        component_group="math-libs",
    ),
    ProjectMetadata(
        artifact_name="rand",
        cmake_target="rocRAND",
        build_subdir="math-libs/RAND",
        component_group="math-libs",
        library_soname="librocrand.so*",
    ),
    ProjectMetadata(
        artifact_name="fft",
        cmake_target="rocFFT",
        build_subdir="math-libs/FFT",
        component_group="math-libs",
        library_soname="librocfft.so*",
    ),
]


def list_all_projects() -> list[ProjectMetadata]:
    """Get a list of all registered projects.

    Returns:
        List of all ProjectMetadata objects in the registry.
    """
    return list(_PROJECT_REGISTRY)


def get_by_artifact_name(artifact_name: str) -> Optional[ProjectMetadata]:
    """Look up project metadata by artifact name (lowercase).

    Args:
        artifact_name: Artifact name to search for (e.g., "hipdnn")

    Returns:
        ProjectMetadata if found, None otherwise.
    """
    for project in _PROJECT_REGISTRY:
        if project.artifact_name == artifact_name:
            return project
    return None


def get_by_cmake_target(cmake_target: str) -> Optional[ProjectMetadata]:
    """Look up project metadata by CMake target name.

    Args:
        cmake_target: CMake target name to search for (e.g., "hipDNN")

    Returns:
        ProjectMetadata if found, None otherwise.
    """
    for project in _PROJECT_REGISTRY:
        if project.cmake_target == cmake_target:
            return project
    return None


def get_by_uppercase_name(uppercase_name: str) -> Optional[ProjectMetadata]:
    """Look up project metadata by uppercase name (THEROCK_ENABLE_* style).

    This handles conversion from uppercase with underscores to lowercase with hyphens.
    For example: "COMPOSABLE_KERNEL" -> "composable-kernel"

    Args:
        uppercase_name: Uppercase name to search for (e.g., "HIPDNN", "COMPOSABLE_KERNEL")

    Returns:
        ProjectMetadata if found, None otherwise.
    """
    # Convert uppercase with underscores to lowercase with hyphens
    lowercase_name = uppercase_name.lower().replace("_", "-")

    # Try direct lookup first
    result = get_by_artifact_name(lowercase_name)
    if result:
        return result

    # Also try without replacement (for names without hyphens)
    if "_" in uppercase_name:
        result = get_by_artifact_name(uppercase_name.lower())
        if result:
            return result

    return None


# Legacy compatibility: NAME_MAPPING dictionary
# Maps artifact names to CMake targets
NAME_MAPPING = {project.artifact_name: project.cmake_target for project in _PROJECT_REGISTRY}


def get_cmake_target(artifact_name: str) -> Optional[str]:
    """Legacy function to get CMake target from artifact name.

    Args:
        artifact_name: Artifact name (e.g., "hipdnn")

    Returns:
        CMake target name if found, None otherwise.
    """
    metadata = get_by_artifact_name(artifact_name)
    return metadata.cmake_target if metadata else None
