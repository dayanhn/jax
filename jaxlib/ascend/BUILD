# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ascend backend

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library")
load(
    "//jaxlib:jax.bzl",
    "if_ascend_is_configured",
    "nanobind_extension",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

cc_library(
    name = "ascend_vendor",
    hdrs = [
        "//jaxlib/gpu:vendor.h",
    ],
    defines = ["JAX_GPU_ASCEND=1"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "versions_helpers",
    srcs = ["versions_helpers.cc"],
    hdrs = ["versions_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base:dynamic_annotations",
    ],
)

nanobind_extension(
    name = "_versions",
    srcs = ["versions.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_versions",
    deps = [
        ":versions_helpers",
        "//jaxlib:absl_status_casters",
        "//jaxlib:kernel_nanobind_helpers",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
    ],
)

py_library(
    name = "ascend_gpu_support",
    deps = [
        ":_versions",
    ],
)

nanobind_extension(
    name = "ascend_plugin_extension",
    srcs = ["ascend_plugin_extension.cc"],
    module_name = "ascend_plugin_extension",
    deps = [
        "//jaxlib:kernel_nanobind_helpers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@nanobind",
        "@xla//xla/pjrt:status_casters",
    ],
)

# We cannot nest select and if_ascend_is_configured so we introduce
# a standalone py_library target.
py_library(
    name = "gpu_only_test_deps",
    # `if_ascend_is_configured` will default to `[]`.
    deps = if_ascend_is_configured([
        ":ascend_gpu_support",
        ":ascend_plugin_extension",
    ]),
)
