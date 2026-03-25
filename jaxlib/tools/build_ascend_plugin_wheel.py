# Copyright 2023 The JAX Authors.
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

# Script that builds a jax ascend plugin wheel, intended to be run via bazel run
# as part of the jax ascend plugin build process.

# Most users should not run this script directly; use build.py instead.

import argparse
import functools
import os
import pathlib
import tempfile

from bazel_tools.tools.python.runfiles import runfiles
from jaxlib.tools import build_utils

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--sources_path",
    default=None,
    help="Path in which the wheel's sources should be prepared. Optional. If "
    "omitted, a temporary directory will be used.",
)
parser.add_argument(
    "--output_path",
    default=None,
    required=True,
    help="Path to which the output wheel should be written. Required.",
)
parser.add_argument(
    "--jaxlib_git_hash",
    default="",
    required=True,
    help="Git hash. Empty if unknown. Optional.",
)
parser.add_argument(
    "--cpu", default=None, required=True, help="Target CPU architecture. Required."
)
parser.add_argument(
    "--platform_version",
    default=None,
    required=True,
    help="Target Ascend version. Required.",
)
parser.add_argument(
    "--editable",
    action="store_true",
    help="Create an 'editable' jax ascend plugin build instead of a wheel.",
)
parser.add_argument(
    "--enable-ascend",
    default=False,
    help="Should we build with Ascend enabled? Requires Ascend toolkit.")
parser.add_argument(
    "--srcs", help="source files for the wheel", action="append"
)
args = parser.parse_args()

r = runfiles.Create()


def write_setup_cfg(sources_path, cpu):
  tag = build_utils.platform_tag(cpu)
  with open(sources_path / "setup.cfg", "w") as f:
    f.write(
        f"""[metadata]
license_files = LICENSE.txt

[bdist_wheel]
plat_name={tag}
python_tag=py3
"""
    )


def prepare_ascend_plugin_wheel(
    wheel_sources_path: pathlib.Path,
    *, 
    cpu,
    ascend_version,
    wheel_sources,
):
  """Assembles a source tree for the wheel in `wheel_sources_path`"""
  source_file_prefix = build_utils.get_source_file_prefix(wheel_sources)
  wheel_sources_map = build_utils.create_wheel_sources_map(
      wheel_sources, root_packages=["jax_plugins", "jaxlib"]
  )
  copy_files = functools.partial(
      build_utils.copy_file,
      runfiles=r,
      wheel_sources_map=wheel_sources_map,
  )

  # Copy plugin pyproject.toml and setup.py
  # Try plugin-specific files first (for plugin wheel), fallback to generic names (for PJRT wheel)
  pyproject_toml = f"{source_file_prefix}jax_plugins/ascend/plugin_pyproject.toml"
  setup_py = f"{source_file_prefix}jax_plugins/ascend/plugin_setup.py"
  
  # Check if plugin-specific files exist in wheel_sources_map
  is_plugin_wheel = pyproject_toml in wheel_sources_map
  if is_plugin_wheel:
    copy_files(
        pyproject_toml,
        dst_dir=wheel_sources_path,
        dst_filename="pyproject.toml",
    )
    copy_files(
        setup_py,
        dst_dir=wheel_sources_path,
        dst_filename="setup.py",
    )
  else:
    # Fallback to generic files for PJRT wheel
    copy_files(
        f"{source_file_prefix}jax_plugins/ascend/pyproject.toml",
        dst_dir=wheel_sources_path,
        dst_filename="pyproject.toml",
    )
    copy_files(
        f"{source_file_prefix}jax_plugins/ascend/setup.py",
        dst_dir=wheel_sources_path,
        dst_filename="setup.py",
    )
  
  # Copy LICENSE.txt
  copy_files(
      f"{source_file_prefix}jaxlib/tools/LICENSE.txt",
      dst_dir=wheel_sources_path,
  )
  build_utils.update_setup_with_ascend_version(wheel_sources_path, ascend_version)
  write_setup_cfg(wheel_sources_path, cpu)

  # Determine the correct package directory based on wheel type
  if is_plugin_wheel:
    # For plugin wheel, use jax_ascend{version}_plugin
    plugin_dir = wheel_sources_path / f"jax_ascend{ascend_version}_plugin"
  else:
    # For PJRT wheel, use jax_plugins.xla_ascend{version}
    plugin_dir = wheel_sources_path / f"jax_plugins" / f"xla_ascend{ascend_version}"
    # Create parent directory
    plugin_dir.parent.mkdir(exist_ok=True)
  
  if is_plugin_wheel:
    # For plugin wheel, copy plugin extension modules
    copy_files(
        f"{source_file_prefix}jaxlib/ascend/_versions.so",
        dst_dir=plugin_dir,
    )
    copy_files(
        f"{source_file_prefix}jaxlib/ascend/ascend_plugin_extension.so",
        dst_dir=plugin_dir,
    )
  else:
    # For PJRT wheel, copy __init__.py and ffi_ops.py for plugin package registration
    copy_files(
        f"{source_file_prefix}jax_plugins/ascend/__init__.py",
        dst_dir=plugin_dir,
        dst_filename="__init__.py",
    )
    copy_files(
        f"{source_file_prefix}jax_plugins/ascend/ffi_ops.py",
        dst_dir=plugin_dir,
        dst_filename="ffi_ops.py",
    )
  
  # Only copy PJRT C API plugin if it's available
  pjrt_plugin_path = f"{source_file_prefix}jax_plugins/ascend/pjrt_c_api_ascend_plugin.so"
  if pjrt_plugin_path in wheel_sources_map:
      copy_files(
          pjrt_plugin_path,
          dst_dir=plugin_dir,
          dst_filename="xla_ascend_plugin.so",
      )


tmpdir = None
sources_path = args.sources_path
if sources_path is None:
  tmpdir = tempfile.TemporaryDirectory(prefix="jaxascendpjrt")
  sources_path = tmpdir.name

try:
  os.makedirs(args.output_path, exist_ok=True)

  if args.enable_ascend:
    prepare_ascend_plugin_wheel(
        pathlib.Path(sources_path),
        cpu=args.cpu,
        ascend_version=args.platform_version,
        wheel_sources=args.srcs,
    )
    package_name = "jax ascend plugin"
  else:
    raise ValueError("Unsupported backend. Choose 'ascend'.")

  if args.editable:
    build_utils.build_editable(sources_path, args.output_path, package_name)
  else:
    build_utils.build_wheel(
        sources_path,
        args.output_path,
        package_name,
        git_hash=args.jaxlib_git_hash,
    )
finally:
  if tmpdir:
    tmpdir.cleanup()
