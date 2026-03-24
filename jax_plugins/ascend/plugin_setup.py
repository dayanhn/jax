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

import importlib
import os
from setuptools import setup
from setuptools.dist import Distribution

__version__ = None
ascend_version = 0  # placeholder
project_name = f"jax-ascend{ascend_version}-plugin"
package_name = f"jax_ascend{ascend_version}_plugin"

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

try:
  _version_module = load_version_module(package_name)
  __version__ = _version_module._get_version_for_build()
  _cmdclass = _version_module._get_cmdclass(package_name)
except Exception:
  __version__ = "0.0.1.dev0"
  _cmdclass = {}

class BinaryDistribution(Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True

setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description="JAX Plugin for Ascend NPUs",
    long_description="",
    long_description_content_type="text/markdown",
    author="JAX team",
    author_email="jax-dev@google.com",
    packages=[package_name],
    python_requires=">=3.11",
    # install_requires=[f"jax-ascend{ascend_version}-pjrt>={__version__}"],
    extras_require={
      'with-cann': [
          # Add CANN-specific Python dependencies here if needed
          # For example, if Huawei provides Python packages for CANN
      ],
    },
    url="https://github.com/jax-ml/jax",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Free Threading :: 3 - Stable",
    ],
    package_data={
        package_name: [
            "*",
        ],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)
