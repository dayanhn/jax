# Copyright 2018 The JAX Authors.
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

"""Repository rule for Ascend configuration."""

load(
    "@rules_ml_toolchain//common:common.bzl",
    "get_host_environ",
)

_TF_NEED_ASCEND = "TF_NEED_ASCEND"
ASCEND_TOOLKIT_HOME = "ASCEND_TOOLKIT_HOME"

def _enable_ascend(repository_ctx):
    """Returns whether to build with Ascend support."""
    return int(get_host_environ(repository_ctx, _TF_NEED_ASCEND, False))

def _create_local_ascend_repository(repository_ctx):
    """Creates the repository containing files set up to build with Ascend."""
    
    # Create BUILD file with config_setting
    repository_ctx.file(
        "ascend/BUILD",
        content = """
# Config setting to enable Ascend
config_setting(
    name = "enable_ascend",
    values = {"define": "using_ascend=true"},
)
""",
    )
    
    # Create ascend_config.h
    ascend_toolkit_home = get_host_environ(repository_ctx, ASCEND_TOOLKIT_HOME, "")
    repository_ctx.file(
        "ascend/ascend/ascend_config.h",
        content = """
#ifndef ASCEND_CONFIG_H_
#define ASCEND_CONFIG_H_

#define ASCEND_TOOLKIT_PATH "{ascend_toolkit_home}"

#endif  // ASCEND_CONFIG_H_
""".format(ascend_toolkit_home=ascend_toolkit_home),
    )
    
    # Create ascend_config.py
    repository_ctx.file(
        "ascend/ascend/ascend_config.py",
        content = """
ascend_toolkit_home = "{ascend_toolkit_home}"
""".format(ascend_toolkit_home=ascend_toolkit_home),
    )

def _create_dummy_ascend_repository(repository_ctx):
    """Creates a dummy repository when Ascend is disabled."""
    repository_ctx.file(
        "ascend/BUILD",
        content = """
config_setting(
    name = "enable_ascend",
    values = {"define": "using_ascend=false"},
)
""",
    )
    
    repository_ctx.file(
        "ascend/ascend/ascend_config.h",
        content = """
#ifndef ASCEND_CONFIG_H_
#define ASCEND_CONFIG_H_

#define ASCEND_TOOLKIT_PATH ""

#endif  // ASCEND_CONFIG_H_
""",
    )
    
    repository_ctx.file(
        "ascend/ascend/ascend_config.py",
        content = "ascend_toolkit_home = \"\"\n",
    )

def _ascend_configure_impl(repository_ctx):
    """Implementation of the ascend_configure repository rule."""
    if _enable_ascend(repository_ctx):
        _create_local_ascend_repository(repository_ctx)
    else:
        _create_dummy_ascend_repository(repository_ctx)

ascend_configure = repository_rule(
    implementation = _ascend_configure_impl,
    attrs = {
        "environ": attr.string_dict(),
    },
    environ = [_TF_NEED_ASCEND, ASCEND_TOOLKIT_HOME],
)

"""Detects and configures the Ascend environment.

Add the following to your WORKSPACE file:

```python
ascend_configure(name = "local_config_ascend")
```

Args:
  name: A unique name for this workspace rule.
"""
