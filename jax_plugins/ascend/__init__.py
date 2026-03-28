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

"""JAX Ascend plugin."""

import ctypes
import functools
import importlib
import logging
import os
import pathlib
import traceback
from typing import Any

from jax._src.lib import xla_client
import jax._src.xla_bridge as xb

# Import FFI operators
from .ffi_ops import gelu, matmul

ascend_plugin_extension = None
_initialized = False  # Flag to prevent duplicate initialization


def _import_extensions():
  """Import the ascend plugin extension module."""
  global ascend_plugin_extension

  # Try to import from different package names
  for pkg_name in ['jax_ascend910_plugin', 'jax_ascend_plugin', 'jaxlib.ascend']:
    try:
      ascend_plugin_extension = importlib.import_module(
          f'{pkg_name}.ascend_plugin_extension'
      )
    except ImportError:
      ascend_plugin_extension = None
    else:
      break


logger = logging.getLogger(__name__)


def _get_library_path():
  """Get the path to the ascend plugin shared library."""
  installed_path = (
      pathlib.Path(__file__).resolve().parent / 'xla_ascend_plugin.so'
  )
  if installed_path.exists():
    return installed_path

  local_path = os.path.join(
      os.path.dirname(__file__), 'pjrt_c_api_ascend_plugin.so'
  )
  if not os.path.exists(local_path):
    runfiles_dir = os.getenv('RUNFILES_DIR', None)
    if runfiles_dir:
      local_path = os.path.join(
          runfiles_dir, '__main__/jax_plugins/ascend/pjrt_c_api_ascend_plugin.so'
      )

  if os.path.exists(local_path):
    logger.debug(
        'Native library %s does not exist. This most likely indicates an issue'
        ' with how %s was built or installed. Fallback to local test'
        ' library %s',
        installed_path,
        __package__,
        local_path,
    )
    return local_path

  logger.debug(
      'WARNING: Native library %s and local test library path %s do not'
      ' exist. This most likely indicates an issue with how %s was built or'
      ' installed or missing src files.',
      installed_path,
      local_path,
      __package__,
  )
  return None


def _load_ascend_libraries():
  """Load Ascend CANN libraries if needed.
  
  This function ensures that the Ascend CANN libraries are properly loaded.
  The ASCEND_TOOLKIT_HOME environment variable should be set by sourcing:
    source $ASCEND_TOOLKIT_HOME/../set_env.sh
  """
  ascend_toolkit_home = os.environ.get('ASCEND_TOOLKIT_HOME')
  if ascend_toolkit_home:
    logger.info(f"Using Ascend toolkit from: {ascend_toolkit_home}")
    # Add library path for runtime loading
    library_path = os.path.join(ascend_toolkit_home, 'lib64')
    if os.path.exists(library_path):
      # Preload critical libraries to avoid dlopen issues
      try:
        ctypes.cdll.LoadLibrary(os.path.join(library_path, 'libascendcl.so'))
        ctypes.cdll.LoadLibrary(os.path.join(library_path, 'libnnopbase.so'))
        ctypes.cdll.LoadLibrary(os.path.join(library_path, 'libopapi_nn.so'))
        ctypes.cdll.LoadLibrary(os.path.join(library_path, 'libhccl.so'))
        ctypes.cdll.LoadLibrary(os.path.join(library_path, 'libhcomm.so'))
        logger.debug("Loaded libascendcl.so")
      except OSError as e:
        logger.warning(f"Failed to preload libascendcl.so: {e}")
  else:
    logger.warning(
        "ASCEND_TOOLKIT_HOME is not set. Please source the environment script:\n"
        "  source $ASCEND_TOOLKIT_HOME/../set_env.sh"
    )


def initialize(options=None):
  """Initialize the Ascend PJRT plugin.
  
  Args:
    options: Optional dictionary of plugin options. If not provided,
             options will be read from environment variables.
  """
  global _initialized
  
  # Prevent duplicate initialization
  if _initialized:
    logger.debug("Ascend PJRT plugin already initialized, skipping")
    return
  
  _import_extensions()
  path = _get_library_path()
  
  if path is None:
    logger.error("Ascend plugin library not found!")
    return

  # Load Ascend CANN libraries first
  _load_ascend_libraries()

  try:
    # Generate plugin options from environment or use provided options
    if options is None:
      options = {}
    
    # Read options from environment variables (similar to CUDA plugin)
    # Use lowercase 'visible_devices' key to match _options_from_jax_configs
    if 'visible_devices' not in options:
      visible_devices = os.getenv('ASCEND_VISIBLE_DEVICES')
      if visible_devices is not None and visible_devices != 'all':
        options['visible_devices'] = [int(x) for x in visible_devices.split(',')]
    
    if 'ASCEND_DEVICE_COUNT' not in options:
      device_count = os.getenv('JAX_ASCEND_DEVICE_COUNT')
      if device_count is not None:
        options['ASCEND_DEVICE_COUNT'] = device_count
    
    # Check if we should skip initialization
    if not os.getenv("JAX_SKIP_ASCEND_INIT", False):
      # Initialize the plugin if extension is available
      if ascend_plugin_extension:
        try:
          ascend_plugin_extension.initialize_ascend()
          logger.info("Ascend plugin extension initialized successfully")
        except Exception as e:
          logger.warning(f"Failed to initialize ascend plugin extension: {e}")
    
    # Register the plugin with XLA
    c_api = xb.register_plugin(
        'ascend', 
        priority=500, 
        library_path=str(path), 
        options=options
    )
    logger.info(f"Ascend PJRT plugin registered successfully from: {path}")
    _initialized = True
    
    # Register custom type handlers if extension is available
    if ascend_plugin_extension:
      try:
        xla_client.register_custom_type_handler(
            "ASCEND",
            functools.partial(
                ascend_plugin_extension.register_custom_type, c_api
            ),
        )
        logger.debug("Registered custom type handler for ASCEND")
      except Exception as e:
        logger.warning(f"Failed to register custom type handler: {e}")
      
      try:
        xla_client.register_custom_call_handler(
            "ASCEND",
            functools.partial(
                ascend_plugin_extension.register_custom_call_target, c_api
            ),
        )
        logger.debug("Registered custom call handler for ASCEND")
      except Exception as e:
        logger.warning(f"Failed to register custom call handler: {e}")
        
  except Exception as e:
    logger.error(f"Failed to initialize Ascend plugin: {e}\n{traceback.format_exc()}")
    raise


# Auto-initialize when module is imported
try:
  initialize()
except Exception as e:
  logger.warning(f"Ascend plugin initialization failed: {e}")


__all__ = [
    "gelu",
    "matmul",
]

