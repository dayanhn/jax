# Copyright 2026 The JAX Authors.
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
"""Ascend FFI operators for JAX."""

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct


def gelu(x):
  """GELU activation function using Ascend FFI.
  
  Args:
    x: Input tensor.
  
  Returns:
    Output tensor after applying GELU activation.
  """
  assert x.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
  out_type = ShapeDtypeStruct(x.shape, x.dtype)
  return jax.ffi.ffi_call("ascend.gelu", (out_type,))(x)


def matmul(a, b):
  """Matrix multiplication using Ascend FFI.
  
  Args:
    a: First input tensor.
    b: Second input tensor.
  
  Returns:
    Output tensor after matrix multiplication.
  """
  assert a.dtype == b.dtype
  assert a.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]
  assert len(a.shape) >= 2
  assert len(b.shape) >= 2
  
  # Calculate output shape
  out_shape = a.shape[:-1] + b.shape[1:]
  out_type = ShapeDtypeStruct(out_shape, a.dtype)
  return jax.ffi.ffi_call("ascend.matmul", (out_type,))(a, b)
