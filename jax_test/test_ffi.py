import jax
import jax.numpy as jnp
from jax_plugins.xla_ascend910 import gelu, matmul
import numpy as np

# 通过numpy来创建输入数据，避免触发XLA编译
m, n, k = 2048, 2048, 2048
np_A = np.full((m, k), 2.0, dtype=np.float32)
np_B = np.full((k, n), 2.0, dtype=np.float32)

A = jax.device_put(np_A, jax.devices()[0])
B = jax.device_put(np_B, jax.devices()[0])

c = matmul(A, B)
y = gelu(c)
np_y = jax.device_get(y)
print(f"Result shape: {np_y.shape}")
print(f"Result[:2, :2] = \n{np_y[:2, :2]}")
