import jax
import jax.numpy as jnp
from jax_plugins.xla_ascend910 import gelu, matmul

# 测试GELU
x = jnp.array([-1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
y = gelu(x)
print(f"GELU output: {y}")

# 测试矩阵乘法
# a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
# b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)
# c = matmul(a, b)
# print(f"Matmul output:\n{c}")