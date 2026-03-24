import os
os.system("clear")

# 只使用 6、7 号 GPU 卡
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

print("pid = ",os.getpid())

import jax
import jax.numpy as jnp
from jax import jit, pmap, make_jaxpr

# 导入 cuda_examples 模块，它会自动注册 MatMul 函数
from jax_ffi_example import cuda_examples

print("Testing JAX FFI with Matrix Multiplication...")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

@jit
def test_ffi_matmul(A, B):
    """测试函数：调用 matmul 执行矩阵乘法"""
    return cuda_examples.matmul_fwd(A, B)

# 创建输入数据
m, k, n = 16, 16, 16
A = jnp.arange(m * k, dtype=jnp.float32).reshape((m, k)) / (m * k)
B = jnp.arange(k * n, dtype=jnp.float32).reshape((k, n)) / (k * n)

print(f"A shape: {A.shape}, dtype: {A.dtype}")
print(f"B shape: {B.shape}, dtype: {B.dtype}")
print(f"A[:2, :2] = \n{A[:2, :2]}")
print(f"B[:2, :2] = \n{B[:2, :2]}")

# 1. 获取 Traced 对象
traced_obj = test_ffi_matmul.trace(A, B)
#print("\n=== JAXPR ===")
#print(traced_obj.jaxpr)

# 2. 获取 StableHLO 表示
lowerd_obj = traced_obj.lower()
print("\n=== STABLEHLO ===")
stablehlo = lowerd_obj.as_text('stablehlo')
print(stablehlo[:2000])  # 只打印前 2000 个字符，避免输出过长

# 3. 获取 HLO 表示
print("\n=== HLO ===")
hlo = lowerd_obj.as_text('hlo')
print(hlo[:2000])  # 只打印前 2000 个字符，避免输出过长

# 4. 编译并执行
print("\n=== Compiling and executing ===")
compiled = lowerd_obj.compile()
result = compiled(A, B)
print(f"Result shape: {result.shape}")
print(f"Result[:2, :2] = \n{result[:2, :2]}")

# 5. 验证结果（与 JAX 内置的 dot 函数比较）
print("\n=== Verifying result ===")
expected = jnp.dot(A, B)
print(f"Expected[:2, :2] = \n{expected[:2, :2]}")
print(f"Results match: {jnp.allclose(result, expected)}")
print(f"Max difference: {jnp.max(jnp.abs(result - expected))}")
