import os
os.system("clear")

# 只使用 6、7 号 GPU 卡
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

print("pid = ", os.getpid())

import jax
import jax.numpy as jnp
from jax import jit, pmap, make_jaxpr

# 导入 cuda_examples 模块，它会自动注册 MatMul 和 MyGelu 函数
from jax_ffi_example import cuda_examples

print("Testing JAX FFI with Matrix Multiplication + GELU...")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

@jit
def test_ffi_matmul_then_gelu(A, B):
    """测试函数：先调用 matmul，然后将结果作为 gelu 的输入
    
    维度信息会自动从输入 Buffer 中提取。
    """
    # Step 1: 执行矩阵乘法 C = A @ B
    matmul_result = cuda_examples.matmul_fwd(A, B)
    
    # Step 2: 对矩阵乘法结果执行 GELU activation
    gelu_result = cuda_examples.gelu_fwd(matmul_result)
    
    return gelu_result

# 创建输入数据
m, k, n = 16, 16, 16
A = jnp.arange(m * k, dtype=jnp.float32).reshape((m, k)) / (m * k)
B = jnp.arange(k * n, dtype=jnp.float32).reshape((k, n)) / (k * n)

print(f"A shape: {A.shape}, dtype: {A.dtype}")
print(f"B shape: {B.shape}, dtype: {B.dtype}")
print(f"A[:2, :2] = \n{A[:2, :2]}")
print(f"B[:2, :2] = \n{B[:2, :2]}")

# 1. 获取 Traced 对象
traced_obj = test_ffi_matmul_then_gelu.trace(A, B)
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
print(f"Result dtype: {result.dtype}")
print(f"Result first few elements: {result[:2, :2]}")
print(f"Result min: {result.min():.6f}, max: {result.max():.6f}")
print(f"Result mean: {result.mean():.6f}")

# 5. 验证结果（与 JAX 内置函数比较）
print("\n=== Verifying result ===")
matmul_expected = jnp.dot(A, B)
gelu_expected = jax.nn.gelu(matmul_expected)
print(f"Expected first few elements: {gelu_expected[:2, :2]}")
print(f"Results match: {jnp.allclose(result, gelu_expected)}")
print(f"Max difference: {jnp.max(jnp.abs(result - gelu_expected))}")

