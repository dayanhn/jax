import os
#os.system("clear")

# 只使用 6、7 号 GPU 卡
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

print("pid = ",os.getpid())

import jax
import jax.numpy as jnp
from jax import jit, pmap, make_jaxpr

print("Testing JAX FFI with ASCEND GELU...")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

# 导入 cuda_examples 模块，它会自动注册 MyGelu 函数
from jax_ffi_example import cuda_examples

@jit
def test_ffi_gelu_fwd(x):
    """测试函数：调用 gelu_fwd 函数执行计算"""
    return cuda_examples.gelu_fwd(x)

# 创建输入数据
size = 1024
x = jnp.linspace(-5.0, 5.0, size * size, dtype=jnp.float32).reshape((size, size))

print(f"Input shape: {x.shape}")
print(f"Input dtype: {x.dtype}")
print(f"Input first few elements: {x[:2, :2]}")

# 1. 获取 Traced 对象
traced_obj = test_ffi_gelu_fwd.trace(x)
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
result = compiled(x)
print(f"Result shape: {result.shape}")
print(f"Result first few elements: {result[:2, :2]}")

# 5. 验证结果（与 JAX 内置的 gelu 函数比较）
print("\n=== Verifying result ===")
expected = jax.nn.gelu(x)
print(f"Expected first few elements: {expected[:2, :2]}")
print(f"Results match: {jnp.allclose(result, expected)}")