import os
os.system("clear")

# 只使用 6、7 号 卡
os.environ["ASCEND_VISIBLE_DEVICES"] = "6,7"


#os.system("clear")
os.system("rm -rf ./tmp/xla_dump/*")
os.system("rm -rf ./tmp/jax_dump/*")

dump_out = True

if dump_out:
    # Ensure C++/Abseil logging flags are set before any C++ extensions are loaded.
    # Use TF_CPP_* env vars which logging_initializer.cc reads and applies.
    # Increase max vlog so XLA_VLOG_LINES(3, ...) will print
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
    os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "4"


    # Enable verbose logging for HLO-related sources and MLIR conversion passes.
    # Use basenames (file without extension), e.g. hlo_module for hlo_module.cc.
    os.environ["TF_CPP_VMODULE"] = ("thunk_emitter=4,compile_module_to_llvm_ir=4")

# Move XLA_FLAGS here so they take effect before importing JAX/C++ backends.
# --xla_gpu_force_compilation_parallelism=1 禁用多线程编译，方便调试
'''
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_experimental_enable_triton_heroless_priority_fusion=true '
    '--xla_dump_to=./tmp/xla_dump '
    '--xla_gpu_force_compilation_parallelism=1 '
    '--xla_dump_hlo_as_text=true '
    '--xla_dump_hlo_as_proto=false '
    '--xla_dump_hlo_pass_re=.* '
    '--xla_dump_hlo_module_re=jit_matmul_with_elementwise '
    # 自动调优日志配置
    '--xla_gpu_dump_autotune_logs_to=./tmp/autotune_logs.txt '
    )
'''

print("pid = ",os.getpid())

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, pmap, make_jaxpr

print("Testing JAX operator fusion with Triton...")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

x = jnp.ones((32, 32,32))
print(f"JAX 实际值:   {x[0, ]:.10f}")

# 示例 2: 矩阵操作与 element-wise 操作的组合（适合 Triton 处理）
@jit
def matmul_with_elementwise(x, y):
    """矩阵乘法与 element-wise 操作的组合"""
    # 矩阵乘法通常会触发 Triton 优化
    matmul_result = jnp.matmul(x, y)
    # 后续的 element-wise 操作应该与矩阵乘法融合
    activated = jnp.tanh(matmul_result)
    scaled = jnp.multiply(activated, 2.0)
    biased = jnp.add(scaled, 0.1)
    return biased


# 创建输入
x = jnp.ones((32, 32))
y = jnp.ones((32, 32))

# ============================================
# 执行并验证结果
# ============================================

# 编译并执行
compiled = matmul_with_elementwise.lower(x, y).compile()
z = compiled(x, y)

# 使用 NumPy 计算预期结果
x_np = np.ones((32, 32))
y_np = np.ones((32, 32))
expected = np.tanh(np.matmul(x_np, y_np)) * 2.0 + 0.1

print("\n" + "="*60)
print("结果验证:")
print("="*60)
print(f"NumPy 预期值: {expected[0, 0]:.10f}")
print(f"JAX 实际值:   {z[0, ]:.10f}")
print(f"是否匹配:     {np.allclose(expected, z)}")
print(f"最大误差:     {np.max(np.abs(expected - z)):.2e}")
