import os

os.system("clear")
print("pid = ", os.getpid())

import jax
import jax.numpy as jnp
from jax import pmap
from jax_plugins.xla_ascend910 import gelu, matmul

import numpy as np

print(f"Number of devices: {len(jax.devices())}")
devices = jax.devices()
n_devices = len(devices)

# 创建大输入数据
m, n, k = 2048, 2048, 2048
np_A = np.full((m, k), 2.0, dtype=np.float32)
np_B = np.full((k, n), 2.0, dtype=np.float32)

# ========== NumPy 层面分片 ==========
np_A_split = np.split(np_A, n_devices, axis=0)
np_A_batched = np.stack(np_A_split)
np_B_replicated = np.stack([np_B] * n_devices)

print(f"np_A_batched shape: {np_A_batched.shape}")
print(f"np_B_replicated shape: {np_B_replicated.shape}")

# ========== 正确分片放到设备上 ==========
# 直接用 pmap 兼容的分片方式，JAX 会自动分发
A_sharded = np_A_batched
B_sharded = np_B_replicated

# ========== 定义 pmap 函数 ==========
@pmap
def parallel_matmul(A, B):
    return matmul(A, B)

# ========== 运行并行计算 ==========
try:
    parallel_result = parallel_matmul(A_sharded, B_sharded)
    
    print(f"Parallel result shape: {parallel_result.shape}")
    print(f"Parallel result devices: {parallel_result.devices()}")
    merged_on_gpu = jnp.concatenate(parallel_result, axis=0)
    print(f"Parallel result shape: {merged_on_gpu.shape}")
    print(f"Parallel result devices: {merged_on_gpu.devices()}")

    # ========== 最简单安全的结果合并 ==========
    result_np = np.array(parallel_result)
    merged_np = np.concatenate(result_np, axis=0)

    print(f"\n✅ 合并成功！")
    print(f"Merged result shape: {merged_np.shape}")
    print(f"Result[:2, :2] = \n{merged_np[:2, :2]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
