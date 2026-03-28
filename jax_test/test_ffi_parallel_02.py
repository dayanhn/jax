import os
# 必须在导入 jax 之前设置环境变量！
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ["ASCEND_VISIBLE_DEVICES"] = "6,7"
os.system("clear")

print("pid = ", os.getpid())

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, pmap

# 根据后端类型动态导入 matmul 函数
backend = jax.default_backend().lower()
print(f"Using backend: {backend}")

if backend == 'cuda' or backend == 'gpu':
    from jax_ffi_example.cuda_examples import matmul_fwd as matmul
elif backend == 'ascend':
    from jax_plugins.xla_ascend910 import matmul
else:
    raise RuntimeError(f"Unsupported backend: {backend}")

n_devices = len(jax.devices())
print(f"Number of devices: {n_devices}")

m, n, k = 128, 128, 128
np_A = np.full((m, k), 2.0, dtype=np.float32)
np_B = np.full((k, n), 2.0, dtype=np.float32)

# ====================== 集合通信归约示例 ======================
# 目标：两个设备都计算完整的 A@B，然后通过 psum 累加
# 最终结果 = 2 * (A @ B)
def parallel_matmul_gather_raw(A, B):
    """
    每个设备接收完整的 A 和 B（通过复制）
    各自计算 matmul(A, B)
    通过 psum 将所有设备的结果求和
    返回：每个设备都有相同的累加结果
    """
    res = matmul(A, B)
    # psum 会对所有设备的结果求和
    # 输出形状保持不变，但每个设备的值都是全局和
    res = jax.lax.psum(res, axis_name='devices')
    return res

# 使用 pmap 装饰器（带 axis_name 参数）
parallel_matmul_gather = pmap(parallel_matmul_gather_raw, axis_name='devices')

# ======================================================================

try:
    # 复制数据到多个设备
    A_replicated = np.stack([np_A] * n_devices) 
    B_replicated = np.stack([np_B] * n_devices)
    
    print(f"A_replicated shape: {A_replicated.shape}")
    print(f"B_replicated shape: {B_replicated.shape}")
    
    # 运行并行计算
    parallel_result = parallel_matmul_gather(A_replicated, B_replicated)
    
    print(f"\n✅ Parallel result shape: {parallel_result.shape}")
    print(f"   Expected: (2, 128, 128) - 第 0 维是设备轴")
    
    # 验证：每个设备都应该有相同的结果（2 倍的单卡结果）
    print(f"\n✅ Device 0 result[:2,:2]:\n{parallel_result[0][:2,:2]}")
    print(f"✅ Device 1 result[:2,:2]:\n{parallel_result[1][:2,:2]}")

    
except Exception as e:
    print(f"Pmap failed: {e}")
    import traceback
    traceback.print_exc()