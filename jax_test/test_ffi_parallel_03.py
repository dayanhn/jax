import os
# 必须在导入 jax 之前设置环境变量！
os.environ["ASCEND_VISIBLE_DEVICES"] = "6,7"
os.system("clear")
print("pid = ", os.getpid())

import jax
import jax.numpy as jnp
from jax import pmap, jit
from jax_plugins.xla_ascend910 import matmul
import numpy as np

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

# ========== 方法：使用 lower_for_tracing 导出 IR ==========
# 对于 pmap 函数，需要使用 lower 直接获取 Lowered 对象，而不是 trace+lower
print("\n" + "="*80)
print("Step 1: Creating Lowered Object for PMAP function")
print("="*80)

lowered_obj = None

# 使用 pmap 的 lowering 方法
try:
    A_replicated = np.stack([np_A] * n_devices)  # [n_devices, 128, 128]
    B_replicated = np.stack([np_B] * n_devices)  # [n_devices, 128, 128]
    lowered_obj = parallel_matmul_gather.lower(A_replicated, B_replicated)

    print("✓ Lowered object created successfully using pmap.lower()")
except Exception as e:
    print(f"Error with pmap.lower(): {e}")
    print("Trying alternative approach with jit...")
    
if lowered_obj is None:
    raise RuntimeError("Failed to create lowered object with both pmap and jit approaches.")

# ========== 2. 获取 StableHLO 表示 ==========
print("\n" + "="*80)
print("Step 2: Exporting STABLEHLO")
print("="*80)
stablehlo = lowered_obj.as_text('stablehlo')
print(f"StableHLO length: {len(stablehlo)} characters")
print("\n=== STABLEHLO (first 3000 chars) ===")
print(stablehlo[:3000])
if len(stablehlo) > 3000:
    print(f"\n... (truncated, total {len(stablehlo)} chars)")


# ========== 3. 获取 HLO 表示 ==========
print("\n" + "="*80)
print("Step 3: Exporting HLO")
print("="*80)
hlo = lowered_obj.as_text('hlo')
print(f"HLO length: {len(hlo)} characters")
print("\n=== HLO (first 3000 chars) ===")
print(hlo[:3000])
if len(hlo) > 3000:
    print(f"\n... (truncated, total {len(hlo)} chars)")

# ========== 4. 编译并执行 ==========
print("\n" + "="*80)
print("Step 4: Compiling and Executing")
print("="*80)
compiled = lowered_obj.compile()
print("✓ Compilation successful")

parallel_result = compiled(A_replicated, B_replicated)
print(f"✓ Execution successful")
print(f"Parallel result shape: {parallel_result.shape}")
print(f"Parallel result devices: {parallel_result.devices()}")
