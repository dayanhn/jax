import jax
import jax.numpy as jnp
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import update, init_mlp_params_numpy

print(os.getpid())

# 根据后端类型打印信息
backend = jax.default_backend().lower()
print(f"Using backend: {backend}")

# 初始化参数（与 train.py 中相同的配置）
layer_sizes = [784, 512, 256, 10]
params_numpy = init_mlp_params_numpy(layer_sizes, seed=32)

# 将参数搬入设备内存
device = jax.devices()[0]
params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params_numpy)

# 创建示例输入数据
batch_size = 128
x = jnp.zeros((batch_size, 784))  # MNIST 图像展平为 784 维
y = jnp.zeros((batch_size,), dtype=jnp.int32)  # 标签
lr = 0.01

# ========== 1. 获取 Lowered 对象 ==========
print("\n" + "="*80)
print("Step 1: Creating Lowered Object for update function")
print("="*80)

# 使用 jit 来包装函数并获取 lowered 对象
# update 函数已经被 @jax.jit 装饰，但我们仍然可以调用 .lower()
lowered_obj = update.lower(params, x, y, lr)

print("✓ Lowered object created successfully")
print(f"Input params structure: {jax.tree_util.tree_structure(params)}")
print(f"Input x shape: {x.shape}, dtype: {x.dtype}")
print(f"Input y shape: {y.shape}, dtype: {y.dtype}")
print(f"Learning rate: {lr}")

# ========== 2. 获取 StableHLO 表示 ==========
print("\n" + "="*80)
print("Step 2: Exporting STABLEHLO")
print("="*80)
stablehlo = lowered_obj.as_text('stablehlo')
print(f"StableHLO length: {len(stablehlo)} characters")
print("\n=== STABLEHLO ===")
print(stablehlo)

# 保存到文件
with open('update_stablehlo.txt', 'w') as f:
    f.write(stablehlo)
print("\n✓ StableHLO saved to update_stablehlo.txt")

# ========== 3. 获取 HLO 表示 ==========
print("\n" + "="*80)
print("Step 3: Exporting HLO")
print("="*80)
hlo = lowered_obj.as_text('hlo')
print(f"HLO length: {len(hlo)} characters")
print("\n=== HLO ===")
print(hlo)

# 保存到文件
with open('update_hlo.txt', 'w') as f:
    f.write(hlo)
print("\n✓ HLO saved to update_hlo.txt")

# ========== 4. 编译并执行（可选验证）==========
print("\n" + "="*80)
print("Step 4: Compiling and Executing (validation)")
print("="*80)
compiled = lowered_obj.compile()
print("✓ Compilation successful")

result = compiled()
print(f"✓ Execution successful")
print(f"Result type: {type(result)}")
print(f"Number of layers in result: {len(result)}")
for i, layer in enumerate(result):
    print(f"  Layer {i}: W shape={layer['W'].shape}, b shape={layer['b'].shape}")
