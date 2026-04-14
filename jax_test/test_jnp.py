import jax
import jax.numpy as jnp
import os

print(os.getpid())


# 根据后端类型打印信息
backend = jax.default_backend().lower()
print(f"Using backend: {backend}")

m, n, k = 128, 128, 128

# ========== 1. 创建函数并获取 Lowered 对象 ==========
def create_full_matrix():
    """创建值为 2.0 的矩阵"""
    return jnp.full((m, n, k), 2.0)

print("\n" + "="*80)
print("Step 1: Creating Lowered Object")
print("="*80)

# 使用 jit 来包装函数并获取 lowered 对象
lowered_obj = jax.jit(create_full_matrix).lower()

print("✓ Lowered object created successfully using jit.lower()")

# ========== 2. 获取 StableHLO 表示 ==========
print("\n" + "="*80)
print("Step 2: Exporting STABLEHLO")
print("="*80)
stablehlo = lowered_obj.as_text('stablehlo')
print(f"StableHLO length: {len(stablehlo)} characters")
print("\n=== STABLEHLO ===")
print(stablehlo)


# ========== 3. 获取 HLO 表示 ==========
print("\n" + "="*80)
print("Step 3: Exporting HLO")
print("="*80)
hlo = lowered_obj.as_text('hlo')
print(f"HLO length: {len(hlo)} characters")
print("\n=== HLO ===")
print(hlo)

# ========== 4. 编译并执行 ==========
print("\n" + "="*80)
print("Step 4: Compiling and Executing")
print("="*80)
compiled = lowered_obj.compile()
print("✓ Compilation successful")

result = compiled()
print(f"✓ Execution successful")
print(f"Result shape: {result.shape}")
print(f"Result dtype: {result.dtype}")
print(f"Result[0, 0]: {result[0, 0]}")
