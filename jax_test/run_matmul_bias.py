import os

os.system('clear')
print(os.getpid())

dump_out = True

if dump_out:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
    os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "4"
    os.environ["TF_CPP_VMODULE"] = ("hlo_pass_pipeline=4,thunk_emitter=4,compile_module_to_llvm_ir=4")

os.environ['XLA_FLAGS'] = (
    '--xla_dump_to=./tmp/xla_dump/run_matmul_bias '
    '--xla_gpu_force_compilation_parallelism=1 '
    '--xla_dump_hlo_as_text=true '
    '--xla_dump_hlo_as_proto=false '
    '--xla_dump_hlo_pass_re=.* '
    '--xla_dump_hlo_module_re=.*  '
    #'--xla_disable_hlo_passes=aclnn-gemm-rewriter'  # 禁用 ACLNN GEMM rewriter
    )
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def matmul_with_bias(x, weight, bias):
    """
    矩阵乘法 + bias
    
    Args:
        x: 输入矩阵, shape [batch_size, input_dim]
        weight: 权重矩阵, shape [input_dim, output_dim]
        bias: 偏置向量, shape [output_dim]
    
    Returns:
        输出矩阵, shape [batch_size, output_dim]
    """
    return jnp.dot(x, weight) + bias


def main():
    # 获取设备信息
    devices = jax.devices()
    device = devices[0]
    print(f"Using device: {device}")
    print(f"Backend: {device.platform}")
    print(f"PID: {os.getpid()}")

    # 使用固定种子确保跨后端一致性
    rng = np.random.default_rng(42)
    
    # 定义维度
    batch_size = 4
    input_dim = 128
    output_dim = 64
    
    # 生成固定输入数据
    x_np = rng.standard_normal((batch_size, input_dim)).astype(np.float32)
    weight_np = rng.standard_normal((input_dim, output_dim)).astype(np.float32)
    bias_np = rng.standard_normal((output_dim,)).astype(np.float32)
    
    # 将数据放到设备上
    x = jax.device_put(x_np, device)
    weight = jax.device_put(weight_np, device)
    bias = jax.device_put(bias_np, device)
    
    # 执行 JIT 编译的计算
    result = matmul_with_bias(x, weight, bias)
    jax.block_until_ready(result)
    
    # 打印结果
    print(f"\nInput shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Bias shape: {bias.shape}")
    print(f"Output shape: {result.shape}")
    print(f"\nExpected output value (all elements): {input_dim + 1}")  # 128 + 1 = 129
    print(f"\nActual output statistics:")
    print(f"  Mean: {result.mean():.10f}")
    print(f"  Std:  {result.std():.10f}")
    print(f"  Min:  {result.min():.10f}")
    print(f"  Max:  {result.max():.10f}")
    print(f"\nFirst row of output (first 10 elements):")
    print(f"  {result[0, :10]}")


if __name__ == '__main__':
    main()
