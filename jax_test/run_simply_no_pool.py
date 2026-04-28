import os

os.system('clear')
print(os.getpid())

dump_out = True

if dump_out:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
    os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "4"
    os.environ["TF_CPP_VMODULE"] = ("hlo_pass_pipeline=4,thunk_emitter=4,compile_module_to_llvm_ir=4")

os.environ['XLA_FLAGS'] = (
    '--xla_dump_to=./tmp/xla_dump/run_simply_no_pool '
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

def create_model():
    rng = np.random.default_rng(42)
    W1 = rng.standard_normal((128, 256)).astype(np.float32)
    b1 = rng.standard_normal((256,)).astype(np.float32)
    W2 = rng.standard_normal((256, 64)).astype(np.float32)
    b2 = rng.standard_normal((64,)).astype(np.float32)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def forward(params, x):
    h = jnp.dot(x, params['W1']) + params['b1']
    out = jnp.dot(h, params['W2']) + params['b2']
    return out

def loss_fn(params, x, y):
    logits = forward(params, x)
    target = jax.nn.one_hot(y, logits.shape[-1])
    loss = jnp.mean((logits - target) ** 2)
    return loss

@jax.jit
def forward_and_backward(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    return loss, grads

@jax.jit
def update(params, x, y, lr):
    """使用梯度下降更新参数"""
    grads = jax.grad(loss_fn)(params, x, y)
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, grads

def main():
    params_np = create_model() 
    device = jax.devices()[0]
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params_np)

    batch_size = 2
    # 使用固定种子的 RNG 生成输入数据,确保跨后端一致性
    input_rng = np.random.default_rng(123)
    x = jax.device_put(input_rng.standard_normal((batch_size, 128)).astype(np.float32), device)
    y = jax.device_put(jnp.array([0, 1], dtype=jnp.int32), device)

    # 设置学习率
    lr = 0.01
    
    #loss, grads = forward_and_backward(params, x, y)
    #jax.block_until_ready((loss, grads))
    
    params, grads = update(params, x, y, lr)
    jax.block_until_ready(params)
    

if __name__ == '__main__':
    main()
