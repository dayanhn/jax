import os

print(os.getpid())

dump_out = True

if dump_out:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
    os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "4"
    os.environ["TF_CPP_VMODULE"] = ("hlo_pass_pipeline=4,thunk_emitter=4,compile_module_to_llvm_ir=4")

os.environ['XLA_FLAGS'] = (
    '--xla_dump_to=./tmp/xla_dump '
    '--xla_gpu_force_compilation_parallelism=1 '
    '--xla_dump_hlo_as_text=true '
    '--xla_dump_hlo_as_proto=false '
    '--xla_dump_hlo_pass_re=.* '
    '--xla_dump_hlo_module_re=.*  '
    )

import jax
import jax.numpy as jnp
import numpy as np

def create_model():
    rng = np.random.default_rng(42)
    # 第一层卷积：3x3 kernel, 输入通道3, 输出通道16
    W1 = rng.standard_normal((3, 3, 3, 16)).astype(np.float32)
    b1 = np.zeros(16, dtype=np.float32)  # 第一层bias
    # 第二层卷积：3x3 kernel, 输入通道16, 输出通道32
    W2 = rng.standard_normal((3, 3, 16, 32)).astype(np.float32)
    b2 = np.zeros(32, dtype=np.float32)  # 第二层bias
    # 全连接层：输入特征数32*32*32=32768, 输出通道10
    W3 = rng.standard_normal((32768, 10)).astype(np.float32)
    b3 = np.zeros(10, dtype=np.float32)  # 第三层bias
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

def forward(params, x):
    # 第一层卷积
    h = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=params['W1'],
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    h = h + params['b1']  # 添加bias
    
    # 第二层卷积
    h = jax.lax.conv_general_dilated(
        lhs=h,
        rhs=params['W2'],
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    h = h + params['b2']  # 添加bias
    
    # 展平
    h = h.reshape(h.shape[0], -1)
    # 全连接层
    out = jnp.dot(h, params['W3']) + params['b3']  # 添加bias
    return out

def loss_fn(params, x, y):
    logits = forward(params, x)
    target = jax.nn.one_hot(y, logits.shape[-1])
    loss = jnp.mean((logits - target) ** 2)
    return loss

@jax.jit
def convnet_forward_and_backward(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    return loss, grads

def main():
    params_np = create_model()
    device = jax.devices()[0]
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params_np)

    batch_size = 2
    # 输入形状：(batch_size, height, width, channels) = (2, 32, 32, 3)
    x = jax.device_put(np.random.randn(batch_size, 32, 32, 3).astype(np.float32), device)
    y = jax.device_put(jnp.array([0, 1], dtype=jnp.int32), device)

    # 前向和反向传播
    loss, grads = convnet_forward_and_backward(params, x, y)
    jax.block_until_ready((loss, grads))

    print("loss:", loss)
    print("W1 grad shape:", grads['W1'].shape)
    print("W2 grad shape:", grads['W2'].shape)
    print("W3 grad shape:", grads['W3'].shape)

if __name__ == '__main__':
    main()
