import os

os.system('clear')
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
    W1 = rng.standard_normal((10, 16)).astype(np.float32)
    W2 = rng.standard_normal((4, 4)).astype(np.float32)  
    return {'W1': W1, 'W2': W2}

def forward(params, x):
    h = jnp.dot(x, params['W1'])
    h = h.reshape(h.shape[0], 4, 4, 1)
    h = jax.lax.reduce_window(
        h, -jnp.inf, jax.lax.max,
        window_dimensions=(1,2,2,1),
        window_strides=(1,2,2,1),
        padding='SAME'
    )
    h = h.reshape(h.shape[0], -1)
    out = jnp.dot(h, params['W2'])
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

def main():
    params_np = create_model() 
    device = jax.devices()[0]
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params_np)

    batch_size = 2
    x = jax.device_put(np.random.randn(batch_size, 10).astype(np.float32), device)
    y = jax.device_put(jnp.array([0, 1], dtype=jnp.int32), device)

    loss, grads = forward_and_backward(params, x, y)
    jax.block_until_ready((loss, grads))

    print("loss:", loss)
    print("W1 grad:", grads['W1'].shape)
    print("W2 grad:", grads['W2'].shape)

if __name__ == '__main__':
    main()
