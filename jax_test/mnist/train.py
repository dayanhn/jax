
import os
os.system('clear')
print(os.getpid())

dump_out = True

if dump_out:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
    os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "4"
    # os.environ["TF_CPP_VMODULE"] = ("hlo_pass_pipeline=4,thunk_emitter=4,compile_module_to_llvm_ir=4")

os.environ['XLA_FLAGS'] = (
    '--xla_dump_to=./tmp/xla_dump '
    '--xla_gpu_force_compilation_parallelism=1 '
    '--xla_dump_hlo_as_text=true '
    '--xla_dump_hlo_as_proto=false '
    '--xla_dump_hlo_pass_re=.* '
    '--xla_dump_hlo_module_re=.*  '
    # 自动调优日志配置
    '--xla_gpu_dump_autotune_logs_to=./tmp/autotune_logs.txt '
    #'--xla_disable_hlo_passes=aclnn-gemm-rewriter'  # 禁用 ACLNN GEMM rewriter
    )

import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import pickle

from data import get_datasets
from model import init_mlp_params_numpy, update, accuracy, predict, loss_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=32)
    p.add_argument('--manual_download', action='store_true', default=True)
    return p.parse_args()


def main():
    args = parse_args()
    train_iter_fn, test_iter_fn = get_datasets(batch_size=args.batch_size, 
                                               use_manual_download=args.manual_download)
    # key = jax.random.PRNGKey(args.seed)
    # params = init_mlp_params(layer_sizes, key)

    # 使用 NumPy 初始化参数
    layer_sizes = [784, 512, 256, 10]
    params_numpy = init_mlp_params_numpy(layer_sizes, args.seed)
    
    # 将参数搬入设备内存
    device = jax.devices()[0]
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params_numpy)

    # 调试代码:使用全1输入测试每个函数的输出

    batch_size = 2
    input_dim = layer_sizes[0]  # 784
    num_classes = layer_sizes[-1]  # 10
    
    x_ones = jax.device_put(jnp.ones((batch_size, input_dim), dtype=jnp.float32), device)
    y_ones = jax.device_put(jnp.ones(batch_size, dtype=jnp.int32), device)
    
    
    # 1. 测试 predict 函数
    #logits = predict(params, x_ones)
    #jax.block_until_ready(logits)
    
    # 2. 测试 loss_fn 函数
    #loss = loss_fn(params, x_ones, y_ones)
    #jax.block_until_ready(loss)
    
    # 3. 测试 accuracy 函数
    #acc = accuracy(params, x_ones, y_ones)
    #jax.block_until_ready(acc)

    # 4. 测试 update 函数
    #lr = 0.01
    #new_params = update(params, x_ones, y_ones, lr)
    #jax.block_until_ready(new_params)
    

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # Training
        train_iter = train_iter_fn()
        for xb, yb in train_iter:
            xb = jnp.array(xb)
            yb = jnp.array(yb)
            params = update(params, xb, yb, args.lr)

        # Evaluation
        test_iter = test_iter_fn()
        accs = []
        for xb, yb in test_iter:
            xb = jnp.array(xb)
            yb = jnp.array(yb)
            accs.append(float(accuracy(params, xb, yb)))
        test_acc = float(np.mean(accs)) if accs else 0.0
        print(f'Epoch {epoch}  test_acc={test_acc:.4f}  epoch_time={time.time()-t0:.1f}s')

    # Save params
    serializable = [ {'W': np.array(p['W']), 'b': np.array(p['b'])} for p in params ]
    with open('mnist_params.pkl', 'wb') as f:
        pickle.dump(serializable, f)
    print('Saved parameters to mnist_params.pkl')


if __name__ == '__main__':
    main()
