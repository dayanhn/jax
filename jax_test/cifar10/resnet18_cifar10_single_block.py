import os
os.system('clear')
print(os.getpid())

dump_out = True

if dump_out:
    # Ensure C++/Abseil logging flags are set before any C++ extensions are loaded.
    # Use TF_CPP_* env vars which logging_initializer.cc reads and applies.
    # Increase max vlog so XLA_VLOG_LINES(3, ...) will print
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
    os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "4"


    # Enable verbose logging for HLO-related sources and MLIR conversion passes.
    # Use basenames (file without extension), e.g. hlo_module for hlo_module.cc.
    os.environ["TF_CPP_VMODULE"] = ("hlo_pass_pipeline=4,thunk_emitter=4,compile_module_to_llvm_ir=4")

import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import optax  # 导入 optax 优化器库


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=32)
    p.add_argument('--data_dir', type=str, default='/data3/zhongzhw/code/google/jax/jax_test/cifar10/datasets/')
    return p.parse_args()


def get_cifar10_datasets(batch_size=128, data_dir='./datasets'):
    """Return train and test iterators yielding (images, labels) as NumPy arrays.
    
    Images are normalized to float32 in [0,1] with shape (B, 32, 32, 3).
    """
    import os
    import pickle
    import urllib.request
    import tarfile
    
    print("Loading CIFAR10 dataset...")
    print(f"Data directory: {data_dir}")
    
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)
    
    # CIFAR10数据集URL
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(data_dir, "cifar-10-python.tar.gz")
    
    # 如果文件不存在，下载数据集
    if not os.path.exists(filename):
        print("Downloading CIFAR10 dataset...")
        try:
            urllib.request.urlretrieve(url, filename)
            print("Download completed!")
        except Exception as e:
            print(f"Download failed: {e}")
            raise
    
    # 解压数据集
    extract_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extract_dir):
        print("Extracting dataset...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(data_dir)
        print("Extraction completed!")
    
    # 加载训练数据
    print("Loading training data...")
    train_images = []
    train_labels = []
    
    # 只加载一个批次的数据，加快调试
    batch_file = os.path.join(extract_dir, "data_batch_1")
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    train_images.extend(batch[b'data'])
    train_labels.extend(batch[b'labels'])
    
    train_images = np.array(train_images, dtype=np.float32) / 255.0
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, H, W, C)
    train_labels = np.array(train_labels, dtype=np.int32)
    
    # 加载测试数据
    print("Loading test data...")
    test_file = os.path.join(extract_dir, "test_batch")
    with open(test_file, 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')
    test_images = np.array(test_batch[b'data'], dtype=np.float32) / 255.0
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, H, W, C)
    test_labels = np.array(test_batch[b'labels'], dtype=np.int32)
    
    # 只使用部分测试数据
    test_images = test_images[:1000]
    test_labels = test_labels[:1000]
    
    print(f"Train set: {train_images.shape[0]} samples")
    print(f"Test set: {test_images.shape[0]} samples")
    
    def train_iter():
        num_samples = len(train_images)
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]
            yield train_images[batch_idx], train_labels[batch_idx]
    
    def test_iter():
        num_samples = len(test_images)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            yield test_images[start:end], test_labels[start:end]
    
    return train_iter, test_iter

# ResNet18 模型定义 (简化为只有一个block)
def init_resnet18_params_numpy(seed):
    """使用 NumPy 初始化 ResNet18 参数"""
    rng = np.random.default_rng(seed)
    
    def conv_block(in_channels, out_channels, stride=1):
        """卷积块"""
        # 卷积层 - He 初始化
        w1 = rng.standard_normal((3, 3, in_channels, out_channels)) * np.sqrt(2.0 / (3*3*in_channels))
        b1 = np.zeros((out_channels,))
        # 批量归一化参数
        gamma1 = np.ones((out_channels,))
        beta1 = np.zeros((out_channels,))
        moving_mean1 = np.zeros((out_channels,))
        moving_var1 = np.ones((out_channels,))
        return {
            'conv': {'W': w1, 'b': b1},
            'bn': {'gamma': gamma1, 'beta': beta1, 'moving_mean': moving_mean1, 'moving_var': moving_var1}
        }
    
    def residual_block(in_channels, out_channels, stride=1):
        """残差块"""
        # 主路径
        conv1 = conv_block(in_channels, out_channels, stride)
        conv2 = conv_block(out_channels, out_channels, 1)
        
        #  shortcuts路径
        shortcut = {}
        if stride != 1 or in_channels != out_channels:
            # 需要下采样
            w = rng.standard_normal((1, 1, in_channels, out_channels)) * np.sqrt(2.0 / (1*1*in_channels))
            b = np.zeros((out_channels,))
            shortcut = {'conv': {'W': w, 'b': b}}
        
        return {'conv1': conv1, 'conv2': conv2, 'shortcut': shortcut}
    
    # 初始化各层参数 (简化为只有一个block)
    params = {
        'conv1': conv_block(3, 64, 1),  # CIFAR10使用3x3卷积，步长1
        'maxpool': {},
        'layer1': {
            'block1': residual_block(64, 64, 1)  # 只保留一个block
        },
        'avgpool': {},
        'fc': {
            'W': rng.standard_normal((64, 10)) * np.sqrt(2.0 / 64),  # 输出通道改为64
            'b': np.zeros((10,))
        }
    }
    
    return params


def batch_norm(x, params, is_training):
    """批量归一化
    
    Returns:
        tuple: (normalized_output, updated_params) 
               训练时返回更新后的参数，推理时返回原参数
    """
    gamma = params['gamma']
    beta = params['beta']
    moving_mean = params['moving_mean']
    moving_var = params['moving_var']
    
    if is_training:
        mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(x, axis=(0, 1, 2), keepdims=True)
        # 更新移动平均值
        decay = 0.9
        new_moving_mean = decay * moving_mean + (1 - decay) * mean.reshape(-1)
        new_moving_var = decay * moving_var + (1 - decay) * var.reshape(-1)
        
        # 创建更新后的参数字典
        updated_params = {
            'gamma': gamma,
            'beta': beta,
            'moving_mean': new_moving_mean,
            'moving_var': new_moving_var
        }
    else:
        mean = moving_mean.reshape(1, 1, 1, -1)
        var = moving_var.reshape(1, 1, 1, -1)
        updated_params = params
    
    epsilon = 1e-5
    x_norm = (x - mean) / jnp.sqrt(var + epsilon)
    output = gamma.reshape(1, 1, 1, -1) * x_norm + beta.reshape(1, 1, 1, -1)
    
    return output, updated_params


def conv2d(x, w, b, stride=1, padding='SAME'):
    """卷积操作"""
    return jax.lax.conv_general_dilated(
        x, w, (stride, stride), padding, 
        rhs_dilation=(1, 1), lhs_dilation=(1, 1), 
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    ) + b.reshape(1, 1, 1, -1)


def predict(params, x, is_training=True):
    """模型预测
    
    Returns:
        tuple: (logits, updated_params) 训练时返回更新后的参数
    """
    # 第一层：卷积 + BN + ReLU
    x, conv1_bn_params = batch_norm(
        conv2d(x, params['conv1']['conv']['W'], params['conv1']['conv']['b']),
        params['conv1']['bn'], 
        is_training
    )
    if is_training:
        params['conv1']['bn'] = conv1_bn_params
    x = jax.nn.relu(x)
    
    # 最大池化
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'SAME'
    )
    
    # 残差块
    def residual_block_forward(x, block_params, is_training):
        # 主路径
        shortcut = x
        if 'conv' in block_params['shortcut']:
            shortcut = conv2d(
                shortcut, 
                block_params['shortcut']['conv']['W'], 
                block_params['shortcut']['conv']['b']
            )
        
        x = conv2d(x, block_params['conv1']['conv']['W'], block_params['conv1']['conv']['b'])
        x, bn1_params = batch_norm(x, block_params['conv1']['bn'], is_training)
        if is_training:
            block_params['conv1']['bn'] = bn1_params
        x = jax.nn.relu(x)
        
        x = conv2d(x, block_params['conv2']['conv']['W'], block_params['conv2']['conv']['b'])
        x, bn2_params = batch_norm(x, block_params['conv2']['bn'], is_training)
        if is_training:
            block_params['conv2']['bn'] = bn2_params
        
        x = x + shortcut
        x = jax.nn.relu(x)
        return x, block_params
    
    # 只使用一个block
    x, params['layer1']['block1'] = residual_block_forward(x, params['layer1']['block1'], is_training)
    
    # 全局平均池化
    x = jnp.mean(x, axis=(1, 2))
    
    # 全连接层
    logits = jnp.dot(x, params['fc']['W']) + params['fc']['b']
    return logits, params


def loss_fn(params, x, y):
    """损失函数
    
    Returns:
        tuple: (loss, updated_params) 包含更新后的 BatchNorm 统计量
    """
    logits, updated_params = predict(params, x, is_training=True)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
    return loss, updated_params


def accuracy(params, x, y):
    """计算准确率"""
    logits, _ = predict(params, x, is_training=False)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


@jax.jit
def compute_grads(params, x, y):
    """JIT 编译的梯度计算部分
    
    Args:
        params: 模型参数
        x, y: 输入数据和标签
    
    Returns:
        grads: 梯度
        updated_params: 包含更新后 BN 统计量的参数
        loss: 当前批次的损失值
    """
    # 计算损失和梯度（has_aux=True 获取更新后的 BatchNorm 参数）
    (loss, updated_params), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y)
    return grads, updated_params, loss


def apply_updates(params, grads, updated_params, opt_state, tx):
    """应用梯度更新（在 JIT 外部执行）
    
    Args:
        params: 当前参数
        grads: 梯度
        updated_params: 包含更新后 BN 统计量的参数
        opt_state: 优化器状态
        tx: 优化器变换
    
    Returns:
        new_params: 更新后的参数
        new_opt_state: 新的优化器状态
    """
    # 使用 optax 计算参数更新（自动处理 Adam 的动量和偏差校正）
    updates, new_opt_state = tx.update(grads, opt_state, params)
    
    # 应用更新到可训练参数（跳过 BatchNorm 的移动统计量）
    def apply_update(param, update, updated_param):
        if isinstance(param, dict) and 'moving_mean' in param:
            return updated_param  # 使用更新后的 BatchNorm 统计量
        return optax.apply_updates(param, update)
    
    new_params = jax.tree_util.tree_map(apply_update, params, updates, updated_params)
    
    return new_params, new_opt_state


def main():
    args = parse_args()
    train_iter_fn, test_iter_fn = get_cifar10_datasets(
        batch_size=args.batch_size, 
        data_dir=args.data_dir
    )
    
    # 初始化 ResNet18 参数（简化版）
    params_numpy = init_resnet18_params_numpy(args.seed)
    
    # 将参数搬入设备内存
    device = jax.devices()[0]
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), params_numpy)
    
    # 使用 optax 创建 Adam 优化器
    tx = optax.adam(learning_rate=args.lr)
    opt_state = tx.init(params)  # 初始化优化器状态

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # Training
        train_iter = train_iter_fn()
        for step, (xb, yb) in enumerate(train_iter):
            xb = jnp.array(xb)
            yb = jnp.array(yb).astype(jnp.int32)  # 确保标签是 int32 类型
            
            # JIT 编译的梯度计算
            grads, updated_params, loss = compute_grads(params, xb, yb)
            
            # 应用参数更新（包含优化器状态更新）
            params, opt_state = apply_updates(params, grads, updated_params, opt_state, tx)
            
            if step % 10 == 0:  # 增加打印频率
                acc = float(accuracy(params, xb, yb))
                print(f'Epoch {epoch}, Step {step}, Batch Acc: {acc:.4f}, Loss: {float(loss):.4f}')

        # Evaluation
        test_iter = test_iter_fn()
        accs = []
        for xb, yb in test_iter:
            xb = jnp.array(xb)
            yb = jnp.array(yb).astype(jnp.int32)  # 确保标签是 int32 类型
            accs.append(float(accuracy(params, xb, yb)))
        test_acc = float(np.mean(accs)) if accs else 0.0
        print(f'Epoch {epoch}  test_acc={test_acc:.4f}  epoch_time={time.time()-t0:.1f}s')

    # Save params
    print('Saving model parameters...')
    # 转换参数为可序列化格式
    def serialize_params(params):
        if isinstance(params, dict):
            return {k: serialize_params(v) for k, v in params.items()}
        elif isinstance(params, jnp.ndarray):
            return np.array(params)
        else:
            return params
    
    serializable = serialize_params(params)
    with open('resnet18_cifar10_single_block_params.pkl', 'wb') as f:
        pickle.dump(serializable, f)
    print('Saved parameters to resnet18_cifar10_single_block_params.pkl')


if __name__ == '__main__':
    main()