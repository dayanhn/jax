简单的 JAX MNIST 训练示例

- 位置: 本目录下含 `data.py`, `model.py`, `train.py`。
- 依赖: `jax` 和 `tensorflow-datasets`（用于下载/加载 MNIST 数据）。

运行示例：

```bash
python train.py --epochs 3 --batch_size 128 --lr 0.01
```

训练结束后会将模型参数保存到 `mnist_params.pkl`。
