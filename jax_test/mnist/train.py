import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
print(os.getpid())

from data import get_datasets
from model import init_mlp_params, update, accuracy


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

    key = jax.random.PRNGKey(args.seed)
    layer_sizes = [784, 512, 256, 10]
    params = init_mlp_params(layer_sizes, key)

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
