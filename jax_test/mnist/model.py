import jax
import jax.numpy as jnp


def init_mlp_params(layer_sizes, key):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for k, n_in, n_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        w = jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(2.0 / max(1, n_in))
        b = jnp.zeros((n_out,))
        params.append({'W': w, 'b': b})
    return params


def predict(params, x):
    h = x
    for layer in params[:-1]:
        h = jnp.dot(h, layer['W']) + layer['b']
        h = jax.nn.relu(h)
    logits = jnp.dot(h, params[-1]['W']) + params[-1]['b']
    return logits


def loss_fn(params, x, y):
    logits = predict(params, x)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
    return loss


def accuracy(params, x, y):
    logits = predict(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


@jax.jit
def update(params, x, y, lr):
    grads = jax.grad(loss_fn)(params, x, y)
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params
