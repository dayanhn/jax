import os
#os.system("clear")
print(os.getpid())

import jax
import jax.numpy as jnp
from jax import jit, pmap, make_jaxpr


print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")