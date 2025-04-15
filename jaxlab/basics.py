import jax.numpy as jnp


a = jnp.array([1, 2, 3])
b = jnp.array([4, 5, 6])

print(jnp.sqrt(a + b).reshape(3, 1))

x = jnp.sqrt(a + b).reshape(3, 1)
x.at[0, 0].set(0)
print(x)