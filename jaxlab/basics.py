import time
import jax
import jax.numpy as jnp


a = jnp.array([1, 2, 3])
b = jnp.array([4, 5, 6])

print(jnp.sqrt(a + b).reshape(3, 1))

x = jnp.sqrt(a + b).reshape(3, 1)
x.at[0, 0].set(0)
print(x)

# we need to use block_until_ready() when we want to force coordination
# between the cpu and the accelerator, because the jax library
# is lazy and will not execute the code until it is needed,
# so if we don't block we will only count time time to enqueue the
# 1,000 jit calls, not the time to execute it

@jax.jit
def f(x):
    x = jnp.sqrt(a + b).reshape(3, 1)
    return x

start = time.perf_counter()
for _ in range(1000):
    f(x).block_until_ready()
end = time.perf_counter()
print(f"Execution time: {end - start} seconds")

print(jax.make_jaxpr(f)(x))