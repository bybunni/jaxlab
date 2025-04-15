import time
import jax
import jax.numpy as jnp
from jaxmarl.environments.mpe.simple_world_comm import SimpleWorldCommMPE
import time
from jaxlab.benchmark import make_benchmark

MPE_ENV = "MPE_simple_reference_v3"

config = {
    "NUM_STEPS": 1000,
    "NUM_ENVS": 1000,
    "ACTIVATION": "relu",
    "ENV_KWARGS": {},
    "ENV_NAME": MPE_ENV,
    "NUM_SEEDS": 1,
    "SEED": 0,
}

### JAXMARL BENCHMARK
num_envs = [1, 100, 1000, 10000]
jaxmarl_sps = []
for num in num_envs:
    config["NUM_ENVS"] = num
    benchmark_fn = jax.jit(make_benchmark(config))
    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng = jax.random.split(rng)
    benchmark_jit = jax.jit(benchmark_fn).lower(_rng).compile()
    before = time.perf_counter_ns()
    runner_state = jax.block_until_ready(benchmark_jit(_rng))
    after = time.perf_counter_ns()
    total_time = (after - before) / 1e9

    sps = config["NUM_STEPS"] * config["NUM_ENVS"] / total_time
    jaxmarl_sps.append(sps)

    print(f"Num Envs: {num}, Total Time (s): {total_time}")
    print(f"Num Envs: {num}, Total Steps: {config['NUM_STEPS'] * config['NUM_ENVS']}")
    print(f"Num Envs: {num}, SPS: {sps}")
