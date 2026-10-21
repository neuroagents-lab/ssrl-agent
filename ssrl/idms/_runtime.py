"""JAX helpers adapted from DreamerV3's embodied.jax runtime."""

import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32


def sg(xs, skip=False):
    return xs if skip else jax.lax.stop_gradient(xs)


def act(name):
    if name == "none":
        return lambda x: x
    if name == "mish":
        return lambda x: x * jnp.tanh(jax.nn.softplus(x))
    if name == "relu2":
        return lambda x: jnp.square(jax.nn.relu(x))
    if name == "swiglu":
        def swiglu(x):
            x, y = jnp.split(x, 2, -1)
            return jax.nn.silu(x) * y
        return swiglu
    return getattr(jax.nn, name)


def get_data_axes():
    axes = ("d", "f")
    for axis in axes:
        try:
            jax.lax.axis_index(axis)
        except NameError:
            return ()
    return axes


class SlowModel:
    """Track a Ninjax module with an exponential moving average."""

    def __init__(self, model, *, source, rate=1.0, every=1):
        assert rate == 1 or rate < 0.5, rate
        self.source = source
        self.model = model
        self.rate = rate
        self.every = every
        name = self.model.name + "_count"
        self.count = nj.Variable(jnp.zeros, (), jnp.int32, name=name)

    def __getattr__(self, name):
        self._initonce()
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        self._initonce()
        return self.model(*args, **kwargs)

    def update(self):
        self._initonce()
        mix = jnp.where(self.count.read() % self.every == 0, self.rate, 0)
        values = jax.tree.map(
            lambda src, dst: mix * src + (1 - mix) * dst,
            self.source.values,
            self.model.values,
        )
        for key, value in values.items():
            self.model.write(key, value)
        self.count.write(self.count.read() + 1)

    def _initonce(self, *args, method=None, **kwargs):
        assert self.source.values, "no parameters to track"
        if not self.model.values:
            prefix = self.model.path + "/"
            nj.context().update({prefix + key: value for key, value in self.source.values.items()})
        assert self.model.values.keys() == self.source.values.keys(), (
            self.model.values.keys(), self.source.values.keys()
        )
