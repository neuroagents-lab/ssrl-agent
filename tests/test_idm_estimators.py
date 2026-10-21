"""Numerical and gradient equivalence to the original dense estimators."""

import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import ninjax as nj
from ssrl.idms.competency.dads import DADS
from ssrl.idms.curiosity.max import MAX


def dense_jrd(model, mean, log_var):
    var = model._rescaled_var(log_var)
    omega = var[:, None] + var[None]
    delta = jnp.square(mean[:, None] - mean[None])
    log_d = -0.5 * jnp.log(omega).sum(-1) - 0.5 * (delta / omega).sum(-1)
    value = (2 * math.log(model._models) - jax.nn.logsumexp(log_d, (0, 1))
             - 0.5 * jnp.log(var).sum(-1).mean(0)
             - 0.5 * mean.shape[-1] * math.log(2))
    return jnp.maximum(value, 0)[..., None]


def dense_max_loss(model, obs, action, target, mask, bootstrap):
    x = model._inputs(obs, action)
    if bootstrap:
        indices = jax.random.randint(nj.seed(), (model._models, x.shape[0]), 0, x.shape[0])
        x, target, mask = x[indices], target[indices], mask[indices]
    else:
        x, target, mask = [jnp.broadcast_to(v[None], (model._models, *v.shape))
                           for v in (x, target, mask)]
    mean, lv = model._split(model._forward_models(x))
    nll = 0.5 * (jnp.square(mean - target) * jnp.exp(-lv) + lv).sum(-1)
    axes = tuple(range(1, mask.ndim))
    losses = (nll * mask).sum(axes) / jnp.maximum(mask.sum(axes), 1)
    return losses.mean(), (nll.mean(), lv.mean(), losses)


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("shape", [(3, 5, 4), (7, 4)])
@pytest.mark.parametrize("chunk_size", [0, 4])
def test_max_loss_values_and_gradients(bootstrap, shape, chunk_size):
    model = MAX(input_dim=4, action_dim=2, target_dim=4, hidden_dim=7,
                   n_models=3, mlp_layers=2, chunk_size=chunk_size, name="max")
    obs = jax.random.normal(jax.random.PRNGKey(1), shape)
    action = jnp.ones((*shape[:-1], 2))
    target = obs * 0.7 + 0.2
    mask = (jnp.arange(math.prod(shape[:-1])).reshape(shape[:-1]) % 3 != 0).astype(jnp.float32)

    def dense(x):
        return dense_max_loss(model, x, action, target, mask, bootstrap)

    def bounded(x):
        loss, mets = model.loss(x, action, target, mask, bootstrap)
        return loss, (mets["nll"], mets["mean_log_var"],
                      jnp.stack([mets[f"model{i}_loss"] for i in range(3)]))

    state = nj.init(dense)({}, obs, seed=0)
    compare_functions(dense, bounded, state, obs)


def compare_functions(dense, bounded, state, obs):
    reference, changed = nj.pure(dense), nj.pure(bounded)
    _, expected = reference(state, obs, seed=42)
    _, actual = jax.jit(changed)(state, obs, seed=42)
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-5)
    gradients = []
    for fn in (reference, changed):
        gradients.append(jax.jit(jax.grad(
            lambda params, x: fn(params, x, seed=42)[1][0], argnums=(0, 1)
        ))(state, obs))
    for a, b in zip(jax.tree.leaves(gradients[0]), jax.tree.leaves(gradients[1])):
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=1e-4)


@pytest.mark.parametrize("models", [2, 3, 5])
@pytest.mark.parametrize("chunk_size", [0, 4])
def test_max_reward_values_and_gradients(models, chunk_size):
    model = MAX(input_dim=4, action_dim=2, hidden_dim=7, n_models=models,
                   mlp_layers=2, chunk_size=chunk_size, name="max_reward")
    obs = jax.random.normal(jax.random.PRNGKey(3), (3, 5, 4))
    action = jnp.ones((3, 5, 2))

    def dense(x):
        reward = dense_jrd(model, *model.predict(x, action))
        return reward.sum(), reward

    def bounded(x):
        reward = model.compute_reward(x, action)
        return reward.sum(), reward

    state = nj.init(dense)({}, obs, seed=0)
    compare_functions(dense, bounded, state, obs)


@pytest.mark.parametrize("categorical", [False, True])
def test_dads_marginal_values_and_gradients(categorical):
    model = DADS(obs_dim=4, num_skills=3 if categorical else 2,
                    skill_dim=0 if categorical else 2, hidden_dim=7, n_layers=2,
                    prior_name="uniform_categorical" if categorical else "uniform_box",
                    marginal_samples=7, chunk_size=4, name="dads")
    obs = jax.random.normal(jax.random.PRNGKey(2), (3, 5, 4))
    following = obs * 0.6 + 0.1

    def dense(x):
        current, target = x.reshape((-1, 4)), following.reshape((-1, 4))
        count = model._num_skills if categorical else model._marginal_samples
        expanded = jnp.broadcast_to(current[:, None], (15, count, 4))
        if categorical:
            skills = jnp.broadcast_to(jnp.eye(count)[None], (15, count, count))
        else:
            skills = model._skill_vectors(model.prior.sample((15, count)), expanded)
        context = jnp.concatenate([expanded, skills], -1)
        log_p = model.dynamics.log_prob(jnp.broadcast_to(target[:, None], expanded.shape), context)[..., 0]
        result = jax.nn.logsumexp(log_p, -1) - math.log(count)
        return result.sum(), result.reshape((3, 5, 1))

    def bounded(x):
        result = model._log_marginal(x, following)
        return result.sum(), result

    state = nj.init(dense)({}, obs, seed=0)
    compare_functions(dense, bounded, state, obs)


def test_chunked_initialization_and_zero_masks():
    # Exercise parameter creation inside scan, as full learner initialization does.
    model = MAX(input_dim=3, n_models=2, hidden_dim=5, mlp_layers=1,
                   chunk_size=2, name="init_max")
    obs = jnp.zeros((3, 5, 3))
    fn = lambda x: model.loss(x, None, x, jnp.zeros(x.shape[:-1]), bootstrap=True)
    state = nj.init(fn)({}, obs, seed=0)
    _, (loss, _) = jax.jit(nj.pure(fn))(state, obs, seed=1)
    assert loss == 0
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(state))
    dads = DADS(obs_dim=3, num_skills=2, skill_dim=2, prior_name="uniform_box",
                   hidden_dim=5, n_layers=1, marginal_samples=7, chunk_size=2,
                   name="init_dads")
    fn = lambda x: dads._log_marginal(x, x)
    state = nj.init(fn)({}, obs, seed=0)
    _, out = jax.jit(nj.pure(fn))(state, obs, seed=1)
    assert np.isfinite(out).all()
