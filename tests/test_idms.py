"""Numerical contracts for intrinsic rewards and skill distributions."""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from agent.p2e.agent import _route_idm_repr_gradient
from ssrl.idms.competency import DADS, DIAYN, RDADS, VALOR, VIC, get_skill_prior
from ssrl.idms.curiosity.cig import conditional_information_gain
from ssrl.idms.curiosity.disagreement import Disagreement
from ssrl.idms.curiosity.icm import ICM
from ssrl.idms.curiosity.rnd import RND
from ssrl.idms.curiosity.simple_curiosity import SimpleCuriosity
from ssrl.idms.entropy import APT
from ssrl.idms.entropy.common import masked_knn_mean_lp_power, masked_kth_l2_distance


def evaluate(function, *args):
    state = nj.init(function)({}, *args, seed=0)
    return nj.pure(function)(state, *args, seed=1)[1]


@pytest.mark.parametrize("mode,expected", [("detached", 0), ("head_only", 0), ("full", 1)])
def test_intrinsic_gradient_routing(mode, expected):
    gradient = jax.grad(lambda x: _route_idm_repr_gradient(x, mode).sum())(jnp.ones(3))
    np.testing.assert_array_equal(gradient, jnp.full(3, expected))


@pytest.mark.parametrize("action_dim", (0, 2))
def test_disagreement_is_scaled_ensemble_variance(action_dim):
    module = Disagreement(
        input_dim=3, action_dim=action_dim, hidden_dim=8, n_models=3,
        mlp_layers=1, scale=2.0, name="disagreement",
    )
    observations = jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3) / 18
    actions = jnp.ones((2, 3, action_dim)) if action_dim else None

    def run():
        predictions = module.predict(observations, actions)
        reward = module.compute_reward(observations, actions)
        loss, _ = module.loss(observations, actions, observations, jnp.zeros((2, 3)), False)
        return predictions, reward, loss

    predictions, reward, loss = evaluate(run)
    assert reward.shape == (2, 3, 1)
    np.testing.assert_allclose(reward, 2 * np.var(predictions, axis=0).mean(-1, keepdims=True))
    assert loss == 0


def test_simple_curiosity_reward_and_prediction_losses():
    module = SimpleCuriosity(
        obs_dim=3, action_dim=2, hidden_dim=6, n_layers=1,
        scale=1.7, name="simple_curiosity",
    )
    current = jnp.array([[1.0, -2.0, 0.5], [0.2, 0.7, -1.0]])
    following = jnp.array([[0.4, -0.1, 1.2], [-0.3, 0.8, 0.6]])
    action = jnp.array([[0.25, -0.5], [-0.1, 0.75]])

    def run():
        prediction = module.predict_next(current, action)
        predicted_action = module.predict_action(current, following)
        expected_reward = jnp.linalg.norm(prediction - following, axis=-1, keepdims=True) * 1.7
        expected_loss = (
            jnp.square(prediction - following).mean()
            + jnp.square(predicted_action - action).mean()
        )
        loss, _ = module.loss(current, action, following)
        return module.compute_reward(current, action, following), expected_reward, loss, expected_loss

    reward, expected_reward, loss, expected_loss = evaluate(run)
    np.testing.assert_allclose(reward, expected_reward, rtol=1e-6)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)


def test_icm_reward_and_weighted_prediction_objective():
    module = ICM(
        obs_dim=3, action_dim=2, proj_dim=4, proj_hidden_dim=5, proj_layers=1,
        hidden_dim=6, n_layers=1, forward_loss_weight=0.3, scale=1.7, name="icm",
    )
    current = jnp.array([[1.0, -2.0, 0.5], [0.2, 0.7, -1.0]])
    following = jnp.array([[0.4, -0.1, 1.2], [-0.3, 0.8, 0.6]])
    action = jnp.array([[0.25, -0.5], [-0.1, 0.75]])

    def run():
        features, target = module.project_pair(current, following)
        prediction = module.predict_next_features(features, action)
        expected = 0.5 * jnp.square(prediction - target).sum(-1, keepdims=True) * 1.7
        forward, inverse, _ = module.loss_terms(current, action, following)
        loss, _ = module.loss(current, action, following)
        return module.compute_reward(current, action, following), expected, loss, forward, inverse

    reward, expected, loss, forward, inverse = evaluate(run)
    np.testing.assert_allclose(reward, expected, rtol=1e-6)
    np.testing.assert_allclose(loss, 0.3 * forward + 0.7 * inverse, rtol=1e-6)


def test_reward_evaluation_does_not_update_rnd_statistics():
    module = RND(obs_dim=3, hidden_dim=8, output_dim=4, mlp_layers=1, name="rnd")
    observations = jnp.ones((2, 3, 3))

    def run():
        module.compute_reward(observations, update_reward_norm=True)
        before = module.reward_rms.count.read()
        reward = module.compute_reward(observations * 2, update_reward_norm=False)
        return before, module.reward_rms.count.read(), reward

    before, after, reward = evaluate(run)
    assert before == after == 6
    assert reward.shape == (2, 3, 1)
    assert np.isfinite(reward).all()


def test_conditional_information_gain_has_diminishing_returns():
    predictions = jnp.broadcast_to(jnp.array([-1.0, 1.0])[:, None, None, None], (2, 1, 3, 1))
    reward = conditional_information_gain(predictions, aleatoric_var=1.0)[0, :, 0]
    np.testing.assert_allclose(reward, np.log([2.0, 1.5, 4 / 3]), rtol=1e-5)
    assert reward[0] > reward[1] > reward[2] > 0
    gradient = jax.grad(lambda x: conditional_information_gain(x, 1.0).sum())(predictions)
    np.testing.assert_array_equal(gradient, 0)


def test_entropy_memory_is_unchanged_during_reward_evaluation():
    module = APT(
        obs_dim=1, rep_dim=1, hidden_dim=4, n_layers=1, k=1,
        normalize_phi=False, reward_normalize=False, memory_size=8,
        memory_sample_size=8, name="apt",
    )

    def run():
        module.update_memory(jnp.array([[1.0]]))
        before = module._mem.read()
        module.update(next_obs=jnp.array([[3.0]]), update_memory=False, train_phi=False)
        return before, module._mem.read(), module.memory_count()

    before, after, count = evaluate(run)
    np.testing.assert_array_equal(before, after)
    assert count == 1


@pytest.mark.parametrize("estimator", (masked_knn_mean_lp_power, masked_kth_l2_distance))
def test_duplicate_particles_have_finite_entropy_gradients(estimator):
    particles = jnp.zeros((4, 3))
    options = {"power": 1.0} if estimator is masked_knn_mean_lp_power else {}
    objective = lambda value: estimator(value, particles, jnp.ones(4, bool), k=2, **options).sum()
    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.zeros((1, 3)))
    assert value == 0
    np.testing.assert_array_equal(gradient, 0)


@pytest.mark.parametrize("prior_name", ("uniform_categorical", "uniform_hypersphere", "uniform_box"))
def test_skill_priors_sample_their_support(prior_name):
    prior = get_skill_prior(prior_name, num_skills=3, skill_dim=3)
    def draw():
        samples = prior.sample((32,))
        return samples, prior.log_prob(samples)

    samples, log_probability = evaluate(draw)
    assert np.isfinite(log_probability).all()
    assert log_probability.shape == (32, 1)
    if prior_name == "uniform_categorical":
        assert samples.shape == (32,)
        assert np.all((samples >= 0) & (samples < 3))
        np.testing.assert_allclose(log_probability, -np.log(3), rtol=1e-6)
    elif prior_name == "uniform_hypersphere":
        np.testing.assert_allclose(np.linalg.norm(samples, axis=-1), 1, atol=1e-6)
    else:
        assert np.all((samples >= -1) & (samples <= 1))
        assert np.isneginf(prior.log_prob(jnp.full((1, 3), 2.0))).all()


@pytest.mark.parametrize("kind", ("diayn", "vic", "dads", "rdads", "valor"))
def test_skill_discovery_reward_and_loss(kind):
    classes = {"diayn": DIAYN, "vic": VIC, "dads": DADS, "rdads": RDADS, "valor": VALOR}
    kwargs = {"num_subseq": 3} if kind == "valor" else {}
    module = classes[kind](obs_dim=4, num_skills=3, hidden_dim=8, n_layers=1, name=kind, **kwargs)
    observations = jnp.ones((2, 3, 4))
    following = observations + 0.1
    skills = jnp.broadcast_to(jnp.eye(3)[None], (2, 3, 3))

    def run():
        if kind == "diayn":
            loss, _ = module.loss(observations, skills)
            reward = module.compute_reward(observations, skill=skills)
        elif kind == "vic":
            loss, _ = module.loss(observations[:, 0], following[:, -1], skills[:, 0])
            reward = module.compute_reward(following, skill=skills[:, 0])
        elif kind == "valor":
            loss, _ = module.loss(observations, skills[:, 0])
            reward = module.compute_reward(observations, skill=skills[:, 0])
        else:
            loss, _ = module.loss(observations, following, skills)
            reward = module.compute_reward(observations, next_obs=following, skill=skills)
        return loss, reward

    loss, reward = evaluate(run)
    assert np.isfinite(loss)
    assert reward.shape == (2, 3, 1)
    assert np.isfinite(reward).all()
    if kind in {"vic", "valor"}:
        np.testing.assert_array_equal(reward[:, :-1], 0)


def test_random_control_actions_are_seeded_and_bounded():
    import elements
    from agent.p2e.random_control import sample_random_actions

    spaces = {
        "continuous": elements.Space(np.float32, (2,), -1, 1),
        "discrete": elements.Space(np.int32, (), 0, 3),
    }
    seed = jax.random.PRNGKey(7)
    first = sample_random_actions(spaces, 32, seed)
    second = sample_random_actions(spaces, 32, seed)
    for name, space in spaces.items():
        action = np.asarray(first[name])
        np.testing.assert_array_equal(action, second[name])
        assert action.shape == (32, *space.shape)
        assert np.all(action >= space.low)
        if space.discrete:
            assert np.all(action < space.high)
        else:
            assert np.all(action <= space.high)
