"""Small end-to-end learner updates; run with ``pytest -m training``."""

from types import SimpleNamespace

import elements
import jax
import ninjax as nj
import numpy as np
from omegaconf import OmegaConf
import pytest

from agent import launcher
from agent.dreamer.agent import DreamerAgent
from agent.p2e.agent import Plan2ExploreAgent
from agent.p2e.random_control import RandomControlAgent


pytestmark = pytest.mark.training
CASES = [(backend, None) for backend in ("dreamerv3", "ne_dreamer", "r2dreamer")]
CASES += [("dreamerv3", idm) for idm in (
    "disagreement", "simple_curiosity", "icm", "max", "apt", "diayn", "dads", "csf", "random",
)]


@pytest.fixture(autouse=True)
def release_compiled_executables():
    yield
    jax.clear_caches()


def tiny_model(config):
    values = OmegaConf.to_container(config.model, resolve=True)
    # Preserve each preset's objective and skill distribution at small dimensions.
    for key in values:
        if key.endswith(("hidden_dim", "rep_dim", "pred_dim", "output_dim", "latent_dim")):
            values[key] = 8
        elif key.endswith(("memory_size", "memory_sample_size", "knn_chunk_size")):
            values[key] = 8
        elif key.endswith("replay_batch_size"):
            values[key] = 4
        elif key.endswith(("_layers", "_n_layers")) and isinstance(values[key], int):
            values[key] = 1
    values.update(
        units=16, depth=4, deter=8, hidden=16, discrete=4, imag_horizon=3,
        horizon=10, compile=False, warmup=0, log_grads=False,
        imag_start_ratio=4.0, recon_start_ratio=4.0,
        plan_repr_source="stoch", expl_repr_mode="raw_plan", expl_head_type="mlp",
        disag_models=3, cig_models=3, max_models=3, apt_k=2, re3_k=2,
        proto_rl_num_prototypes=4, proto_rl_candidates_per_prototype=2,
        proto_rl_knn_k=2, csf_num_negatives=4, empowerment_particles=2,
        ne_dreamer={"hidden_dim": 16, "num_layers": 1, "num_heads": 2},
        loss_scales={},
    )
    if values.get("use_skills"):
        values["skill_horizon"] = 3 if values["idm_type"] in ("vic", "valor") else 0
        if values["skill_space"] == "categorical":
            values.update(num_skills=3, skill_dim=0)
        else:
            values["skill_dim"] = 3
    return SimpleNamespace(**values)


@pytest.mark.parametrize("backend,idm", CASES, ids=[f"{backend}-{idm or 'task'}" for backend, idm in CASES])
def test_learner_update_and_policy(compose_config, tmp_path, backend, idm):
    overrides = [f"agent_name={backend}", "envs=dmc_proprio"]
    if idm:
        overrides += ["algo=p2e", f"idm={idm}"]
    configured = tiny_model(compose_config(*overrides))
    translated = launcher._agent_translation(
        configured, agent_name=backend, train_ratio=8, batch_length=4,
    )
    config = elements.Config(
        **translated, logdir=str(tmp_path), seed=0, jax={"jit": False, "precompile": False, "compute_dtype": "float32"},
        batch_size=2, batch_length=4, replay_context=0, report_length=4, replica=0, replicas=1,
    )
    observations = {
        "vector": elements.Space(np.float32, (3,)),
        "reward": elements.Space(np.float32, ()),
        **{key: elements.Space(bool, ()) for key in ("is_first", "is_last", "is_terminal")},
    }
    discrete = backend == "r2dreamer"
    action_space = (elements.Space(np.int32, (), 0, 3) if discrete
                    else elements.Space(np.float32, (2,), -1, 1))
    agent_class = RandomControlAgent if idm == "random" else Plan2ExploreAgent if idm else DreamerAgent
    agent = agent_class(observations, {"action": action_space}, config)
    model = agent.model
    data = {key: np.zeros((2, 4, *space.shape), space.dtype) for key, space in observations.items()}
    data["vector"][:] = np.random.default_rng(0).normal(size=(2, 4, 3))
    data["reward"][:] = 0.1
    data["is_first"][:, 0] = True
    data["action"] = np.zeros((2, 4, *action_space.shape), action_space.dtype)
    data["stepid"] = np.zeros((2, 4, 20), np.uint8)
    if configured.use_skills:
        # Unit basis vectors are valid categorical, sphere, and box skill samples.
        skills = np.eye(3, dtype=np.float32)[:2]
        data["skill"] = np.broadcast_to(skills[:, None], (2, 4, 3)).copy()
        data["skill_start"] = np.broadcast_to(np.array([True, False, False, True]), (2, 4)).copy()
    carry = model.init_train(2)
    initial = agent.params
    before = {key: np.array(value) for key, value in initial.items() if key.startswith("dyn/")}
    updated, (_, _, metrics) = nj.pure(model.train)(initial, carry, data, seed=1)
    losses = {key: value for key, value in metrics.items() if key.startswith("loss/")}
    assert losses
    assert all(np.isfinite(value).all() for value in losses.values())
    assert before
    assert any(not np.array_equal(value, updated[key]) for key, value in before.items())
    if idm and idm != "random":
        loss_name = "disag" if idm == "disagreement" else idm
        assert f"loss/{loss_name}" in losses
        assert np.isfinite(metrics["expl/intr_reward"])
    obs = {key: value[:, 0] for key, value in data.items() if key in observations}
    for mode in ("train", "eval", "eval_greedy"):
        _, (_, actions, _) = nj.pure(model.policy)(updated, model.init_policy(2), obs, mode, seed=2)
        action = np.asarray(actions["action"])
        assert action.shape == (2, *action_space.shape)
        assert np.isfinite(action).all()
        if discrete:
            assert np.all(action >= action_space.low)
            assert np.all(action < action_space.high)
        elif mode == "eval_greedy":
            # Sampled continuous actions are clipped by the environment wrapper.
            assert np.all(action >= action_space.low)
            assert np.all(action <= action_space.high)
