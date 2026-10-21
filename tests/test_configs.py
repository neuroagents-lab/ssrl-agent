"""Paper presets must compose into valid runtime configurations."""

import pytest

from agent import launcher
from ssrl.idms import SUPPORTED_IDMS, validate_supported_idm


BACKENDS = ("dreamerv3", "ne_dreamer", "r2dreamer")
ENVIRONMENTS = (
    "dmc_proprio", "craftax_classic", "dmc_walker_multitask",
    "dmc_walker_multitask_vision", "dmc_walker_adaptation",
    "dmc_walker_adaptation_vision",
)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_paper_presets_build_runtime_config(compose_config, backend, environment):
    config = compose_config(f"agent_name={backend}", f"envs={environment}")
    reference = launcher.build_reference_config(config, launcher._load_reference_deps())
    assert reference.agent.rep_loss == ("dreamer" if backend == "dreamerv3" else backend)
    assert reference.run.steps == int(config.envs.steps)
    assert reference.batch_length == config.batch_length
    assert reference.agent.imag_last > 0


@pytest.mark.parametrize("idm", sorted(SUPPORTED_IDMS))
@pytest.mark.parametrize("environment", ("dmc_proprio", "craftax_classic"))
def test_idm_presets_validate_and_translate(compose_config, idm, environment):
    config = compose_config(f"envs={environment}", "algo=p2e", f"idm={idm}")
    validate_supported_idm(config.model)
    translated = launcher._agent_translation(
        config.model, agent_name="dreamerv3", env_config=config.envs,
        train_ratio=config.envs.train_ratio, batch_length=config.batch_length,
    )
    assert translated["p2e"]["idm_type"] == idm
    assert translated["p2e"]["expl_extr_scale"] == 0


@pytest.mark.parametrize("vision", (False, True))
@pytest.mark.parametrize("algorithm", ("dreamer", "p2e"))
def test_multitask_walker_has_separate_reward_heads(compose_config, vision, algorithm):
    profile = "dmc_walker_multitask" + ("_vision" if vision else "")
    config = compose_config(f"envs={profile}", f"algo={algorithm}")
    reference = launcher.build_reference_config(config, launcher._load_reference_deps())
    tasks = ("stand", "walk", "run") if algorithm == "p2e" else ("walk", "run")
    assert reference.agent.eval_task_heads == tuple(
        f"walker_{task}=dmc_walker_{task}" for task in tasks
    )
    assert reference.agent.train_base_task_behavior is (algorithm == "dreamer")
    assert reference.agent.walker_task_reward_labels is True


@pytest.mark.parametrize("overrides", (
    ["model.expl_extr_scale=1"],
    ["model.expl_until=100"],
    ["idm=diayn", "model.use_skills=false"],
))
def test_invalid_exploration_configuration_fails(compose_config, overrides):
    config = compose_config("algo=p2e", "model.strict_task_blind_pretraining=true", *overrides)
    with pytest.raises((ValueError, NotImplementedError)):
        validate_supported_idm(config.model)
        launcher._agent_translation(config.model)


@pytest.mark.parametrize("idm,scale", [("simple_curiosity", 1.0), ("icm", 0.01)])
def test_curiosity_presets_preserve_reward_scales(compose_config, idm, scale):
    config = compose_config("algo=p2e", f"idm={idm}")
    translated = launcher._agent_translation(config.model)
    assert translated["p2e"]["idm_type"] == idm
    assert translated["p2e"][f"{idm}_scale"] == scale
    if idm == "icm":
        assert translated["p2e"]["icm_forward_loss_weight"] == 0.2


def test_max_preset_uses_dense_five_member_ensemble(compose_config):
    config = compose_config("algo=p2e", "idm=max")
    translated = launcher._agent_translation(config.model)["p2e"]
    expected = {
        "max_models": 5,
        "max_hidden_dim": 256,
        "max_layers": 2,
        "max_chunk_size": 0,
        "max_scale": 1.0,
        "max_lambda": 0.1,
        "max_min_log_var": -10.0,
        "max_max_log_var": 2.0,
        "max_normalize": False,
        "idm_act": "relu",
    }
    for name, value in expected.items():
        assert translated[name] == value
