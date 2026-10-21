"""Verify that the IDM distribution works independently of the experiment repo."""

from email.parser import BytesParser
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_python(*arguments, cwd):
    result = subprocess.run(
        [sys.executable, "-I", "-B", *arguments],
        cwd=cwd,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "PIP_NO_INDEX": "1"},
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def installed_idm_package(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("idm_package")
    source = workspace / "source"
    source.mkdir()
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / filename, source / filename)
    for directory in ("ssrl", "agent", "configs", "envs", "experiments", "tests"):
        shutil.copytree(
            ROOT / directory,
            source / directory,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
        )
    wheels = workspace / "wheels"
    run_python(
        "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
        "--no-cache-dir", "--wheel-dir", str(wheels), str(source), cwd=workspace,
    )
    wheel, = wheels.glob("*.whl")
    target = workspace / "installed"
    run_python(
        "-m", "pip", "install", "--no-deps", "--no-compile", "--no-cache-dir",
        "--target", str(target), str(wheel), cwd=workspace,
    )
    return wheel, target


def test_wheel_contains_only_idms_and_minimal_dependencies(installed_idm_package):
    wheel, _ = installed_idm_package
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_path, = (name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
    assert metadata["Name"] == "ssrl"
    assert "ssrl/idms/curiosity/icm.py" in names
    assert all(
        name.startswith("ssrl/") or ".dist-info/" in name
        for name in names
    )
    base_requirements = [
        requirement for requirement in metadata.get_all("Requires-Dist", [])
        if "extra ==" not in requirement
    ]
    assert {
        re.match(r"[\w.-]+", requirement).group().lower()
        for requirement in base_requirements
    } == {"jax", "ninjax"}


def test_installed_idms_train_without_experiment_imports(installed_idm_package):
    _, target = installed_idm_package
    script = textwrap.dedent(
        """
        import importlib.abc
        from pathlib import Path
        import sys

        target = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(target))

        class BlockExperimentImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                blocked = {
                    "agent", "dreamer", "embodied", "envs", "hydra", "omegaconf",
                    "elements", "portal", "optax", "chex", "dm_control", "mujoco",
                    "craftax", "torch", "torchvision", "wandb",
                }
                if fullname.split(".", 1)[0] in blocked:
                    raise AssertionError(f"IDMs imported experiment dependency: {fullname}")
                return None

        sys.meta_path.insert(0, BlockExperimentImports())
        import ssrl
        from ssrl import idms
        from ssrl.idms import validate_supported_idm

        validate_supported_idm(None)
        assert "jax" not in sys.modules

        import jax
        import jax.numpy as jnp
        import ninjax as nj
        from ssrl.idms import ICM, SimpleCuriosity, MAX
        from ssrl.idms.competency import DIAYN

        assert Path(ssrl.__file__).is_relative_to(target)
        for name in idms.__all__:
            getattr(idms, name)

        observations = jnp.array([[0.2, -0.4, 0.8], [-0.3, 0.9, 0.1]])
        next_observations = observations + 0.25
        actions = jnp.array([[0.3, -0.1], [-0.2, 0.4]])
        skills = jnp.eye(2)

        for cls in (SimpleCuriosity, ICM, DIAYN):
            settings = dict(obs_dim=3, hidden_dim=8, n_layers=1, name=cls.__name__)
            if cls is DIAYN:
                settings["num_skills"] = 2
            else:
                settings["action_dim"] = 2
            module = cls(**settings)

            def evaluate():
                if cls is DIAYN:
                    loss, _ = module.loss(observations, skills)
                    reward = module.compute_reward(observations, skill=skills)
                else:
                    loss, _ = module.loss(observations, actions, next_observations)
                    reward = module.compute_reward(observations, actions, next_observations)
                return loss, reward

            state = nj.init(evaluate)({}, seed=0)
            assert state
            pure_evaluate = nj.pure(evaluate)
            _, (loss, reward) = pure_evaluate(state, seed=1)
            assert loss.shape == () and bool(jnp.isfinite(loss))
            assert reward.shape == (2, 1) and bool(jnp.isfinite(reward).all())
            objective = lambda parameters: pure_evaluate(parameters, seed=1)[1][0]
            gradients = jax.grad(objective)(state)
            leaves = jax.tree_util.tree_leaves(gradients)
            assert all(bool(jnp.isfinite(gradient).all()) for gradient in leaves)
            assert any(bool(jnp.any(gradient != 0)) for gradient in leaves)
            updated = jax.tree_util.tree_map(lambda p, g: p - 1e-3 * g, state, gradients)
            assert bool(jnp.isfinite(objective(updated)))
        """
    )
    run_python("-c", script, str(target), cwd=target.parent)
