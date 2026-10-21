# Self-supervised Reinforcement Learning (SSRL) Agent

Code for training intrinsically motivated world-model agents.

## Use the IDMs

With Python 3.11 or newer, install from a checkout:

```bash
python -m pip install .
```

This installs the IDM library and its JAX/Ninjax dependencies. For NVIDIA CUDA 12,
use `python -m pip install '.[cuda]'`. Install directly from GitHub with
`python -m pip install 'git+https://github.com/neuroagents-lab/ssrl-agent.git'`.

Import modules through `ssrl.idms`:

```python
import jax.numpy as jnp
import ninjax as nj
from ssrl.idms import SimpleCuriosity

idm = SimpleCuriosity(obs_dim=8, action_dim=2, name="curiosity")
obs, action, next_obs = jnp.zeros((4, 8)), jnp.zeros((4, 2)), jnp.ones((4, 8))

state = nj.init(idm.loss)({}, obs, action, next_obs, seed=0)
state, (loss, metrics) = nj.pure(idm.loss)(state, obs, action, next_obs, seed=1)
state, reward = nj.pure(idm.compute_reward)(state, obs, action, next_obs, seed=2)
```

`ICM`, `MAX`, and the other methods are available through the same import
path and the `curiosity`, `entropy`, and `competency` subpackages. Supply features
and actions from your own agent; optimization stays in your training loop.
Skill methods also require skill inputs, and CIP requires a compatible Dreamer
dynamics model. Importing the library does not load the experiment environments.

## Install experiment dependencies

Requires Linux, Python 3.11, and an NVIDIA GPU with a CUDA 12-compatible driver.
Run these commands from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run an experiment

Train a DreamerV3 agent with simple curiosity on DMC Walker Walk:

```bash
MUJOCO_GL=egl python train.py \
  envs=dmc_proprio envs.task=dmc_walker_walk \
  model=size12M agent_name=dreamerv3 \
  algo=p2e idm=simple_curiosity \
  model.strict_task_blind_pretraining=true \
  device=cuda:0 seed=0 \
  logdir=logdir/walker_simple_curiosity_seed0
```

Metrics and checkpoints are saved in the specified `logdir`. W&B is disabled by
default. Change `idm` to select another preset from `configs/idm/`, such as `icm`
or `max`, and use a separate `logdir` for each run.

Slurm launchers are in `experiments/`; use `--help` for options and `--dry-run`
to preview a submission.

Built on [DreamerV3](https://github.com/danijar/dreamerv3).

Licensed under the [MIT License](LICENSE). Third-party assets retain their
respective licenses; see [the asset notices](envs/assets/robokitchen/THIRD_PARTY_NOTICES).
