"""P2E random data collection with a learned world model and task behavior."""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from agent.dreamer.agent import DreamerAgent
from agent.dreamer.base import getcfg
from ssrl.idms import validate_supported_idm


def sample_random_actions(act_space, batch_size, seed, distribution="p2e", noise=1.0):
    """Sample integer actions or bounded floats, independently across a batch.

    P2E mode clips N(0, 1), adds independent N(0, noise), then clips again.
    Affine rescaling maps the result to the action bounds. Uniform mode
    samples within those bounds without additive noise.
    """
    actions = {}
    keys = jax.random.split(seed, len(act_space))
    for (name, space), key in zip(act_space.items(), keys):
        shape = (batch_size, *space.shape)
        if space.discrete:
            action = jax.random.randint(key, shape, space.low, space.high)
        elif distribution == "uniform":
            action = jax.random.uniform(
                key, shape, minval=space.low, maxval=space.high)
        else:
            base_key, noise_key = jax.random.split(key)
            action = jnp.clip(jax.random.normal(base_key, shape), -1.0, 1.0)
            action = jnp.clip(
                action + noise * jax.random.normal(noise_key, shape), -1.0, 1.0)
            midpoint = (space.high + space.low) / 2.0
            half_range = (space.high - space.low) / 2.0
            action = midpoint + half_range * action
        actions[name] = action.astype(space.dtype)
    return actions


class RandomControlAgent(DreamerAgent):
    """Train Dreamer normally; replace only real training actions with random ones."""

    def __init__(self, obs_space, act_space, config):
        settings = dict(config.p2e)
        settings.update(expl_behavior="plan2explore", idm_type="random")
        validate_supported_idm(settings)
        self.random_distribution = settings.get(
            "random_action_distribution", "p2e").lower()
        self.random_noise = float(settings.get("random_action_noise", 1.0))
        for name, space in act_space.items():
            if not (np.isfinite(space.low).all() and np.isfinite(space.high).all()):
                raise ValueError(f"Random control requires finite action bounds: {name}.")
        if getcfg(config, "strict_task_blind_pretraining", False):
            conflicts = [
                name for name in ("reward_grad", "repval_loss", "repval_grad", "ac_grads")
                if getcfg(config, name, False)
            ]
            if conflicts:
                raise ValueError(
                    f"Strict task-blind random control requires {conflicts} false.")
        super().__init__(obs_space, act_space, config)
        if not self.train_base_task_behavior and not self.task_heads["specs"]:
            raise ValueError(
                "Random control requires a trained base or evaluation task policy.")

    def _policy_action(self, feat, mode):
        if mode in {"train", "explore"}:
            return sample_random_actions(
                self.act_space, feat["deter"].shape[0], nj.seed(),
                self.random_distribution, self.random_noise,
            )
        return super()._policy_action(feat, mode)
