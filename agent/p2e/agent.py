import re

import numpy as np

import embodied.jax
import embodied.jax.nets as nn
import elements
import jax
import jax.numpy as jnp
import ninjax as nj

from dreamer.agent import DreamerAgent
from dreamer.base import (
    concat,
    f32,
    imag_loss,
    policy_action,
    prefix,
    sample,
    sg,
)

from ssrl.idms.curiosity import (
    CIG,
    CTEC,
    CIP,
    Disagreement,
    GammaLearningProgress,
    SimpleCuriosity,
    MAX,
    NGU,
    ICM,
    RND,
)
from ssrl.idms.curiosity.common import RunningMeanNormalizer, padded_next_sequence
from ssrl.idms.entropy import APT, ProtoRL, RE3, SMM
from ssrl.idms.competency import (
    CSF,
    DADS,
    DIAYN,
    EmpowermentGain,
    ForwardEmpowerment,
    PotentialEmpowerment,
    ProcessEmpowerment,
    RDADS,
    ReverseEmpowerment,
    VALOR,
    VIC,
    get_skill_prior,
    normalize_skill_space,
    skill_event_dim,
    skill_to_vectors,
)
from ssrl.idms.competency.common import (
    Linear,
    fixed_horizon_skill_segments,
    infer_skill_starts,
)


def _get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_expl_grad_mode(value):
    aliases = {
        "detached": "detached",
        "detach": "detached",
        "stopgrad": "detached",
        "stop_grad": "detached",
        "head_only": "head_only",
        "head": "head_only",
        "encoder_only": "encoder_only",
        "encoder": "encoder_only",
        "enc": "encoder_only",
        "full": "full",
        "shared": "full",
    }
    mode = aliases.get(str(value).lower())
    if mode is None:
        raise ValueError(
            "expl_grad_mode must be detached, head_only, encoder_only, or full, "
            f"got {value!r}."
        )
    return mode


def _route_idm_repr_gradient(value, mode):
    """Lets replay IDM loss shape the shared encoder/RSSM only in full mode."""
    return value if _normalize_expl_grad_mode(mode) == "full" else sg(value)


def _space_flat_dim(space):
    shape = tuple(getattr(space, "shape", ()) or ())
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if getattr(space, "discrete", False):
        classes = np.asarray(space.classes).reshape(-1)
        if not classes.size:
            raise ValueError("Discrete action spaces must expose class counts.")
        if not np.all(classes == classes[0]):
            raise ValueError("JAX P2E requires uniform discrete action classes per action entry.")
        size *= int(classes[0])
    return size


def _action_flat_dim(act_space):
    return sum(_space_flat_dim(space) for space in act_space.values())


def _all_actions_discrete(act_space):
    return bool(act_space) and all(getattr(space, "discrete", False) for space in act_space.values())


_ARRIVAL_ALIGNED_TRANSITION_IDMS = frozenset({"dads", "rdads"})


def _cig_outgoing_transitions(reprs, action):
    """Select the H departure states/actions scored by an H-step CIG rollout."""
    return reprs[:, :-1], jax.tree.map(lambda x: x[:, :-1], action)


def _arrival_align_cig_reward(reward):
    """Convert CIG's outgoing-transition rewards to Dreamer's arrival layout."""
    zero = jnp.zeros((reward.shape[0], 1, *reward.shape[2:]), reward.dtype)
    return jnp.concatenate([zero, reward], axis=1)


def _normalize_reward_norm_source(source):
    aliases = {
        "replay_only": "replay_only",
        "replay": "replay_only",
        "replay_and_imagination": "replay_and_imagination",
        "both": "replay_and_imagination",
    }
    normalized = aliases.get(str(source).lower())
    if normalized is None:
        raise ValueError(
            "idm_reward_norm_source must be replay_only or "
            f"replay_and_imagination, got {source!r}."
        )
    return normalized


def _sample_skill_transition_batch(obs, next_obs, skill, mask, sample_size):
    """Flatten and optionally subsample valid replay transitions."""
    obs = jnp.asarray(obs)
    next_obs = jnp.asarray(next_obs)
    skill = jnp.asarray(skill)
    mask = jnp.asarray(mask, bool)
    lead_shape = obs.shape[:-1]
    if next_obs.shape[:-1] != lead_shape or mask.shape != lead_shape:
        raise ValueError("Transition observations and mask must share leading dimensions.")
    if skill.shape[: len(lead_shape)] != lead_shape:
        raise ValueError("Transition skills must share observation leading dimensions.")

    total = int(np.prod(lead_shape, dtype=np.int64))
    obs = obs.reshape((total, obs.shape[-1]))
    next_obs = next_obs.reshape((total, next_obs.shape[-1]))
    skill = skill.reshape((total, *skill.shape[len(lead_shape) :]))
    mask = mask.reshape((total,))
    sample_size = int(sample_size or 0)
    if sample_size <= 0 or total <= sample_size:
        return obs, next_obs, skill, mask

    # Sample valid transitions first; mask padding to preserve a static batch size.
    random_order = jax.random.uniform(nj.seed(), (total,), dtype=jnp.float32)
    priority = random_order + (~mask).astype(jnp.float32) * 2.0
    indices = jnp.argsort(priority)[:sample_size]
    return obs[indices], next_obs[indices], skill[indices], mask[indices]


class Plan2ExploreAgent(DreamerAgent):

    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)
        if not self.train_base_task_behavior and not self.task_heads["specs"]:
            raise ValueError(
                "train_base_task_behavior=false requires at least one "
                "eval_task_heads entry; otherwise unlabeled evaluation would "
                "use an untrained main actor."
            )
        self.p2e = config.p2e
        self._idm_type = str(_get(self.p2e, "idm_type", "disagreement")).lower()
        self._idm_loss_key = str(_get(self.p2e, "idm_loss_key", self._default_idm_loss_key()))
        self._plan_repr_source = str(_get(self.p2e, "plan_repr_source", "stoch"))
        self._expl_grad_mode = _normalize_expl_grad_mode(
            _get(self.p2e, "expl_grad_mode", "head_only")
        )
        self._disag_target = str(_get(self.p2e, "disag_target", "same"))
        self._disag_action_cond = bool(_get(self.p2e, "disag_action_cond", True))
        self._disag_bootstrap = bool(_get(self.p2e, "disag_bootstrap", True))
        self._expl_intr_scale = float(_get(self.p2e, "expl_intr_scale", 1.0))
        self._expl_extr_scale = float(_get(self.p2e, "expl_extr_scale", 0.0))
        self._idm_reward_norm_source = _normalize_reward_norm_source(
            _get(self.p2e, "idm_reward_norm_source", "replay_only")
        )
        self._global_intrinsic_reward_normalizer = (
            RunningMeanNormalizer(
                eps=float(_get(self.p2e, "idm_reward_normalize_eps", 1e-8)),
                name="idm_reward_norm",
            )
            if bool(_get(self.p2e, "idm_reward_normalize", False))
            else None
        )
        self.strict_task_blind_pretraining = bool(
            _get(config, "strict_task_blind_pretraining", False)
        )
        if self.strict_task_blind_pretraining:
            conflicts = []
            if bool(_get(config, "reward_grad", True)):
                conflicts.append("reward_grad=true")
            if bool(_get(config, "repval_loss", True)):
                conflicts.append("repval_loss=true")
            if bool(_get(config, "repval_grad", True)):
                conflicts.append("repval_grad=true")
            if bool(_get(config, "ac_grads", False)):
                conflicts.append("ac_grads=true")
            if self._expl_extr_scale:
                conflicts.append(f"expl_extr_scale={self._expl_extr_scale}")
            if conflicts:
                raise ValueError(
                    "Strict task-blind P2E configuration is internally "
                    "inconsistent: " + ", ".join(conflicts)
                )
        self._ngu_update_tail = int(_get(self.p2e, "ngu_update_tail", 0) or 0)
        self._ctec_replay_batch_size = int(_get(self.p2e, "ctec_replay_batch_size", 256) or 0)
        self._diayn_replay_batch_size = int(
            _get(self.p2e, "diayn_replay_batch_size", 128) or 0
        )
        self._dads_replay_batch_size = int(
            _get(self.p2e, "dads_replay_batch_size", 256) or 0
        )
        self._rdads_replay_batch_size = int(
            _get(self.p2e, "rdads_replay_batch_size", 256) or 0
        )
        self._csf_replay_batch_size = int(
            _get(self.p2e, "csf_replay_batch_size", 256) or 0
        )
        self._cip_imag_last = max(0, int(_get(self.p2e, "cip_imag_last", 0) or 0))
        self._skill_idm_types = {
            "diayn",
            "vic",
            "dads",
            "rdads",
            "valor",
            "csf",
        }
        self._empowerment_idm_types = {
            "empowerment_forward",
            "empowerment_reverse",
            "empowerment_gain",
            "empowerment_potential",
            "empowerment_process",
        }
        self._uses_skills = bool(_get(self.p2e, "use_skills", False)) and self._idm_type in self._skill_idm_types
        self._num_skills = int(_get(self.p2e, "num_skills", 10))
        self._skill_space = normalize_skill_space(_get(self.p2e, "skill_space", "categorical"))
        self._skill_dim_config = int(_get(self.p2e, "skill_dim", 0) or 0)
        self._skill_prior_name = self._configured_skill_prior_name()
        self._skill_prior_kwargs = self._configured_skill_prior_kwargs()
        self._skill_prior = (
            get_skill_prior(
                self._skill_prior_name,
                num_skills=self._num_skills,
                skill_dim=self._skill_dim_config,
                **(self._skill_prior_kwargs or {}),
            )
            if self._uses_skills
            else None
        )
        self._skill_event_dim = (
            skill_event_dim(self._skill_prior, self._num_skills, self._skill_dim_config)
            if self._uses_skills
            else 0
        )
        self._skill_embed_dim = int(_get(self.p2e, "skill_embed_dim", 0) or 0)
        self._skill_input_dim = self._skill_embed_dim if self._skill_embed_dim > 0 else self._skill_event_dim
        self._skill_horizon = int(_get(self.p2e, "skill_horizon", 0) or 0)
        if self._idm_type in {"vic", "valor"}:
            if self._skill_space != "categorical":
                raise ValueError(f"{self._idm_type.upper()} requires categorical skills.")
            if self._skill_horizon <= 0:
                raise ValueError(
                    f"{self._idm_type.upper()} requires a positive skill_horizon "
                    "to define complete option trajectories."
                )
            if self._skill_horizon != int(config.imag_length):
                raise ValueError(
                    f"{self._idm_type.upper()} requires skill_horizon "
                    f"({self._skill_horizon}) to equal imag_length "
                    f"({int(config.imag_length)}) so imagined rollouts are complete options."
                )
        self.skill_embed = (
            Linear(
                input_dim=self._skill_event_dim,
                output_dim=self._skill_input_dim,
                name="skill_embed",
            )
            if self._uses_skills and self._skill_input_dim != self._skill_event_dim
            else None
        )

        input_dim = self._repr_dim(self._plan_repr_source)
        if self._idm_type == "disagreement":
            target_source = self._plan_repr_source if self._disag_target == "same" else self._disag_target
            target_dim = self._repr_dim(target_source)
        else:
            target_dim = input_dim
        full_action_dim = _action_flat_dim(act_space)
        action_dim = full_action_dim if self._uses_actions else 0
        self.curiosity = self._make_curiosity(input_dim, action_dim, target_dim, act_space)
        self.disag = self.curiosity if self._idm_type == "disagreement" else None

        scalar = elements.Space(np.float32, ())
        d1, d2 = config.policy_dist_disc, config.policy_dist_cont
        outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
        self.expl_pol = embodied.jax.MLPHead(act_space, outs, **config.policy, name="expl_pol")
        self.expl_val = embodied.jax.MLPHead(scalar, **config.value, name="expl_val")
        self.slowexplval = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name="slowexplval"),
            source=self.expl_val,
            **config.slowvalue,
        )
        self.expl_retnorm = embodied.jax.Normalize(**config.retnorm, name="expl_retnorm")
        self.expl_valnorm = embodied.jax.Normalize(**config.valnorm, name="expl_valnorm")
        self.expl_advnorm = embodied.jax.Normalize(**config.advnorm, name="expl_advnorm")

        curiosity_modules = [self.curiosity]
        if self._idm_type == "apt":
            curiosity_modules = (
                [self.curiosity.phi]
                if bool(_get(self.p2e, "apt_train_phi", True))
                else []
            )
        elif self._idm_type == "cig":
            # CIG's aleatoric and reward EMAs are state, not optimizer targets.
            curiosity_modules = [self.curiosity.ensemble]
        if self._idm_type == "proto_rl":
            curiosity_modules = [
                self.curiosity.online_encoder,
                self.curiosity.online_projector,
                self.curiosity.predictor,
                self.curiosity.prototype_bank,
            ]
        elif self._idm_type == "re3":
            curiosity_modules = []
        elif self._idm_type == "cip":
            curiosity_modules = []
        elif self._idm_type == "gamma_progress":
            curiosity_modules = [self.curiosity.dynamics]
        elif self._idm_type == "empowerment_gain":
            curiosity_modules = list(
                self.curiosity.get_trainable_modules().values()
            )
        if self.skill_embed is not None:
            curiosity_modules.append(self.skill_embed)
        rep_modules = self._representation_modules()
        base_behavior = (
            [self.rew, self.pol, self.val]
            if self.train_base_task_behavior
            else []
        )
        self.modules = [
            self.dyn,
            *self._encoder_modules,
            self.con,
            *base_behavior,
            *rep_modules,
            *curiosity_modules,
            self.expl_pol,
            self.expl_val,
        ]
        self.main_opt = embodied.jax.Optimizer(
            self.modules,
            self._make_opt(**config.opt),
            adaptive=embodied.jax.opt.adaptive_critic(
                **(_get(config, "adaptive_critic", {}) or {})),
            summary_depth=1,
            log_grads=_get(config, "log_grads", True),
            name="p2e_opt",
        )
        self.opt = self.main_opt

    def _representation_modules(self):
        if self.rep_loss == "dreamer":
            return [self.dec]
        modules = []
        if self.r2_projector is not None:
            modules.append(self.r2_projector)
        if self.ne is not None:
            modules.append(self.ne)
        if self.dreamerpro is not None:
            modules.extend(self.dreamerpro.modules)
        return modules

    def _default_idm_loss_key(self):
        if self._idm_type == "disagreement":
            return "disag"
        return self._idm_type

    def _default_prior_for_skill_space(self):
        return {
            "categorical": "uniform_categorical",
            "sphere": "uniform_hypersphere",
            "box": "uniform_box",
        }[self._skill_space]

    def _configured_skill_prior_name(self):
        if self._idm_type not in {
            "diayn",
            "vic",
            "dads",
            "rdads",
            "valor",
            "csf",
        }:
            return self._default_prior_for_skill_space()
        value = _get(self.p2e, f"{self._idm_type}_prior", None)
        return str(value) if value else self._default_prior_for_skill_space()

    def _configured_skill_prior_kwargs(self):
        if self._idm_type not in {
            "diayn",
            "vic",
            "dads",
            "rdads",
            "valor",
            "csf",
        }:
            return None
        return _get(self.p2e, f"{self._idm_type}_prior_kwargs", None)

    @property
    def _uses_actions(self):
        if self._idm_type in {
            "apt",
            "proto_rl",
            "re3",
            "smm",
            "rnd",
            "diayn",
            "vic",
            "dads",
            "rdads",
            "valor",
            "csf",
        }:
            return False
        if self._idm_type == "disagreement":
            return self._disag_action_cond
        return True

    def _make_curiosity(self, input_dim, action_dim, target_dim, act_space):
        hidden = int(_get(self.p2e, "idm_hidden_dim", _get(self.p2e, "disag_hidden_dim", 256)))
        layers = int(_get(self.p2e, "idm_layers", _get(self.p2e, "disag_layers", 2)))
        act = str(_get(self.p2e, "idm_act", _get(self.p2e, "disag_act", "relu")))
        discrete_actions = _all_actions_discrete(act_space)
        if self._idm_type == "disagreement":
            return Disagreement(
                input_dim=input_dim,
                action_dim=action_dim,
                target_dim=target_dim,
                hidden_dim=int(_get(self.p2e, "disag_hidden_dim", hidden)),
                mlp_layers=int(_get(self.p2e, "disag_layers", layers)),
                n_models=int(_get(self.p2e, "disag_models", 5)),
                scale=float(_get(self.p2e, "disag_scale", 10000.0)),
                use_log_reward=bool(_get(self.p2e, "disag_log", False)),
                normalize_rewards=bool(_get(self.p2e, "disag_normalize", False)),
                eps=float(_get(self.p2e, "disag_eps", 1e-8)),
                act=str(_get(self.p2e, "disag_act", act)),
                name="disag",
            )
        if self._idm_type == "max":
            return MAX(
                input_dim=input_dim,
                action_dim=action_dim,
                target_dim=target_dim,
                hidden_dim=int(_get(self.p2e, "max_hidden_dim", 256)),
                mlp_layers=int(_get(self.p2e, "max_layers", 2)),
                n_models=int(_get(self.p2e, "max_models", 5)),
                chunk_size=int(_get(self.p2e, "max_chunk_size", 0)),
                scale=float(_get(self.p2e, "max_scale", 1.0)),
                jrd_lambda=float(_get(self.p2e, "max_lambda", 0.1)),
                min_log_var=float(_get(self.p2e, "max_min_log_var", -10.0)),
                max_log_var=float(_get(self.p2e, "max_max_log_var", 2.0)),
                normalize_rewards=bool(_get(self.p2e, "max_normalize", False)),
                eps=float(_get(self.p2e, "max_eps", 1e-8)),
                act=act,
                name="max",
            )
        if self._idm_type == "cig":
            return CIG(
                input_dim=input_dim,
                action_dim=action_dim,
                target_dim=target_dim,
                hidden_dim=int(_get(self.p2e, "cig_hidden_dim", 400)),
                mlp_layers=int(_get(self.p2e, "cig_layers", 4)),
                n_models=int(_get(self.p2e, "cig_models", 5)),
                scale=float(_get(self.p2e, "cig_scale", 1.0)),
                ridge_scale=float(_get(self.p2e, "cig_ridge_scale", 1.0)),
                aleatoric_ema=float(_get(self.p2e, "cig_aleatoric_ema", 0.99)),
                normalize_rewards=bool(_get(self.p2e, "cig_normalize", True)),
                reward_ema=float(_get(self.p2e, "cig_reward_ema", 0.99)),
                eps=float(_get(self.p2e, "cig_eps", 1e-8)),
                act=str(_get(self.p2e, "cig_act", "elu")),
                name="cig",
            )
        if self._idm_type == "simple_curiosity":
            return SimpleCuriosity(
                obs_dim=input_dim,
                action_dim=action_dim,
                hidden_dim=hidden,
                n_layers=layers,
                scale=float(_get(self.p2e, "simple_curiosity_scale", 1.0)),
                discrete_actions=discrete_actions,
                act=act,
                name="simple_curiosity",
            )
        if self._idm_type == "icm":
            return ICM(
                obs_dim=input_dim,
                action_dim=action_dim,
                proj_dim=int(_get(self.p2e, "icm_proj_dim", 0)),
                proj_hidden_dim=int(
                    _get(self.p2e, "icm_proj_hidden_dim", hidden) or hidden
                ),
                proj_layers=int(_get(self.p2e, "icm_proj_layers", 1)),
                hidden_dim=hidden,
                n_layers=layers,
                forward_loss_weight=float(
                    _get(self.p2e, "icm_forward_loss_weight", 0.2)
                ),
                scale=float(_get(self.p2e, "icm_scale", 0.01)),
                discrete_actions=discrete_actions,
                act=act,
                name="icm",
            )
        if self._idm_type == "rnd":
            rnd_hidden = int(_get(self.p2e, "rnd_hidden_dim", hidden) or hidden)
            rnd_output = int(_get(self.p2e, "rnd_output_dim", 512) or rnd_hidden)
            return RND(
                obs_dim=input_dim,
                hidden_dim=rnd_hidden,
                output_dim=rnd_output,
                mlp_layers=int(_get(self.p2e, "rnd_layers", layers)),
                scale=float(_get(self.p2e, "rnd_scale", 1.0)),
                normalize_rewards=bool(_get(self.p2e, "rnd_normalize", True)),
                eps=float(_get(self.p2e, "rnd_eps", 1e-8)),
                act=act,
                name="rnd",
            )
        if self._idm_type == "apt":
            return APT(
                obs_dim=input_dim,
                rep_dim=int(_get(self.p2e, "apt_rep_dim", 256)),
                hidden_dim=hidden,
                n_layers=layers,
                k=int(_get(self.p2e, "apt_k", 16)),
                knn_clip=float(_get(self.p2e, "apt_knn_clip", 0.0005)),
                normalize_phi=bool(_get(self.p2e, "apt_normalize_phi", True)),
                train_phi=bool(_get(self.p2e, "apt_train_phi", True)),
                reward_normalize=bool(_get(self.p2e, "apt_reward_normalize", True)),
                eps=float(_get(self.p2e, "apt_eps", 1e-8)),
                knn_chunk_size=int(_get(self.p2e, "apt_knn_chunk_size", 1024)),
                memory_size=int(_get(self.p2e, "apt_memory_size", 2048)),
                memory_sample_size=int(_get(self.p2e, "apt_memory_sample_size", 2048)),
                act=act,
                name="apt",
            )
        if self._idm_type == "re3":
            return RE3(
                obs_dim=input_dim,
                rep_dim=int(_get(self.p2e, "re3_rep_dim", 50)),
                hidden_dim=hidden,
                n_layers=layers,
                k=int(_get(self.p2e, "re3_k", 53)),
                scale=float(_get(self.p2e, "re3_scale", 0.1)),
                log_reward=bool(_get(self.p2e, "re3_log_reward", False)),
                reward_normalize=bool(_get(self.p2e, "re3_reward_normalize", True)),
                eps=float(_get(self.p2e, "re3_eps", 1e-8)),
                knn_chunk_size=int(_get(self.p2e, "re3_knn_chunk_size", 1024)),
                memory_size=int(_get(self.p2e, "re3_memory_size", 2048)),
                memory_sample_size=int(_get(self.p2e, "re3_memory_sample_size", 2048)),
                act=act,
                name="re3",
            )
        if self._idm_type == "smm":
            return SMM(
                obs_dim=input_dim,
                latent_dim=int(_get(self.p2e, "smm_latent_dim", 128)),
                hidden_dim=int(_get(self.p2e, "smm_hidden_dim", 150)),
                n_layers=int(_get(self.p2e, "smm_layers", 2)),
                scale=float(_get(self.p2e, "smm_scale", 1.0)),
                kl_scale=float(_get(self.p2e, "smm_kl_scale", 0.5)),
                target=str(_get(self.p2e, "smm_target", "uniform")),
                log_target_const=float(_get(self.p2e, "smm_log_target_const", 0.0)),
                min_log_var=float(_get(self.p2e, "smm_min_log_var", -7.0)),
                max_log_var=float(_get(self.p2e, "smm_max_log_var", 2.0)),
                reward_normalize=bool(_get(self.p2e, "smm_reward_normalize", False)),
                eps=float(_get(self.p2e, "smm_eps", 1e-8)),
                act=act,
                name="smm",
            )
        if self._idm_type == "proto_rl":
            return ProtoRL(
                obs_dim=input_dim,
                rep_dim=int(_get(self.p2e, "proto_rl_rep_dim", 128)),
                hidden_dim=hidden,
                pred_dim=int(_get(self.p2e, "proto_rl_pred_dim", 512)),
                n_layers=layers,
                num_prototypes=int(_get(self.p2e, "proto_rl_num_prototypes", 512)),
                candidates_per_prototype=int(_get(self.p2e, "proto_rl_candidates_per_prototype", 4)),
                temperature=float(_get(self.p2e, "proto_rl_temperature", 0.1)),
                sinkhorn_iters=int(_get(self.p2e, "proto_rl_sinkhorn_iters", 3)),
                target_momentum=float(_get(self.p2e, "proto_rl_target_momentum", 0.95)),
                k=int(_get(self.p2e, "proto_rl_knn_k", 3)),
                reward_normalize=bool(_get(self.p2e, "proto_rl_reward_normalize", False)),
                eps=float(_get(self.p2e, "proto_rl_eps", 1e-8)),
                act=act,
                name="proto_rl",
            )
        if self._idm_type == "ngu":
            rnd_hidden = int(_get(self.p2e, "ngu_rnd_hidden_dim", hidden) or hidden)
            rnd_output = int(_get(self.p2e, "ngu_rnd_output_dim", rnd_hidden) or rnd_hidden)
            return NGU(
                obs_dim=input_dim,
                action_dim=action_dim,
                rep_dim=int(_get(self.p2e, "ngu_rep_dim", 32)),
                hidden_dim=hidden,
                n_layers=layers,
                k=int(_get(self.p2e, "ngu_k", 10)),
                scale=float(_get(self.p2e, "ngu_scale", 0.3)),
                kernel_epsilon=float(_get(self.p2e, "ngu_kernel_epsilon", 1e-4)),
                cluster_distance=float(_get(self.p2e, "ngu_cluster_distance", 8e-3)),
                pseudo_count_c=float(_get(self.p2e, "ngu_pseudo_count_c", 1e-3)),
                max_similarity=float(_get(self.p2e, "ngu_max_similarity", 8.0)),
                lifelong_clip=float(_get(self.p2e, "ngu_lifelong_clip", 5.0)),
                use_rnd=bool(_get(self.p2e, "ngu_use_rnd", True)),
                rnd_hidden_dim=rnd_hidden,
                rnd_output_dim=rnd_output,
                rnd_layers=int(_get(self.p2e, "ngu_rnd_layers", 2)),
                inverse_coef=float(_get(self.p2e, "ngu_inverse_coef", 1.0)),
                rnd_coef=float(_get(self.p2e, "ngu_rnd_coef", 1.0)),
                vectorized_scan=bool(_get(self.p2e, "ngu_vectorized_scan", True)),
                discrete_actions=discrete_actions,
                eps=float(_get(self.p2e, "ngu_eps", 1e-8)),
                act=act,
                name="ngu",
            )
        if self._idm_type == "gamma_progress":
            return GammaLearningProgress(
                obs_dim=input_dim,
                action_dim=action_dim,
                hidden_dim=hidden,
                n_layers=layers,
                scale=float(_get(self.p2e, "progress_scale", 1.0)),
                gamma=float(_get(self.p2e, "gamma_progress_gamma", 0.9995)),
                act=act,
                name="progress",
            )
        if self._idm_type == "cip":
            return CIP(
                compute_dtype=jnp.dtype(nn.COMPUTE_DTYPE).name,
                state_source=str(_get(self.p2e, "cip_state_source", "deter")),
                probes=int(_get(self.p2e, "cip_probes", 8)),
                scale=float(_get(self.p2e, "cip_scale", 1.0)),
                eps=float(_get(self.p2e, "cip_eps", 1e-6)),
                positive=bool(_get(self.p2e, "cip_positive", True)),
                normalize_rewards=bool(_get(self.p2e, "cip_normalize", False)),
                seed=int(_get(self.p2e, "cip_seed", 0)),
                stop_reward_grad=bool(_get(self.p2e, "cip_stop_reward_grad", True)),
                name="cip",
            )
        if self._idm_type == "ctec":
            return CTEC(
                obs_dim=input_dim,
                action_dim=action_dim,
                rep_dim=int(_get(self.p2e, "ctec_rep_dim", 64)),
                hidden_dim=int(_get(self.p2e, "ctec_hidden_dim", hidden)),
                n_layers=int(_get(self.p2e, "ctec_layers", 2)),
                scale=float(_get(self.p2e, "ctec_scale", 1.0)),
                discount=float(_get(self.p2e, "ctec_gamma", 0.99)),
                temperature=float(_get(self.p2e, "ctec_temperature", 1.0)),
                learn_temperature=bool(_get(self.p2e, "ctec_learn_temperature", True)),
                similarity=str(_get(self.p2e, "ctec_similarity", "l1")),
                loss_type=str(_get(self.p2e, "ctec_loss_type", "infonce")),
                normalize_embeddings=bool(_get(self.p2e, "ctec_normalize_embeddings", True)),
                reward_estimator=str(_get(self.p2e, "ctec_reward_estimator", "single")),
                reward_normalize_truncation=bool(_get(self.p2e, "ctec_reward_normalize_truncation", False)),
                replay_batch_size=self._ctec_replay_batch_size,
                act=act,
                name="ctec",
            )
        if self._idm_type == "diayn":
            return DIAYN(
                obs_dim=input_dim,
                num_skills=self._num_skills,
                hidden_dim=hidden,
                n_layers=layers,
                entropy_coef=float(_get(self.p2e, "diayn_entropy_coef", 0.0)),
                prior_name=self._skill_prior_name,
                prior_kwargs=self._skill_prior_kwargs,
                skill_dim=self._skill_dim_config,
                posterior_type=str(_get(self.p2e, "continuous_skill_posterior", "dot_product")),
                temperature=float(_get(self.p2e, "continuous_skill_temperature", 0.1)),
                act=act,
                name="diayn",
            )
        if self._idm_type == "vic":
            return VIC(
                obs_dim=input_dim,
                num_skills=self._num_skills,
                hidden_dim=hidden,
                n_layers=layers,
                entropy_coef=float(_get(self.p2e, "vic_entropy_coef", 0.0)),
                prior_name=self._skill_prior_name,
                prior_kwargs=self._skill_prior_kwargs,
                skill_dim=self._skill_dim_config,
                posterior_type=str(_get(self.p2e, "continuous_skill_posterior", "dot_product")),
                temperature=float(_get(self.p2e, "continuous_skill_temperature", 0.1)),
                condition_on_start=bool(_get(self.p2e, "vic_condition_on_start", True)),
                reward_at_end=bool(_get(self.p2e, "vic_reward_at_end", True)),
                act=act,
                name="vic",
            )
        if self._idm_type == "dads":
            return DADS(
                obs_dim=input_dim,
                num_skills=self._num_skills,
                hidden_dim=hidden,
                n_layers=layers,
                entropy_coef=float(_get(self.p2e, "dads_entropy_coef", 0.0)),
                prior_name=self._skill_prior_name,
                prior_kwargs=self._skill_prior_kwargs,
                skill_dim=self._skill_dim_config,
                use_marginal_model=bool(_get(self.p2e, "dads_use_marginal_model", False)),
                marginal_samples=int(_get(self.p2e, "dads_marginal_samples", 100)),
                act=act,
                name="dads",
            )
        if self._idm_type == "rdads":
            return RDADS(
                obs_dim=input_dim,
                num_skills=self._num_skills,
                hidden_dim=hidden,
                n_layers=layers,
                entropy_coef=float(_get(self.p2e, "rdads_entropy_coef", 0.0)),
                prior_name=self._skill_prior_name,
                prior_kwargs=self._skill_prior_kwargs,
                skill_dim=self._skill_dim_config,
                posterior_type=str(_get(self.p2e, "continuous_skill_posterior", "dot_product")),
                temperature=float(_get(self.p2e, "continuous_skill_temperature", 0.1)),
                act=act,
                name="rdads",
            )
        if self._idm_type == "valor":
            return VALOR(
                obs_dim=input_dim,
                num_skills=self._num_skills,
                hidden_dim=hidden,
                n_layers=int(_get(self.p2e, "valor_mlp_layers", 1)),
                num_subseq=int(_get(self.p2e, "valor_num_subseq", 11)),
                entropy_coef=float(_get(self.p2e, "valor_entropy_coef", 0.0)),
                prior_name=self._skill_prior_name,
                prior_kwargs=self._skill_prior_kwargs,
                skill_dim=self._skill_dim_config,
                posterior_type=str(_get(self.p2e, "continuous_skill_posterior", "dot_product")),
                temperature=float(_get(self.p2e, "continuous_skill_temperature", 0.1)),
                subtract_log_prior=bool(_get(self.p2e, "valor_subtract_log_prior", True)),
                reward_at_end=bool(_get(self.p2e, "valor_reward_at_end", True)),
                head_hidden_dim=int(_get(self.p2e, "valor_head_hidden_dim", 0)),
                act=act,
                name="valor",
            )
        if self._idm_type == "csf":
            return CSF(
                obs_dim=input_dim,
                num_skills=self._num_skills,
                proj_dim=int(_get(self.p2e, "csf_proj_dim", 0)),
                proj_hidden_dim=int(_get(self.p2e, "csf_proj_hidden_dim", 1024)),
                proj_layers=int(_get(self.p2e, "csf_proj_layers", 2)),
                entropy_coef=float(_get(self.p2e, "csf_entropy_coef", 0.0)),
                prior_name=self._skill_prior_name,
                prior_kwargs=self._skill_prior_kwargs,
                skill_dim=self._skill_dim_config,
                reward_mode=str(_get(self.p2e, "csf_reward_mode", "inner")),
                posterior_type=str(_get(self.p2e, "csf_posterior", "dot_product")),
                temperature=float(_get(self.p2e, "csf_temperature", 1.0)),
                normalize_features=bool(_get(self.p2e, "csf_normalize_features", False)),
                normalize_skills=bool(_get(self.p2e, "csf_normalize_skills", False)),
                input_norm_type=str(_get(self.p2e, "csf_input_norm", "none")),
                input_norm_eps=float(_get(self.p2e, "csf_input_norm_eps", 1e-3)),
                xi=float(_get(self.p2e, "csf_xi", 5.0)),
                num_negatives=int(_get(self.p2e, "csf_num_negatives", 256)),
                sample_prior_negatives=bool(
                    _get(self.p2e, "csf_sample_prior_negatives", True)
                ),
                exclude_positive=bool(_get(self.p2e, "csf_exclude_positive_from_negatives", True)),
                act=act,
                name="csf",
            )
        if self._idm_type == "empowerment_forward":
            return ForwardEmpowerment(
                obs_dim=input_dim,
                action_dim=action_dim,
                hidden_dim=hidden,
                n_layers=layers,
                action_concat=bool(_get(self.p2e, "empowerment_action_concat", False)),
                act=act,
                name="empowerment_forward",
            )
        if self._idm_type == "empowerment_reverse":
            return ReverseEmpowerment(
                obs_dim=input_dim,
                action_dim=action_dim,
                hidden_dim=hidden,
                n_layers=layers,
                act=act,
                name="empowerment_reverse",
            )
        if self._idm_type == "empowerment_gain":
            return EmpowermentGain(
                obs_dim=input_dim,
                action_dim=action_dim,
                hidden_dim=hidden,
                n_layers=layers,
                gamma=float(_get(self.p2e, "empowerment_gain_gamma", 0.9995)),
                gce_latent_dim=int(
                    _get(self.p2e, "empowerment_gce_latent_dim", 64)
                ),
                gce_power=float(_get(self.p2e, "empowerment_gce_power", 1.0)),
                gce_waterfill_iters=int(
                    _get(self.p2e, "empowerment_gce_waterfill_iters", 32)
                ),
                gce_recon_scale=float(
                    _get(self.p2e, "empowerment_gce_recon_scale", 1.0)
                ),
                gce_state_noise=bool(
                    _get(self.p2e, "empowerment_gce_state_noise", False)
                ),
                act=act,
                name="empowerment_gain",
            )
        if self._idm_type == "empowerment_potential":
            return PotentialEmpowerment(
                obs_dim=input_dim,
                action_dim=action_dim,
                hidden_dim=hidden,
                n_layers=layers,
                particles=int(_get(self.p2e, "empowerment_particles", 16)),
                next_samples=int(_get(self.p2e, "empowerment_next_samples", 1)),
                q_loss_weight=float(_get(self.p2e, "empowerment_q_loss_weight", 1.0)),
                action_concat=bool(_get(self.p2e, "empowerment_action_concat", False)),
                reward_chunk_size=int(_get(self.p2e, "empowerment_reward_chunk_size", 0)),
                estimator=str(_get(self.p2e, "empowerment_estimator", "gce")),
                gce_latent_dim=int(_get(self.p2e, "empowerment_gce_latent_dim", 64)),
                gce_power=float(_get(self.p2e, "empowerment_gce_power", 1.0)),
                gce_waterfill_iters=int(_get(self.p2e, "empowerment_gce_waterfill_iters", 32)),
                gce_recon_scale=float(_get(self.p2e, "empowerment_gce_recon_scale", 1.0)),
                gce_state_noise=bool(_get(self.p2e, "empowerment_gce_state_noise", False)),
                act=act,
                name="empowerment_potential",
            )
        if self._idm_type == "empowerment_process":
            return ProcessEmpowerment(
                obs_dim=input_dim,
                action_dim=action_dim,
                hidden_dim=hidden,
                n_layers=layers,
                particles=int(_get(self.p2e, "empowerment_particles", 16)),
                next_samples=int(_get(self.p2e, "empowerment_next_samples", 1)),
                action_concat=bool(_get(self.p2e, "empowerment_action_concat", False)),
                reward_chunk_size=int(_get(self.p2e, "empowerment_reward_chunk_size", 0)),
                estimator=str(_get(self.p2e, "empowerment_estimator", "gce")),
                gce_latent_dim=int(_get(self.p2e, "empowerment_gce_latent_dim", 64)),
                gce_waterfill_iters=int(_get(self.p2e, "empowerment_gce_waterfill_iters", 32)),
                gce_recon_scale=float(_get(self.p2e, "empowerment_gce_recon_scale", 1.0)),
                gce_state_noise=bool(_get(self.p2e, "empowerment_gce_state_noise", False)),
                act=act,
                name="empowerment_process",
            )
        raise NotImplementedError(f"Unsupported JAX curiosity IDM: {self._idm_type!r}")

    @property
    def policy_keys(self):
        task_pols = "|".join(
            re.escape(head.name) for head in self.task_heads["pols"].values()
        )
        task_pols = f"|{task_pols}" if task_pols else ""
        return rf"^(enc|dyn|pol|expl_pol|skill_embed{task_pols})/"

    @property
    def ext_space(self):
        spaces = super().ext_space
        if self._uses_skills:
            spaces["skill"] = elements.Space(np.float32, (self._skill_event_dim,))
            spaces["skill_start"] = elements.Space(bool, ())
        return spaces

    def init_policy(self, batch_size):
        carry = super().init_policy(batch_size)
        if not self._uses_skills:
            return carry
        skill = self._zero_skill((batch_size,))
        skill_step = jnp.full((batch_size,), -1, jnp.int32)
        return (*carry, skill, skill_step)

    def init_train(self, batch_size):
        return super().init_policy(batch_size)

    def init_report(self, batch_size):
        return super().init_policy(batch_size)

    def _zero_skill(self, batch_shape):
        return jnp.zeros((*tuple(batch_shape), self._skill_event_dim), jnp.float32)

    def _sample_skill_vectors(self, batch_shape):
        if not self._uses_skills:
            return None
        sample = self._skill_prior.sample(tuple(batch_shape))
        ref = self._zero_skill(batch_shape)
        return skill_to_vectors(
            sample,
            ref,
            self._skill_prior.skill_space,
            self._num_skills,
            self._skill_event_dim,
            normalize=self._skill_prior.skill_space == "sphere",
        ).astype(jnp.float32)

    def _update_policy_skill_vectors(self, reset, skill, skill_step):
        batch_shape = tuple(reset.shape)
        if skill is None or skill_step is None:
            skill = self._zero_skill(batch_shape)
            skill_step = jnp.full(batch_shape, -1, jnp.int32)
        else:
            skill = jnp.asarray(skill, jnp.float32).reshape((*batch_shape, self._skill_event_dim))
            skill_step = jnp.asarray(skill_step, jnp.int32).reshape(batch_shape)
        reset = jnp.asarray(reset).reshape(batch_shape)
        next_step = skill_step + 1
        resample = reset | (skill_step < 0)
        if self._skill_horizon > 0:
            resample = resample | (next_step >= self._skill_horizon)
        new_skill = self._sample_skill_vectors(batch_shape)
        skill = jnp.where(resample[..., None], new_skill, skill)
        next_step = jnp.where(resample, jnp.zeros_like(next_step), next_step)
        return skill, next_step, resample

    def _skill_input(self, skill, ref):
        if not self._uses_skills:
            return None
        if skill is None:
            skill = self._zero_skill(ref.shape[:-1])
        skill = skill_to_vectors(
            skill,
            ref,
            self._skill_prior.skill_space,
            self._num_skills,
            self._skill_event_dim,
            normalize=self._skill_prior.skill_space == "sphere",
        )
        if self.skill_embed is not None:
            skill = self.skill_embed(skill)
        return skill

    def _expl_input(self, tensor, skill):
        tensor = f32(tensor)
        if not self._uses_skills:
            return tensor
        return jnp.concatenate([tensor, self._skill_input(skill, tensor)], axis=-1)

    def policy(self, carry, obs, mode="train"):
        skill_state = None
        skill_step = None
        if self._uses_skills and len(carry) == 6:
            (enc_carry, dyn_carry, dec_carry, prevact, skill_state, skill_step) = carry
        else:
            (enc_carry, dyn_carry, dec_carry, prevact) = carry
        kw = dict(training=False, single=True)
        reset = obs["is_first"]
        enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
        dyn_carry, dyn_entry, feat = self.dyn.observe(
            dyn_carry, tokens, prevact, reset, **kw
        )
        dec_entry = {}
        if dec_carry:
            dec_carry, dec_entry, _ = self.dec(dec_carry, feat, reset, **kw)
        tensor = self.feat2tensor(feat)
        skill = None
        skill_start = jnp.zeros(reset.shape, bool)
        if self._uses_skills:
            if mode == "train":
                skill, skill_step, skill_start = self._update_policy_skill_vectors(
                    reset, skill_state, skill_step
                )
                skill_state = skill
            else:
                skill = self._zero_skill(reset.shape)
                if skill_state is None or skill_step is None:
                    skill_state = skill
                    skill_step = jnp.full(reset.shape, -1, jnp.int32)
        if mode == "train":
            head = self.expl_pol
            head_inp = self._expl_input(tensor, skill)
        else:
            head = self._policy_head_for_mode(mode)
            head_inp = tensor
        policy = head(head_inp, bdims=1)
        act = policy_action(policy, mode)
        out = {}
        if getattr(self.config, "policy_checks", False):
            out["finite"] = elements.tree.flatdict(
                jax.tree.map(
                    lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
                    dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act),
                )
            )
        carry = (enc_carry, dyn_carry, dec_carry, act)
        if self._uses_skills:
            carry = (*carry, skill_state, skill_step)
        if self._uses_skills:
            out["skill"] = skill
            out["skill_start"] = skill_start
        if self.config.replay_context:
            out.update(elements.tree.flatdict(dict(enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
        return carry, act, out

    def train(self, carry, data):
        carry, obs, prevact, stepid, task_labels = self._apply_replay_context(
            carry, data
        )
        departact = self._departure_actions(data, obs["is_first"].shape[1])
        departskill = self._departure_skill(data, obs["is_first"].shape[1])
        departskill_start = self._departure_skill_start(
            data,
            obs["is_first"].shape[1],
            departskill,
        )
        self._reset_optimizer_state_on_create()
        metrics, (carry, entries, outs, mets) = self.main_opt(
            self.loss,
            carry,
            obs,
            prevact,
            departact,
            departskill,
            departskill_start,
            training=True,
            has_aux=True,
        )
        metrics.update(mets)
        if self.train_base_task_behavior:
            self.slowval.update()
        if self.task_opt is not None:
            task_metrics, task_mets = self.task_opt(
                self.task_loss,
                outs["repfeat"],
                entries[1],
                carry[1],
                obs,
                prevact,
                task_labels,
                training=True,
                has_aux=True,
            )
            metrics.update(task_metrics)
            metrics.update(task_mets)
            for slowval in self.task_heads["slowvals"].values():
                slowval.update()
        self.slowexplval.update()
        if self.rep_loss == "dreamerpro":
            self.dreamerpro.update_targets()
        if hasattr(self.curiosity, "post_update"):
            self.curiosity.post_update()
        outs = {}
        if self.config.replay_context:
            updates = elements.tree.flatdict(dict(
                stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
            B, T = obs["is_first"].shape
            assert all(x.shape[:2] == (B, T) for x in updates.values()), (
                (B, T), {k: v.shape for k, v in updates.items()})
            outs["replay"] = updates
        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, outs, metrics

    def _reset_optimizer_state_on_create(self, optimizers=None):
        if not nj.creating():
            return
        if optimizers is None:
            optimizers = [self.main_opt]
            if self.task_opt is not None:
                optimizers.append(self.task_opt)
        for optimizer in optimizers:
            for name in ("state", "adaptive"):
                state = optimizer._submodules.get(name)
                if state is not None:
                    state.treedef = None

    def _departure_actions(self, data, length):
        start = int(self.config.replay_context)
        return {key: data[key][:, start : start + length] for key in self.act_space}

    def _departure_skill(self, data, length):
        if not self._uses_skills:
            return None
        start = int(self.config.replay_context)
        if "skill" not in data:
            return self._zero_skill((data["is_first"].shape[0], length))
        return data["skill"][:, start : start + length]

    def _departure_skill_start(self, data, length, skill):
        if not self._uses_skills:
            return None
        start = int(self.config.replay_context)
        reset = data["is_first"][:, start : start + length]
        explicit = (
            data["skill_start"][:, start : start + length]
            if "skill_start" in data
            else None
        )
        if start <= 0 or "skill" not in data:
            return infer_skill_starts(skill, reset, explicit)
        previous_skill = data["skill"][:, start - 1 : start]
        skill_with_context = jnp.concatenate([previous_skill, skill], axis=1)
        reset_with_context = jnp.concatenate(
            [jnp.zeros_like(reset[:, :1]), reset],
            axis=1,
        )
        explicit_with_context = (
            jnp.concatenate([jnp.zeros_like(explicit[:, :1]), explicit], axis=1)
            if explicit is not None
            else None
        )
        return infer_skill_starts(
            skill_with_context,
            reset_with_context,
            explicit_with_context,
        )[:, 1:]

    def _repr_dim(self, source):
        if source == "feat":
            return int(self.dyn.deter + self.dyn.stoch * self.dyn.classes)
        if source == "deter":
            return int(self.dyn.deter)
        if source == "stoch":
            return int(self.dyn.stoch * self.dyn.classes)
        raise ValueError(f"Unsupported P2E representation source: {source}")

    def _repr(self, feat, source):
        if source == "feat":
            return self.feat2tensor(feat)
        if source == "deter":
            return nn.cast(feat["deter"])
        if source == "stoch":
            shape = feat["stoch"].shape
            return nn.cast(feat["stoch"].reshape((*shape[:-2], shape[-2] * shape[-1])))
        raise ValueError(f"Unsupported P2E representation source: {source}")

    def _target_repr(self, feat):
        source = self._plan_repr_source if self._disag_target == "same" else self._disag_target
        return self._repr(feat, source)

    def _action_tensor(self, action):
        if not self._uses_actions:
            return None
        return nn.DictConcat(self.act_space, 1)(action)

    def _process_action_particles(self, expinp, skill=None):
        if self._idm_type != "empowerment_process":
            return None, None
        particles = int(_get(self.p2e, "empowerment_particles", 16))
        samples = []
        logits = []
        for _ in range(particles):
            dist = self.expl_pol(self._expl_input(expinp, skill), 2)
            act = sample(dist)
            samples.append(nn.DictConcat(self.act_space, 1)(act))
            logits.append(jnp.zeros(expinp.shape[:-1], jnp.float32))
        return jnp.stack(samples, axis=-2), jnp.stack(logits, axis=-1)

    def _intrinsic_reward(
        self,
        reprs,
        action,
        update,
        sequence_layout="batch_major",
        skill=None,
        process_action_particles=None,
        action_logits=None,
        is_first=None,
        valid=None,
        policy_entropy=None,
        update_global_reward_norm=None,
    ):
        action = self._action_tensor(action) if isinstance(action, dict) else action
        arrival_aligned = False
        if self._idm_type == "cip":
            if not isinstance(reprs, dict):
                raise ValueError("CIP intrinsic reward expects an RSSM feature tree, not a flat representation.")
            reward = self.curiosity.compute_reward(
                self.dyn,
                reprs,
                action,
                update_reward_norm=update,
                sequence_layout=sequence_layout,
            )
        elif self._idm_type == "cig":
            reward = self.curiosity.compute_reward(
                reprs,
                action,
                update_reward_norm=update,
                sequence_layout=sequence_layout,
                is_first=is_first,
                valid=valid,
            )
        elif self._idm_type == "rnd":
            reward = self.curiosity.compute_reward(reprs, update_reward_norm=update)
        elif self._idm_type == "ngu":
            reward_repr = reprs
            reward_action = action
            transposed = reprs.ndim == 3 and sequence_layout == "batch_major"
            if transposed:
                reward_repr = jnp.swapaxes(reward_repr, 0, 1)
                reward_action = (
                    jnp.swapaxes(reward_action, 0, 1)
                    if reward_action is not None and reward_action.ndim == 3
                    else reward_action
                )
            reward = self.curiosity.compute_reward(
                reward_repr,
                reward_action,
                update_reward_norm=update,
            )
            if transposed:
                reward = jnp.swapaxes(reward, 0, 1)
        elif self._idm_type == "ctec":
            reward_repr = reprs
            reward_action = action
            transposed = reprs.ndim == 3 and sequence_layout == "batch_major"
            if transposed:
                reward_repr = jnp.swapaxes(reward_repr, 0, 1)
                reward_action = jnp.swapaxes(reward_action, 0, 1)
            reward = self.curiosity.compute_sequence_reward(
                reward_repr,
                reward_action,
                estimator=str(_get(self.p2e, "ctec_reward_estimator", "single")),
                normalize_truncation=bool(
                    _get(self.p2e, "ctec_reward_normalize_truncation", False)
                ),
                discount=float(_get(self.p2e, "ctec_gamma", 0.99)),
            )
            if transposed:
                reward = jnp.swapaxes(reward, 0, 1)
        elif self._idm_type == "diayn":
            reward = self.curiosity.compute_reward(reprs, skill=skill)
        elif self._idm_type == "vic":
            reward = self.curiosity.compute_reward(reprs, skill=skill)
        elif self._idm_type in _ARRIVAL_ALIGNED_TRANSITION_IDMS:
            if sequence_layout != "batch_major" or reprs.ndim != 3:
                raise ValueError(
                    f"{self._idm_type.upper()} transition rewards require "
                    "batch-major sequences [B,T,D]."
                )
            transition_action = action[:, :-1] if action is not None else None
            transition_skill = skill[:, :-1] if skill is not None else None
            transition_entropy = (
                policy_entropy[:, :-1] if policy_entropy is not None else None
            )
            reward = self.curiosity.compute_reward(
                reprs[:, :-1],
                transition_action,
                reprs[:, 1:],
                skill=transition_skill,
                policy_entropy=transition_entropy,
                update_reward_norm=update,
            )
            arrival_aligned = True
        elif self._idm_type == "csf":
            next_repr = padded_next_sequence(reprs)
            reward = self.curiosity.compute_reward(
                reprs,
                action,
                next_repr,
                skill=skill,
                update_reward_norm=update,
            )
        elif self._idm_type == "valor":
            reward = self.curiosity.compute_reward(reprs, skill=skill)
        elif self._idm_type == "empowerment_process":
            next_repr = padded_next_sequence(reprs)
            reward = self.curiosity.compute_reward(
                reprs,
                action,
                next_repr,
                action_particles=process_action_particles,
                action_logits=action_logits,
            )
        elif self._idm_type in self._empowerment_idm_types:
            next_repr = padded_next_sequence(reprs)
            reward = self.curiosity.compute_reward(reprs, action, next_repr)
        else:
            next_repr = padded_next_sequence(reprs)
            reward = self.curiosity.compute_reward(
                reprs,
                action,
                next_repr,
                update_reward_norm=update,
            )
        normalizer = getattr(self, "_global_intrinsic_reward_normalizer", None)
        if normalizer is not None:
            update_global = (
                update
                if update_global_reward_norm is None
                else bool(update_global_reward_norm)
            )
            reward = normalizer.normalize(
                reward,
                update=update_global,
                mask=valid,
            )
        reward = reward * self._expl_intr_scale
        if arrival_aligned:
            zero = jnp.zeros(
                (reward.shape[0], 1, *reward.shape[2:]), reward.dtype
            )
            reward = jnp.concatenate([zero, reward], axis=1)
        return reward

    def _curiosity_replay_loss(
        self,
        reprs,
        target_repr,
        action,
        reset,
        last,
        training,
        skill=None,
        skill_start=None,
    ):
        if self._idm_type == "cip":
            return self.curiosity.loss()
        mask = ~reset[:, 1:]
        obs = reprs[:, :-1]
        next_obs = reprs[:, 1:]
        act = action[:, :-1] if action is not None else None
        if self._idm_type in {"vic", "valor"}:
            trajectories, labels, segment_mask = fixed_horizon_skill_segments(
                reprs,
                skill,
                skill_start,
                reset,
                last,
                self._skill_horizon,
            )
            if self._idm_type == "vic":
                loss, metrics = self.curiosity.loss(
                    trajectories[:, 0],
                    trajectories[:, -1],
                    labels,
                    segment_mask,
                )
            else:
                loss, metrics = self.curiosity.loss(
                    trajectories,
                    labels,
                    segment_mask,
                )
            metrics["valid_segments"] = f32(segment_mask).sum()
            metrics["segment_reward"] = metrics[f"{self._idm_type}_segment_reward"]
            return loss, metrics
        if self._idm_type == "disagreement":
            return self.curiosity.loss(
                obs,
                act,
                target_repr[:, 1:],
                mask,
                bootstrap=self._disag_bootstrap and training,
            )
        if self._idm_type == "max":
            return self.curiosity.loss(
                obs,
                act,
                next_obs,
                mask,
                bootstrap=self._disag_bootstrap and training,
            )
        if self._idm_type == "cig":
            return self.curiosity.loss(
                obs,
                act,
                target_repr[:, 1:],
                mask,
                update_aleatoric=training,
            )
        if self._idm_type == "smm":
            return self.curiosity.loss(obs, mask)
        if self._idm_type in {"simple_curiosity", "icm"}:
            return self.curiosity.loss(obs, act, next_obs, mask)
        if self._idm_type == "rnd":
            return self.curiosity.loss(obs, mask)
        if self._idm_type == "apt":
            return self.curiosity.loss(
                next_obs,
                mask,
                train_phi=bool(_get(self.p2e, "apt_train_phi", True)),
                update_memory=training,
                update_reward_norm=training,
            )
        if self._idm_type == "re3":
            return self.curiosity.loss(
                obs,
                mask,
                update_memory=training,
                update_reward_norm=training,
            )
        if self._idm_type == "proto_rl":
            return self.curiosity.loss(
                obs,
                next_obs,
                mask,
                update_queue=training,
            )
        if self._idm_type == "ngu":
            if self._ngu_update_tail > 0:
                tail = min(self._ngu_update_tail, obs.shape[1])
                obs, next_obs, act, mask = obs[:, -tail:], next_obs[:, -tail:], act[:, -tail:], mask[:, -tail:]
            return self.curiosity.loss(obs, act, next_obs, mask)
        if self._idm_type == "gamma_progress":
            return self.curiosity.loss(obs, act, next_obs, mask)
        if self._idm_type == "ctec":
            return self.curiosity.sample_future_loss(
                reprs,
                action,
                reset,
                sample_size=self._ctec_replay_batch_size,
            )
        if self._idm_type == "diayn":
            obs, _, replay_skill, mask = _sample_skill_transition_batch(
                obs,
                next_obs,
                skill[:, :-1],
                mask,
                self._diayn_replay_batch_size,
            )
            return self.curiosity.loss(obs, replay_skill, mask)
        if self._idm_type in _ARRIVAL_ALIGNED_TRANSITION_IDMS:
            replay_batch_size = (
                self._dads_replay_batch_size
                if self._idm_type == "dads"
                else self._rdads_replay_batch_size
            )
            obs, next_obs, replay_skill, mask = _sample_skill_transition_batch(
                obs,
                next_obs,
                skill[:, :-1],
                mask,
                replay_batch_size,
            )
            loss, metrics = self.curiosity.loss(
                obs, next_obs, replay_skill, mask
            )
            metrics[f"{self._idm_type}_transition_batch"] = jnp.asarray(
                obs.shape[0], jnp.float32
            )
            metrics[f"{self._idm_type}_valid_transitions"] = f32(mask).sum()
            return loss, metrics
        if self._idm_type == "csf":
            obs, next_obs, replay_skill, mask = _sample_skill_transition_batch(
                obs,
                next_obs,
                skill[:, :-1],
                mask,
                self._csf_replay_batch_size,
            )
            return self.curiosity.loss(obs, next_obs, replay_skill, mask)
        if self._idm_type in self._empowerment_idm_types:
            return self.curiosity.loss(obs, act, next_obs, mask)
        raise NotImplementedError(f"Unsupported JAX curiosity IDM: {self._idm_type!r}")

    def loss(
        self,
        carry,
        obs,
        prevact,
        departact=None,
        departskill=None,
        departskill_start=None,
        training=False,
    ):
        enc_carry, dyn_carry, dec_carry = carry
        start_enc_carry, start_dyn_carry = enc_carry, dyn_carry
        reset = obs["is_first"]
        B, T = reset.shape
        losses = {}
        metrics = {}

        if self.rep_loss == "dreamer":
            reconK = min(int(_get(self.config, "recon_last", 0)) or T, T)
            recon_indices = self._sequence_selection(
                B,
                T,
                reconK,
                _get(self.config, "recon_start_strategy", "stratified"),
            )
        else:
            reconK, recon_indices = T, None

        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
        dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
            dyn_carry, tokens, prevact, reset, training
        )
        losses.update(los)
        metrics.update(mets)
        if departact is None:
            departact = prevact
        dec_carry, dec_entries, replosses, repmetrics = self._representation_loss(
            dec_carry,
            start_enc_carry,
            start_dyn_carry,
            obs,
            prevact,
            departact,
            tokens,
            repfeat,
            training,
            reconK,
            recon_indices,
        )
        for key, value in replosses.items():
            losses[key] = value if value.shape == (B, T) else jnp.ones((B, T), f32) * value
        metrics.update(repmetrics)
        inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
        if self.train_base_task_behavior:
            losses["rew"] = self.rew(inp, 2).loss(obs["reward"])
        con = f32(~obs["is_terminal"])
        if self.config.contdisc:
            con *= 1 - 1 / self.config.horizon
        losses["con"] = self.con(self.feat2tensor(repfeat), 2).loss(con)

        shapes = {k: v.shape for k, v in losses.items()}
        assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

        if departskill is None and self._uses_skills:
            departskill = self._zero_skill(reset.shape)
        idm_repr = (
            sg(repfeat)
            if self._idm_type == "cip"
            else _route_idm_repr_gradient(
                self._repr(repfeat, self._plan_repr_source),
                self._expl_grad_mode,
            )
        )
        if self._idm_type == "disagreement":
            idm_target = sg(self._target_repr(repfeat))
        elif self._idm_type == "cig":
            # CIG uses Plan2Explore's detached next-embedding regression target.
            idm_target = sg(self._repr(repfeat, self._plan_repr_source))
        else:
            idm_target = idm_repr
        idm_action = self._action_tensor(departact)
        idm_loss, idm_mets = self._curiosity_replay_loss(
            idm_repr,
            idm_target,
            idm_action,
            reset,
            obs["is_last"],
            training,
            departskill,
            departskill_start,
        )
        losses[self._idm_loss_key] = idm_loss
        metrics.update(prefix(idm_mets, self._idm_type))
        metrics[f"{self._idm_type}_loss"] = idm_loss
        update_replay_normalizers = self._idm_reward_norm_source in {
            "replay_only",
            "replay_and_imagination",
        }
        replay_reward_updates = (
            training
            and update_replay_normalizers
            and not bool(
                getattr(
                    self.curiosity,
                    "updates_reward_normalizer_in_update",
                    False,
                )
            )
        )
        replay_global_reward_updates = training and update_replay_normalizers
        skip_cip_replay = (
            self._idm_type == "cip"
            and not bool(_get(self.p2e, "cip_replay_metric", False))
            and not bool(_get(self.p2e, "cip_normalize", False))
        )
        # Process empowerment needs on-policy action particles (unavailable on the
        # single-sample replay actions), so its replay-reward diagnostic is skipped.
        segment_replay_metric = self._idm_type in {"vic", "valor"}
        skip_replay_metric = skip_cip_replay or self._idm_type == "empowerment_process"
        if skip_replay_metric:
            metrics["idm/replay_reward"] = jnp.zeros((), jnp.float32)
        elif segment_replay_metric:
            metrics["idm/replay_reward"] = (
                idm_mets["segment_reward"] * self._expl_intr_scale
            )
        else:
            arrival_aligned = self._idm_type in _ARRIVAL_ALIGNED_TRANSITION_IDMS
            if arrival_aligned:
                replay_repr = idm_repr
                replay_action = departact
                replay_skill = departskill
            else:
                replay_repr = (
                    jax.tree.map(lambda x: x[:, :-1], idm_repr)
                    if self._idm_type == "cip"
                    else idm_repr[:, :-1]
                )
                replay_action = {k: v[:, :-1] for k, v in departact.items()}
                replay_skill = (
                    departskill[:, :-1] if departskill is not None else None
                )
            replay_first = None
            replay_valid = ~reset[:, 1:] if arrival_aligned else None
            if self._idm_type == "cig":
                # Bound the quadratic kernel by the imagination horizon.
                # Reset conditioning at the window start and episode boundaries.
                window = min(int(self.config.imag_length), replay_repr.shape[1])
                start = replay_repr.shape[1] - window
                replay_repr = replay_repr[:, start:]
                replay_action = jax.tree.map(
                    lambda x: x[:, start:], replay_action
                )
                replay_first = reset[:, :-1][:, start:]
                replay_valid = (~reset[:, 1:])[:, start:]
            replay_reward = self._intrinsic_reward(
                replay_repr,
                replay_action,
                update=replay_reward_updates,
                skill=replay_skill,
                is_first=replay_first,
                valid=replay_valid,
                update_global_reward_norm=replay_global_reward_updates,
            )
            if arrival_aligned:
                replay_reward = replay_reward[:, 1:, 0]
                valid_reward = f32(replay_valid)
                metrics["idm/replay_reward"] = (
                    replay_reward * valid_reward
                ).sum() / jnp.maximum(valid_reward.sum(), 1.0)
            else:
                metrics["idm/replay_reward"] = replay_reward.mean()
        if self._global_intrinsic_reward_normalizer is not None:
            metrics["idm/reward_norm_mean"] = (
                self._global_intrinsic_reward_normalizer.mean.read()
            )
            metrics["idm/reward_norm_count"] = (
                self._global_intrinsic_reward_normalizer.count.read()
            )
        if self._idm_type == "disagreement":
            metrics["disagreement_loss"] = idm_loss
            metrics["disag/replay_reward"] = metrics["idm/replay_reward"]

        K = self._imag_start_count(T)
        H = self.config.imag_length
        imag_indices = self._sequence_selection(
            B, T, K, _get(self.config, "imag_start_strategy", "last")
        )
        starts, first = self._imag_start_data(
            repfeat, dyn_entries, dyn_carry, K, imag_indices
        )

        if self.train_base_task_behavior:
            policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
            _, imgfeat, imgprevact = self.dyn.imagine(
                starts, policyfn, H, training
            )
            imgfeat = concat(
                [sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1
            )
            lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
            lastact = jax.tree.map(lambda x: x[:, None], lastact)
            imgact = concat([imgprevact, lastact], 1)
            inp = self.feat2tensor(imgfeat)
            los, _, mets = imag_loss(
                imgact,
                self.rew(inp, 2).pred(),
                self.con(inp, 2).prob(1),
                self.pol(inp, 2),
                self.val(inp, 2),
                self.slowval(inp, 2),
                self.retnorm,
                self.valnorm,
                self.advnorm,
                update=training,
                contdisc=self.config.contdisc,
                horizon=self.config.horizon,
                **self.config.imag_loss,
            )
            losses.update(
                {k: v.mean(1).reshape((B, K)) for k, v in los.items()}
            )
            metrics.update(mets)

        cip_override = bool(self._idm_type == "cip" and self._cip_imag_last)
        expK = (
            min(self._cip_imag_last, T)
            if cip_override
            else K
        )
        if not cip_override:
            exp_indices = imag_indices
            exp_starts, exp_first = starts, first
        else:
            exp_indices = self._sequence_selection(
                B,
                T,
                expK,
                _get(self.p2e, "cip_imag_start_strategy", "last"),
            )
            exp_starts, exp_first = self._imag_start_data(
                repfeat, dyn_entries, dyn_carry, expK, exp_indices
            )
        exp_skill_base = self._sample_skill_vectors((B * expK,)) if self._uses_skills else None

        def expl_policyfn(feat):
            tensor = self.feat2tensor(feat)
            return sample(self.expl_pol(self._expl_input(tensor, exp_skill_base), 1))

        _, expfeat, expprevact = self.dyn.imagine(exp_starts, expl_policyfn, H, training)
        expfeat = concat([sg(exp_first, skip=self.config.ac_grads), sg(expfeat)], 1)
        explastact = expl_policyfn(jax.tree.map(lambda x: x[:, -1], expfeat))
        explastact = jax.tree.map(lambda x: x[:, None], explastact)
        expact = concat([expprevact, explastact], 1)
        expinp = self.feat2tensor(expfeat)
        exp_skill = (
            jnp.broadcast_to(exp_skill_base[:, None], (B * expK, H + 1, self._skill_event_dim))
            if self._uses_skills
            else None
        )
        exp_policy = self.expl_pol(self._expl_input(expinp, exp_skill), 2)
        exp_policy_entropy = sum(
            distribution.entropy() for distribution in exp_policy.values()
        )
        while exp_policy_entropy.ndim < expinp.ndim:
            exp_policy_entropy = exp_policy_entropy[..., None]
        process_particles, process_logits = self._process_action_particles(expinp, exp_skill)
        reward_repr = (
            sg(expfeat)
            if self._idm_type == "cip"
            else sg(self._repr(expfeat, self._plan_repr_source))
        )
        reward_action = expact
        if self._idm_type == "cig":
            # CIG scores outgoing transitions (s_t, a_t) -> s_{t+1}, whereas
            # Dreamer rewards are arrival-aligned and lambda_return ignores
            # column zero. Score exactly H transitions and pad a leading zero.
            reward_repr, reward_action = _cig_outgoing_transitions(
                reward_repr, reward_action
            )
        update_imagination_normalizers = (
            training
            and self._idm_reward_norm_source == "replay_and_imagination"
        )
        intr_rew = self._intrinsic_reward(
            reward_repr,
            reward_action,
            update=update_imagination_normalizers,
            skill=exp_skill,
            process_action_particles=process_particles,
            action_logits=process_logits,
            policy_entropy=(
                exp_policy_entropy
                if self._idm_type in _ARRIVAL_ALIGNED_TRANSITION_IDMS
                else None
            ),
            update_global_reward_norm=update_imagination_normalizers,
        )
        if self._idm_type == "cig":
            intr_rew = _arrival_align_cig_reward(intr_rew)
        intr_rew = intr_rew[..., 0]
        if self._expl_extr_scale:
            if not self.train_base_task_behavior:
                raise ValueError(
                    "expl_extr_scale requires train_base_task_behavior=true."
                )
            extr_rew = self.rew(expinp, 2).pred()
            exp_rew = intr_rew + self._expl_extr_scale * extr_rew
        else:
            extr_rew = jnp.zeros_like(intr_rew)
            exp_rew = intr_rew
        exp_con = self.con(expinp, 2).prob(1)
        exp_los, _, exp_mets = imag_loss(
            expact,
            exp_rew,
            exp_con,
            exp_policy,
            self.expl_val(expinp, 2),
            self.slowexplval(expinp, 2),
            self.expl_retnorm,
            self.expl_valnorm,
            self.expl_advnorm,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss,
        )
        losses["expl_policy"] = exp_los["policy"].mean(1).reshape((B, expK))
        losses["expl_value"] = exp_los["value"].mean(1).reshape((B, expK))
        metrics.update(prefix(exp_mets, "expl_imag"))
        metrics["expl/intr_reward"] = (
            intr_rew[:, 1:].mean()
            if self._idm_type == "cig"
            or self._idm_type in _ARRIVAL_ALIGNED_TRANSITION_IDMS
            else intr_rew.mean()
        )
        metrics["expl/extr_reward"] = extr_rew.mean()

        if self.train_base_task_behavior and self.config.repval_loss:
            last, term, rew = [obs[k] for k in ("is_last", "is_terminal", "reward")]
            los, _, mets = self._replay_value_loss(
                repfeat,
                last,
                term,
                rew,
                self.val,
                self.slowval,
                self.valnorm,
                K,
                training,
                repval_grad=self.config.repval_grad,
            )
            losses.update(los)
            metrics.update(prefix(mets, "reploss"))

        metrics["imag/start_count"] = jnp.asarray(K, f32)
        metrics["imag/start_fraction"] = jnp.asarray(K / T, f32)
        metrics["expl_imag/start_count"] = jnp.asarray(expK, f32)
        metrics["expl_imag/start_fraction"] = jnp.asarray(expK / T, f32)
        if self.rep_loss == "dreamer":
            metrics["recon/start_count"] = jnp.asarray(reconK, f32)
            metrics["recon/start_fraction"] = jnp.asarray(reconK / T, f32)

        assert set(losses.keys()) == set(self.scales.keys()), (sorted(losses), sorted(self.scales))
        metrics.update({f"loss/{k}": v.mean() for k, v in losses.items()})
        loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

        carry = (enc_carry, dyn_carry, dec_carry)
        entries = (enc_entries, dyn_entries, dec_entries)
        outs = {"tokens": tokens, "repfeat": repfeat, "losses": losses}
        return loss, (carry, entries, outs, metrics)
