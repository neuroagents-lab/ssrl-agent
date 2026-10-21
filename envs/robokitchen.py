"""Native MuJoCo 3 port of the LEXA RoboKitchen benchmark.

The policy observation is the 64 x 64 RGB image.  Reward labels and metrics
are emitted in reserved namespaces so the experiment driver can keep them out
of policy and representation inputs while retaining the labels in replay.

The model assets originate from LEXA commit
674d56ea7454f7a3804a7ffb987930f64d6133e6.  See
``assets/robokitchen/PROVENANCE.md`` for licensing and local XML changes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

import gymnasium as gym
import numpy as np


PORT_VERSION = "lexa-674d56e-mujoco3-v1"
ASSET_MANIFEST_SHA256 = "3d746c79d77faa1963223fea78a6359e0d7ec501678d6dcd57bb96683b1c00f9"
FRAME_SKIP = 16
DEFAULT_ACTION_REPEAT = 2
IMAGE_SIZE = (64, 64)
CAMERA_SETTINGS: Mapping[str, object] = MappingProxyType(
    {
        "distance": 1.86,
        "lookat": (-0.3, 0.5, 2.0),
        "azimuth": 90.0,
        "elevation": -60.0,
    }
)
MODEL_SIGNATURE: Mapping[str, int] = MappingProxyType(
    {
        "nq": 30,
        "nv": 29,
        "nu": 9,
        "nbody": 46,
        "nmocap": 1,
        "neq": 6,
        "nmesh": 43,
        "ntex": 14,
    }
)
_VISUAL_STATE_FIELDS = (
    "cam_xmat",
    "cam_xpos",
    "flexvert_xpos",
    "geom_xmat",
    "geom_xpos",
    "light_xdir",
    "light_xpos",
    "site_xmat",
    "site_xpos",
    "wrap_xpos",
    "xmat",
    "xpos",
    "xquat",
)


@dataclass(frozen=True)
class TaskSpec:
    """Reward definition for one atomic RoboKitchen task."""

    name: str
    indices: tuple[int, ...]
    target: tuple[float, ...]


_TASK_SPECS = (
    TaskSpec("bottom_burner", (9, 10), (-0.88, -0.01)),
    TaskSpec("light_switch", (17, 18), (-0.69, -0.05)),
    TaskSpec("slide_cabinet", (19,), (0.37,)),
    TaskSpec("hinge_cabinet", (20, 21), (0.0, 0.5)),
    TaskSpec("microwave", (22,), (-0.5,)),
    TaskSpec("kettle", (23, 24, 25), (-0.23, 0.75, 1.62)),
)
TASKS = tuple(spec.name for spec in _TASK_SPECS)
TASK_REGISTRY: Mapping[str, TaskSpec] = MappingProxyType(
    {spec.name: spec for spec in _TASK_SPECS}
)
LABEL_KEYS = tuple(f"label/reward/{task}" for task in TASKS)

_REGISTRY_PAYLOAD = json.dumps(
    [asdict(spec) for spec in _TASK_SPECS],
    sort_keys=True,
    separators=(",", ":"),
)
TASK_REGISTRY_HASH = hashlib.sha256(_REGISTRY_PAYLOAD.encode("utf-8")).hexdigest()


# Exact reset used by LEXA's
# KitchenMicrowaveKettleLightTopLeftBurnerV0 environment.
INIT_QPOS = np.array(
    [
        1.48388023e-01,
        -1.76848573e00,
        1.84390296e00,
        -2.47685760e00,
        2.60252026e-01,
        7.12533105e-01,
        1.59515394e00,
        4.79267505e-02,
        3.71350919e-02,
        -2.66279850e-04,
        -5.18043486e-05,
        3.12877220e-05,
        -4.51199853e-05,
        -3.90842156e-06,
        -4.22629655e-05,
        6.28065475e-05,
        4.04984708e-05,
        4.62730939e-04,
        -2.26906415e-04,
        -4.65501369e-04,
        -6.44129196e-03,
        -1.77048263e-03,
        1.08009684e-03,
        -2.69397440e-01,
        3.50383255e-01,
        1.61944683e00,
        1.00618764e00,
        4.06395120e-03,
        -6.62095997e-03,
        -2.68278933e-04,
    ],
    dtype=np.float64,
)


def _validate_qpos(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape != INIT_QPOS.shape:
        raise ValueError(f"Expected qpos shape {INIT_QPOS.shape}, got {qpos.shape}.")
    return qpos


def task_distance(qpos: np.ndarray, task: str) -> float:
    """Return the L2 goal distance for an atomic task."""

    qpos = _validate_qpos(qpos)
    try:
        spec = TASK_REGISTRY[str(task)]
    except KeyError as exc:
        raise ValueError(f"Unknown RoboKitchen task {task!r}; expected one of {TASKS}.") from exc
    indices = np.asarray(spec.indices, dtype=np.int32)
    target = np.asarray(spec.target, dtype=np.float64)
    return float(np.linalg.norm(qpos[indices] - target))


def task_reward(qpos: np.ndarray, task: str) -> float:
    """Return LEXA's dense reward (negative task goal distance)."""

    return -task_distance(qpos, task)


def all_task_rewards(qpos: np.ndarray) -> dict[str, float]:
    """Compute all six dense labels from one simulator state."""

    qpos = _validate_qpos(qpos)
    return {task: task_reward(qpos, task) for task in TASKS}


def _same_threshold_side(value: float, goal: float, threshold: float) -> bool:
    # Keep LEXA's strict comparisons.  A value exactly on the threshold is
    # unsuccessful even if the goal is on one side of it.
    return bool(
        (value < threshold and goal < threshold)
        or (value > threshold and goal > threshold)
    )


def task_success(qpos: np.ndarray, task: str) -> tuple[int, int]:
    """Return ``(task_relevant, strict_all_objects)`` LEXA success values.

    ``strict_all_objects`` includes the benchmark's top burner even though it
    is not one of the six atomic reward tasks.  The kettle check intentionally
    uses qpos[23:25], preserving the reference implementation's two-coordinate
    success test while the dense reward uses all three coordinates.
    """

    qpos = _validate_qpos(qpos)
    try:
        spec = TASK_REGISTRY[str(task)]
    except KeyError as exc:
        raise ValueError(f"Unknown RoboKitchen task {task!r}; expected one of {TASKS}.") from exc

    goal_qpos = INIT_QPOS.copy()
    goal_qpos[np.asarray(spec.indices, dtype=np.int32)] = np.asarray(
        spec.target, dtype=np.float64
    )
    per_object = {
        "bottom_burner": _same_threshold_side(qpos[9], goal_qpos[9], -0.38),
        "top_burner": _same_threshold_side(qpos[13], goal_qpos[13], -0.38),
        "light_switch": _same_threshold_side(qpos[17], goal_qpos[17], -0.25),
        "slide_cabinet": abs(qpos[19] - goal_qpos[19]) < 0.1,
        "hinge_cabinet": abs(qpos[21] - goal_qpos[21]) < 0.2,
        "microwave": abs(qpos[22] - goal_qpos[22]) < 0.2,
        "kettle": np.linalg.norm(qpos[23:25] - goal_qpos[23:25]) < 0.2,
    }
    return int(per_object[spec.name]), int(all(per_object.values()))


def _quat_to_zyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Match ``numpy-quaternion.as_euler_angles`` used by LEXA."""

    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"Invalid end-effector quaternion {quat}.")
    w, x, y, z = quat
    ratio = np.clip((w * w + z * z) / norm, 0.0, 1.0)
    return np.array(
        [
            np.arctan2(z, w) + np.arctan2(-x, y),
            2.0 * np.arccos(np.sqrt(ratio)),
            np.arctan2(z, w) - np.arctan2(-x, y),
        ],
        dtype=np.float64,
    )


def _zyz_to_quat(angles: np.ndarray) -> np.ndarray:
    """Match ``numpy-quaternion.from_euler_angles`` in MuJoCo wxyz order."""

    alpha, beta, gamma = np.asarray(angles, dtype=np.float64)
    return np.array(
        [
            np.cos(beta / 2.0) * np.cos((alpha + gamma) / 2.0),
            -np.sin(beta / 2.0) * np.sin((alpha - gamma) / 2.0),
            np.sin(beta / 2.0) * np.cos((alpha - gamma) / 2.0),
            np.cos(beta / 2.0) * np.sin((alpha + gamma) / 2.0),
        ],
        dtype=np.float64,
    )


class RoboKitchen(gym.Env):
    """LEXA RoboKitchen with native MuJoCo and image-only policy input."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 31}

    def __init__(
        self,
        task: str,
        action_repeat: int = DEFAULT_ACTION_REPEAT,
        size: tuple[int, int] = IMAGE_SIZE,
        seed: int = 0,
    ):
        super().__init__()
        task = str(task)
        if task not in (*TASKS, "explore"):
            raise ValueError(
                f"Unknown RoboKitchen task {task!r}; expected one of {TASKS} or 'explore'."
            )
        size = tuple(int(value) for value in size)
        if size != IMAGE_SIZE:
            raise ValueError(
                f"RoboKitchen benchmark observations are fixed at {IMAGE_SIZE}, got {size}."
            )
        action_repeat = int(action_repeat)
        if action_repeat < 1:
            raise ValueError(f"action_repeat must be positive, got {action_repeat}.")

        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - depends on optional runtime.
            raise ImportError(
                "RoboKitchen requires the native 'mujoco' package (version 3.x)."
            ) from exc

        self._mujoco = mujoco
        self.task = task
        self.action_repeat = action_repeat
        self.size = size
        self._seed = int(seed)
        self._renderer = None
        self._step_count = 0

        model_path = (
            pathlib.Path(__file__).resolve().parent
            / "assets"
            / "robokitchen"
            / "adept_envs"
            / "franka"
            / "assets"
            / "franka_kitchen_ee_ctrl.xml"
        )
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        actual_signature = {
            name: int(getattr(self.model, name)) for name in MODEL_SIGNATURE
        }
        if actual_signature != dict(MODEL_SIGNATURE):
            raise RuntimeError(
                "Unexpected RoboKitchen model signature: "
                f"expected {dict(MODEL_SIGNATURE)}, got {actual_signature}."
            )
        self.data = mujoco.MjData(self.model)
        self._link7_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "panda0_link7"
        )
        if self._link7_id < 0:
            raise RuntimeError("RoboKitchen model has no panda0_link7 body.")
        self._state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self._state_size = mujoco.mj_stateSize(self.model, self._state_spec)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        spaces: dict[str, gym.Space] = {
            "image": gym.spaces.Box(0, 255, shape=self.size + (3,), dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, shape=(), dtype=np.bool_),
            "is_last": gym.spaces.Box(0, 1, shape=(), dtype=np.bool_),
            "is_terminal": gym.spaces.Box(0, 1, shape=(), dtype=np.bool_),
        }
        for key in LABEL_KEYS:
            spaces[key] = gym.spaces.Box(-np.inf, 0.0, shape=(), dtype=np.float32)
        spaces.update(
            {
                "log/success_task_relevant": gym.spaces.Box(
                    0.0, 1.0, shape=(), dtype=np.float32
                ),
                "log/success_all_objects": gym.spaces.Box(
                    0.0, 1.0, shape=(), dtype=np.float32
                ),
                "log/distance": gym.spaces.Box(
                    0.0, np.inf, shape=(), dtype=np.float32
                ),
            }
        )
        for task_name in TASKS:
            spaces[f"log/success_task_relevant/{task_name}"] = gym.spaces.Box(
                0.0, 1.0, shape=(), dtype=np.float32
            )
            spaces[f"log/success_all_objects/{task_name}"] = gym.spaces.Box(
                0.0, 1.0, shape=(), dtype=np.float32
            )
            spaces[f"log/distance/{task_name}"] = gym.spaces.Box(
                0.0, np.inf, shape=(), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)
        self.reward_range = (-np.inf, 0.0)

        # Seed Gymnasium's generator now, while still honoring reset(seed=...).
        super().reset(seed=self._seed)

    @property
    def unwrapped_qpos(self) -> np.ndarray:
        """A copy of simulator qpos for parity checks and diagnostics."""

        return self.data.qpos.copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._seed = int(seed)
        super().reset(seed=self._seed if seed is not None else None)
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = INIT_QPOS
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        self._sync_mocap_to_link7()
        self._mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        labels = {task: 0.0 for task in TASKS}
        return self._observation(labels, is_first=True), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}.")
        if not np.isfinite(action).all():
            raise ValueError(f"RoboKitchen action must be finite, got {action}.")
        action = np.clip(action, -1.0, 1.0)

        labels = {task: 0.0 for task in TASKS}
        for _ in range(self.action_repeat):
            self._physics_step(action)
            instant = all_task_rewards(self.data.qpos)
            for task_name, value in instant.items():
                labels[task_name] += value

        self._step_count += 1
        reward = 0.0 if self.task == "explore" else labels[self.task]
        obs = self._observation(labels, is_first=False)
        info = {"is_terminal": False}
        return obs, np.float32(reward), False, False, info

    def _physics_step(self, action: np.ndarray) -> None:
        # This is a direct native-MuJoCo translation of LEXA's end-effector
        # branch.  The desired orientation is fixed for the 16 physics frames,
        # while mocap deltas are recomputed relative to link7 each frame.
        rotation = _quat_to_zyz(self.data.xquat[self._link7_id]) - action[3:6]
        for _ in range(FRAME_SKIP):
            target_quat = _zyz_to_quat(rotation)
            quat_delta = target_quat - self.data.xquat[self._link7_id]
            self._sync_mocap_to_link7()
            self.data.mocap_pos[0] += 0.05 * action[:3]
            self.data.mocap_quat[0] += 0.05 * quat_delta
            self.data.ctrl[7] = action[-1]
            self.data.ctrl[8] = -action[-1]
            self._mujoco.mj_step(self.model, self.data)

    def _sync_mocap_to_link7(self) -> None:
        self.data.mocap_pos[0] = self.data.xpos[self._link7_id]
        self.data.mocap_quat[0] = self.data.xquat[self._link7_id]

    def _observation(
        self, labels: Mapping[str, float], *, is_first: bool
    ) -> dict[str, np.ndarray | np.generic]:
        obs: dict[str, np.ndarray | np.generic] = {
            "image": self.render(),
            "is_first": np.asarray(is_first, dtype=np.bool_),
            "is_last": np.asarray(False, dtype=np.bool_),
            "is_terminal": np.asarray(False, dtype=np.bool_),
        }
        for task_name in TASKS:
            obs[f"label/reward/{task_name}"] = np.asarray(
                labels[task_name], dtype=np.float32
            )

        metrics = {}
        for task_name in TASKS:
            distance = task_distance(self.data.qpos, task_name)
            relevant, strict = task_success(self.data.qpos, task_name)
            metrics[task_name] = (distance, relevant, strict)
            obs[f"log/distance/{task_name}"] = np.asarray(distance, dtype=np.float32)
            obs[f"log/success_task_relevant/{task_name}"] = np.asarray(
                relevant, dtype=np.float32
            )
            obs[f"log/success_all_objects/{task_name}"] = np.asarray(
                strict, dtype=np.float32
            )

        if self.task == "explore":
            active = np.mean(np.asarray(tuple(metrics.values()), dtype=np.float64), axis=0)
            distance, relevant, strict = active
        else:
            distance, relevant, strict = metrics[self.task]
        obs["log/distance"] = np.asarray(distance, dtype=np.float32)
        obs["log/success_task_relevant"] = np.asarray(relevant, dtype=np.float32)
        obs["log/success_all_objects"] = np.asarray(strict, dtype=np.float32)
        return obs

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("RoboKitchen supports only render mode 'rgb_array'.")
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                self.model, height=self.size[0], width=self.size[1]
            )
        camera = self._mujoco.MjvCamera()
        self._mujoco.mjv_defaultCamera(camera)
        camera.type = self._mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = float(CAMERA_SETTINGS["distance"])
        camera.lookat[:] = CAMERA_SETTINGS["lookat"]
        camera.azimuth = float(CAMERA_SETTINGS["azimuth"])
        camera.elevation = float(CAMERA_SETTINGS["elevation"])
        self._renderer.update_scene(self.data, camera=camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def get_state(self) -> dict[str, object]:
        """Return a complete integration state suitable for deterministic restore."""

        state = np.empty(self._state_size, dtype=np.float64)
        self._mujoco.mj_getState(self.model, self.data, state, self._state_spec)
        rng_state = None
        if getattr(self, "np_random", None) is not None:
            rng_state = copy.deepcopy(self.np_random.bit_generator.state)
        return {
            "version": PORT_VERSION,
            "task": self.task,
            "step_count": self._step_count,
            "mj_state": state,
            # mj_step's derived kinematic arrays describe the state just
            # before Euler integration. Preserve the compact visual closure
            # so an immediate render is exact; this also retains the lagged
            # link7 pose consumed by the reference controller's next step.
            "visual_state": {
                name: getattr(self.data, name).copy()
                for name in _VISUAL_STATE_FIELDS
            },
            "rng_state": rng_state,
        }

    def set_state(self, state: Mapping[str, object]) -> None:
        """Restore a state produced by :meth:`get_state`."""

        if state.get("version") != PORT_VERSION:
            raise ValueError(
                f"RoboKitchen state version {state.get('version')!r} does not match "
                f"{PORT_VERSION!r}."
            )
        if state.get("task") != self.task:
            raise ValueError(
                f"Cannot restore task {state.get('task')!r} into {self.task!r}."
            )
        mj_state = np.asarray(state["mj_state"], dtype=np.float64)
        if mj_state.shape != (self._state_size,):
            raise ValueError(
                f"Expected MuJoCo state shape {(self._state_size,)}, got {mj_state.shape}."
            )
        self._mujoco.mj_setState(self.model, self.data, mj_state, self._state_spec)
        self._mujoco.mj_forward(self.model, self.data)
        visual_state = state.get("visual_state")
        if not isinstance(visual_state, Mapping):
            raise ValueError("RoboKitchen state has no visual_state mapping.")
        if set(visual_state) != set(_VISUAL_STATE_FIELDS):
            raise ValueError(
                "Unexpected RoboKitchen visual state fields: expected "
                f"{sorted(_VISUAL_STATE_FIELDS)}, got {sorted(visual_state)}."
            )
        for name in _VISUAL_STATE_FIELDS:
            target = getattr(self.data, name)
            value = np.asarray(visual_state[name], dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(
                    f"Expected {name} restore shape {target.shape}, got "
                    f"{value.shape}."
                )
            target[:] = value
        self._step_count = int(state.get("step_count", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.np_random.bit_generator.state = copy.deepcopy(rng_state)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


__all__ = [
    "ASSET_MANIFEST_SHA256",
    "CAMERA_SETTINGS",
    "DEFAULT_ACTION_REPEAT",
    "FRAME_SKIP",
    "IMAGE_SIZE",
    "INIT_QPOS",
    "LABEL_KEYS",
    "MODEL_SIGNATURE",
    "PORT_VERSION",
    "RoboKitchen",
    "TASK_REGISTRY",
    "TASK_REGISTRY_HASH",
    "TASKS",
    "TaskSpec",
    "all_task_rewards",
    "task_distance",
    "task_reward",
    "task_success",
]
