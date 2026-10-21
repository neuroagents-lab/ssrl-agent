from __future__ import annotations

from collections.abc import Mapping

import numpy as np


FEATURE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def getcfg(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def encoder_enabled(model_config) -> bool:
    encoder = getcfg(model_config, "encoder", None)
    return str(getcfg(encoder, "typ", "simple")).lower() == "dinov2"


def feature_dim(model_name: str = "dinov2_vits14", feature: str = "cls", override=0) -> int:
    if override:
        return int(override)
    name = str(model_name)
    if name not in FEATURE_DIMS:
        raise ValueError(
            f"Unknown DINOv2 model_name {name!r}; configure "
            "model.encoder.dinov2.feature_dim explicitly for custom models."
        )
    feat = str(feature).lower()
    base = FEATURE_DIMS[name]
    if feat in {"cls", "mean_patch"}:
        return base
    if feat in {"cls_mean", "cls_plus_mean"}:
        return 2 * base
    raise ValueError(
        "model.encoder.dinov2.feature must be one of cls, mean_patch, "
        f"or cls_mean; got {feature!r}."
    )


def config_from_model(model_config) -> dict:
    encoder = getcfg(model_config, "encoder", None)
    dinov2 = getcfg(encoder, "dinov2", None)
    cfg = {
        "model_name": str(getcfg(dinov2, "model_name", "dinov2_vits14")),
        "source": str(getcfg(dinov2, "source", "torchhub")),
        "image_key": str(getcfg(dinov2, "image_key", "image")),
        "feature_key": str(getcfg(dinov2, "feature_key", "dinov2")),
        "feature": str(getcfg(dinov2, "feature", "cls")),
        "image_size": int(getcfg(dinov2, "image_size", 224)),
        "device": str(getcfg(dinov2, "device", "auto")),
        "batch_size": int(getcfg(dinov2, "batch_size", 64)),
        "keep_image_log": bool(getcfg(dinov2, "keep_image_log", True)),
        "feature_dim": int(getcfg(dinov2, "feature_dim", 0) or 0),
    }
    cfg["output_dim"] = feature_dim(
        cfg["model_name"], cfg["feature"], override=cfg["feature_dim"]
    )
    return cfg


def may_use_cuda(model_config) -> bool:
    if not encoder_enabled(model_config):
        return False
    device = config_from_model(model_config)["device"].lower()
    return device == "auto" or device.startswith("cuda")


class DINOv2FeatureWrapper:

    def __init__(self, env, elements, config, extractor=None):
        self.env = env
        self._elements = elements
        self._config = dict(config)
        self._image_key = self._config["image_key"]
        self._feature_key = self._config["feature_key"]
        self._keep_image_log = bool(self._config["keep_image_log"])
        self._feature_dim = int(self._config["output_dim"])
        if self._feature_key == self._image_key:
            raise ValueError("DINOv2 feature_key must differ from image_key.")
        if self._image_key not in env.obs_space:
            raise ValueError(
                "DINOv2 encoder requires image observations: missing "
                f"{self._image_key!r} in obs_space {sorted(env.obs_space)}."
            )
        image_space = env.obs_space[self._image_key]
        if len(image_space.shape) != 3:
            raise ValueError(
                f"DINOv2 image key {self._image_key!r} must have HWC shape; "
                f"got {image_space.shape}."
            )
        self._extractor = extractor
        self._obs_space = None

    @property
    def obs_space(self):
        if self._obs_space is None:
            spaces = dict(self.env.obs_space)
            image_space = spaces.pop(self._image_key)
            spaces[self._feature_key] = self._elements.Space(
                np.float32, (self._feature_dim,), -np.inf, np.inf
            )
            if self._keep_image_log:
                spaces[f"log/{self._image_key}"] = image_space
            self._obs_space = spaces
        return self._obs_space

    @property
    def act_space(self):
        return self.env.act_space

    def __len__(self):
        return len(self.env)

    def step(self, action):
        obs = dict(self.env.step(action))
        if self._image_key not in obs:
            raise KeyError(
                f"DINOv2 encoder expected observation key {self._image_key!r}; "
                f"got {sorted(obs)}."
            )
        image = np.asarray(obs.pop(self._image_key))
        obs[self._feature_key] = self._encode(image)
        if self._keep_image_log:
            obs[f"log/{self._image_key}"] = image
        return obs

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def _encode(self, image):
        extractor = self._extractor
        if extractor is None:
            extractor = DINOv2FeatureExtractor(**self._config)
            self._extractor = extractor
        features = np.asarray(extractor(image), dtype=np.float32)
        return features.reshape((*image.shape[:-3], self._feature_dim))


class DINOv2FeatureExtractor:

    def __init__(
        self,
        model_name="dinov2_vits14",
        source="torchhub",
        image_key="image",
        feature_key="dinov2",
        feature="cls",
        image_size=224,
        device="auto",
        batch_size=64,
        keep_image_log=True,
        feature_dim=0,
        output_dim=None,
    ):
        del image_key, feature_key, keep_image_log, feature_dim, output_dim
        self.model_name = str(model_name)
        self.source = str(source).lower()
        self.feature = str(feature).lower()
        self.image_size = int(image_size)
        self.device = str(device).lower()
        self.batch_size = max(1, int(batch_size))
        self._torch = None
        self._model = None
        self._resolved_device = None

    def __call__(self, images):
        self._load()
        single = np.asarray(images).ndim == 3
        batch = self._prepare_images(images)
        chunks = []
        torch = self._torch
        with torch.no_grad():
            for start in range(0, len(batch), self.batch_size):
                inp = batch[start : start + self.batch_size].to(self._resolved_device)
                out = self._forward(inp)
                chunks.append(out.detach().cpu().numpy().astype(np.float32))
        features = np.concatenate(chunks, axis=0)
        return features[0] if single else features

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise RuntimeError(
                "DINOv2 encoder requires optional PyTorch dependencies. "
                "Install them with `python -m pip install -r requirements.txt`."
            ) from exc
        if self.source != "torchhub":
            raise ValueError(
                "Only model.encoder.dinov2.source=torchhub is supported in v1; "
                f"got {self.source!r}."
            )
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2", self.model_name, trust_repo=True
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINOv2 from torchhub. Ensure network/cache access "
                f"for facebookresearch/dinov2 and model {self.model_name!r}."
            ) from exc
        model.eval().to(device)
        for param in model.parameters():
            param.requires_grad_(False)
        self._torch = torch
        self._F = F
        self._model = model
        self._resolved_device = device

    def _prepare_images(self, images):
        arr = np.asarray(images)
        if arr.ndim == 3:
            arr = arr[None]
        if arr.ndim != 4:
            raise ValueError(f"DINOv2 images must have shape HWC or NHWC; got {arr.shape}.")
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.shape[-1] != 3:
            raise ValueError(f"DINOv2 images must have 1 or 3 channels; got {arr.shape}.")
        if np.issubdtype(arr.dtype, np.floating):
            max_value = np.nanmax(arr) if arr.size else 0.0
            arr = arr * 255.0 if max_value <= 1.0 else arr
        arr = np.clip(arr, 0, 255).astype(np.float32) / 255.0
        torch = self._torch
        x = torch.from_numpy(arr).permute(0, 3, 1, 2)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = self._F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def _forward(self, x):
        if hasattr(self._model, "forward_features"):
            out = self._model.forward_features(x)
            if isinstance(out, Mapping):
                cls = out.get("x_norm_clstoken")
                patch = out.get("x_norm_patchtokens")
                if self.feature == "cls" and cls is not None:
                    return cls
                if self.feature == "mean_patch" and patch is not None:
                    return patch.mean(dim=1)
                if (
                    self.feature in {"cls_mean", "cls_plus_mean"}
                    and cls is not None
                    and patch is not None
                ):
                    return self._torch.cat([cls, patch.mean(dim=1)], dim=-1)
        out = self._model(x)
        if self.feature == "cls":
            return out
        raise RuntimeError(
            f"DINOv2 model {self.model_name!r} did not expose patch tokens for "
            f"feature={self.feature!r}."
        )
