import ctypes
import os
import pathlib
import warnings


def _configure_madrona_cuda_heap():
    heap_size = os.environ.get("MADRONA_MWGPU_DEVICE_HEAP_SIZE")
    if not heap_size or os.environ.get("MADRONA_DISABLE_CUDA_HEAP_SIZE") == "1":
        return
    try:
        requested = int(heap_size)
    except ValueError:
        warnings.warn(
            f"Ignoring invalid MADRONA_MWGPU_DEVICE_HEAP_SIZE={heap_size!r}.",
            RuntimeWarning,
        )
        return
    if requested <= 0:
        return

    cudart = None
    for name in ("libcudart.so.12", "libcudart.so"):
        try:
            cudart = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if cudart is None:
        return

    cuda_limit_malloc_heap_size = 2
    cudart.cudaDeviceSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    cudart.cudaDeviceSetLimit.restype = ctypes.c_int

    candidates = [requested, 1 << 30, 1 << 29, 1 << 28, 1 << 27]
    seen = set()
    for size in candidates:
        if size in seen:
            continue
        seen.add(size)
        err = cudart.cudaDeviceSetLimit(cuda_limit_malloc_heap_size, size)
        if err == 0:
            os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"] = str(size)
            os.environ["MADRONA_DISABLE_CUDA_HEAP_SIZE"] = "1"
            return

    warnings.warn(
        "Could not preconfigure the Madrona CUDA device heap; disabling "
        "Madrona's late cudaDeviceSetLimit() call so the run can proceed with "
        "the CUDA default heap.",
        RuntimeWarning,
    )
    os.environ["MADRONA_DISABLE_CUDA_HEAP_SIZE"] = "1"


_configure_madrona_cuda_heap()

import hydra

from agent.launcher import run

warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    logdir = pathlib.Path(str(config.logdir)).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    run(config)


if __name__ == "__main__":
    main()
