import torch


def mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def resolve_torch_device(preferred: str = "auto") -> str:
    preferred = preferred.lower()
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"

    if preferred == "mps":
        if not mps_available():
            raise RuntimeError("MPS was requested but is not available.")
        return "mps"

    if preferred == "cpu":
        return "cpu"

    raise ValueError(
        f"Unsupported device '{preferred}'. Use one of: auto, cuda, mps, cpu."
    )


def describe_torch_backends() -> dict:
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "mps_available": mps_available(),
    }
