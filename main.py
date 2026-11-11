"""Runtime hardware diagnostics for PRISM perception execution."""

from perception.detection import resolve_device


def get_runtime_summary() -> dict[str, object]:
    """Inspect Torch CUDA availability without crashing on CPU-only machines."""
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device = resolve_device(cuda_available)
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "cuda_available": cuda_available,
        "device": device,
        "device_name": device_name,
    }


def main() -> None:
    """Print resolved execution device summary."""
    summary = get_runtime_summary()
    print(f"CUDA available: {summary['cuda_available']}")
    print(f"Active device: {summary['device']} ({summary['device_name']})")


if __name__ == "__main__":
    main()
