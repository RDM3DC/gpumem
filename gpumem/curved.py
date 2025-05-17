"""Curved memory compression utilities."""

from __future__ import annotations

import torch
import cupy as cp


def curve_memory_compress(tensor: torch.Tensor, factor: int = 4):
    """Compress tensor using an adaptive curve approach.

    Parameters
    ----------
    tensor:
        Input tensor located on CUDA device.
    factor:
        Segment length used for compression.

    Returns
    -------
    tuple[cupy.ndarray, tuple[int, ...]]
        Compressed representation and original shape.
    """
    tensor_cp = cp.asarray(tensor.detach())
    original_shape = tensor_cp.shape
    reshaped = tensor_cp.reshape(-1, factor)
    means = reshaped.mean(axis=1)
    diffs = reshaped - means[:, cp.newaxis]
    compressed = cp.hstack([means[:, cp.newaxis], diffs[:, 1:]])
    return compressed, original_shape


def curve_memory_decompress(
    compressed: cp.ndarray,
    factor: int = 4,
    original_shape: tuple[int, ...] | None = None,
):
    """Restore tensor compressed with ``curve_memory_compress``."""
    means = compressed[:, 0]
    diffs = compressed[:, 1:]
    first_diffs = -diffs.sum(axis=1, keepdims=True)
    full_diffs = cp.hstack([first_diffs, diffs])
    restored = full_diffs + means[:, cp.newaxis]
    if original_shape is None:
        raise ValueError("original_shape must be provided")
    return restored.reshape(original_shape)


if __name__ == "__main__":
    device_compute = torch.device("cuda:0")
    device_memory = torch.device("cuda:1")

    original_shape = (128, 512, 512)
    tensor_gpu0 = torch.randn(original_shape, device=device_compute)

    compressed_gpu0, original_shape = curve_memory_compress(tensor_gpu0)
    compressed_size_mb = compressed_gpu0.nbytes / (1024 ** 2)
    print(f"Curved compression size on GPU0: {compressed_size_mb:.2f} MB")

    with cp.cuda.Device(1):
        compressed_gpu1 = cp.asarray(compressed_gpu0, order="C").copy()
    print("Compressed tensor transferred to GPU1 via NVLink")

    tensor_gpu1_restored = curve_memory_decompress(
        compressed_gpu1, factor=4, original_shape=original_shape
    )
    print("Tensor decompressed on GPU1.")

    tensor_gpu1_restored_torch = torch.as_tensor(
        tensor_gpu1_restored.get(), device=device_compute
    )
    error_norm = torch.norm(tensor_gpu0 - tensor_gpu1_restored_torch)
    print(f"Integrity Check - Norm difference: {error_norm.item():.6f}")

