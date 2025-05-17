import torch
import cupy as cp
import numpy as np


def compress_tensor(tensor: torch.Tensor) -> cp.ndarray:
    """Compress a PyTorch tensor on the GPU using a simple mask.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor located on a CUDA device.

    Returns
    -------
    cupy.ndarray
        Compressed representation stored as a CuPy array.
    """
    tensor_cp = cp.asarray(tensor.detach())
    compressed = cp.compress(cp.abs(tensor_cp) > 0.01, tensor_cp)
    return compressed


def decompress_tensor(compressed: cp.ndarray, original_shape) -> cp.ndarray:
    """Decompress a tensor previously compressed with ``compress_tensor``."""
    decompressed = cp.zeros(np.prod(original_shape), dtype=compressed.dtype)
    mask = cp.abs(decompressed) > 0.01
    decompressed[mask] = compressed
    return decompressed.reshape(original_shape)


def run_test() -> None:
    """Run the GPU memory transfer validation test."""
    print("Available GPUs:", torch.cuda.device_count())

    device_compute = torch.device('cuda:0')

    tensor_size = (256, 1024, 1024)  # ~1 GB float32 tensor
    tensor_gpu0 = torch.randn(tensor_size, device=device_compute)
    print("Tensor allocated on GPU0")

    compressed_gpu0 = compress_tensor(tensor_gpu0)
    compressed_size_mb = compressed_gpu0.nbytes / (1024 ** 2)
    print(f"Compressed size on GPU0: {compressed_size_mb:.2f} MB")

    compressed_gpu1 = cp.asarray(compressed_gpu0, order='C').copy(order='C')
    with cp.cuda.Device(1):
        compressed_gpu1_gpu = cp.asarray(compressed_gpu1)
    print("Compressed data transferred to GPU1 via NVLink")

    tensor_gpu1_decompressed = decompress_tensor(compressed_gpu1_gpu, tensor_size)
    print("Decompressed data on GPU1")

    tensor_gpu1_torch = torch.as_tensor(tensor_gpu1_decompressed.get(), device=device_compute)
    difference = torch.norm(tensor_gpu0 - tensor_gpu1_torch)
    print(f"Data integrity check - Norm difference: {difference:.4f}")


if __name__ == "__main__":
    run_test()
