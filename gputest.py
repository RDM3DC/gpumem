import torch
import cupy as cp
import numpy as np

# Verify GPUs
print("Available GPUs:", torch.cuda.device_count())

# Select devices
device_compute = torch.device('cuda:0')  # Primary GPU
device_memory = torch.device('cuda:1')   # Secondary GPU (memory expansion)

# Step 1: Generate large tensor (e.g., 1 GB)
tensor_size = (256, 1024, 1024)  # ~1 GB float32 tensor
tensor_gpu0 = torch.randn(tensor_size, device=device_compute)
print("Tensor allocated on GPU0")

# Step 2: Compress tensor on GPU0 using CuPy
def compress_tensor(tensor):
    tensor_cp = cp.asarray(tensor.detach())
    compressed = cp.compress(cp.abs(tensor_cp) > 0.01, tensor_cp)
    return compressed

compressed_gpu0 = compress_tensor(tensor_gpu0)
compressed_size_mb = compressed_gpu0.nbytes / (1024 ** 2)
print(f"Compressed size on GPU0: {compressed_size_mb:.2f} MB")

# Step 3: Transfer compressed data to GPU1
compressed_gpu1 = cp.asarray(compressed_gpu0, order='C')
compressed_gpu1 = compressed_gpu1.copy(order='C')
with cp.cuda.Device(1):
    compressed_gpu1_gpu = cp.asarray(compressed_gpu1)
print("Compressed data transferred to GPU1 via NVLink")

# Step 4: Decompress tensor on GPU1
def decompress_tensor(compressed, original_shape):
    decompressed = cp.zeros(np.prod(original_shape), dtype=compressed.dtype)
    mask = cp.abs(decompressed) > 0.01
    decompressed[mask] = compressed
    return decompressed.reshape(original_shape)

tensor_gpu1_decompressed = decompress_tensor(compressed_gpu1_gpu, tensor_size)
print("Decompressed data on GPU1")

# Step 5: Validate Data Integrity (GPU0 vs GPU1 decompressed)
tensor_gpu1_torch = torch.as_tensor(tensor_gpu1_decompressed.get(), device=device_compute)
difference = torch.norm(tensor_gpu0 - tensor_gpu1_torch)
print(f"Data integrity check - Norm difference: {difference:.4f}")
