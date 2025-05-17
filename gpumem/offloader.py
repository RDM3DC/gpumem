from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import cupy as cp
import torch


@dataclass
class OffloadedTensor:
    """Representation of a compressed tensor stored on the secondary GPU."""

    compressed: cp.ndarray
    original_shape: tuple[int, ...]
    original_size_mb: float
    compressed_size_mb: float


@dataclass
class LayerOffloader:
    """Manage compression and offloading of model layers between GPUs."""

    device_compute: torch.device | int = 0
    device_storage: int = 1
    compression_factor: int = 4
    offloaded_layers: Dict[str, OffloadedTensor] = field(default_factory=dict)
    total_saved_mb: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.device_compute, torch.device):
            self.device_compute = torch.device(f"cuda:{self.device_compute}")

    def compress_and_offload(self, layer_name: str, tensor: torch.Tensor) -> float:
        """Compress ``tensor`` and move it to ``device_storage``."""
        original_size_mb = tensor.nelement() * tensor.element_size() / (1024 ** 2)

        tensor_cp = cp.asarray(tensor.detach())
        original_shape = tensor_cp.shape

        reshaped = tensor_cp.reshape(-1, self.compression_factor)
        means = reshaped.mean(axis=1)
        diffs = reshaped - means[:, cp.newaxis]
        compressed = cp.hstack([means[:, cp.newaxis], diffs[:, 1:]])

        with cp.cuda.Device(self.device_storage):
            compressed_storage = cp.asarray(compressed, order="C").copy()

        compressed_size_mb = compressed_storage.nbytes / (1024 ** 2)
        memory_saved = original_size_mb - compressed_size_mb

        self.offloaded_layers[layer_name] = OffloadedTensor(
            compressed=compressed_storage,
            original_shape=tuple(int(x) for x in original_shape),
            original_size_mb=original_size_mb,
            compressed_size_mb=compressed_size_mb,
        )
        self.total_saved_mb += memory_saved
        return memory_saved

    def reload_layer(
        self, layer_name: str, target_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decompress a previously offloaded layer."""
        if layer_name not in self.offloaded_layers:
            raise ValueError(f"Layer '{layer_name}' was not previously offloaded")

        layer_data = self.offloaded_layers[layer_name]
        compressed = cp.ascontiguousarray(layer_data.compressed)
        original_shape = layer_data.original_shape

        with cp.cuda.Device(self.device_storage):
            means = compressed[:, 0]
            diffs = compressed[:, 1:]
            first_diffs = -diffs.sum(axis=1, keepdims=True)
            full_diffs = cp.hstack([first_diffs, diffs])
            restored = full_diffs + means[:, cp.newaxis]
            decompressed = cp.ascontiguousarray(restored.reshape(original_shape))
            decompressed_np = cp.asnumpy(decompressed)

        decompressed_torch = torch.tensor(
            decompressed_np, device=self.device_compute, dtype=torch.float16
        )

        if target_tensor is not None:
            target_tensor.copy_(decompressed_torch)
            return target_tensor
        return decompressed_torch

    def offload_model_layer(self, model: torch.nn.Module, layer_path: str) -> float:
        """Offload ``layer_path`` from ``model``."""
        parts = layer_path.split(".")
        layer: torch.nn.Module = model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)

        if hasattr(layer, "weight") and layer.weight is not None:
            saved = self.compress_and_offload(layer_path, layer.weight.data)
            layer.weight.data = torch.zeros(1, device=self.device_compute)
            return saved
        raise ValueError(f"Layer at {layer_path} does not have accessible weights")

    def reload_model_layer(self, model: torch.nn.Module, layer_path: str) -> None:
        """Reload an offloaded ``layer_path`` into ``model``."""
        parts = layer_path.split(".")
        layer: torch.nn.Module = model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)

        if hasattr(layer, "weight"):
            decompressed = self.reload_layer(layer_path)
            layer.weight.data = decompressed
        else:
            raise ValueError(f"Layer at {layer_path} does not have accessible weights")

    def get_memory_stats(self) -> dict:
        """Return detailed statistics about offloaded layers."""
        return {
            "total_layers_offloaded": len(self.offloaded_layers),
            "total_saved_mb": self.total_saved_mb,
            "layers": {
                name: {
                    "original_size_mb": data.original_size_mb,
                    "compressed_size_mb": data.compressed_size_mb,
                    "compression_ratio": data.original_size_mb / data.compressed_size_mb,
                }
                for name, data in self.offloaded_layers.items()
            },
        }

    def get_memory_savings(self) -> dict:
        """Return summary statistics for memory savings."""
        total_orig = sum(d.original_size_mb for d in self.offloaded_layers.values())
        total_comp = sum(d.compressed_size_mb for d in self.offloaded_layers.values())
        return {
            "num_layers_offloaded": len(self.offloaded_layers),
            "total_saved_mb": self.total_saved_mb,
            "compression_ratio": total_orig / (total_comp + 1e-10),
        }

    def offload_multiple_layers(
        self, model: torch.nn.Module, layer_paths: list[str]
    ) -> float:
        """Offload several layers in ``layer_paths``."""
        total_saved = 0.0
        for path in layer_paths:
            try:
                saved = self.offload_model_layer(model, path)
                total_saved += saved
                print(f"Offloaded {path}: {saved:.2f} MB saved")
            except Exception as e:  # noqa: BLE001
                print(f"Error offloading {path}: {e}")
        return total_saved

    def reload_all_layers(self, model: torch.nn.Module) -> None:
        """Reload all previously offloaded layers into ``model``."""
        for name in list(self.offloaded_layers.keys()):
            try:
                self.reload_model_layer(model, name)
                print(f"Reloaded {name}")
            except Exception as e:  # noqa: BLE001
                print(f"Error reloading {name}: {e}")

