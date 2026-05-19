"""Neural network layer definitions."""

from dataclasses import dataclass


@dataclass
class LinearLayer:
    """Fully connected layer."""
    in_features: int
    out_features: int

    def num_macs(self) -> int:
        return self.in_features * self.out_features


@dataclass
class Conv2DLayer:
    """2D Convolutional layer."""
    in_channels: int
    out_channels: int
    kernel_size: int
    input_spatial_size: int  # Assuming square input (H=W)

    def num_macs(self) -> int:
        # Simplified MAC count assuming stride 1, no padding, or just a rough estimate
        # spatial output size = input_spatial_size - kernel_size + 1
        out_spatial = self.input_spatial_size - self.kernel_size + 1
        if out_spatial <= 0:
            out_spatial = 1
        pixels = out_spatial * out_spatial
        macs_per_pixel = self.in_channels * self.kernel_size * self.kernel_size * self.out_channels
        return pixels * macs_per_pixel
