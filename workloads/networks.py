"""Example neural network workloads."""

from dataclasses import dataclass

from .cnn_layers import Conv2DLayer, LinearLayer


@dataclass
class SimpleMLP:
    """A simple Multi-Layer Perceptron (e.g., for MNIST)."""
    layers: list[LinearLayer]

    @classmethod
    def get_mnist_mlp(cls) -> "SimpleMLP":
        return cls([
            LinearLayer(in_features=784, out_features=128),
            LinearLayer(in_features=128, out_features=64),
            LinearLayer(in_features=64, out_features=10),
        ])

    def total_macs(self) -> int:
        return sum(layer.num_macs() for layer in self.layers)


@dataclass
class SimpleCNN:
    """A simple Convolutional Neural Network."""
    conv_layers: list[Conv2DLayer]
    linear_layers: list[LinearLayer]

    @classmethod
    def get_simple_cnn(cls) -> "SimpleCNN":
        return cls(
            conv_layers=[
                Conv2DLayer(in_channels=1, out_channels=16, kernel_size=3, input_spatial_size=28),
                Conv2DLayer(in_channels=16, out_channels=32, kernel_size=3, input_spatial_size=14),
            ],
            linear_layers=[
                LinearLayer(in_features=32 * 5 * 5, out_features=128),
                LinearLayer(in_features=128, out_features=10),
            ]
        )

    def total_macs(self) -> int:
        conv_macs = sum(layer.num_macs() for layer in self.conv_layers)
        linear_macs = sum(layer.num_macs() for layer in self.linear_layers)
        return conv_macs + linear_macs
