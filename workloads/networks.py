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
        """A simple Convolutional Neural Network.
        
        Note on Dimension Propagation:
        Under pure conv mapping, a 28x28 input with kernel_size=3 outputs a 26x26 map.
        The second layer expects 14x14 input. This transition implicitly assumes a 2x2
        MaxPooling layer with padding (or a stride/padding combination) that reduces 26x26 -> 14x14.
        Likewise, the second layer (14x14, kernel_size=3) outputs 12x12. The first linear layer
        expects 32 * 5 * 5 (5x5 spatial resolution), which implicitly assumes another MaxPooling
        step (or pooling + padding) reducing 12x12 -> 5x5.
        """
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

    @classmethod
    def get_medmnist_cnn(cls, in_channels: int = 3, num_classes: int = 10) -> "SimpleCNN":
        """A slightly larger CNN for 28x28x3 MedMNIST datasets.
        
        Note on Dimension Propagation:
        Like get_simple_cnn(), this network implicitly assumes 2x2 MaxPooling and padding
        steps to transition spatial feature maps from 26x26 -> 14x14 and from 12x12 -> 5x5.
        """
        return cls(
            conv_layers=[
                Conv2DLayer(in_channels=in_channels, out_channels=32, kernel_size=3, input_spatial_size=28),
                Conv2DLayer(in_channels=32, out_channels=64, kernel_size=3, input_spatial_size=14),
                # Outputs 12x12 -> pooled to 5x5
            ],
            linear_layers=[
                LinearLayer(in_features=64 * 5 * 5, out_features=128),
                LinearLayer(in_features=128, out_features=num_classes),
            ]
        )

    def total_macs(self) -> int:
        conv_macs = sum(layer.num_macs() for layer in self.conv_layers)
        linear_macs = sum(layer.num_macs() for layer in self.linear_layers)
        return conv_macs + linear_macs
