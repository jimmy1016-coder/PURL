import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

from modules.thin_resnet import ResNetSE


class Encoder(nn.Module):
    """Encoder base class."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, F), where B is the batch size,
                        L is the sequence length, and F is the feature size (e. g. 80 mels).

        Returns:
            Tensor: Output of shape (B, L_1, F_1), where B is the batch size,
                    L_1 is the sequence length, and F_1 is the feature size.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class Projector(nn.Module):
    """Projector base class."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, 1, F), where B is the batch size
                        and F is the feature size.

        Returns:
            Tensor: Output of shape (B, 1, F_p), where B is the batch size and F_P
            denotes the projector feature size.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class ECAPAEncoder(Encoder):
    def __init__(
        self,
        input_size: int,
        channels: list[int],
        kernel_sizes: list[int],
        dilations: list[int],
        attention_channels: int,
        lin_neurons: int,
        res2net_scale: int,
        se_channels: int,
    ):
        super().__init__()
        self.model = ECAPA_TDNN(
            input_size=input_size,
            channels=channels,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            attention_channels=attention_channels,
            lin_neurons=lin_neurons,
            res2net_scale=res2net_scale,
            se_channels=se_channels,
        )

    def forward(self, x):
        return self.model(x)


class ResNetEncoder(Encoder):
    def __init__(
        self,
        layers: list[int],
        num_filters: list[int],
        nOut: int,
        encoder_type: str = 'SAP'
    ):
        super().__init__()
        self.model = ResNetSE(
            layers=layers,
            num_filters=num_filters,
            nOut=nOut,
            encoder_type=encoder_type
        )

    def forward(self, x):
        return self.model(x.unsqueeze(1))


class IdentityProjector(Projector):
    def forward(self, x):
        return x
