import torch.nn as nn

from speechbrain.lobes.features import Fbank


class FeatureExtractor(nn.Module):
    def forward(self, x):
        """
        Returns data in format [channels, sequence_length, features]
        """
        return NotImplementedError


class SBFbank(FeatureExtractor):
    """Wrapper for speechbrain's Fbank (retuns log-mel features)."""

    def __init__(
        self,
        sample_rate: int,
        f_min: int,
        f_max: int,
        n_fft: int,
        n_mels: int,
        win_length: int,
        hop_length: int,
        postprocessor: nn.Module = nn.Identity(),
    ):
        super().__init__()

        # convert win_length and hop_length to miliseconds for Fbank class
        win_length = int(win_length / sample_rate * 1000)
        hop_length = int(hop_length / sample_rate * 1000)

        self.fbank = Fbank(
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_fft=n_fft,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length,
        )

        self.postprocessor = postprocessor

    def forward(self, x):
        features = self.fbank(x)
        features = self.postprocessor(features)

        return features
    