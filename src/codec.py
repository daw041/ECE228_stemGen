"""EnCodec wrapper for waveform <-> discrete token conversion.

The wrapper keeps the model/trainer/decoder codebook count aligned.  Missing
RVQ levels are not padded with token 0 because token 0 is a real codeword, not
silence.
"""
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio


class AudioCodec:
    """Wrapper around pretrained EnCodec with configurable codebook count."""

    BANDWIDTH_BY_CODEBOOKS_24KHZ = {
        1: 1.5,  # first codebook only; sliced from the 1.5 kbps setting
        2: 1.5,
        4: 3.0,
        8: 6.0,
        16: 12.0,
        32: 24.0,
    }

    def __init__(self, bandwidth: float = None, num_codebooks: int = 4, device: str = "cpu"):
        self.device = device
        self.num_codebooks = int(num_codebooks)
        self.model = EncodecModel.encodec_model_24khz()
        self.bandwidth = bandwidth or self._infer_bandwidth(self.num_codebooks)
        self.model.set_target_bandwidth(self.bandwidth)
        self.model = self.model.to(device)
        self.model.eval()
        self.sample_rate = self.model.sample_rate

    def _infer_bandwidth(self, num_codebooks: int) -> float:
        """Choose the smallest 24kHz EnCodec bandwidth covering num_codebooks."""
        for n_q, bandwidth in sorted(self.BANDWIDTH_BY_CODEBOOKS_24KHZ.items()):
            if num_codebooks <= n_q:
                return bandwidth
        return self.BANDWIDTH_BY_CODEBOOKS_24KHZ[max(self.BANDWIDTH_BY_CODEBOOKS_24KHZ)]

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, input_sr: int = None) -> torch.Tensor:
        """Encode waveform to discrete codec tokens.

        Returns:
            tokens: [batch, num_codebooks, time_frames] discrete token indices
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        src_sr = input_sr or self.sample_rate
        if src_sr != self.model.sample_rate:
            waveform = waveform.cpu()
            waveform = convert_audio(
                waveform, src_sr, self.model.sample_rate, self.model.channels
            )
        waveform = waveform.to(self.device)
        encoded = self.model.encode(waveform)
        codes = encoded[0][0]  # [batch, n_q, time]
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        if codes.shape[1] < self.num_codebooks:
            raise ValueError(
                f"Codec produced {codes.shape[1]} codebooks at bandwidth {self.bandwidth}, "
                f"but {self.num_codebooks} were requested."
            )
        return codes[:, :self.num_codebooks, :]

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode discrete codec tokens back to waveform.

        Args:
            tokens: [batch, num_codebooks, time_frames] token indices
        """
        tokens = tokens.to(self.device)
        if tokens.shape[1] != self.num_codebooks:
            raise ValueError(
                f"Decode received {tokens.shape[1]} codebooks, expected {self.num_codebooks}."
            )
        frames = [(tokens, None)]
        decoded = self.model.decode(frames)
        return decoded.squeeze(0)


def load_codec(num_codebooks: int = 4, device: str = "cpu", bandwidth: float = None) -> AudioCodec:
    """Load EnCodec.

    Keyword arguments are preferred.  A legacy positional call such as
    ``load_codec("cuda")`` is accepted to avoid a silent device/codebook swap.
    """
    if isinstance(num_codebooks, str):
        device = num_codebooks
        num_codebooks = 4
    return AudioCodec(bandwidth=bandwidth, num_codebooks=num_codebooks, device=device)
