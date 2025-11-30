import librosa
import torch
import torch.nn.functional as F
from torch import Tensor

_mel_basis_cache: dict[
    tuple[torch.device, torch.dtype, int, int, int, float, float], Tensor
] = {}
_hann_window_cache: dict[tuple[torch.device, torch.dtype, int], Tensor] = {}


def _next_power_of_two(value: int) -> int:
    power = 1
    while power < value:
        power <<= 1
    return power


def log_mel_spectrogram(
    wave: Tensor,
    *,
    frame_size: int,
    spec_size: int,
    sampling_rate: int | None = None,
    mel_fmin: float = 0.0,
    mel_fmax: float | None = None,
) -> Tensor:
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    elif wave.dim() != 2:
        raise ValueError("wave must be 1D or 2D tensor")

    if sampling_rate is None:
        sampling_rate = frame_size * 100
    if mel_fmax is None:
        mel_fmax = sampling_rate / 2

    hop_size = frame_size
    win_size = frame_size * 4
    n_fft = _next_power_of_two(win_size)
    win_size = min(win_size, n_fft)

    device = wave.device
    dtype = wave.dtype
    mel_key = (device, dtype, n_fft, spec_size, sampling_rate, mel_fmin, mel_fmax)
    hann_key = (device, dtype, win_size)

    if mel_key not in _mel_basis_cache:
        mel_basis = torch.from_numpy(
            librosa.filters.mel(
                sr=sampling_rate,
                n_fft=n_fft,
                n_mels=spec_size,
                fmin=mel_fmin,
                fmax=mel_fmax,
            )
        )
        _mel_basis_cache[mel_key] = mel_basis.to(device=device, dtype=dtype)

    if hann_key not in _hann_window_cache:
        _hann_window_cache[hann_key] = torch.hann_window(win_size).to(
            device=device, dtype=dtype
        )

    mel_basis = _mel_basis_cache[mel_key]
    hann_window = _hann_window_cache[hann_key]

    pad = int((n_fft - hop_size) // 2)
    wave = F.pad(wave.unsqueeze(1), (pad, pad), mode="reflect")
    wave = wave.squeeze(1)

    spec = torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec_magnitude = torch.sqrt(torch.clamp(spec.pow(2).sum(-1), min=1e-9))
    mel_spec = torch.matmul(mel_basis.unsqueeze(0), spec_magnitude)
    return torch.log(torch.clamp(mel_spec, min=1e-5))
