from pathlib import Path

import librosa
import numpy
import soundfile

from .sampling_data import SamplingData


class Wave:
    def __init__(self, wave: numpy.ndarray, sampling_rate: int):
        self.wave = wave
        self.sampling_rate = sampling_rate

    @staticmethod
    def load(
        path: Path,
        sampling_rate: int | None = None,
        dtype: numpy.dtype | type = numpy.float32,
    ):
        if path.suffix in {".npy", ".npz"}:
            d = SamplingData.load(path)
            d.array = numpy.squeeze(d.array)
            if sampling_rate is not None:
                d.array = librosa.resample(
                    d.array,
                    orig_sr=d.rate,
                    target_sr=sampling_rate,
                    res_type="soxr_vhq",
                )
                d.rate = sampling_rate
            return Wave(wave=d.array, sampling_rate=int(d.rate))
        else:
            wave, loaded_sr = librosa.load(
                path, sr=sampling_rate, dtype=dtype, res_type="soxr_vhq"
            )
            return Wave(wave=wave, sampling_rate=int(loaded_sr))

    def save(self, path: Path):
        soundfile.write(str(path), data=self.wave, samplerate=self.sampling_rate)
