from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

import numpy
import torch
from torch import ScriptModule, Tensor, nn

from .config import Config
from .network.predictor import Predictor, create_predictor


class GeneratorOutput(TypedDict):
    spec: Tensor
    wave: Tensor


def to_tensor(array: Tensor | numpy.ndarray | Any):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    else:
        return array


class Generator(nn.Module):
    def __init__(
        self,
        config: Config,
        predictor: Predictor | ScriptModule | str | Path | BytesIO,
        use_gpu: bool,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, (str, Path, BytesIO)):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def forward(
        self,
        f0_list: list[numpy.ndarray | torch.Tensor],
        phoneme_list: list[numpy.ndarray | torch.Tensor],
        speaker_id: numpy.ndarray | torch.Tensor | None = None,
    ):
        f0_list = [to_tensor(f0).to(self.device) for f0 in f0_list]
        phoneme_list = [to_tensor(phoneme).to(self.device) for phoneme in phoneme_list]
        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id).to(self.device)

        batch_size = len(f0_list)
        if speaker_id is None:
            device = f0_list[0].device
            speaker_id_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            speaker_id_tensor = to_tensor(speaker_id).to(f0_list[0].device).long()

        with torch.inference_mode():
            _, spec_list, wave_list = self.predictor(
                f0_list=f0_list,
                phoneme_list=phoneme_list,
                speaker_id=speaker_id_tensor,
            )

        return [
            GeneratorOutput(spec=spec, wave=wave)
            for spec, wave in zip(spec_list, wave_list, strict=True)
        ]
