from typing import TypedDict

import torch
from torch import Tensor, nn

from yukarin_sosoav.dataset import DatasetOutput

from .generator import Generator, GeneratorOutput
from .utility.audio_utility import log_mel_spectrogram


class EvaluatorOutput(TypedDict):
    value: Tensor
    spec_loss: Tensor
    wave_spec_loss: Tensor
    data_num: int


class Evaluator(nn.Module):
    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    def forward(self, data: DatasetOutput) -> EvaluatorOutput:
        output_list: list[GeneratorOutput] = self.generator(
            f0_list=data["f0"],
            phoneme_list=data["phoneme"],
            speaker_id=(
                torch.stack(data["speaker_id"])
                if data["speaker_id"] is not None
                else None
            ),
        )

        spec_all = torch.cat([output["spec"] for output in output_list], dim=0)
        target_spec_all = torch.cat(data["spec"], dim=0)

        spec_loss = torch.abs(spec_all - target_spec_all).mean()

        num_mels = target_spec_all.size(-1)
        frame_size = self.generator.predictor.frame_size
        sampling_rate = self.generator.predictor.sampling_rate

        pred_wave_spec_list = [
            log_mel_spectrogram(
                output["wave"].unsqueeze(0),
                frame_size=frame_size,
                spec_size=num_mels,
                sampling_rate=sampling_rate,
            )
            .squeeze(0)
            .transpose(0, 1)
            for output in output_list
        ]

        pred_wave_spec = torch.cat(pred_wave_spec_list, dim=0)
        target_wave_spec = torch.cat(data["spec"], dim=0)
        wave_spec_loss = torch.abs(pred_wave_spec - target_wave_spec).mean()

        return EvaluatorOutput(
            value=wave_spec_loss,
            spec_loss=spec_loss,
            wave_spec_loss=wave_spec_loss,
            data_num=len(data["spec"]),
        )
