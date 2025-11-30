from torch import Tensor, nn

from ..config import NetworkConfig
from .acoustic_predictor import AcousticPredictor, create_acoustic_predictor
from .vocoder import Vocoder


class Predictor(nn.Module):
    def __init__(
        self,
        acoustic_predictor: AcousticPredictor,
        vocoder: Vocoder,
        frame_size: int,
        sampling_rate: int,
    ):
        super().__init__()
        self.acoustic_predictor = acoustic_predictor
        self.vocoder = vocoder
        self.frame_size = frame_size
        self.sampling_rate = sampling_rate

    def forward(  # noqa: D102
        self,
        *,
        f0_list: list[Tensor],  # [(fL,)]
        phoneme_list: list[Tensor],  # [(fL,)]
        speaker_id: Tensor,  # (B,)
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        spec1_list, spec2_list = self.acoustic_predictor.forward_list(
            f0_list=f0_list,
            phoneme_list=phoneme_list,
            speaker_id=speaker_id,
        )
        wave_list = self.vocoder.forward_list(spec2_list)
        return spec1_list, spec2_list, wave_list


def create_predictor(config: NetworkConfig) -> Predictor:
    acoustic_predictor = create_acoustic_predictor(config.acoustic)
    vocoder = Vocoder(
        input_channels=acoustic_predictor.output_size,
        upsample_rates=config.vocoder.upsample_rates,
        upsample_kernel_sizes=config.vocoder.upsample_kernel_sizes,
        upsample_initial_channel=config.vocoder.upsample_initial_channel,
        resblock=config.vocoder.resblock,
        resblock_kernel_sizes=config.vocoder.resblock_kernel_sizes,
        resblock_dilation_sizes=config.vocoder.resblock_dilation_sizes,
    )
    return Predictor(
        acoustic_predictor=acoustic_predictor,
        vocoder=vocoder,
        frame_size=config.frame_size,
        sampling_rate=config.sampling_rate,
    )
