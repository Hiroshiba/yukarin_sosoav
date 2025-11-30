from typing import Any, TypedDict

import torch
from torch import Tensor, nn
from torch.nn.functional import l1_loss

from yukarin_sosoav.dataset import DatasetOutput

from .config import ModelConfig
from .network.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from .network.predictor import Predictor
from .utility.audio_utility import log_mel_spectrogram


class GeneratorModelOutput(TypedDict):
    loss: Tensor
    spec_loss1: Tensor
    spec_loss2: Tensor
    wave_spec_loss: Tensor
    adversarial_loss: Tensor
    feature_matching_loss: Tensor
    data_num: int


class DiscriminatorModelOutput(TypedDict):
    loss: Tensor
    mpd_loss: Tensor
    msd_loss: Tensor
    data_num: int


def reduce_result(results: list[dict[str, Any]]):
    if not results:
        raise ValueError("results is empty")
    result: dict[str, Any] = {}
    sum_data_num = sum(int(r["data_num"]) for r in results)
    keys = set(results[0].keys()) - {"data_num"}
    for key in keys:
        values = [r[key] * int(r["data_num"]) for r in results]
        if isinstance(values[0], Tensor):
            stacked = torch.stack([v.to(dtype=torch.float32) for v in values])
            result[key] = stacked.sum() / sum_data_num
        else:
            result[key] = sum(float(v) for v in values) / sum_data_num
    result["data_num"] = sum_data_num
    return result


def _slice_specs_for_wave(
    spec_list: list[Tensor],
    wave_start_frame: Tensor,
    framed_wave_list: list[Tensor],
) -> list[Tensor]:
    start_list = wave_start_frame.detach().cpu().tolist()
    return [
        spec[start : start + framed_wave.size(0)]
        for spec, start, framed_wave in zip(
            spec_list, start_list, framed_wave_list, strict=True
        )
    ]


def _log_mel_list(
    wave_list: list[Tensor],
    *,
    frame_size: int,
    spec_size: int,
    sampling_rate: int,
) -> list[Tensor]:
    return [
        log_mel_spectrogram(
            wave.unsqueeze(0),
            frame_size=frame_size,
            spec_size=spec_size,
            sampling_rate=sampling_rate,
        )
        .squeeze(0)
        .transpose(0, 1)
        for wave in wave_list
    ]


def _feature_loss(
    fmap_real: list[list[Tensor]], fmap_generated: list[list[Tensor]]
) -> Tensor:
    loss = fmap_real[0][0].new_zeros(())
    for real_layers, generated_layers in zip(fmap_real, fmap_generated, strict=True):
        for real_feature, generated_feature in zip(
            real_layers, generated_layers, strict=True
        ):
            loss = loss + torch.mean(torch.abs(real_feature - generated_feature))
    return loss * 2.0


def _generator_loss(disc_generated_outputs: list[Tensor]) -> Tensor:
    loss = disc_generated_outputs[0].new_zeros(())
    for generated_output in disc_generated_outputs:
        loss = loss + torch.mean((1.0 - generated_output) ** 2)
    return loss


def _discriminator_loss(
    disc_real_outputs: list[Tensor], disc_generated_outputs: list[Tensor]
) -> Tensor:
    loss = disc_real_outputs[0].new_zeros(())
    for real_output, generated_output in zip(
        disc_real_outputs, disc_generated_outputs, strict=True
    ):
        loss = (
            loss
            + torch.mean((1.0 - real_output) ** 2)
            + torch.mean(generated_output**2)
        )
    return loss


class Model(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        predictor: Predictor,
        *,
        mpd: MultiPeriodDiscriminator,
        msd: MultiScaleDiscriminator,
    ):
        super().__init__()
        self.model_config = model_config
        self.acoustic_predictor = predictor.acoustic_predictor
        self.vocoder = predictor.vocoder
        self.mpd = mpd
        self.msd = msd

        self.sampling_rate = predictor.sampling_rate
        self.frame_size = predictor.frame_size

    def forward(
        self, data: DatasetOutput
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        batch_size = len(data["spec"])

        spec1_list, spec2_list = self.acoustic_predictor.forward_list(
            f0_list=data["f0"],
            phoneme_list=data["phoneme"],
            speaker_id=(
                torch.stack(data["speaker_id"])
                if data["speaker_id"] is not None
                else None
            ),
        )

        pred_segment_spec_list = _slice_specs_for_wave(
            spec_list=spec2_list,
            wave_start_frame=torch.stack(data["wave_start_frame"]).long(),
            framed_wave_list=data["framed_wave"],
        )
        pred_wave_list = self.vocoder.forward_list(pred_segment_spec_list)

        return spec1_list, spec2_list, pred_wave_list

    def calc_generator(
        self,
        data: DatasetOutput,
        *,
        spec1_list: list[Tensor],
        spec2_list: list[Tensor],
        pred_wave_list: list[Tensor],
    ) -> GeneratorModelOutput:
        spec_list: list[Tensor] = data["spec"]
        framed_wave_list: list[Tensor] = data["framed_wave"]
        wave_start_frame = torch.stack(data["wave_start_frame"]).long()

        spec1_all = torch.cat(spec1_list, dim=0)
        spec2_all = torch.cat(spec2_list, dim=0)
        target_spec_all = torch.cat(spec_list, dim=0)

        spec_loss1 = l1_loss(spec1_all, target_spec_all)
        spec_loss2 = l1_loss(spec2_all, target_spec_all)

        target_segment_spec_list = _slice_specs_for_wave(
            spec_list=spec_list,
            wave_start_frame=wave_start_frame,
            framed_wave_list=framed_wave_list,
        )
        num_mels = target_spec_all.size(-1)

        pred_wave_spec = torch.cat(
            _log_mel_list(
                pred_wave_list,
                frame_size=self.frame_size,
                spec_size=num_mels,
                sampling_rate=self.sampling_rate,
            ),
            dim=0,
        )
        target_wave_spec = torch.cat(target_segment_spec_list, dim=0)
        wave_spec_loss = l1_loss(pred_wave_spec, target_wave_spec)

        target_wave = torch.stack(
            [frames.reshape(-1) for frames in framed_wave_list], dim=0
        ).unsqueeze(1)
        pred_wave = torch.stack(pred_wave_list, dim=0).unsqueeze(1)

        _, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = self.mpd(target_wave, pred_wave)
        _, y_d_gs_msd, fmap_rs_msd, fmap_gs_msd = self.msd(target_wave, pred_wave)

        adversarial_loss = _generator_loss(y_d_gs_mpd) + _generator_loss(y_d_gs_msd)
        feature_matching_loss = _feature_loss(fmap_rs_mpd, fmap_gs_mpd) + _feature_loss(
            fmap_rs_msd, fmap_gs_msd
        )

        loss = (
            self.model_config.acoustic_loss1_weight * spec_loss1
            + self.model_config.acoustic_loss2_weight * spec_loss2
            + self.model_config.vocoder_spec_loss_weight * wave_spec_loss
            + self.model_config.vocoder_adv_loss_weight * adversarial_loss
            + self.model_config.vocoder_fm_loss_weight * feature_matching_loss
        )

        return GeneratorModelOutput(
            loss=loss,
            spec_loss1=spec_loss1,
            spec_loss2=spec_loss2,
            wave_spec_loss=wave_spec_loss,
            adversarial_loss=adversarial_loss,
            feature_matching_loss=feature_matching_loss,
            data_num=len(spec_list),
        )

    def calc_discriminator(
        self,
        data: DatasetOutput,
        *,
        pred_wave_list: list[Tensor],
    ) -> DiscriminatorModelOutput:
        framed_wave_list: list[Tensor] = data["framed_wave"]

        target_wave = torch.stack(
            [frames.reshape(-1) for frames in framed_wave_list], dim=0
        ).unsqueeze(1)
        pred_wave = torch.stack(pred_wave_list, dim=0).unsqueeze(1).detach()

        y_d_rs_mpd, y_d_gs_mpd, _, _ = self.mpd(target_wave, pred_wave)
        y_d_rs_msd, y_d_gs_msd, _, _ = self.msd(target_wave, pred_wave)

        mpd_loss = _discriminator_loss(y_d_rs_mpd, y_d_gs_mpd)
        msd_loss = _discriminator_loss(y_d_rs_msd, y_d_gs_msd)

        loss = mpd_loss + msd_loss

        return DiscriminatorModelOutput(
            loss=loss,
            mpd_loss=mpd_loss,
            msd_loss=msd_loss,
            data_num=len(framed_wave_list),
        )
