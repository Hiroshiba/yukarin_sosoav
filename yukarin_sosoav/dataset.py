import json
from dataclasses import dataclass
from enum import Enum
from functools import partial
from os import PathLike
from pathlib import Path
from typing import TypedDict

import numpy
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data._utils.collate import default_convert

from .config import DatasetConfig
from .data.phoneme import OjtPhoneme
from .data.sampling_data import SamplingData
from .data.wave import Wave
from .utility.dataset_utility import CachePath

mora_phoneme_list = ["a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N", "cl", "pau"]
voiced_phoneme_list = (
    ["a", "i", "u", "e", "o", "N"]
    + ["n", "m", "y", "r", "w", "g", "z", "j", "d", "b"]
    + ["ny", "my", "ry", "gy", "by", "gw"]
)


@dataclass
class DatasetInput:
    f0: SamplingData
    phoneme: SamplingData
    spec: SamplingData
    silence: SamplingData
    phoneme_list: list[OjtPhoneme] | None
    volume: SamplingData | None
    wave: Wave


@dataclass
class LazyDatasetInput:
    f0_path: PathLike
    phoneme_path: PathLike
    spec_path: PathLike
    silence_path: PathLike
    phoneme_list_path: PathLike | None
    volume_path: PathLike | None
    wave_path: PathLike

    def generate(self):
        return DatasetInput(
            f0=SamplingData.load(self.f0_path),
            phoneme=SamplingData.load(self.phoneme_path),
            spec=SamplingData.load(self.spec_path),
            silence=SamplingData.load(self.silence_path),
            phoneme_list=(
                OjtPhoneme.load_julius_list(self.phoneme_list_path, verify=False)
                if self.phoneme_list_path is not None
                else None
            ),
            volume=(
                SamplingData.load(self.volume_path)
                if self.volume_path is not None
                else None
            ),
            wave=Wave.load(Path(self.wave_path)),
        )


class DatasetOutput(TypedDict):
    f0: Tensor
    phoneme: Tensor
    spec: Tensor
    speaker_id: Tensor | None
    wave: Tensor
    framed_wave: Tensor
    wave_start_frame: Tensor


class F0ProcessMode(str, Enum):
    normal = "normal"
    phoneme_mean = "phoneme_mean"
    mora_mean = "mora_mean"
    voiced_mora_mean = "voiced_mora_mean"


def f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: list[float],
    weight: numpy.ndarray | None,
):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    if weight is None:
        for a in numpy.split(f0, indexes):
            a[:] = numpy.mean(a[a > 0])
    else:
        for a, b in zip(numpy.split(f0, indexes), numpy.split(weight, indexes)):
            a[:] = numpy.sum(a[a > 0] * b[a > 0]) / numpy.sum(b[a > 0])
    f0[numpy.isnan(f0)] = 0
    return f0


def get_notsilence_range(silence: numpy.ndarray, prepost_silence_length: int):
    """
    最初と最後の無音を除去したrangeを返す。
    一番最初や最後が無音でない場合はノイズとみなしてその区間も除去する。
    最小でもprepost_silence_lengthだけは確保する。
    """
    length = len(silence)

    ps = numpy.argwhere(numpy.logical_and(silence[:-1], ~silence[1:]))
    pre_length = ps[0][0] + 1 if len(ps) > 0 else 0
    pre_index = max(0, pre_length - prepost_silence_length)

    ps = numpy.argwhere(numpy.logical_and(~silence[:-1], silence[1:]))
    post_length = length - (ps[-1][0] + 1) if len(ps) > 0 else 0
    post_index = length - max(0, post_length - prepost_silence_length)
    return range(pre_index, post_index)


def preprocess(
    d: DatasetInput,
    prepost_silence_length: int,
    max_sampling_length: int,
    wave_frame_length: int,
    f0_process_mode: F0ProcessMode,
    time_mask_max_second: float,
    time_mask_rate: float,
):
    rate = d.spec.rate

    f0 = d.f0.resample(rate)
    phoneme = d.phoneme.resample(rate)
    silence = d.silence.resample(rate)
    volume = d.volume.resample(rate) if d.volume is not None else None
    spec = d.spec.array
    wave = d.wave.wave
    wave_rate = float(d.wave.sampling_rate)

    frame_rate = float(rate)
    if wave_rate % frame_rate != 0:
        raise ValueError(
            f"wave_rate ({wave_rate}) must be an integer multiple of frame_rate ({frame_rate})"
        )

    frame_size = int(wave_rate / frame_rate)
    trimmed_wave_length = (len(wave) // frame_size) * frame_size
    wave = wave[:trimmed_wave_length]
    framed_wave = wave.reshape(-1, frame_size)

    assert numpy.abs(len(spec) - len(f0)) < 5
    assert numpy.abs(len(spec) - len(phoneme)) < 5
    assert numpy.abs(len(spec) - len(silence)) < 5
    assert volume is None or numpy.abs(len(spec) - len(silence)) < 5
    assert numpy.abs(len(spec) - len(framed_wave)) < 5

    length = min(len(spec), len(f0), len(phoneme), len(silence), len(framed_wave))
    if volume is not None:
        length = min(length, len(volume))

    # 最初と最後の無音を除去する
    notsilence_range = get_notsilence_range(
        silence=silence[:length],
        prepost_silence_length=prepost_silence_length,
    )
    f0 = f0[notsilence_range]
    silence = silence[notsilence_range]
    phoneme = phoneme[notsilence_range]
    spec = spec[notsilence_range]
    framed_wave = framed_wave[notsilence_range]
    if volume is not None:
        volume = volume[notsilence_range]
    length = len(f0)

    # サンプリング長調整
    if length > max_sampling_length:
        offset = numpy.random.randint(length - max_sampling_length + 1)
        offset_slice = slice(offset, offset + max_sampling_length)
        f0 = f0[offset_slice]
        silence = silence[offset_slice]
        phoneme = phoneme[offset_slice]
        spec = spec[offset_slice]
        framed_wave = framed_wave[offset_slice]
        if volume is not None:
            volume = volume[offset_slice]
        length = max_sampling_length

    if f0_process_mode == F0ProcessMode.normal:
        pass
    else:
        assert d.phoneme_list is not None
        weight = volume

        if f0_process_mode == F0ProcessMode.phoneme_mean:
            split_second_list = [p.end for p in d.phoneme_list[:-1]]
        else:
            split_second_list = [
                p.end for p in d.phoneme_list[:-1] if p.phoneme in mora_phoneme_list
            ]

        if f0_process_mode == F0ProcessMode.voiced_mora_mean:
            if weight is None:
                weight = numpy.ones_like(f0)

            for p in d.phoneme_list:
                if p.phoneme not in voiced_phoneme_list:
                    weight[int(p.start * rate) : int(p.end * rate)] = 0

        weight = weight[:length]

        f0 = f0_mean(
            f0=f0,
            rate=rate,
            split_second_list=split_second_list,
            weight=weight,
        )

    if silence.ndim == 2:
        silence = numpy.squeeze(silence, axis=1)

    if time_mask_max_second > 0 and time_mask_rate > 0:
        expected_num = time_mask_rate * length
        num = int(expected_num) + int(
            numpy.random.rand() < (expected_num - int(expected_num))
        )
        for _ in range(num):
            mask_length = numpy.random.randint(int(rate * time_mask_max_second))
            mask_offset = numpy.random.randint(len(f0) - mask_length + 1)
            f0[mask_offset : mask_offset + mask_length] = 0
            phoneme[mask_offset : mask_offset + mask_length] = 0

    # 波形セグメントの切り出し
    if length >= wave_frame_length:
        wave_start_frame = numpy.random.randint(length - wave_frame_length + 1)
        framed_wave = framed_wave[
            wave_start_frame : wave_start_frame + wave_frame_length
        ]
    else:
        wave_start_frame = 0
        pad_length = wave_frame_length - length

        spec_padding = numpy.zeros((pad_length, spec.shape[1]), dtype=spec.dtype)
        spec = numpy.concatenate([spec, spec_padding], axis=0)

        f0_padding = numpy.zeros(pad_length, dtype=f0.dtype)
        f0 = numpy.concatenate([f0, f0_padding], axis=0)

        phoneme_padding = numpy.zeros(
            (pad_length, phoneme.shape[1]), dtype=phoneme.dtype
        )
        phoneme = numpy.concatenate([phoneme, phoneme_padding], axis=0)

        wave_padding = numpy.zeros(
            (pad_length, framed_wave.shape[1]), dtype=framed_wave.dtype
        )
        framed_wave = numpy.concatenate([framed_wave, wave_padding], axis=0)

    output_data = DatasetOutput(
        f0=torch.from_numpy(f0).float(),
        phoneme=torch.from_numpy(phoneme).float(),
        spec=torch.from_numpy(spec).float(),
        speaker_id=None,
        wave=torch.from_numpy(wave).float(),
        framed_wave=torch.from_numpy(framed_wave).float(),
        wave_start_frame=torch.tensor(wave_start_frame).long(),
    )
    return output_data


class FeatureTargetDataset(Dataset):
    def __init__(
        self,
        datas: list[DatasetInput | LazyDatasetInput],
        prepost_silence_length: int,
        max_sampling_length: int,
        wave_frame_length: int,
        f0_process_mode: F0ProcessMode,
        time_mask_max_second: float,
        time_mask_rate: float,
    ):
        self.datas = datas
        self.preprocessor = partial(
            preprocess,
            prepost_silence_length=prepost_silence_length,
            max_sampling_length=max_sampling_length,
            wave_frame_length=wave_frame_length,
            f0_process_mode=f0_process_mode,
            time_mask_max_second=time_mask_max_second,
            time_mask_rate=time_mask_rate,
        )

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        if isinstance(data, LazyDatasetInput):
            data = data.generate()
        return self.preprocessor(data)


class SpeakerFeatureDataset(Dataset):
    def __init__(self, dataset: FeatureTargetDataset, speaker_ids: list[int]):
        assert len(dataset) == len(speaker_ids)
        self.dataset = dataset
        self.speaker_ids = speaker_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        d = self.dataset[i]
        d["speaker_id"] = torch.from_numpy(numpy.array(self.speaker_ids[i])).long()
        return d


class UnbalancedSpeakerFeatureDataset(SpeakerFeatureDataset):
    def __init__(
        self,
        dataset: FeatureTargetDataset,
        speaker_ids: list[int],
        weighted_speaker_id: int,
        weight: int,
    ):
        super().__init__(dataset=dataset, speaker_ids=speaker_ids)

        self.weighted_indexes = [
            i
            for i, speaker_id in enumerate(speaker_ids)
            if speaker_id == weighted_speaker_id
        ]
        self.weight = weight

        assert len(self.weighted_indexes) > 0

    def __len__(self):
        return super().__len__() + len(self.weighted_indexes) * (self.weight - 1)

    def __getitem__(self, i):
        if i >= super().__len__():
            i = self.weighted_indexes[
                (i - super().__len__()) % len(self.weighted_indexes)
            ]
        return super().__getitem__(i)


class TensorWrapperDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return default_convert(self.dataset[i])


def _load_pathlist(path: Path, root_dir: Path) -> dict[str, Path]:
    path_list = [root_dir / p for p in path.read_text().splitlines()]
    return {p.stem: p for p in path_list}


def create_dataset(config: DatasetConfig):
    f0_paths = _load_pathlist(config.f0_pathlist_path, config.root_dir)
    fn_list = sorted(f0_paths.keys())
    assert len(fn_list) > 0

    phoneme_paths = _load_pathlist(config.phoneme_pathlist_path, config.root_dir)
    assert set(fn_list) == set(phoneme_paths.keys())

    spec_paths = _load_pathlist(config.spec_pathlist_path, config.root_dir)
    assert set(fn_list) == set(spec_paths.keys())

    silence_paths = _load_pathlist(config.silence_pathlist_path, config.root_dir)
    assert set(fn_list) == set(silence_paths.keys())

    wave_paths = _load_pathlist(config.wave_pathlist_path, config.root_dir)
    assert set(fn_list) == set(wave_paths.keys())

    phoneme_list_paths: dict[str, Path] | None = None
    if config.phoneme_list_pathlist_path is not None:
        phoneme_list_paths = _load_pathlist(
            config.phoneme_list_pathlist_path, config.root_dir
        )
        fn_list = sorted(phoneme_list_paths.keys())
        assert len(fn_list) > 0

    volume_paths: dict[str, Path] | None = None
    if config.volume_pathlist_path is not None:
        volume_paths = _load_pathlist(config.volume_pathlist_path, config.root_dir)
        fn_list = sorted(volume_paths.keys())
        assert len(fn_list) > 0

    speaker_ids: dict[str, int] | None = None
    if config.speaker_dict_path is not None:
        fn_each_speaker: dict[str, list[str]] = json.loads(
            config.speaker_dict_path.read_text()
        )
        assert config.num_speaker == len(fn_each_speaker)

        speaker_ids = {
            fn: speaker_id
            for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
            for fn in fns
        }
        assert set(fn_list).issubset(set(speaker_ids.keys()))

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    test_num = config.test_num
    trains = fn_list[test_num:]
    tests = fn_list[:test_num]

    def _dataset(fns, for_eval=False):
        inputs = [
            LazyDatasetInput(
                f0_path=CachePath(f0_paths[fn]),
                phoneme_path=CachePath(phoneme_paths[fn]),
                spec_path=CachePath(spec_paths[fn]),
                silence_path=CachePath(silence_paths[fn]),
                phoneme_list_path=(
                    CachePath(phoneme_list_paths[fn])
                    if phoneme_list_paths is not None
                    else None
                ),
                volume_path=(
                    CachePath(volume_paths[fn]) if volume_paths is not None else None
                ),
                wave_path=CachePath(wave_paths[fn]),
            )
            for fn in fns
        ]

        dataset = FeatureTargetDataset(
            datas=inputs,
            prepost_silence_length=config.prepost_silence_length,
            max_sampling_length=config.max_sampling_length,
            wave_frame_length=config.wave_frame_length,
            f0_process_mode=F0ProcessMode(config.f0_process_mode),
            time_mask_max_second=(config.time_mask_max_second if not for_eval else 0),
            time_mask_rate=(config.time_mask_rate if not for_eval else 0),
        )

        if speaker_ids is not None:
            if config.weighted_speaker_id is None or config.speaker_weight is None:
                dataset = SpeakerFeatureDataset(
                    dataset=dataset,
                    speaker_ids=[speaker_ids[fn] for fn in fns],
                )
            else:
                dataset = UnbalancedSpeakerFeatureDataset(
                    dataset=dataset,
                    speaker_ids=[speaker_ids[fn] for fn in fns],
                    weighted_speaker_id=config.weighted_speaker_id,
                    weight=config.speaker_weight,
                )

        dataset = TensorWrapperDataset(dataset)

        if for_eval:
            dataset = ConcatDataset([dataset] * config.test_trial_num)

        return dataset

    valid_dataset = (
        create_validation_dataset(config) if config.valid_num is not None else None
    )

    return {
        "train": _dataset(trains),
        "test": _dataset(tests, for_eval=False),
        "eval": _dataset(tests, for_eval=True),
        "valid": valid_dataset,
    }


def create_validation_dataset(config: DatasetConfig):
    assert config.valid_f0_pathlist_path is not None
    assert config.valid_phoneme_pathlist_path is not None
    assert config.valid_spec_pathlist_path is not None
    assert config.valid_silence_pathlist_path is not None
    assert config.valid_trial_num is not None

    f0_paths = _load_pathlist(config.valid_f0_pathlist_path, config.root_dir)
    fn_list = sorted(f0_paths.keys())
    assert len(fn_list) > 0

    phoneme_paths = _load_pathlist(config.valid_phoneme_pathlist_path, config.root_dir)
    assert set(fn_list) == set(phoneme_paths.keys())

    spec_paths = _load_pathlist(config.valid_spec_pathlist_path, config.root_dir)
    assert set(fn_list) == set(spec_paths.keys())

    silence_paths = _load_pathlist(config.valid_silence_pathlist_path, config.root_dir)
    assert set(fn_list) == set(silence_paths.keys())

    if config.valid_wave_pathlist_path is not None:
        wave_paths = _load_pathlist(config.valid_wave_pathlist_path, config.root_dir)
    else:
        wave_paths = _load_pathlist(config.wave_pathlist_path, config.root_dir)
    assert set(fn_list) == set(wave_paths.keys())

    phoneme_list_paths: dict[str, Path] | None = None
    if config.valid_phoneme_list_pathlist_path is not None:
        phoneme_list_paths = _load_pathlist(
            config.valid_phoneme_list_pathlist_path, config.root_dir
        )
        fn_list = sorted(phoneme_list_paths.keys())
        assert len(fn_list) > 0

    volume_paths: dict[str, Path] | None = None
    if config.valid_volume_pathlist_path is not None:
        volume_paths = _load_pathlist(
            config.valid_volume_pathlist_path, config.root_dir
        )
        fn_list = sorted(volume_paths.keys())
        assert len(fn_list) > 0

    speaker_ids: dict[str, int] | None = None
    if config.valid_speaker_dict_path is not None:
        fn_each_speaker: dict[str, list[str]] = json.loads(
            config.valid_speaker_dict_path.read_text()
        )

        speaker_ids = {
            fn: speaker_id
            for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
            for fn in fns
        }
        assert set(fn_list).issubset(set(speaker_ids.keys()))

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    valids = fn_list[: config.valid_num]

    inputs = [
        LazyDatasetInput(
            f0_path=CachePath(f0_paths[fn]),
            phoneme_path=CachePath(phoneme_paths[fn]),
            spec_path=CachePath(spec_paths[fn]),
            silence_path=CachePath(silence_paths[fn]),
            phoneme_list_path=(
                CachePath(phoneme_list_paths[fn])
                if phoneme_list_paths is not None
                else None
            ),
            volume_path=(
                CachePath(volume_paths[fn]) if volume_paths is not None else None
            ),
            wave_path=CachePath(wave_paths[fn]),
        )
        for fn in valids
    ]

    dataset = FeatureTargetDataset(
        datas=inputs,
        prepost_silence_length=config.prepost_silence_length,
        max_sampling_length=config.max_sampling_length,
        wave_frame_length=config.wave_frame_length,
        f0_process_mode=F0ProcessMode(config.f0_process_mode),
        time_mask_max_second=0,
        time_mask_rate=0,
    )

    if speaker_ids is not None:
        if config.weighted_speaker_id is None or config.speaker_weight is None:
            dataset = SpeakerFeatureDataset(
                dataset=dataset,
                speaker_ids=[speaker_ids[fn] for fn in valids],
            )
        else:
            dataset = UnbalancedSpeakerFeatureDataset(
                dataset=dataset,
                speaker_ids=[speaker_ids[fn] for fn in valids],
                weighted_speaker_id=config.weighted_speaker_id,
                weight=config.speaker_weight,
            )

    dataset = TensorWrapperDataset(dataset)
    dataset = ConcatDataset([dataset] * config.valid_trial_num)
    return dataset
