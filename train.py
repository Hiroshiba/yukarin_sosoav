import argparse
from pathlib import Path

import torch
import yaml
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from yukarin_sosoav.config import Config
from yukarin_sosoav.dataset import create_dataset
from yukarin_sosoav.evaluator import Evaluator
from yukarin_sosoav.generator import Generator
from yukarin_sosoav.model import (
    DiscriminatorModelOutput,
    GeneratorModelOutput,
    Model,
    reduce_result,
)
from yukarin_sosoav.network.discriminator import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from yukarin_sosoav.network.predictor import create_predictor
from yukarin_sosoav.utility.pytorch_utility import (
    collate_list,
    detach_cpu,
    init_weights,
    make_optimizer,
    make_scheduler,
    to_device,
)
from yukarin_sosoav.utility.train_utility import Logger, SaveManager


def train(config_yaml_path: Path, output_dir: Path):
    # config
    with config_yaml_path.open() as f:
        config_dict = yaml.safe_load(f)
    config = Config.from_dict(config_dict)
    config.add_git_info()

    # dataset
    def _create_loader(dataset, for_train: bool, for_eval: bool):
        if dataset is None:
            return None
        batch_size = (
            config.train.eval_batch_size if for_eval else config.train.batch_size
        )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.train.num_processes,
            collate_fn=collate_list,
            pin_memory=config.train.use_gpu,
            drop_last=for_train,
            timeout=0 if config.train.num_processes == 0 else 30,
            persistent_workers=config.train.num_processes > 0,
        )

    datasets = create_dataset(config.dataset)
    train_loader = _create_loader(datasets["train"], for_train=True, for_eval=False)
    test_loader = _create_loader(datasets["test"], for_train=False, for_eval=False)
    eval_loader = _create_loader(datasets["eval"], for_train=False, for_eval=True)
    valid_loader = _create_loader(datasets["valid"], for_train=False, for_eval=True)

    # predictor
    predictor = create_predictor(config.network)
    device = "cuda" if config.train.use_gpu else "cpu"
    if config.train.pretrained_predictor_path is not None:
        state_dict = torch.load(
            config.train.pretrained_predictor_path, map_location=device
        )
        predictor.load_state_dict(state_dict)
    if config.train.pretrained_vocoder_path is not None:
        state_dict = torch.load(
            config.train.pretrained_vocoder_path, map_location=device
        )
        if "generator" in state_dict:
            predictor.vocoder.load_state_dict(state_dict["generator"])
        else:
            raise ValueError(
                "pretrained_vocoder_path state_dict does not have 'generator' key"
            )
    print("predictor:", predictor)

    # model
    mpd = MultiPeriodDiscriminator(
        initial_channel=config.network.discriminator.mpd_initial_channel
    )
    msd = MultiScaleDiscriminator(
        initial_channel=config.network.discriminator.msd_initial_channel
    )
    if config.train.pretrained_discriminator_path is not None:
        state_dict = torch.load(
            config.train.pretrained_discriminator_path, map_location=device
        )
        if "mpd" in state_dict and "msd" in state_dict:
            mpd.load_state_dict(state_dict["mpd"])
            msd.load_state_dict(state_dict["msd"])
        else:
            raise ValueError(
                "pretrained_discriminator_path state_dict does not have 'mpd' and 'msd' keys"
            )

    model = Model(model_config=config.model, predictor=predictor, mpd=mpd, msd=msd)
    if config.train.weight_initializer is not None:
        init_weights(model, name=config.train.weight_initializer)
    model.to(device)
    model.train()

    # evaluator
    generator = Generator(
        config=config, predictor=predictor, use_gpu=config.train.use_gpu
    )
    evaluator = Evaluator(generator=generator)

    # optimizer
    generator_optimizer = make_optimizer(
        config_dict=config.train.generator_optimizer, model=predictor
    )
    discriminator_optimizer = make_optimizer(
        config_dict=config.train.discriminator_optimizer,
        model=torch.nn.ModuleList([mpd, msd]),
    )
    generator_scaler = GradScaler(device, enabled=config.train.use_amp)
    discriminator_scaler = GradScaler(device, enabled=config.train.use_amp)

    # logger
    logger = Logger(
        config_dict=config_dict,
        project_category=config.project.category,
        project_name=config.project.name,
        output_dir=output_dir,
    )

    # snapshot
    snapshot_path = output_dir / "snapshot.pth"
    if not snapshot_path.exists():
        iteration = -1
        epoch = -1
    else:
        snapshot = torch.load(snapshot_path, map_location=device)

        model.load_state_dict(snapshot["model"])
        generator_optimizer.load_state_dict(snapshot["generator_optimizer"])
        discriminator_optimizer.load_state_dict(snapshot["discriminator_optimizer"])
        generator_scaler.load_state_dict(snapshot["generator_scaler"])
        discriminator_scaler.load_state_dict(snapshot["discriminator_scaler"])
        logger.load_state_dict(snapshot["logger"])

        iteration = snapshot["iteration"]
        epoch = snapshot["epoch"]
        print(f"Loaded snapshot from {snapshot_path} (epoch: {epoch})")

    # scheduler
    generator_scheduler = None
    if config.train.generator_scheduler is not None:
        generator_scheduler = make_scheduler(
            config_dict=config.train.generator_scheduler,
            optimizer=generator_optimizer,
            last_epoch=epoch,
        )
    discriminator_scheduler = None
    if config.train.discriminator_scheduler is not None:
        discriminator_scheduler = make_scheduler(
            config_dict=config.train.discriminator_scheduler,
            optimizer=discriminator_optimizer,
            last_epoch=epoch,
        )

    # save
    save_manager = SaveManager(
        predictor=predictor,
        prefix="predictor_",
        output_dir=output_dir,
        top_num=config.train.model_save_num,
        last_num=config.train.model_save_num,
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    with (output_dir / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(config.to_dict(), f)

    # loop
    assert config.train.eval_epoch % config.train.log_epoch == 0
    assert config.train.snapshot_epoch % config.train.eval_epoch == 0

    for _ in range(config.train.stop_epoch):
        epoch += 1
        if epoch > config.train.stop_epoch:
            break

        model.train()

        train_generator_results: list[GeneratorModelOutput] = []
        train_discriminator_results: list[DiscriminatorModelOutput] = []

        for batch in train_loader:
            iteration += 1

            # Discriminatorの更新
            with autocast(device, enabled=config.train.use_amp):
                batch = to_device(batch, device, non_blocking=True)
                spec1_list, spec2_list, pred_wave_list = model(batch)

                discriminator_result = model.calc_discriminator(
                    batch, pred_wave_list=pred_wave_list
                )
                discriminator_loss = discriminator_result["loss"]

            if discriminator_loss.isnan():
                raise ValueError("discriminator loss is NaN")

            discriminator_optimizer.zero_grad()
            discriminator_scaler.scale(discriminator_loss).backward()
            discriminator_scaler.step(discriminator_optimizer)
            discriminator_scaler.update()

            # Generatorの更新
            with autocast(device, enabled=config.train.use_amp):
                generator_result = model.calc_generator(
                    batch,
                    spec1_list=spec1_list,
                    spec2_list=spec2_list,
                    pred_wave_list=pred_wave_list,
                )
                generator_loss = generator_result["loss"]

            if generator_loss.isnan():
                raise ValueError("generator loss is NaN")

            generator_optimizer.zero_grad()
            generator_scaler.scale(generator_loss).backward()
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()

            train_generator_results.append(detach_cpu(generator_result))
            train_discriminator_results.append(detach_cpu(discriminator_result))

        if generator_scheduler is not None:
            generator_scheduler.step()
        if discriminator_scheduler is not None:
            discriminator_scheduler.step()

        if epoch % config.train.log_epoch == 0:
            model.eval()

            with torch.inference_mode():
                test_generator_results: list[GeneratorModelOutput] = []
                test_discriminator_results: list[DiscriminatorModelOutput] = []
                for batch in test_loader:
                    batch = to_device(batch, device, non_blocking=True)
                    spec1_list, spec2_list, pred_wave_list = model(batch)
                    discriminator_result = model.calc_discriminator(
                        batch, pred_wave_list=pred_wave_list
                    )
                    generator_result = model.calc_generator(
                        batch,
                        spec1_list=spec1_list,
                        spec2_list=spec2_list,
                        pred_wave_list=pred_wave_list,
                    )
                    test_generator_results.append(detach_cpu(generator_result))
                    test_discriminator_results.append(detach_cpu(discriminator_result))

                summary = {
                    "train": {
                        "generator": reduce_result(train_generator_results),
                        "discriminator": reduce_result(train_discriminator_results),
                    },
                    "test": {
                        "generator": reduce_result(test_generator_results),
                        "discriminator": reduce_result(test_discriminator_results),
                    },
                    "iteration": iteration,
                    "lr": generator_optimizer.param_groups[0]["lr"],
                    "lr_discriminator": discriminator_optimizer.param_groups[0]["lr"],
                }

                if epoch % config.train.eval_epoch == 0:
                    if eval_loader is not None:
                        eval_results = []
                        for batch in eval_loader:
                            batch = to_device(batch, device, non_blocking=True)
                            result = evaluator(batch)
                            eval_results.append(detach_cpu(result))
                        summary["eval"] = reduce_result(eval_results)

                    if valid_loader is not None:
                        valid_results = []
                        for batch in valid_loader:
                            batch = to_device(batch, device, non_blocking=True)
                            result = evaluator(batch)
                            valid_results.append(detach_cpu(result))
                        summary["valid"] = reduce_result(valid_results)

                    if epoch % config.train.snapshot_epoch == 0:
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "generator_optimizer": generator_optimizer.state_dict(),
                                "discriminator_optimizer": discriminator_optimizer.state_dict(),
                                "generator_scaler": generator_scaler.state_dict(),
                                "discriminator_scaler": discriminator_scaler.state_dict(),
                                "logger": logger.state_dict(),
                                "iteration": iteration,
                                "epoch": epoch,
                            },
                            snapshot_path,
                        )

                        if "valid" in summary:
                            save_manager.save(
                                value=float(summary["valid"]["value"]),
                                step=epoch,
                                judge="min",
                            )

                logger.log(summary=summary, step=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    train(**vars(parser.parse_args()))
