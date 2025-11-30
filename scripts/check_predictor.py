import argparse
from pathlib import Path

import torch
import yaml

from yukarin_sosoav.config import Config
from yukarin_sosoav.network.predictor import create_predictor


def check_predictor(config_yaml_path: Path):
    with config_yaml_path.open() as f:
        config_dict = yaml.safe_load(f)

    config = Config.from_dict(config_dict)

    # predictor
    predictor = create_predictor(config.network)
    torch.jit.script(predictor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    check_predictor(**vars(parser.parse_args()))
