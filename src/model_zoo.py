import json

from pathlib import Path


def get_model_zoo() -> list:
    with open(Path(__file__).parent.joinpath("models.json")) as f:
        model_zoo = json.load(f)
    return model_zoo
