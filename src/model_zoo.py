import json


def get_model_zoo() -> list:
    with open("src/models.json") as f:
        model_zoo = json.load(f)
    return model_zoo
