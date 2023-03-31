import json


def get_model_zoo():
    with open("src/models.json") as f:
        model_zoo = json.load(f)
    return model_zoo
